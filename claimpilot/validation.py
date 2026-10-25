"""추가 평가만 수행한다. 모델 저장 및 기존 MVP 산출물 갱신은 하지 않는다."""
import hashlib
import json
from itertools import combinations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from claimpilot.modeling import FEATURES, TARGET, choose_threshold, evaluate, feature_columns, prepare

SEED = 42
CANDIDATES = (
    {'name': 'Logistic Regression C=0.1', 'family': 'lr', 'C': .1},
    {'name': 'Logistic Regression C=1.0', 'family': 'lr', 'C': 1.},
    {'name': 'CatBoost depth=4', 'family': 'catboost', 'depth': 4},
    {'name': 'CatBoost depth=6', 'family': 'catboost', 'depth': 6},
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_input_groups(df, features=FEATURES):
    """원문 열값 tuple의 정확한 동등성으로 묶는다. 해시 충돌에 의존하지 않는다."""
    return pd.factorize(pd.MultiIndex.from_frame(df[list(features)]), sort=False)[0]


def audit_input_groups(df, features=FEATURES):
    groups = raw_input_groups(df, features)
    work = df[list(features)].copy()
    work.insert(0, 'input_group_id', groups)
    work['label'] = df[TARGET].to_numpy()
    grouped = work.groupby('input_group_id', sort=True, dropna=False)
    table = grouped[list(features)].first()
    table['size'] = grouped.size()
    table['positives'] = grouped.label.sum().astype(int)
    table['negatives'] = table['size'] - table['positives']
    table['label_count'] = grouped.label.nunique()
    table['conflicting_labels'] = table.label_count.gt(1)
    table['duplicate_excess'] = table['size'] - 1
    table = table.reset_index()
    duplicate_rows = int(df.duplicated(subset=list(features)).sum())
    assert duplicate_rows == int(table.duplicate_excess.sum())
    summary = {
        'definition': '승인된 19개 입력의 원문 값 조합. 실제 고객/사건 그룹이 아님.',
        'rows': len(df), 'unique_input_groups': len(table),
        'duplicate_rows': duplicate_rows,
        'duplicate_groups': int(table['size'].gt(1).sum()),
        'rows_in_duplicate_groups': int(table.loc[table['size'].gt(1), 'size'].sum()),
        'max_group_size': int(table['size'].max()),
        'size_distribution': {str(k): int(v) for k, v in table['size'].value_counts().sort_index().items()},
        'conflicting_groups': int(table.conflicting_labels.sum()),
        'rows_in_conflicting_groups': int(table.loc[table.conflicting_labels, 'size'].sum()),
        'previous_observation': {'duplicate_rows': 51, 'conflicting_groups': 2},
        'previous_observation_matches': duplicate_rows == 51 and int(table.conflicting_labels.sum()) == 2,
        'grouping_uses_labels': False,
    }
    return groups, table, summary


def group_split(indices, groups, seed=SEED):
    """약 60/20/20, 원문 입력 그룹을 모든 split 사이에서 격리한다."""
    indices = np.asarray(indices, dtype=int)
    groups = np.asarray(groups)
    dev_pos, test_pos = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(indices, groups=groups[indices]))
    dev, test = indices[dev_pos], indices[test_pos]
    train_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=seed).split(dev, groups=groups[dev]))
    result = {'train': dev[train_pos], 'validation': dev[val_pos], 'test': test}
    assert_disjoint_groups(result, groups)
    return result


def assert_disjoint_groups(splits, groups):
    for a, b in combinations(splits, 2):
        if set(np.asarray(groups)[splits[a]]) & set(np.asarray(groups)[splits[b]]):
            raise ValueError('입력 그룹 분할 중복: %s/%s' % (a, b))


def temporal_split(df, groups=None, seed=SEED):
    """순수 연도 cutoff. 미래 입력/라벨로 과거 모집단을 고르지 않는다."""
    years = df.Year.to_numpy()
    if not set(np.unique(years)).issubset({1994, 1995, 1996}):
        raise ValueError('승인되지 않은 연도')
    early = np.flatnonzero(np.isin(years, [1994, 1995]))
    future = np.flatnonzero(years == 1996)
    if not len(future) or not len(early):
        raise ValueError('과거와 1996 표본이 모두 필요합니다.')
    train, validation = train_test_split(early, test_size=.25,
                                        stratify=df.iloc[early][TARGET], random_state=seed)
    return {'train': train, 'validation': validation, 'test': future}, {
        'historical_rows': len(early), 'purged_historical_rows': 0,
        'uses_future_input_to_select_historical_rows': False,
        'policy': '1994~1995 전체 모집단을 seed42 stratified 75/25 분할. 1996 입력/라벨로 과거 행을 선택하거나 제외하지 않는다. 입력그룹 중복은 평가 후 감사만 수행.',
    }


def split_diagnostics(df, groups, splits):
    parts = {}
    for name, idx in splits.items():
        idx = np.asarray(idx, dtype=int)
        y = df.iloc[idx][TARGET]
        parts[name] = {'n': len(idx), 'positives': int(y.sum()), 'prevalence': float(y.mean()),
                       'unique_groups': len(set(np.asarray(groups)[idx])),
                       'years': {str(k): int(v) for k, v in df.iloc[idx].Year.value_counts().sort_index().items()},
                       'ordered_claim_id_sha256': hashlib.sha256(json.dumps(df.iloc[idx].PolicyNumber.astype(int).tolist()).encode()).hexdigest()}
    intersections = {}
    row_intersections = {}
    for a, b in combinations(splits, 2):
        key = a + '__' + b
        intersections[key] = len(set(np.asarray(groups)[splits[a]]) & set(np.asarray(groups)[splits[b]]))
        row_intersections[key] = len(set(splits[a]) & set(splits[b]))
    dev = np.concatenate([splits['train'], splits['validation']])
    intersections['development__test'] = len(set(np.asarray(groups)[dev]) & set(np.asarray(groups)[splits['test']]))
    return {'splits': parts, 'group_intersections': intersections, 'row_intersections': row_intersections,
            'assigned_rows': sum(len(v) for v in splits.values())}


def original_random_split(df, manifest=None):
    """기존 순서를 보존한다. 동점 점수의 top20 순서까지 동일해야 한다."""
    if manifest is not None:
        if manifest['seed'] != SEED or manifest['features'] != FEATURES:
            raise ValueError('기존 매니페스트 설정 불일치')
        lookup = {int(claim_id): i for i, claim_id in enumerate(df.PolicyNumber)}
        splits = {name: np.asarray([lookup[int(row['claim_id'])] for row in manifest['splits'][name]], dtype=int)
                  for name in ('train', 'validation', 'test')}
    else:
        dev, test = train_test_split(np.arange(len(df)), test_size=.2, stratify=df[TARGET], random_state=SEED)
        train, val = train_test_split(dev, test_size=.25, stratify=df.iloc[dev][TARGET], random_state=SEED)
        splits = {'train': train, 'validation': val, 'test': test}
    joined = np.concatenate(list(splits.values()))
    if len(joined) != len(df) or len(np.unique(joined)) != len(df):
        raise ValueError('기존 분할의 행 누락 또는 중복')
    return splits


def make_lr(C=.1, class_weight='balanced'):
    return Pipeline([('encode', OneHotEncoder(handle_unknown='ignore')),
                     ('clf', LogisticRegression(C=C, class_weight=class_weight, max_iter=1500, random_state=SEED))])


def fit_candidate(spec, X_train, y_train, X_validation, y_validation):
    if spec['family'] == 'lr':
        model = make_lr(C=spec['C'])
        model.fit(X_train, y_train)
    else:
        model = CatBoostClassifier(iterations=450, depth=spec['depth'], learning_rate=.05,
                                   loss_function='Logloss', eval_metric='PRAUC', random_seed=SEED,
                                   verbose=False, allow_writing_files=False, thread_count=4)
        model.fit(X_train, y_train, cat_features=list(X_train.columns),
                  eval_set=(X_validation, y_validation), early_stopping_rounds=60)
    return model


def select_candidates(X, y, train, validation, specs=CANDIDATES):
    """API 자체가 test를 받지 않는다. 선택/임계값/early stopping 모두 과거 val만."""
    fitted = []
    rows = []
    for spec in specs:
        model = fit_candidate(spec, X.iloc[train], y.iloc[train], X.iloc[validation], y.iloc[validation])
        scores = model.predict_proba(X.iloc[validation])[:, 1]
        threshold = choose_threshold(y.iloc[validation], scores)
        row = {'name': spec['name'], 'params': dict(spec), 'threshold': threshold,
               'validation': evaluate(y.iloc[validation], scores, threshold)}
        if spec['family'] == 'catboost':
            row['params']['best_iteration'] = int(model.best_iteration_)
        rows.append(row)
        fitted.append(model)
    selected_idx = max(range(len(rows)), key=lambda i: (rows[i]['validation']['ap'], rows[i]['validation']['capture_at_20']))
    return fitted[selected_idx], rows, selected_idx


def run_evaluation(X, y, splits):
    # test 조회는 선택을 고정한 뒤만 수행한다. train만 fit하며 전체 재학습하지 않는다.
    model, candidates, selected_idx = select_candidates(X, y, splits['train'], splits['validation'])
    selected = candidates[selected_idx]
    scores = model.predict_proba(X.iloc[splits['test']])[:, 1]
    return {'selected_model': selected['name'], 'selected_candidate_index': selected_idx,
            'threshold': selected['threshold'], 'candidates': candidates,
            'selection_rule': 'validation AP 최대, 동률이면 validation capture_at_20. test로 재선정하지 않음.',
            'threshold_rule': '선택된 모델의 validation F1 최대. test에서는 고정.',
            'test': evaluate(y.iloc[splits['test']], scores, selected['threshold']),
            'test_access_after_selection': True, 'refit_on_train_plus_validation': False,
            'test_evaluated_candidates': 1, 'model_saved': False}


def weight_diagnostic(X, y, splits):
    rows = []
    for weight in ('balanced', None):
        model = make_lr(C=.1, class_weight=weight)
        model.fit(X.iloc[splits['train']], y.iloc[splits['train']])
        val_scores = model.predict_proba(X.iloc[splits['validation']])[:, 1]
        threshold = choose_threshold(y.iloc[splits['validation']], val_scores)
        rows.append({'class_weight': weight, 'C': .1, 'threshold': threshold,
                     'validation': evaluate(y.iloc[splits['validation']], val_scores, threshold),
                     'test': evaluate(y.iloc[splits['test']], model.predict_proba(X.iloc[splits['test']])[:, 1], threshold)})
    return {'purpose': '사전지정 LR C=.1 balanced vs None 진단 비교만 수행. production 선정·조정에 사용하지 않음.',
            'threshold_rule': '각 모델별 validation F1 최대, test 관측 전 고정',
            'models': rows, 'production_model_selection': False,
            'test_delta_none_minus_balanced': {key: rows[1]['test'][key] - rows[0]['test'][key]
                                              for key in ('ap', 'precision', 'recall', 'f1', 'brier', 'capture_at_20', 'precision_at_20', 'lift_at_20')}}


def compare_reference(result, old_metrics):
    old = next(row for row in old_metrics['models'] if row['name'] == old_metrics['selected_model'])
    keys = ('ap', 'precision', 'recall', 'f1', 'brier', 'capture_at_20', 'precision_at_20', 'lift_at_20')
    return {'reference_file': 'reports/metrics.json', 'reference_selected_model': old['name'],
            'reference_random_test': old['test'],
            'delta_vs_reference_random_test': {k: result['test'][k] - old['test'][k] for k in keys},
            'interpretation': '분할·대상 표본이 달라 성능 향상/통계적 유의성의 증거가 아닌 기술적 비교. 기존 지표는 변경하지 않음.'}
