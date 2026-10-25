"""작은 fixture의 계약 및 (존재할 경우) 실제 phase2 보고서 검증."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import claimpilot.validation as v
from claimpilot.modeling import FEATURES, TARGET, prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    rows = []
    for g in range(90):
        for repeat in range(2):
            row = {col: '%s_%s' % (col, g) for col in FEATURES}
            row.update({TARGET: int(g % 3 == 0), 'PolicyNumber': len(rows) + 1,
                        'Year': 1994 if g < 30 else 1995 if g < 60 else 1996})
            rows.append(row)
    # 미래와 입력이 동일한 과거 행. 시간평가에서는 제거하지 않고 사후 감사한다.
    copy = dict(rows[-1]); copy['Year'] = 1995; copy['PolicyNumber'] = len(rows) + 1
    rows.append(copy)
    # 같은 원문 입력에 상반된 라벨을 하나 넣는다.
    rows[1][TARGET] = 1 - rows[0][TARGET]
    return pd.DataFrame(rows)


def test_raw_group_audit(frame):
    groups, table, summary = v.audit_input_groups(frame)
    assert len(table) == 90
    assert summary['duplicate_rows'] == 91
    assert summary['duplicate_groups'] == 90
    assert summary['conflicting_groups'] == 1
    assert summary['max_group_size'] == 3
    assert table['size'].sum() == len(frame)
    assert groups[0] == groups[1]
    assert table.loc[table.conflicting_labels, 'positives'].tolist() == [1]
    changed = frame.copy(); changed[TARGET] = 1 - changed[TARGET]
    np.testing.assert_array_equal(v.raw_input_groups(changed), groups)


def test_raw_not_prepared_sentinel():
    df = pd.DataFrame({'value': [None, '__MISSING__', 'a|b', 'a|b'], TARGET: [0, 1, 0, 1]})
    groups = v.raw_input_groups(df, ['value'])
    assert groups[0] != groups[1]
    assert groups[2] == groups[3]


def test_group_split_disjoint_reproducible(frame):
    groups = v.raw_input_groups(frame)
    splits = v.group_split(np.arange(len(frame)), groups)
    again = v.group_split(np.arange(len(frame)), groups)
    assert sorted(np.concatenate(list(splits.values()))) == list(range(len(frame)))
    for name in splits:
        np.testing.assert_array_equal(splits[name], again[name])
    v.assert_disjoint_groups(splits, groups)
    d = v.split_diagnostics(frame, groups, splits)
    assert set(d['group_intersections'].values()) == {0}
    assert set(d['row_intersections'].values()) == {0}
    with pytest.raises(ValueError):
        v.assert_disjoint_groups({'train': np.array([0]), 'test': np.array([1])}, groups)


def test_temporal_cutoff_no_future_input_or_label_dependency(frame):
    groups = v.raw_input_groups(frame)
    splits, sampling = v.temporal_split(frame, groups)
    assert sampling['purged_historical_rows'] == 0
    assert not sampling['uses_future_input_to_select_historical_rows']
    assert frame.iloc[splits['test']].Year.eq(1996).all()
    historical = np.r_[splits['train'], splits['validation']]
    assert set(historical) == set(np.flatnonzero(frame.Year.isin([1994, 1995])))
    assert len(splits['test']) == 60
    assert len(frame) == sum(map(len, splits.values()))
    changed = frame.copy()
    changed.loc[changed.Year.eq(1996), TARGET] = 1 - changed.loc[changed.Year.eq(1996), TARGET]
    for col in FEATURES:
        changed.loc[changed.Year.eq(1996), col] = 'unrelated_future_value'
    again, again_sampling = v.temporal_split(changed, v.raw_input_groups(changed))
    for name in splits:
        np.testing.assert_array_equal(splits[name], again[name])
    assert sampling == again_sampling
    diag = v.split_diagnostics(frame, groups, splits)
    assert diag['group_intersections']['development__test'] == 1


def test_selection_uses_only_train_validation(frame, monkeypatch):
    groups = v.raw_input_groups(frame)
    splits, _ = v.temporal_split(frame, groups)
    X = prepare(frame, FEATURES); y = frame[TARGET]
    observed = []

    class Fake:
        def __init__(self, c):
            self.c = c
        def predict_proba(self, x):
            assert set(x.index) <= set(splits['validation'])
            scores = .1 + .8 * (x.index.to_numpy() % 4 == 0)
            if self.c == 1:
                scores = 1 - scores
            return np.column_stack([1 - scores, scores])

    def fit(spec, train_x, train_y, val_x, val_y):
        assert set(train_x.index) == set(splits['train'])
        assert set(val_x.index) == set(splits['validation'])
        assert frame.loc[train_x.index].Year.max() < 1996
        assert frame.loc[val_x.index].Year.max() < 1996
        observed.append((train_y.copy(), val_y.copy()))
        return Fake(spec['C'])

    monkeypatch.setattr(v, 'fit_candidate', fit)
    specs = v.CANDIDATES[:2]
    _, before, selected = v.select_candidates(X, y, splits['train'], splits['validation'], specs)
    changed = y.copy(); changed.iloc[splits['test']] = 1 - changed.iloc[splits['test']]
    _, after, selected_after = v.select_candidates(X, changed, splits['train'], splits['validation'], specs)
    assert before == after
    assert selected == selected_after
    assert len(observed) == 4
    assert 'test' not in inspect.signature(v.select_candidates).parameters


def test_original_random_manifest_preserves_order(frame):
    splits = v.original_random_split(frame)
    manifest = {'seed': 42, 'features': FEATURES, 'splits': {
        name: [{'claim_id': int(frame.iloc[i].PolicyNumber)} for i in indices] for name, indices in splits.items()}}
    restored = v.original_random_split(frame, manifest)
    for name in splits:
        np.testing.assert_array_equal(splits[name], restored[name])
    manifest['splits']['test'][0] = manifest['splits']['train'][0]
    with pytest.raises(ValueError):
        v.original_random_split(frame, manifest)


def test_lr_weight_configuration():
    for weight in ('balanced', None):
        model = v.make_lr(.1, weight)
        assert model.named_steps['clf'].C == .1
        assert model.named_steps['clf'].class_weight == weight
        assert model.named_steps['clf'].random_state == 42
        assert model.named_steps['encode'].handle_unknown == 'ignore'


@pytest.fixture
def output():
    path = ROOT/'reports/phase2_validation.json'
    if not path.exists():
        pytest.skip('python scripts/validate_phase2.py 실행 후 실제 산출물 검증')
    return json.loads(path.read_text())


@pytest.fixture
def actual_data():
    path=ROOT/'data/raw/fraud_oracle.csv'
    if not path.exists():pytest.skip('공개 체크아웃에는 원본 데이터 없음: 다운로드·학습 후 통합 검증')
    return pd.read_csv(path)


def test_real_artifact_configuration_protected(output,actual_data):
    assert {'configuration', 'split_diagnostics', 'group_audit', 'group_evaluation', 'temporal_independent', 'weight_diagnostic', 'limitations'} <= set(output)
    c = output['configuration']
    assert c['features'] == FEATURES and c['feature_count'] == 19 and c['seed'] == 42
    assert c['catboost']['thread_count'] == 4
    assert c['existing_artifacts_unchanged'] and not c['model_artifact_written']
    assert c['protected_files_sha256_before'] == c['protected_files_sha256_after']
    if any(not (ROOT/name).exists() for name in c['protected_files_sha256_after']):
        pytest.skip('모델·분할 산출물 생성 후 해시 보호 검증')
    for name, digest in c['protected_files_sha256_after'].items():
        assert v.sha256_file(ROOT/name) == digest
    assert v.sha256_file(ROOT/'data/raw/fraud_oracle.csv') == c['data_sha256']


def test_real_audit_matches_raw_and_csv(output,actual_data):
    df = actual_data
    groups, table, audit = v.audit_input_groups(df)
    actual = pd.read_csv(ROOT/'reports/phase2_group_audit.csv')
    assert audit == output['group_audit']
    assert audit['duplicate_rows'] == 51 and audit['conflicting_groups'] == 2
    assert len(table) == len(actual) == df.shape[0] - 51
    assert actual['size'].sum() == len(df)
    assert actual['positives'].sum() == df[TARGET].sum()
    assert actual['conflicting_labels'].sum() == 2
    assert actual.columns.tolist() == ['input_group_id', 'size', 'positives', 'negatives',
                                       'label_count', 'conflicting_labels', 'duplicate_excess']
    assert not set(FEATURES) & set(actual.columns)


def test_real_splits_and_selection(output,actual_data):
    df = actual_data
    groups = v.raw_input_groups(df)
    for name in ('group_evaluation', 'temporal_independent'):
        diag = output['split_diagnostics'][name]
        if name == 'group_evaluation':
            assert set(diag['group_intersections'].values()) == {0}
        assert set(diag['row_intersections'].values()) == {0}
        if name == 'group_evaluation':
            parts = v.group_split(np.arange(len(df)), groups)
        else:
            parts, _ = v.temporal_split(df)
        assert all('row_indices' not in part for part in diag['splits'].values())
        assert diag == v.split_diagnostics(df, groups, parts)
        result = output[name]
        best = max(result['candidates'], key=lambda r: (r['validation']['ap'], r['validation']['capture_at_20']))
        assert result['selected_model'] == best['name']
        assert result['threshold'] == best['threshold']
        assert result['test_evaluated_candidates'] == 1
        assert not result['model_saved'] and not result['refit_on_train_plus_validation']
        assert result['test']['n'] == len(parts['test'])
        for metric in ('ap', 'precision', 'recall', 'f1', 'brier', 'capture_at_20', 'precision_at_20', 'lift_at_20'):
            assert np.isfinite(result['test'][metric])
    t = output['split_diagnostics']['temporal_independent']['splits']
    assert set(t['train']['years']) <= {'1994', '1995'}
    assert set(t['validation']['years']) <= {'1994', '1995'}
    assert set(t['test']['years']) == {'1996'}
    assert t['test']['n'] == int(df.Year.eq(1996).sum())
    assert t['train']['n'] + t['validation']['n'] == int(df.Year.isin([1994, 1995]).sum())
    assert not output['temporal_independent']['historical_sampling']['uses_future_input_to_select_historical_rows']
    assert output['temporal_independent']['posthoc_input_group_intersections']['development__test'] == 22
    assert not output['temporal_independent']['test_used_for_selection_threshold_or_early_stopping']
    assert any('unseen' in limitation for limitation in output['limitations'])


def test_real_weight_diagnostic_reproduction(output):
    w = output['weight_diagnostic']
    assert not w['production_model_selection']
    assert [row['class_weight'] for row in w['models']] == ['balanced', None]
    assert all(row['C'] == .1 for row in w['models'])
    assert w['balanced_reproduction_max_abs_error'] < 1e-10
    # 기존 개발/test 공통 입력 19그룹은 재현하되 이 진단에서는 변경하지 않는다.
    assert output['split_diagnostics']['original_random']['group_intersections']['development__test'] == 19


def test_real_report_has_no_raw_row_indices_or_japanese_strings(output):
    import re
    text = json.dumps(output, ensure_ascii=False)
    assert 'row_indices' not in text
    assert not re.search('[\u3040-\u30ff]', text)
