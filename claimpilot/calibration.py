"""P2 확률보정 연구 전용. production 모델/선정/기존 보고서를 변경하지 않는다.

balanced는 양성 손실 가중치를 높여 유효 사전분포와 최적화 목표를 바꾼다.
따라서 raw sigmoid 점수는 원 모집단 사건 확률이 아니며, 순위 AP가 유용해도
Brier가 상수 기준선보다 나쁠 수 있다. 고정 sigmoid 보정도 표본/라벨/분포에
의존하므로 실제 사기 확률 또는 자동 지급 판단으로 해석하지 않는다.
"""
import hashlib
import json
from itertools import combinations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from claimpilot.modeling import TARGET, choose_threshold, evaluate
from claimpilot.validation import SEED, make_lr, original_random_split

CALIBRATION_C = 1e6
ARM_NAMES = ('raw_balanced', 'raw_none', 'sigmoid_balanced', 'sigmoid_none')
BASELINE_NAME = 'constant_train_prevalence'


def id_digest(values):
    return hashlib.sha256(json.dumps([int(x) for x in values]).encode()).hexdigest()


def research_split(df, manifest):
    """기존 manifest 순서 보존. validation만 seed42 계층 1:1 분할한다."""
    original = original_random_split(df, manifest)
    if not df.PolicyNumber.is_unique or not df[TARGET].isin([0, 1]).all():
        raise ValueError('고유 claim ID와 이진 라벨이 필요합니다.')
    calibration, selection = train_test_split(
        original['validation'], test_size=.5,
        stratify=df.iloc[original['validation']][TARGET], random_state=SEED)
    parts = {'train': original['train'], 'calibration': calibration,
             'selection': selection, 'test': original['test']}
    intersections = {}
    row_intersections = {}
    for a, b in combinations(parts, 2):
        intersections[a + '__' + b] = len(set(df.iloc[parts[a]].PolicyNumber) & set(df.iloc[parts[b]].PolicyNumber))
        row_intersections[a + '__' + b] = len(set(parts[a]) & set(parts[b]))
    if any(intersections.values()) or any(row_intersections.values()):
        raise ValueError('연구 분할의 claim ID/행 중복')
    if set(calibration) | set(selection) != set(original['validation']):
        raise ValueError('기존 validation 보존 실패')
    diagnostics = {
        'splits': {name: {'n': len(idx), 'positives': int(df.iloc[idx][TARGET].sum()),
                         'prevalence': float(df.iloc[idx][TARGET].mean()),
                         'ordered_claim_id_sha256': id_digest(df.iloc[idx].PolicyNumber)}
                   for name, idx in parts.items()},
        'claim_id_intersections': intersections, 'row_intersections': row_intersections,
        'original_train_order_preserved': np.array_equal(parts['train'], original['train']),
        'original_test_order_preserved': np.array_equal(parts['test'], original['test']),
        'original_validation_partition_preserved': True,
        'original_validation_ordered_claim_id_sha256': id_digest(df.iloc[original['validation']].PolicyNumber),
    }
    return parts, diagnostics


def fit_sigmoid(raw_logits, y):
    """사전고정: raw decision_function 하나에 비가중 logistic C=1e6 fit.

    calibration 행만 받는다. test 기반 방법/규제/하이퍼파라미터 선택은 없다.
    """
    z = np.asarray(raw_logits, dtype=float).reshape(-1, 1)
    y = np.asarray(y)
    if len(z) != len(y) or not np.isfinite(z).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError('유한 logits와 양쪽 이진 클래스가 필요합니다.')
    return LogisticRegression(C=CALIBRATION_C, class_weight=None,
                              solver='lbfgs', max_iter=1500, random_state=SEED).fit(z, y)


def predict_arm(arm, X):
    if arm['model'] is None:
        return np.full(len(X), arm['prevalence'])
    if arm['calibrator'] is None:
        return arm['model'].predict_proba(X)[:, 1]
    logits = arm['model'].decision_function(X).reshape(-1, 1)
    return arm['calibrator'].predict_proba(logits)[:, 1]


def fit_research_arms(X_train, y_train, X_calibration, y_calibration,
                      X_selection, y_selection):
    """test를 받지 않는 API: train fit, calibration 보정, selection F1 고정.

    보정/비보정은 동일하게 train만 학습한 LR 객체를 공유한다. 재학습하지 않는다.
    """
    index_sets = [set(x.index) for x in (X_train, X_calibration, X_selection)]
    if any(a & b for a, b in combinations(index_sets, 2)):
        raise ValueError('fit/보정/선정 행 index 중복')
    for X, y in ((X_train, y_train), (X_calibration, y_calibration), (X_selection, y_selection)):
        if not X.index.equals(y.index):
            raise ValueError('입력과 라벨 index 불일치')
    raw, calibrated = [], []
    for weight, suffix in (('balanced', 'balanced'), (None, 'none')):
        model = make_lr(C=.1, class_weight=weight)
        model.fit(X_train, y_train)
        calibrator = fit_sigmoid(model.decision_function(X_calibration), y_calibration)
        raw.append({'name': 'raw_' + suffix, 'model': model, 'calibrator': None,
                    'class_weight': weight})
        calibrated.append({'name': 'sigmoid_' + suffix, 'model': model,
                           'calibrator': calibrator, 'class_weight': weight})
    arms = raw + calibrated
    arms.append({'name': BASELINE_NAME, 'model': None, 'calibrator': None,
                 'class_weight': None, 'prevalence': float(np.mean(y_train))})
    for arm in arms:
        scores = predict_arm(arm, X_selection)
        arm['threshold'] = choose_threshold(y_selection, scores)
        arm['selection'] = diagnostics(y_selection, scores, arm['threshold'])
    return arms


def calibration_table(y, scores, bins=10):
    """고정폭 [0,.1), ... [.9,1].빈 bin 평균/비율은 null, 수는 0."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if bins < 1 or not len(y) or len(y) != len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('유효한 확률과 평가 행이 필요합니다.')
    assignment = np.minimum((scores * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = assignment == b; n = int(mask.sum())
        rows.append({'bin': b + 1, 'lower': b / bins, 'upper': (b + 1) / bins,
                     'n': n, 'positives': int(y[mask].sum()),
                     'mean_score': float(scores[mask].mean()) if n else None,
                     'observed_rate': float(y[mask].mean()) if n else None})
    return rows


def capture_bins(y, scores, bins=10):
    """점수 내림차순 stable 순서의 누적 ceil(n*b/10) 검토 예산 bin."""
    y, scores = np.asarray(y), np.asarray(scores)
    order = np.argsort(-scores, kind='stable')
    positives = int(y.sum()); rows = []; start = 0; cumulative = 0
    for b in range(1, bins + 1):
        end = int(np.ceil(len(y) * b / bins)); idx = order[start:end]
        captured = int(y[idx].sum()); cumulative += captured
        rows.append({'bin': b, 'n': len(idx), 'captured': captured,
                     'precision': captured / len(idx) if len(idx) else None,
                     'cumulative_reviewed': end, 'cumulative_captured': cumulative,
                     'cumulative_capture': cumulative / positives if positives else 0.})
        start = end
    return rows


def diagnostics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    table = calibration_table(y, scores)
    result = evaluate(y, scores, threshold)
    result.update({'log_loss': float(log_loss(y, scores, labels=[0, 1])),
                   'mean_score': float(scores.mean()), 'prevalence': float(y.mean()),
                   'calibration_table': table, 'capture_bins': capture_bins(y, scores),
                   'ece_10_uniform': float(sum(r['n'] * abs(r['mean_score'] - r['observed_rate'])
                                              for r in table if r['n']) / len(y))})
    return result
