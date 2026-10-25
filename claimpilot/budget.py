"""라벨이 있는 held-out 표본의 검토량별 순위 평가. 비용·사기 판단이 아니다."""
import math
from numbers import Real

import numpy as np

DEFAULT_FRACTIONS = (.05, .10, .15, .20, .30, .40)


def _validated(y, scores):
    """유한 실수 1차원 입력과 이진 라벨만 허용한다."""
    try:
        labels, values = np.asarray(y), np.asarray(scores)
        if (labels.ndim != 1 or values.ndim != 1 or not labels.size
                or labels.size != values.size):
            raise ValueError('비어 있지 않은 동일 길이 1차원 입력이 필요합니다.')
        if labels.dtype.kind not in 'biuf' or values.dtype.kind not in 'biuf':
            raise ValueError('라벨과 점수는 유한 실수여야 합니다.')
        if not np.isfinite(labels).all() or not np.isfinite(values).all():
            raise ValueError('비유한 라벨·점수는 허용하지 않습니다.')
        if not np.isin(labels, [0, 1]).all():
            raise ValueError('라벨은 0 또는 1이어야 합니다.')
    except (TypeError, OverflowError) as exc:
        raise ValueError('유효한 라벨·점수 배열이 필요합니다.') from exc
    # 실수 변환 후 부호 반전: unsigned 정수 점수의 언더플로를 피한다.
    order = np.argsort(-values.astype(np.longdouble), kind='stable')
    return labels.astype(np.int64)[order]


def _fraction(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not math.isfinite(value) or not 0 < value <= 1):
        raise ValueError('검토 비율은 (0, 1] 범위의 유한 실수여야 합니다.')
    return float(value)


def _point(n, positives, k, captured):
    prevalence = positives / n
    precision = captured / k if k else 0.0
    return dict(k=k, n=n, positives=positives, prevalence=prevalence,
                reviewed_fraction=k / n, captured_count=captured,
                capture_rate=captured / positives if positives else 0.0,
                precision=precision, random_expected_count=k * prevalence,
                random_expected_capture=k / n,
                lift=precision / prevalence if positives else 0.0,
                status='ok' if positives else 'no_positives')


def budget_metrics(y, scores, fraction):
    """상위 ceil(n*fraction)건. 동점은 입력 순서, 점수는 확률일 필요 없음.

    양성이 없으면 capture_rate와 lift는 0, status는 no_positives다.
    random_expected_capture는 무작위 검토 비율 k/n(양성 없음에도 유지)이다.
    """
    ranked = _validated(y, scores)
    requested = _fraction(fraction)
    n = len(ranked)
    k = math.ceil(n * requested)
    return dict(requested_fraction=requested,
                **_point(n, int(ranked.sum()), k, int(ranked[:k].sum())))


def budget_curve(y, scores, fractions=DEFAULT_FRACTIONS):
    """요청된 비율의 입력 순서·중복을 보존하는 지표 목록."""
    ranked = _validated(y, scores)
    try:
        requested = [_fraction(f) for f in fractions]
    except TypeError as exc:
        raise ValueError('검토 비율 목록이 필요합니다.') from exc
    if not requested:
        raise ValueError('검토 비율 목록은 비어 있을 수 없습니다.')
    n, positives = len(ranked), int(ranked.sum())
    cumulative = np.cumsum(ranked)
    result = []
    for fraction in requested:
        k = math.ceil(n * fraction)
        result.append(dict(requested_fraction=fraction,
                           **_point(n, positives, k, int(cumulative[k - 1]))))
    return result


def cumulative_capture(y, scores):
    """원점 k=0부터 n까지의 계단형 포집 곡선과 무작위 기준선.

    각 점의 x는 reviewed_fraction=k/n, y는 capture_rate다.
    원점의 precision/lift는 0으로 정의하며 보간·면적 추정은 하지 않는다.
    """
    ranked = _validated(y, scores)
    n, positives = len(ranked), int(ranked.sum())
    counts = np.concatenate(([0], np.cumsum(ranked)))
    return [_point(n, positives, k, int(count)) for k, count in enumerate(counts)]
