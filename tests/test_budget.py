"""수학적 단위 사례는 성능 주장에 사용하지 않는다. 실제 회귀는 별도 검증."""
import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

from claimpilot.budget import budget_metrics, budget_curve, cumulative_capture
from claimpilot.modeling import evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_fractional_ceil_counts_and_random_baseline():
    result = budget_metrics([1, 0, 0, 0, 1, 0], [.9, .8, .7, .6, .5, .4], .2)
    assert result == dict(requested_fraction=.2, k=2, n=6, positives=2,
                          prevalence=2 / 6, reviewed_fraction=2 / 6,
                          captured_count=1, capture_rate=.5, precision=.5,
                          random_expected_count=2 * (2 / 6),
                          random_expected_capture=2 / 6, lift=.5 / (2 / 6), status='ok')


def test_ties_keep_input_order_and_scores_rank_descending():
    assert budget_metrics([0, 1, 1, 0], [.5] * 4, .25)['captured_count'] == 0
    assert budget_metrics([1, 0, 0, 1], [.5] * 4, .25)['captured_count'] == 1
    result = budget_metrics([1, 0, 1], [2., 9., 3.], 1 / 3)
    assert result['captured_count'] == 0
    # unsigned integer scores must not wrap around on negation.
    assert budget_metrics([1, 0], np.array([0, 255], dtype=np.uint8), .5)['captured_count'] == 0


@pytest.mark.parametrize('fraction,expected_k', [(1e-12, 1), (.25, 1), (.2500001, 2), (1., 4)])
def test_boundaries_and_small_budget(fraction, expected_k):
    result = budget_metrics([1, 0, 0, 1], [4, 3, 2, 1], fraction)
    assert result['k'] == expected_k
    assert result['reviewed_fraction'] == expected_k / 4


def test_all_reviewed_and_singleton():
    result = budget_metrics([0, 1, 1], [.9, .2, .1], 1)
    assert (result['k'], result['capture_rate'], result['lift']) == (3, 1., 1.)
    assert result['precision'] == 2 / 3
    assert result['random_expected_count'] == 2
    assert budget_metrics([1], [-5.], .05)['capture_rate'] == 1


def test_no_positives_explicit_status_zero_not_nan():
    for result in budget_curve([0, 0], [.8, .1], [.1, 1]) + cumulative_capture([0, 0], [.8, .1]):
        assert result['status'] == 'no_positives'
        assert result['capture_rate'] == result['lift'] == result['precision'] == 0
        assert result['random_expected_count'] == 0
        assert result['random_expected_capture'] == result['k'] / 2
        assert all(math.isfinite(v) for v in result.values() if isinstance(v, (float, int)))


def test_default_and_dynamic_fractions_order_and_duplicates():
    labels, scores = [1, 0, 1, 0, 0], [.1, .7, .6, .2, .9]
    assert [r['requested_fraction'] for r in budget_curve(labels, scores)] == [.05, .1, .15, .2, .3, .4]
    fractions = [.73, .01, 1, .73]
    assert budget_curve(labels, scores, iter(fractions)) == [budget_metrics(labels, scores, f) for f in fractions]


def test_cumulative_all_steps_origin_and_random_line():
    curve = cumulative_capture([1, 0, 1], [.2, .9, .5])
    assert len(curve) == 4
    assert [r['k'] for r in curve] == [0, 1, 2, 3]
    assert [r['captured_count'] for r in curve] == [0, 0, 1, 2]
    assert [r['capture_rate'] for r in curve] == [0, 0, .5, 1]
    assert [r['reviewed_fraction'] for r in curve] == [0, 1 / 3, 2 / 3, 1]
    assert [r['random_expected_capture'] for r in curve] == [0, 1 / 3, 2 / 3, 1]
    for point in curve[1:]:
        expected = budget_metrics([1, 0, 1], [.2, .9, .5], point['k'] / 3)
        assert point == {key: value for key, value in expected.items() if key != 'requested_fraction'}


@pytest.mark.parametrize('fraction', [0, -1, 1.000001, np.nan, np.inf, -np.inf, '0.2', None, True, [0.2], 1j])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        budget_metrics([1], [.1], fraction)
    with pytest.raises(ValueError):
        budget_curve([1], [.1], [fraction])


@pytest.mark.parametrize('fractions', [[], None, .2, [.1, 0]])
def test_invalid_fraction_sequence(fractions):
    with pytest.raises(ValueError):
        budget_curve([1], [.5], fractions)


@pytest.mark.parametrize('labels,scores', [([], []), ([1], []), ([1, 0], [.1]),
    ([[1]], [[.1]]), ([1], [[.1]]), ([2], [.2]), ([-1], [.2]), ([.5], [.2]),
    ([np.nan], [.2]), ([np.inf], [.2]), ([1], [np.nan]), ([1], [np.inf]),
    ([1], [-np.inf]), (['1'], [.2]), ([1], ['.2']), ([None], [.2]), ([1], [None]),
    ([1], [1j]), (None, [.2]), ([[1], [0, 1]], [.2, .3])])
def test_invalid_arrays_rejected_by_all_functions(labels, scores):
    for function, extra in [(budget_metrics, (.2,)), (budget_curve, ()), (cumulative_capture, ())]:
        with pytest.raises(ValueError):
            function(labels, scores, *extra)


@pytest.mark.parametrize('labels,scores', [([1, 0, 0, 0, 1, 0], [.9, .8, .7, .6, .5, .4]),
    ([1, 0, 0, 0, 0], [.1] * 5), ([0, 1, 1, 0], [.5] * 4)])
def test_existing_top20_regression(labels, scores):
    result, old = budget_metrics(labels, scores, .2), evaluate(labels, scores)
    mapping = {'k': 'reviewed_count', 'captured_count': 'captured_count',
               'capture_rate': 'capture_at_20', 'precision': 'precision_at_20',
               'lift': 'lift_at_20', 'n': 'n', 'positives': 'positives'}
    for new, previous in mapping.items():
        assert result[new] == pytest.approx(old[previous])


def test_real_files_optional_selected_model_top20_regression():
    paths = [ROOT / 'reports/scored_claims.csv', ROOT / 'reports/metrics.json']
    if not all(p.exists() for p in paths):
        pytest.skip('실제 학습 산출물이 있을 때만 held-out 회귀 검증')
    from scripts.analyze_budget import build_report
    report = build_report([.07, .2, .57, 1.])
    assert report['top20_regression']['status'] == 'passed'
    assert len(report['cumulative_capture']) == report['n'] + 1
    assert [r['requested_fraction'] for r in report['budgets']] == [.07, .2, .57, 1.]
    metadata = json.loads(paths[1].read_text())
    assert report['selected_model'] == metadata['selected_model']
    expected = next(m['test'] for m in metadata['models'] if m['name'] == report['selected_model'])
    with paths[0].open() as stream:
        rows = sorted(csv.DictReader(stream), key=lambda row: int(row['priority_rank']))
    labels, scores = [int(row['label']) for row in rows], [float(row['risk_score']) for row in rows]
    assert evaluate(labels, scores)['capture_at_20'] == pytest.approx(expected['capture_at_20'])
    assert report['budgets'][1]['captured_count'] == expected['captured_count']
    assert report['budgets'][-1]['capture_rate'] == 1
