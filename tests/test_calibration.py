"""P2 기능 fixture, fit 출처/분할 누수 방지 및 실제 보고서 계약 테스트."""
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

import claimpilot.calibration as c
from claimpilot.modeling import FEATURES, TARGET, prepare

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def frame():
    n = 400
    df = pd.DataFrame({col: ['v' + str((i + j) % 7) for i in range(n)]
                       for j, col in enumerate(FEATURES)})
    df['PolicyNumber'] = np.arange(1001, 1001 + n)
    df[TARGET] = [int(i % 5 == 0 or i % 13 == 0) for i in range(n)]
    return df


def manifest_for(df):
    from claimpilot.validation import original_random_split
    original = original_random_split(df)
    return {'seed': 42, 'features': FEATURES, 'splits': {
        name: [{'claim_id': int(df.iloc[i].PolicyNumber)} for i in indices]
        for name, indices in original.items()}}


def test_split_claim_ids_disjoint_and_preserved(frame):
    manifest = manifest_for(frame)
    parts, meta = c.research_split(frame, manifest)
    again, _ = c.research_split(frame, manifest)
    assert {k: len(v) for k, v in parts.items()} == {'train': 240, 'calibration': 40, 'selection': 40, 'test': 80}
    assert set(meta['claim_id_intersections'].values()) == {0}
    assert set(meta['row_intersections'].values()) == {0}
    for name in parts:
        np.testing.assert_array_equal(parts[name], again[name])
    for name in ('train', 'test'):
        assert frame.iloc[parts[name]].PolicyNumber.tolist() == [r['claim_id'] for r in manifest['splits'][name]]
    assert set(frame.iloc[np.r_[parts['calibration'], parts['selection']]].PolicyNumber) == {r['claim_id'] for r in manifest['splits']['validation']}
    bad = json.loads(json.dumps(manifest))
    bad['splits']['test'][0] = bad['splits']['train'][0]
    with pytest.raises(ValueError):
        c.research_split(frame, bad)


def test_actual_fit_rows_and_frozen_models(frame, monkeypatch):
    parts, _ = c.research_split(frame, manifest_for(frame))
    X = prepare(frame, FEATURES); y = frame[TARGET]
    observed_base = []; observed_cal = []; observed_threshold = []
    original_lr, original_sigmoid, original_threshold = c.make_lr, c.fit_sigmoid, c.choose_threshold

    def make_lr(*args, **kwargs):
        model = original_lr(*args, **kwargs); original_fit = model.fit
        def spy_fit(fit_X, fit_y):
            observed_base.append(fit_X.index.tolist())
            assert fit_y.index.equals(fit_X.index)
            return original_fit(fit_X, fit_y)
        monkeypatch.setattr(model, 'fit', spy_fit)
        return model

    def sigmoid(logits, fit_y):
        observed_cal.append(fit_y.index.tolist())
        assert len(logits) == len(parts['calibration'])
        return original_sigmoid(logits, fit_y)

    def threshold(fit_y, scores):
        observed_threshold.append(fit_y.index.tolist())
        return original_threshold(fit_y, scores)

    monkeypatch.setattr(c, 'make_lr', make_lr)
    monkeypatch.setattr(c, 'fit_sigmoid', sigmoid)
    monkeypatch.setattr(c, 'choose_threshold', threshold)
    arms = c.fit_research_arms(X.iloc[parts['train']], y.iloc[parts['train']],
        X.iloc[parts['calibration']], y.iloc[parts['calibration']],
        X.iloc[parts['selection']], y.iloc[parts['selection']])
    assert observed_base == [parts['train'].tolist()] * 2
    assert observed_cal == [parts['calibration'].tolist()] * 2
    assert observed_threshold == [parts['selection'].tolist()] * 5
    assert [a['name'] for a in arms] == list(c.ARM_NAMES) + [c.BASELINE_NAME]
    assert arms[0]['model'] is arms[2]['model'] and arms[1]['model'] is arms[3]['model']
    assert 'test' not in inspect.signature(c.fit_research_arms).parameters
    for arm in arms:
        model = arm['model']
        if model is not None:
            assert model.named_steps['clf'].C == .1
            assert model.named_steps['clf'].class_weight == arm['class_weight']
        if arm['calibrator'] is not None:
            assert arm['calibrator'].C == 1e6
            assert arm['calibrator'].class_weight is None
            z = model.decision_function(X.iloc[parts['test']]).reshape(-1, 1)
            np.testing.assert_allclose(c.predict_arm(arm, X.iloc[parts['test']]), arm['calibrator'].predict_proba(z)[:, 1])
        scores = c.predict_arm(arm, X.iloc[parts['selection']])
        actual = f1_score(y.iloc[parts['selection']], scores >= arm['threshold'])
        best = max(f1_score(y.iloc[parts['selection']], scores >= t) for t in np.unique(scores))
        assert actual == best
        result = c.diagnostics(y.iloc[parts['test']], c.predict_arm(arm, X.iloc[parts['test']]), arm['threshold'])
        assert np.isfinite(result['brier']) and np.isfinite(result['log_loss'])
    np.testing.assert_allclose(c.predict_arm(arms[-1], X), y.iloc[parts['train']].mean())
    with pytest.raises(ValueError, match='중복'):
        c.fit_research_arms(X.iloc[:40], y.iloc[:40], X.iloc[:40], y.iloc[:40], X.iloc[40:80], y.iloc[40:80])


def test_sigmoid_fixture_and_invalid_inputs():
    z = np.tile(np.arange(-2., 3.), 8)
    y = np.tile([0, 0, 1, 0, 1], 8)
    model = c.fit_sigmoid(z, y)
    scores = model.predict_proba(np.array([[-2.], [0.], [2.]]))[:, 1]
    assert np.isfinite(scores).all() and np.all(np.diff(scores) > 0)
    with pytest.raises(ValueError):
        c.fit_sigmoid([0, np.nan], [0, 1])
    with pytest.raises(ValueError):
        c.fit_sigmoid([0, 1], [0, 0])


def test_probability_bins_endpoints_empty_and_capture():
    y = np.array([0, 1, 1, 0]); scores = np.array([0., .1, .95, 1.])
    table = c.calibration_table(y, scores)
    assert table[0]['n'] == 1 and table[1]['n'] == 1 and table[-1]['n'] == 2
    assert table[2]['n'] == 0 and table[2]['mean_score'] is None
    assert table[2]['observed_rate'] is None
    assert sum(r['n'] for r in table) == len(y)
    assert sum(r['positives'] for r in table) == y.sum()
    result = c.diagnostics(y, scores, .5)
    assert result['brier'] == pytest.approx(np.mean((y - scores) ** 2))
    assert result['capture_bins'][1]['cumulative_captured'] == result['captured_count']
    assert result['capture_bins'][-1]['cumulative_capture'] == 1.
    # 상위 동점은 원본 순서 유지.
    result = c.diagnostics(np.array([1, 0, 1, 0, 0]), np.full(5, .2), .2)
    assert result['captured_count'] == 1
    for invalid in ([0, np.nan], [0, 1.1], [-.1, .5]):
        with pytest.raises(ValueError):
            c.diagnostics([0, 1], invalid, .5)


@pytest.fixture
def report():
    path = ROOT/'reports/phase2_calibration.json'
    if not path.exists():
        pytest.skip('실제 calibrate_phase2.py 실행 전')
    return json.loads(path.read_text())


def test_real_report_configuration_and_split_provenance(report):
    config = report['configuration']
    assert config['calibration']['C'] == 1e6 and config['calibration']['predeclared']
    assert not config['calibration']['method_selected_using_test']
    assert config['predeclared_test_arms'] == list(c.ARM_NAMES)
    assert not config['model_selection_performed'] and not config['production_model_changed']
    assert not config['model_artifact_written']
    assert not config['test_used_for_fit_calibration_threshold_or_method_choice']
    assert config['existing_artifacts_unchanged']
    assert config['protected_files_sha256_before'] == config['protected_files_sha256_after']
    if not (ROOT/'data/raw/fraud_oracle.csv').exists() or not (ROOT/'reports/split_manifest.json').exists():
        pytest.skip('원본 데이터와 기존 분할 생성 후 보정 출처 통합 검증')
    df = pd.read_csv(ROOT/'data/raw/fraud_oracle.csv')
    manifest = json.loads((ROOT/'reports/split_manifest.json').read_text())
    parts, meta = c.research_split(df, manifest)
    assert report['split_diagnostics'] == meta
    assert [len(parts[k]) for k in ('train', 'calibration', 'selection', 'test')] == [9252, 1542, 1542, 3084]
    assert set(meta['claim_id_intersections'].values()) == {0}
    assert set(meta['row_intersections'].values()) == {0}
    for role, source in [('base_estimator', 'train'), ('calibrator', 'calibration'), ('threshold', 'selection')]:
        assert report['fit_rows'][role] == {'source': source, **meta['splits'][source]}
    assert any('독립 외부검증' in s for s in report['limitations_ko'])
    assert report['numerical_diagnostics']['all_coefficients_finite']
    assert report['numerical_diagnostics']['convergence_warning_count'] == 0
    assert all(w['category'] != 'ConvergenceWarning' for w in report['numerical_diagnostics']['fit_warnings'])


def test_real_results_finite_and_tables_conserve_rows(report):
    def check_finite(obj):
        if isinstance(obj, dict):
            for value in obj.values(): check_finite(value)
        elif isinstance(obj, list):
            for value in obj: check_finite(value)
        elif isinstance(obj, float):
            assert np.isfinite(obj)
    check_finite(report)
    assert [r['name'] for r in report['models']] == list(c.ARM_NAMES) + [c.BASELINE_NAME]
    for row in report['models']:
        assert row['fit_provenance']['threshold'] == 'selection'
        assert row['fit_provenance']['calibrator'] == ('calibration' if row['name'].startswith('sigmoid') else None)
        for split in ('selection', 'test'):
            metrics = row[split]
            assert sum(r['n'] for r in metrics['calibration_table']) == metrics['n']
            assert sum(r['positives'] for r in metrics['calibration_table']) == metrics['positives']
            assert sum(r['captured'] for r in metrics['capture_bins']) == metrics['positives']
            assert metrics['capture_bins'][1]['cumulative_captured'] == metrics['captured_count']
            assert metrics['capture_bins'][-1]['cumulative_reviewed'] == metrics['n']
            for key in ('ap', 'brier', 'precision', 'recall', 'f1', 'capture_at_20'):
                assert 0 <= metrics[key] <= 1
            for bin_row in metrics['calibration_table']:
                if bin_row['n']:
                    assert 0 <= bin_row['mean_score'] <= 1
                    assert bin_row['observed_rate'] == pytest.approx(bin_row['positives'] / bin_row['n'])
    old = report['brier_diagnostic']
    assert old['reproduced_balanced_brier'] == pytest.approx(old['reference_balanced_brier'], abs=1e-10)
