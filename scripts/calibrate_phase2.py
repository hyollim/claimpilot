"""P2 보정 연구만 실행. reports/phase2_calibration.json 외 산출물을 쓰지 않는다."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
from claimpilot.calibration import (ARM_NAMES, CALIBRATION_C, diagnostics,
                                    fit_research_arms, predict_arm, research_split)
from claimpilot.modeling import FEATURES, TARGET, feature_columns, prepare

OUTPUT = ROOT/'reports/phase2_calibration.json'
ALLOWED = {'claimpilot/calibration.py', 'scripts/calibrate_phase2.py',
           'tests/test_calibration.py', 'reports/phase2_calibration.json'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_snapshot():
    """원 모델/bundle 및 원 평가 산출물만 감사. 동시 P0 UI 변경과 분리한다."""
    protected = [ROOT/'reports'/name for name in ('metrics.json', 'data_audit.json', 'split_manifest.json')]
    protected += [p for p in (ROOT/'models').rglob('*') if p.is_file()]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(protected)}


def main():
    started = time.time(); before = protected_snapshot()
    source = ROOT/'data/raw/fraud_oracle.csv'
    df = pd.read_csv(source, encoding='utf-8-sig')
    manifest = json.loads((ROOT/'reports/split_manifest.json').read_text())
    if manifest['data_sha256'] != sha256(source):
        raise ValueError('기존 데이터 체크섬과 manifest 불일치')
    if len(df) != 15420 or feature_columns(df) != FEATURES:
        raise ValueError('승인된 데이터/입력 스키마 불일치')
    parts, split_metadata = research_split(df, manifest)
    if {k: len(v) for k, v in parts.items()} != {'train': 9252, 'calibration': 1542, 'selection': 1542, 'test': 3084}:
        raise ValueError('기존 분할 크기 불일치')
    X, y = prepare(df, FEATURES), df[TARGET]
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter('always')
        arms = fit_research_arms(X.iloc[parts['train']], y.iloc[parts['train']],
                                 X.iloc[parts['calibration']], y.iloc[parts['calibration']],
                                 X.iloc[parts['selection']], y.iloc[parts['selection']])
    warning_rows = [{'category': w.category.__name__, 'message': str(w.message)} for w in fit_warnings]
    if any(w['category'] == 'ConvergenceWarning' for w in warning_rows):
        raise RuntimeError('학습 미수렴: hyperparameter 변경 없이 중단')
    for arm in arms:
        for fitted in (arm['model'].named_steps['clf'] if arm['model'] is not None else None, arm['calibrator']):
            if fitted is not None and not (np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_).all()):
                raise RuntimeError('비유한 학습 계수: 연구 실행 중단')
    # 모든 방법·hyperparameter·각 threshold를 고정한 뒤에만 test를 함께 진단한다.
    rows = []
    for arm in arms:
        row = {'name': arm['name'], 'class_weight': arm['class_weight'],
               'threshold': arm['threshold'], 'selection': arm['selection'],
               'fit_provenance': {'base_estimator': 'train' if arm['model'] is not None else None,
                                  'calibrator': 'calibration' if arm['calibrator'] is not None else None,
                                  'threshold': 'selection',
                                  'constant_prevalence': 'train' if arm['model'] is None else None},
               'test': diagnostics(y.iloc[parts['test']], predict_arm(arm, X.iloc[parts['test']]), arm['threshold'])}
        if arm['model'] is not None:
            row['base_fit_iterations'] = arm['model'].named_steps['clf'].n_iter_.tolist()
        if arm['calibrator'] is not None:
            row['calibration_fit_iterations'] = arm['calibrator'].n_iter_.tolist()
            row['sigmoid_parameters'] = {'slope': float(arm['calibrator'].coef_[0, 0]),
                                         'intercept': float(arm['calibrator'].intercept_[0]), 'C': CALIBRATION_C}
        rows.append(row)
    after = protected_snapshot()
    if before != after:
        raise RuntimeError('허용 목록 외 파일 체크섬 변경 감지: 연구 보고서를 쓰지 않음')
    reference = json.loads((ROOT/'reports/metrics.json').read_text())
    old_balanced = next(r for r in reference['models'] if r['name'] == 'Logistic Regression C=0.1')
    baseline = rows[-1]['test']; balanced = rows[0]['test']
    report = {
        'configuration': {'purpose': 'P0와 분리한 P2 연구 진단. production 전환/모델 저장 없음.',
            'seed': 42, 'features': FEATURES, 'lr': {'C': .1, 'class_weights': ['balanced', None],
            'max_iter': 1500, 'solver': 'lbfgs', 'encoder_fit_source': 'train'},
            'calibration': {'method': 'raw logits에 비가중 sigmoid logistic regression',
                'C': CALIBRATION_C, 'solver': 'lbfgs', 'max_iter': 1500,
                'fit_source': 'calibration', 'predeclared': True, 'method_selected_using_test': False},
            'predeclared_test_arms': list(ARM_NAMES), 'test_diagnostic_arms': 4, 'additional_baselines': 1,
            'threshold_rule': '각 arm selection F1 최대, 동률은 기존 choose_threshold의 첫 threshold. test에서 고정.',
            'model_selection_performed': False, 'production_model_changed': False,
            'model_artifact_written': False, 'test_used_for_fit_calibration_threshold_or_method_choice': False,
            'top20_rule': 'ceil(n*.2), stable 내림차순, 기존 random test 순서 유지',
            'calibration_bins': '고정폭 10개 [0,.1), ... [.9,1]; 빈 bin은 n=0, 평균/관측률 null',
            'capture_bins': '고정 누적 검토율 10% 간격, ceil(n*비율), stable 동점 순서',
            'data_sha256': sha256(source),
            'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__, 'platform': platform.platform()},
            'protection_scope_ko': 'models 디렉터리의 원 모델/bundle 전체 및 기존 reports/metrics.json, data_audit.json, split_manifest.json. 동시 P0 UI 코드 변경은 감사 범위에서 제외한다. P2는 허용된 네 파일만 작성하며 git 작업을 하지 않는다.',
            'existing_artifacts_unchanged': before == after,
            'protected_files_sha256_before': before, 'protected_files_sha256_after': after},
        'numerical_diagnostics': {'fit_warnings': warning_rows,
            'all_coefficients_finite': True, 'convergence_warning_count': 0,
            'note_ko': '현재 플랫폼에서 dense logits 보정 fit 중 NumPy matmul RuntimeWarning이 관측될 수 있다. 경고를 숨기지 않고 기록하며 계수·지표 유한성과 수렴 경고 부재를 별도 확인한다. 다른 플랫폼의 비트 단위 재현을 보장하지 않는다.'},
        'class_weight_diagnostic': {'train_negatives': int((y.iloc[parts['train']] == 0).sum()),
            'train_positives': int(y.iloc[parts['train']].sum()),
            'balanced_negative_weight': len(parts['train']) / (2 * int((y.iloc[parts['train']] == 0).sum())),
            'balanced_positive_weight': len(parts['train']) / (2 * int(y.iloc[parts['train']].sum())),
            'effective_weighted_training_prevalence': .5},
        'split_diagnostics': split_metadata,
        'fit_rows': {'base_estimator': {'source': 'train', **split_metadata['splits']['train']},
                     'calibrator': {'source': 'calibration', **split_metadata['splits']['calibration']},
                     'threshold': {'source': 'selection', **split_metadata['splits']['selection']}},
        'models': rows,
        'brier_diagnostic': {'reference_balanced_brier': old_balanced['test']['brier'],
            'reproduced_balanced_brier': balanced['brier'], 'constant_train_prevalence_brier': baseline['brier'],
            'balanced_mean_probability': balanced['mean_score'], 'test_prevalence': balanced['prevalence'],
            'explanation_ko': '기존 약 .2156 vs 상수 약 .0564는 AP 순위 성능과 확률 정확도가 다른 목표이기 때문이다. balanced는 n/(2*n_class) 가중치로 양성 손실을 크게 하여 유효 사전분포를 1/2로 바꾸고 가중 로그손실을 최소화한다. 원 모집단 약 6% 양성률보다 높은 raw 확률을 내어 Brier를 악화시킬 수 있다. 가중치는 규제하의 기울기도 바꾸므로 단순 사전확률 이동만으로 모든 차이를 설명할 수 없다. 이 비교만으로 인과적 분해를 입증하지 않는다.'},
        'limitations_ko': [
            '기존 random test 3084는 MVP에서 이미 관측되었다. 완전 unseen test 또는 독립 외부검증이 아니다.',
            '사전고정한 4 arms를 같은 test에서 함께 진단한 결과다. test로 보정법/모델/규제/threshold를 선정하거나 production 전환하지 않는다.',
            'calibration 1542와 selection 1542의 양성 표본이 적다. 단일 seed 분할 진단이며 재학습 불확실성이나 외부 모집단 성능을 보장하지 않는다.',
            '행/claim ID 교집합 0은 실제 고객/사건 독립성을 뜻하지 않는다. 기존 random split의 동일 입력그룹 중복은 유지된다.',
            'sigmoid 보정은 calibration 표본에서 관측 라벨과의 관계를 학습할 뿐 실제 사기 확률/자동 지급 판단을 정당화하지 않는다. 라벨 품질/모집단 이동/수집시점 누수 한계가 남는다.',
            '상수 점수 top20은 원 test 순서에 의한 동점 결과이며 업무 순위 가치가 아니다. 확률 구간 표는 기술통계이며 유의성 검정이 아니다.'],
        'elapsed_seconds': round(time.time() - started, 3)}
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    for row in rows:
        print(json.dumps({'name': row['name'], 'threshold': row['threshold'],
            **{k: row['test'][k] for k in ('ap', 'brier', 'log_loss', 'precision', 'recall', 'f1', 'capture_at_20', 'captured_count')}}, ensure_ascii=False))
    print('보호파일 불변:', before == after, '산출물:', OUTPUT)


if __name__ == '__main__':
    main()
