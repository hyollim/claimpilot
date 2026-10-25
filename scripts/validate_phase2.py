"""seed42 사전지정 추가 평가. 허용된 두 보고서만 저장하며 production 모델은 불변."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import catboost
import numpy as np
import pandas as pd
import sklearn
from claimpilot.modeling import FEATURES, TARGET, feature_columns, prepare
from claimpilot.validation import (CANDIDATES, SEED, audit_input_groups, compare_reference,
    group_split, original_random_split, run_evaluation, sha256_file, split_diagnostics,
    temporal_split, weight_diagnostic)


def main():
    started = time.time()
    protected_paths = [ROOT/'reports/metrics.json', ROOT/'reports/split_manifest.json', ROOT/'models/selected.joblib']
    before = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected_paths}
    data_path = ROOT/'data/raw/fraud_oracle.csv'
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    features = feature_columns(df)
    if features != FEATURES or len(features) != 19 or not df[TARGET].isin([0, 1]).all() or not df.PolicyNumber.is_unique:
        raise ValueError('승인 스키마/라벨/식별자 조건 불일치')
    X, y = prepare(df, features), df[TARGET]
    groups, table, audit = audit_input_groups(df)
    split = group_split(np.arange(len(df)), groups)
    temporal, historical_sampling = temporal_split(df)
    manifest_path = ROOT/'reports/split_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['data_sha256'] != sha256_file(data_path):
        raise ValueError('매니페스트 원본 데이터 체크섬 불일치')
    random = original_random_split(df, manifest)
    # 이 순서가 중요: temporal 후보 선정은 과거 X/y만 받는 함수 내부에서 완료된다.
    temporal_result = run_evaluation(X, y, temporal)
    temporal_result.update({'heldout_year': 1996, 'selection_years': [1994, 1995],
                            'test_used_for_selection_threshold_or_early_stopping': False,
                            'historical_sampling': historical_sampling,
                            'independence_scope': '이번 실험 내부 모델선정 독립성. 프로젝트 전체 완전 unseen 주장은 아님.'})
    temporal_result['posthoc_input_group_intersections'] = split_diagnostics(df, groups, temporal)['group_intersections']
    group_result = run_evaluation(X, y, split)
    weights = weight_diagnostic(X, y, random)
    old_metrics = json.loads((ROOT/'reports/metrics.json').read_text())
    group_result['compare_reference'] = compare_reference(group_result, old_metrics)
    temporal_result['compare_reference'] = compare_reference(temporal_result, old_metrics)
    weights['reference_balanced_test'] = next(row['test'] for row in old_metrics['models'] if row['name'] == 'Logistic Regression C=0.1')
    weights['balanced_reproduction_max_abs_error'] = max(abs(weights['models'][0]['test'][k] - v) for k, v in weights['reference_balanced_test'].items())
    after = {str(p.relative_to(ROOT)): sha256_file(p) for p in protected_paths}
    if before != after:
        raise RuntimeError('보호 대상 파일이 변경됨')
    result = {
        'configuration': {'seed': SEED, 'features': features, 'feature_count': len(features),
            'data_sha256': sha256_file(data_path), 'raw_input_grouping': True,
            'split_method': 'GroupShuffleSplit seed42, test .2, validation .25 of development',
            'temporal_split_method': '순수 연도 cutoff: 1994~1995 전체 모집단 stratified validation .25, 1996 전체 holdout; 미래 입력/라벨 기반 과거 표본 선택 없음',
            'candidates': list(CANDIDATES), 'lr': {'class_weight': 'balanced', 'max_iter': 1500, 'encoder': 'OneHotEncoder handle_unknown=ignore; train fit only'},
            'catboost': {'iterations': 450, 'learning_rate': .05, 'loss_function': 'Logloss', 'eval_metric': 'PRAUC', 'early_stopping_rounds': 60, 'thread_count': 4, 'allow_writing_files': False},
            'weight_diagnostic': {'C': .1, 'class_weights': ['balanced', None], 'split_source': 'reports/split_manifest.json'},
            'top20_rule': 'ceil(n*.2), 점수 내림차순 stable 동점 순서. 기존 evaluate와 동일.',
            'ap_definition': 'sklearn average_precision_score; 계단형 AP',
            'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'catboost': catboost.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'platform': platform.platform()},
            'protected_files_sha256_before': before, 'protected_files_sha256_after': after,
            'existing_artifacts_unchanged': before == after, 'model_artifact_written': False},
        'split_diagnostics': {'group_evaluation': split_diagnostics(df, groups, split),
                              'temporal_independent': split_diagnostics(df, groups, temporal),
                              'original_random': split_diagnostics(df, groups, random)},
        'group_audit': audit, 'group_evaluation': group_result, 'temporal_independent': temporal_result,
        'weight_diagnostic': weights,
        'limitations': [
            '동일 19개 원문 입력 그룹은 실제 고객/사건 그룹이 아니다. 고객/사건 ID 부재로 실제 독립성·누수 부재를 입증할 수 없다.',
            '1996은 원래 MVP에서 이미 관측되었다. 프로젝트 단위 완전 unseen 미래검증이 아니다. 이번 실험에서는 1996을 후보선정/threshold/early stopping에 엄격히 사용하지 않는다.',
            '시간 분할은 Year의 업무 의미와 정확한 접수 시점이 검증되지 않은 연도 이동 평가다.',
            '시간 실험은 순수 연도 cutoff로 과거 전체 모집단을 유지한다. 시간 split 간 동일 원문 입력 그룹이 존재할 수 있으며 사후 감사 수치만 보고한다. 그룹격리 실험과 별개다.',
            '작은 고정 후보범위의 단일 seed 평가이며 재학습/표본 불확실성 신뢰구간은 제공하지 않는다. 플랫폼별 CatBoost early stopping 차이 가능.',
            '가중치 비교는 사전지정 진단뿐이며 test로 새 production 모델을 선정하거나 조정하지 않는다. 기존 selected.joblib과 metrics는 보존한다.',
            '비보정 모델 점수의 Brier는 참고 진단이며 실제 사기 확률 또는 자동 사기/지급 판단으로 사용하지 않는다.',
            '무작위 기존 평가와 그룹/시간 평가의 표본 차이로 성능 차이를 인과적 개선으로 해석하지 않는다.'
        ],
        'elapsed_seconds': round(time.time() - started, 3),
    }
    table[['input_group_id', 'size', 'positives', 'negatives', 'label_count',
           'conflicting_labels', 'duplicate_excess']].to_csv(ROOT/'reports/phase2_group_audit.csv', index=False)
    (ROOT/'reports/phase2_validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'group_audit': audit, 'group_evaluation': {'selected_model': group_result['selected_model'], 'test': group_result['test']},
                      'temporal_independent': {'selected_model': temporal_result['selected_model'], 'test': temporal_result['test'], 'historical_sampling': historical_sampling},
                      'weight_diagnostic': weights, 'elapsed_seconds': result['elapsed_seconds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
