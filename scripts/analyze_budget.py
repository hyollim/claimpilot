#!/usr/bin/env python3
"""기존 실제 held-out 점수 파일만 평가. 모델 학습·재선정·비용 환산 없음."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from claimpilot.budget import DEFAULT_FRACTIONS, budget_curve, budget_metrics, cumulative_capture

NOTES = [
    '공개 데이터의 라벨이 있는 held-out test 표본만 평가한다. 합성 데이터 성과를 사용하지 않는다.',
    'label은 평가 전용이며 모델 입력·검토 의사결정 입력이 아니다.',
    'production 신규 데이터에는 label이 없으므로 포집률·정밀도·lift를 계산할 수 없다.',
    '검토 비율은 건수 기준 용량이다. 비용·절감액·ROI로 환산하지 않는다.',
    '점수는 비보정 순위 점수다. 실제 사기 확률·사기 확정·보험금 지급 판단이 아니다.',
    '양성 없음은 status=no_positives, capture_rate=0, lift=0으로 표시한다. 무작위 기준선 k/n은 유지한다.',
    '고정된 선택 모델의 기술적 평가이며 test 결과로 모델이나 최적 예산을 재선정하지 않는다.',
]


def build_report(fractions=DEFAULT_FRACTIONS):
    """정해진 실제 reports 입력을 읽어 출처 및 기존 top20 지표를 검증한다."""
    score_path, metrics_path = ROOT / 'reports/scored_claims.csv', ROOT / 'reports/metrics.json'
    score_bytes, metrics_bytes = score_path.read_bytes(), metrics_path.read_bytes()
    metadata = json.loads(metrics_bytes)
    rows = list(csv.DictReader(io.StringIO(score_bytes.decode('utf-8-sig'))))
    if not rows or not {'priority_rank', 'risk_score', 'label'}.issubset(rows[0]):
        raise ValueError('실제 held-out 점수와 평가 전용 label·priority_rank가 필요합니다.')
    # CSV 내 원래 행 순서를 동점 기준으로 보존하며 기존 priority_rank 순서를 따른다.
    rows.sort(key=lambda row: int(row['priority_rank']))
    if [int(row['priority_rank']) for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError('priority_rank는 중복 없이 1..n이어야 합니다.')
    scores = [float(row['risk_score']) for row in rows]
    labels = [float(row['label']) for row in rows]
    if any(scores[i] < scores[i + 1] for i in range(len(scores) - 1)):
        raise ValueError('priority_rank가 점수 내림차순과 일치하지 않습니다.')
    if any('FraudFound_P' in row and float(row['FraudFound_P']) != label
           for row, label in zip(rows, labels)):
        raise ValueError('평가 라벨 열이 원 라벨과 불일치합니다.')
    top20 = budget_metrics(labels, scores, .20)
    selected_model = metadata['selected_model']
    matches = [m for m in metadata['models'] if m['name'] == selected_model]
    if len(matches) != 1:
        raise ValueError('선택 모델의 기존 지표를 유일하게 찾을 수 없습니다.')
    expected = matches[0]['test']
    mapping = {'k': 'reviewed_count', 'captured_count': 'captured_count',
               'capture_rate': 'capture_at_20', 'precision': 'precision_at_20',
               'lift': 'lift_at_20', 'n': 'n', 'positives': 'positives'}
    checks = {new: math.isclose(top20[new], expected[old], rel_tol=1e-12, abs_tol=1e-12)
              for new, old in mapping.items()}
    test_split = metadata['split']['test']
    if not all(checks.values()) or (top20['n'], top20['positives']) != (test_split['n'], test_split['positives']):
        raise ValueError('실제 점수 파일이 기존 선택 모델 held-out test/top20 지표와 불일치합니다.')
    source_files = {
        'scored_claims': {'path': str(score_path.relative_to(ROOT)), 'sha256': hashlib.sha256(score_bytes).hexdigest()},
        'metrics': {'path': str(metrics_path.relative_to(ROOT)), 'sha256': hashlib.sha256(metrics_bytes).hexdigest()},
    }
    return dict(selected_model=selected_model, evaluation_split='heldout_test',
                source_files=source_files, selection_rule=metadata.get('selection_rule'),
                ranking='priority_rank ascending; descending risk_score; stable CSV row order for score ties',
                rounding='ceil(n * requested_fraction); reviewed_fraction=k/n',
                n=top20['n'], positives=top20['positives'], prevalence=top20['prevalence'],
                notes=NOTES, top20_regression={'status': 'passed', 'checks': checks, 'actual': top20},
                budgets=budget_curve(labels, scores, fractions),
                cumulative_capture=cumulative_capture(labels, scores))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fractions', nargs='+', type=float, default=DEFAULT_FRACTIONS,
                        help='검토 비율 (0,1], 예: --fractions .07 .2 .5')
    args = parser.parse_args(argv)
    report = build_report(args.fractions)
    out_json, out_csv = ROOT / 'reports/review_budget.json', ROOT / 'reports/review_budget.csv'
    context = dict(selected_model=report['selected_model'], evaluation_split=report['evaluation_split'],
                   source_scored_claims=report['source_files']['scored_claims']['path'],
                   source_metrics=report['source_files']['metrics']['path'],
                   evaluation_note='label 평가 전용; production 무라벨 신규 데이터 적용 불가; 비용 환산 없음; 합성 성과 미사용')
    records = [dict(**context, **point) for point in report['budgets']]
    with out_csv.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f"{report['selected_model']} | heldout_test n={report['n']} positives={report['positives']} | top20 regression passed")
    print('requested actual k captured capture precision random_expected_count random_capture lift')
    for row in report['budgets']:
        print(f"{row['requested_fraction']:.2%} {row['reviewed_fraction']:.4%} {row['k']} "
              f"{row['captured_count']} {row['capture_rate']:.4%} {row['precision']:.4%} "
              f"{row['random_expected_count']:.4f} {row['random_expected_capture']:.4%} {row['lift']:.4f}")
    print('reports/review_budget.csv, reports/review_budget.json')


if __name__ == '__main__':
    main()
