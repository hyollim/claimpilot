"""8문서 내부 검색평가. 임계값은 개발만 사용하고 평가 뒤 변경하지 않는다."""
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from claimpilot.retrieval import B, FIELD_WEIGHTS, K1, search_documents
from claimpilot.search import search_sources

# 실행 전 고정한 후보와 목적함수. 평가 문항·점수는 select_threshold에 전달하지 않는다.
THRESHOLD_CANDIDATES = (0, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def ranking_metrics(retrieved, relevant, k=3):
    relevant = set(relevant)
    top = retrieved[:k]
    hits = [int(doc_id in relevant) for doc_id in top]
    recall = len(set(top) & relevant) / len(relevant) if relevant else 0.0
    reciprocal = next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    return {"recall_at_3": recall, "mrr_at_3": reciprocal,
            "ndcg_at_3": dcg / ideal if ideal else 0.0}


def evaluate(questions, no_answer, retrieve):
    rows = []
    for question in questions:
        results = retrieve(question["query"])
        ids = [row["id"] for row in results]
        rows.append({**question, "retrieved_ids": ids,
                     "scores": [row["score"] for row in results],
                     **ranking_metrics(ids, question["relevant_ids"])})
    negatives = []
    for question in no_answer:
        results = retrieve(question["query"])
        negatives.append({**question, "retrieved_ids": [row["id"] for row in results],
                          "false_positive": bool(results),
                          "display": "관련 발췌 없음" if not results else "참고 발췌 반환(오탐)"})
    metrics = {key: sum(row[key] for row in rows) / len(rows) if rows else 0.0
               for key in ("recall_at_3", "mrr_at_3", "ndcg_at_3")}
    metrics.update(answerable_count=len(rows), no_answer_count=len(negatives),
                   no_answer_false_positive_count=sum(row["false_positive"] for row in negatives),
                   no_answer_false_positive_rate=(sum(row["false_positive"] for row in negatives)
                                                  / len(negatives) if negatives else 0.0))
    return {"metrics": metrics, "questions": rows, "no_answer": negatives}


def select_threshold(development, development_no_answer, sources_path):
    trials = []
    for threshold in THRESHOLD_CANDIDATES:
        result = evaluate(development, development_no_answer,
                          lambda q: search_documents(q, 3, sources_path, threshold=threshold))
        metrics = result["metrics"]
        ranking_mean = sum(metrics[key] for key in ("recall_at_3", "mrr_at_3", "ndcg_at_3")) / 3
        objective = (ranking_mean + 1 - metrics["no_answer_false_positive_rate"]) / 2
        trials.append({"threshold": threshold, "objective": objective, **metrics})
    best = max(trials, key=lambda row: (row["objective"], -row["threshold"]))
    return best["threshold"], trials


def main():
    source_path = ROOT / "resources/official_sources.json"
    query_path = ROOT / "resources/retrieval_queries.json"
    source_bytes = source_path.read_bytes()
    query_bytes = query_path.read_bytes()
    sources = json.loads(source_bytes)
    manifest = json.loads(query_bytes)
    if (hashlib.sha256(source_bytes).hexdigest() != manifest["frozen_source_sha256"]
            or [s["id"] for s in sources] != manifest["frozen_source_ids"]
            or len(sources) != 8 or len(manifest["gold_evidence"]) != 8):
        raise ValueError("고정 8문서 불일치: 후보 출처 혼합 금지")
    if (len(manifest["development"]) != 8 or len(manifest["fixed_evaluation"]) != 12
            or len(manifest["no_answer"]) != 4
            or any(sum(q["split"] == split for q in manifest["no_answer"]) != 2
                   for split in ("development", "fixed_evaluation"))):
        raise ValueError("사전 고정 평가 분할 개수 불일치")
    for source, evidence in zip(sources, manifest["gold_evidence"]):
        if any(source[key] != evidence[key] for key in ("id", "url", "section")) or source["text"] != evidence["quote"]:
            raise ValueError("사전 고정 원문 근거가 변경되었습니다.")
    development = manifest["development"]
    dev_no_answer = [q for q in manifest["no_answer"] if q["split"] == "development"]
    threshold, trials = select_threshold(development, dev_no_answer, source_path)
    # 이 지점에서 선택은 완료. 아래 평가 결과로 후보/목적함수/파라미터를 바꾸지 않는다.
    fixed = manifest["fixed_evaluation"]
    fixed_no_answer = [q for q in manifest["no_answer"] if q["split"] == "fixed_evaluation"]
    methods = {
        "substring_baseline": lambda q: search_sources(q, 3, source_path),
        "bm25_threshold_0": lambda q: search_documents(q, 3, source_path, threshold=0),
        "bm25_dev_selected": lambda q: search_documents(q, 3, source_path, threshold=threshold),
    }
    report = {
        "scope": manifest["scope"], "gold_provenance": manifest["declaration"],
        "split_policy": manifest["split_policy"], "gold_evidence": manifest["gold_evidence"],
        "configuration": {"python": platform.python_version(), "implementation": "stdlib",
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "queries_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "source_count": len(sources), "source_ids": manifest["frozen_source_ids"],
            "ngram_range": [2, 4], "word_boundary_crossing": False,
            "k1": K1, "b": B, "field_weights": FIELD_WEIGHTS,
            "query_ngram_deduplication": True,
            "selected_threshold": threshold, "default_api_threshold": 0,
            "selection_data": "development 8 answerable + 2 separately declared no-answer only",
            "threshold_candidates": list(THRESHOLD_CANDIDATES),
            "objective": "0.5 * mean(Recall@3, MRR@3, nDCG@3) + 0.5 * no-answer specificity; ties choose lower threshold",
            "evaluation_used_for_tuning": False,
            "metric_definitions": {"recall_at_3": "macro relevant IDs retrieved / total relevant IDs",
                "mrr_at_3": "macro reciprocal first relevant rank; absent=0",
                "ndcg_at_3": "binary relevance DCG / ideal DCG, log2 discount",
                "no_answer_false_positive_rate": "fraction of no-answer questions returning any document"}},
        "threshold_selection_trials_development_only": trials,
        "development": {name: evaluate(development, dev_no_answer, fn) for name, fn in methods.items()},
        "fixed_evaluation": {name: evaluate(fixed, fixed_no_answer, fn) for name, fn in methods.items()},
        "interpretation": {**manifest["interpretation"],
            "limitations": ["20개 수작업 자율합성 질의와 4개 답 없음 문항의 작은 내부 평가. 외부 독립 gold가 아니다.",
                "동일 8문서의 짧은 발췌를 활용한 표현변경 평가로 일반화 성능을 입증하지 않는다.",
                "ngram 부분일치 및 주제는 비슷하지만 답 없는 질문의 오탐 가능. 임계값은 의미적 답변 가능성 검증이 아니다.",
                "점수 척도가 달라 두 검색 방식 점수를 확률처럼 직접 비교할 수 없다.",
                "관련 발췌 없음은 검색 결과가 없다는 뜻이며 실제 공식 정보의 부재를 단정하지 않는다."],
            "ui_default_decision": "parent 결정. 기존 search.py와 UI 기본값은 변경하지 않음."}}
    report["no_answer_all_four_descriptive"] = {name: {
        "false_positive_count": report["development"][name]["metrics"]["no_answer_false_positive_count"]
            + report["fixed_evaluation"][name]["metrics"]["no_answer_false_positive_count"],
        "count": 4,
        "note": "개발 2개를 포함한 기술통계이며 독립 평가 지표가 아님"}
        for name in methods}
    report["comparison"] = {name: {key: row["metrics"][key] - report["fixed_evaluation"]["substring_baseline"]["metrics"][key]
        for key in ("recall_at_3", "mrr_at_3", "ndcg_at_3", "no_answer_false_positive_rate")}
        for name, row in report["fixed_evaluation"].items() if name != "substring_baseline"}
    if source_bytes != source_path.read_bytes() or query_bytes != query_path.read_bytes():
        raise RuntimeError("평가 중 고정 자료가 변경되었습니다.")
    output = ROOT / "reports/retrieval_evaluation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_threshold": threshold,
        "fixed_evaluation": {name: result["metrics"] for name, result in report["fixed_evaluation"].items()},
        "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
