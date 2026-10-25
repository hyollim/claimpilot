import json
import math
from pathlib import Path

import pytest

from claimpilot.retrieval import (NO_EXCERPT_MESSAGE, character_ngrams,
                                  search_documents)
from claimpilot.search import search_sources
from scripts.evaluate_retrieval import ranking_metrics, select_threshold


@pytest.fixture
def sources_path(tmp_path):
    sources = [
        {"id": "a", "title": "접수", "text": "접수"},
        {"id": "b", "title": "없음", "text": "없음"},
    ]
    for source in sources:
        source.update(organization="가", section="가", limitations="참고용",
                      url="https://www.kbinsure.co.kr/test", accessed_at="2026-10-25")
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    return path


def test_ngrams_normalization_and_boundaries():
    assert character_ngrams("가나다라") == ["가나", "나다", "다라", "가나다", "나다라", "가나다라"]
    assert "나 다" not in character_ngrams("가나 다라")
    assert "나다" not in character_ngrams("가나 다라")
    assert character_ngrams("ＳＭＳ") == ["sm", "ms", "sms"]
    assert character_ngrams("가 0 !") == []


def test_exact_bm25_formula_and_metadata(sources_path):
    result = search_documents("접수", sources_path=sources_path)
    assert [row["id"] for row in result] == ["a"]
    assert result[0]["score"] == round(math.log(2) * 4 * 2.2 / (4 + 1.2), 6)
    original = json.loads(sources_path.read_text(encoding="utf-8"))[0]
    assert {key: result[0][key] for key in original} == original
    assert result[0]["score"] > 1  # 검색 점수는 확률이 아님


def test_threshold_zero_and_empty_display(sources_path):
    assert search_documents("접수", sources_path=sources_path, threshold=0)
    assert search_documents("접수", sources_path=sources_path, threshold=99999) == []
    assert NO_EXCERPT_MESSAGE == "관련 발췌 없음"


@pytest.mark.parametrize("query", ["", " ", "!!!", "0", "수", "가", "xyzxyz"])
def test_noquery_and_single_character_substring_false_hits(sources_path, query):
    assert search_documents(query, sources_path=sources_path) == []


def test_single_character_baseline_remains_unchanged(sources_path):
    assert search_sources("수", sources_path=sources_path)
    assert search_documents("수", sources_path=sources_path) == []


def test_partial_token_overlap_is_not_semantic_no_answer_detection(sources_path):
    # 2자 이상 ngram은 주제만 같은 답 없는 질문도 반환할 수 있다.
    assert search_documents("접수 취소 수수료", sources_path=sources_path)


def test_baseline_wrapper_uses_original_scores(sources_path):
    assert search_documents("접수", sources_path=sources_path, method="substring") == search_sources("접수", sources_path=sources_path)


def test_limit_dedup_determinism_and_nonmutation(sources_path):
    before = sources_path.read_bytes()
    assert search_documents("접수", 0, sources_path) == []
    assert len(search_documents("접수", 1, sources_path)) == 1
    assert search_documents("접수 접수", sources_path=sources_path) == search_documents("접수", sources_path=sources_path)
    assert sources_path.read_bytes() == before


def test_ties_sorted_by_id(sources_path):
    sources = json.loads(sources_path.read_text())
    sources[1] = {**sources[0], "id": "0"}
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    assert [row["id"] for row in search_documents("접수", sources_path=sources_path)] == ["0", "a"]


@pytest.mark.parametrize("kwargs", [
    {"query": None}, {"query": 0}, {"query": "가" * 501},
    {"limit": True}, {"limit": -1}, {"limit": 51}, {"limit": 1.5},
    {"threshold": True}, {"threshold": -1}, {"threshold": float("nan")},
    {"threshold": float("inf")}, {"threshold": "0"}, {"method": "unknown"},
])
def test_invalid_arguments(sources_path, kwargs):
    args = {"query": "접수", "sources_path": sources_path, **kwargs}
    with pytest.raises(ValueError):
        search_documents(**args)


@pytest.mark.parametrize("url", [
    "https://example.org/x", "https://kbinsure.co.kr.evil.example/x",
    "https://evilknia.or.kr/x", "https://www.kbinsure.co.kr@evil.example/x",
    "https://user@www.kbinsure.co.kr/x", "https://www.kbinsure.co.kr:444/x",
    "http://www.kbinsure.co.kr/x", "javascript:alert(1)", "//accident.knia.or.kr/x",
])
@pytest.mark.parametrize("method", ["bm25", "substring"])
def test_strict_domain_allowlist(sources_path, url, method):
    sources = json.loads(sources_path.read_text())
    sources[0]["url"] = url
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(ValueError):
        search_documents("접수", sources_path=sources_path, method=method)


@pytest.mark.parametrize("url", ["https://www.kbinsure.co.kr/x", "https://accident.knia.or.kr/index"])
def test_official_urls_accepted(sources_path, url):
    sources = json.loads(sources_path.read_text())
    sources[0]["url"] = url
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    assert search_documents("접수", sources_path=sources_path)


@pytest.mark.parametrize("content", ["{}", "[{}]", "[null]", "invalid"])
def test_invalid_files(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        search_documents("접수", sources_path=path)


def test_missing_file_and_empty_corpus(tmp_path):
    with pytest.raises(FileNotFoundError):
        search_documents("접수", sources_path=tmp_path / "missing.json")
    path = tmp_path / "empty.json"
    path.write_text("[]")
    assert search_documents("접수", sources_path=path) == []


def test_duplicate_id_rejected(sources_path):
    sources = json.loads(sources_path.read_text())
    sources[1]["id"] = sources[0]["id"]
    sources_path.write_text(json.dumps(sources))
    with pytest.raises(ValueError):
        search_documents("접수", sources_path=sources_path)


def test_metrics_known_ranks():
    assert ranking_metrics(["x", "a", "b"], ["a"]) == {
        "recall_at_3": 1, "mrr_at_3": 0.5, "ndcg_at_3": 1 / math.log2(3)}
    assert ranking_metrics(["x"], ["a"])["recall_at_3"] == 0
    assert ranking_metrics(["a", "b", "x"], ["a", "b"])["ndcg_at_3"] == 1
    assert ranking_metrics([], ["a"])["mrr_at_3"] == 0


def test_threshold_selection_needs_development_only(sources_path):
    threshold, trials = select_threshold(
        [{"id": "D", "query": "접수", "relevant_ids": ["a"]}],
        [{"id": "N", "query": "독립질문", "relevant_ids": []}], sources_path)
    assert threshold == 0
    assert trials[0]["objective"] == 1


def test_fixed_manifest_scope_and_evidence():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "resources/retrieval_queries.json").read_text())
    sources = json.loads((root / "resources/official_sources.json").read_text())
    assert len(manifest["development"]) == 8
    assert len(manifest["fixed_evaluation"]) == 12
    assert len(manifest["no_answer"]) == 4
    assert [row["id"] for row in sources] == manifest["frozen_source_ids"]
    assert len(manifest["gold_evidence"]) == 8
    for source, evidence in zip(sources, manifest["gold_evidence"]):
        assert source["id"] == evidence["id"]
        assert source["text"] == evidence["quote"]
    ids = set(manifest["frozen_source_ids"])
    for question in manifest["development"] + manifest["fixed_evaluation"]:
        assert set(question["relevant_ids"]) <= ids
    assert all(not question["relevant_ids"] for question in manifest["no_answer"])
