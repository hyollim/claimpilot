import json

import pytest

from claimpilot.search import search_sources


@pytest.fixture
def sources_path(tmp_path):
    sources = [
        {"id": "b", "title": "보험금 청구 절차", "text": "사고 접수 및 증빙 서류 안내"},
        {"id": "a", "title": "과실비율 기준", "text": "과실비율은 참고자료이며 개별 판단이 필요합니다."},
        {"id": "c", "title": "사고 접수", "text": "온라인 접수 단계"},
    ]
    for item in sources:
        item.update(organization="공식기관 fixture", url="https://example.org/" + item["id"],
                    accessed_at="2026-10-25", section="안내", limitations="최신 원문 재확인")
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(sources, ensure_ascii=False), encoding="utf-8")
    return path


def test_korean_search_metadata_and_limit(sources_path):
    results = search_sources("접수", 1, sources_path)
    assert len(results) == 1
    assert results[0]["id"] == "c"  # 제목 가중치
    assert results[0]["score"] > 0
    assert results[0]["organization"] == "공식기관 fixture"
    assert results[0]["url"] == "https://example.org/c"
    assert results[0]["limitations"]


def test_no_match_and_empty(sources_path):
    for query in ("", "   ", "!!!", "없는키워드"):
        assert search_sources(query, sources_path=sources_path) == []
    assert search_sources("접수", 0, sources_path) == []


def test_multiple_terms_and_determinism(sources_path):
    first = search_sources("접수 서류", sources_path=sources_path)
    assert {r["id"] for r in first} == {"b", "c"}
    assert first == search_sources("접수 접수 서류", sources_path=sources_path)


@pytest.mark.parametrize("query,limit", [(None, 5), ("a" * 501, 5), ("접수", -1),
                                         ("접수", 51), ("접수", True), ("접수", 1.5)])
def test_invalid_arguments(sources_path, query, limit):
    with pytest.raises(ValueError):
        search_sources(query, limit, sources_path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        search_sources("접수", sources_path=tmp_path / "missing.json")


@pytest.mark.parametrize("content", ["{}", "[{}]", "[null]", "invalid"])
def test_invalid_document(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        search_sources("접수", sources_path=path)


def test_unsafe_url_rejected(sources_path):
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    sources[0]["url"] = "javascript:alert(1)"
    sources_path.write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(ValueError):
        search_sources("접수", sources_path=sources_path)


def test_file_is_not_mutated(sources_path):
    before = sources_path.read_bytes()
    result = search_sources("과실", sources_path=sources_path)
    assert result[0]["id"] == "a"
    assert sources_path.read_bytes() == before
