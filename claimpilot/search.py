"""확인된 공식 문서 발췌만 검색한다. 네트워크 및 생성 API를 사용하지 않는다."""
import json
import math
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_SOURCES_PATH = Path(__file__).resolve().parents[1] / "resources" / "official_sources.json"
REQUIRED_FIELDS = ("id", "title", "organization", "url", "accessed_at", "section", "text", "limitations")


def _normalize(value):
    return unicodedata.normalize("NFKC", value).casefold()


def search_sources(query, limit=5, sources_path=None):
    """가중 어휘 일치 검색. score는 검색 관련도이며 확률/법적 효력이 아니다.

    sources_path는 독립 fixture 또는 검증된 로컬 자료 경로 주입용이다.
    빈 검색어/일치 없음은 [], 파일 누락/형식 오류는 명시적으로 예외를 반환한다.
    """
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("검색어는 500자 이하 문자열이어야 합니다.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 50:
        raise ValueError("limit는 0~50 정수여야 합니다.")
    terms = list(dict.fromkeys(re.findall(r"[\w가-힣]+", _normalize(query))))
    if not terms or limit == 0:
        return []
    path = Path(sources_path) if sources_path is not None else DEFAULT_SOURCES_PATH
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list):
        raise ValueError("공식 출처 파일은 배열이어야 합니다.")
    for source in sources:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key].strip()
            for key in REQUIRED_FIELDS
        ):
            raise ValueError("공식 출처의 필수 메타데이터가 누락되었습니다.")
        url = urlsplit(source["url"])
        if url.scheme not in ("https", "http") or not url.netloc or url.username or url.password:
            raise ValueError("공식 출처 URL이 올바르지 않습니다.")
    fields = (("title", 3.0), ("organization", 2.0), ("section", 2.0), ("text", 1.0))
    documents = [{key: _normalize(source[key]) for key, _ in fields} for source in sources]
    frequencies = {term: sum(any(term in text for text in doc.values()) for doc in documents)
                   for term in terms}
    ranked = []
    for source, doc in zip(sources, documents):
        score = 0.0
        for term in terms:
            idf = math.log(1 + (len(sources) + 1) / (frequencies[term] + 1))
            score += idf * sum(weight * min(text.count(term), 3)
                               for key, weight in fields for text in (doc[key],))
        if score > 0:
            ranked.append({**source, "score": round(score, 6)})
    return sorted(ranked, key=lambda item: (-item["score"], item["id"]))[:limit]
