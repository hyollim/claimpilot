"""고정 공식 발췌 검색. stdlib 한국어 character ngram(2~4) BM25.

점수는 확률이 아니다. []는 '관련 발췌 없음'으로 표시해야 한다.
검색 결과는 업무 참고자료이며 모델 설명/사기·지급 판단과 분리한다.
기본 threshold=0은 순위 검색용이며, 답 없는 질문 판별을 보장하지 않는다.
"""
from collections import Counter
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from claimpilot.search import DEFAULT_SOURCES_PATH, REQUIRED_FIELDS, _normalize, search_sources

NO_EXCERPT_MESSAGE = "관련 발췌 없음"
K1 = 1.2
B = 0.75
DEFAULT_THRESHOLD = 0.0  # UI 기본값 결정은 호출자 소관. 평가 임계값은 보고서에 별도 저장.
FIELD_WEIGHTS = {"title": 3, "organization": 2, "section": 2, "text": 1}
ALLOWED_HOSTS = frozenset({"kbinsure.co.kr", "www.kbinsure.co.kr", "knia.or.kr",
                           "www.knia.or.kr", "accident.knia.or.kr"})


def character_ngrams(text):
    """NFKC/casefold 후 단어 내부 2~4자. 공백/문장 경계를 넘지 않는다.

    1자 입력은 substring 과잉일치를 막기 위해 검색하지 않는다.
    단어 내부 부분일치는 설계상 가능하며 의미적 답변 가능성을 보장하지 않는다.
    """
    return [word[i:i + n] for word in re.findall(r"[^\W_]+", _normalize(text))
            for n in range(2, 5) for i in range(len(word) - n + 1)]


def _load_sources(sources_path):
    path = Path(sources_path) if sources_path is not None else DEFAULT_SOURCES_PATH
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list):
        raise ValueError("공식 출처 파일은 배열이어야 합니다.")
    seen = set()
    for source in sources:
        if not isinstance(source, dict) or any(
            not isinstance(source.get(key), str) or not source[key].strip()
            for key in REQUIRED_FIELDS
        ):
            raise ValueError("공식 출처의 필수 메타데이터가 누락되었습니다.")
        url = urlsplit(source["url"])
        if (url.scheme != "https" or url.hostname not in ALLOWED_HOSTS
                or url.username is not None or url.password is not None
                or url.port not in (None, 443)):
            raise ValueError("KB/KNIA 공식 HTTPS 도메인만 허용합니다.")
        if source["id"] in seen:
            raise ValueError("중복 출처 ID입니다.")
        seen.add(source["id"])
    return sources


def search_documents(query, limit=5, sources_path=None, method="bm25", *, threshold=None):
    """기존 메타데이터+score 목록. bm25 또는 기존 substring 기준선.

    threshold는 원시 검색 점수의 하한(포함)이며 0도 유효하다. 점수 0은 제외.
    가중 title/organization/section/text의 빈도로 BM25를 계산한다.
    k1=1.2, b=.75 고정. 질의 ngram은 중복 제거, 동점은 ID 오름차순.
    파일 누락/형식 오류/비공식 URL은 명시적 예외. 네트워크·생성 API 없음.
    """
    if not isinstance(query, str) or len(query) > 500:
        raise ValueError("검색어는 500자 이하 문자열이어야 합니다.")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 50:
        raise ValueError("limit는 0~50 정수여야 합니다.")
    if method not in ("bm25", "substring"):
        raise ValueError("method는 bm25 또는 substring이어야 합니다.")
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold < 0):
        raise ValueError("threshold는 유한한 0 이상 숫자여야 합니다.")
    if limit == 0 or not re.search(r"\w", query):
        return []
    terms = set(character_ngrams(query))
    if method == "bm25" and not terms:
        return []
    sources = _load_sources(sources_path)
    if method == "substring":
        # 기존 구현을 그대로 호출. 추가 allowlist만 이 래퍼에서 적용한다.
        return [row for row in search_sources(query, 50, sources_path)
                if row["score"] >= threshold][:limit]
    if not sources:
        return []
    counts = []
    for source in sources:
        counter = Counter()
        for field, weight in FIELD_WEIGHTS.items():
            counter.update({term: freq * weight
                            for term, freq in Counter(character_ngrams(source[field])).items()})
        counts.append(counter)
    lengths = [sum(counter.values()) for counter in counts]
    average_length = sum(lengths) / len(sources)
    if not average_length:
        return []
    dfs = {term: sum(term in counter for counter in counts) for term in terms}
    ranked = []
    for source, counter, length in zip(sources, counts, lengths):
        score = 0.0
        for term in sorted(terms):
            frequency = counter[term]
            if frequency:
                idf = math.log(1 + (len(sources) - dfs[term] + 0.5) / (dfs[term] + 0.5))
                score += idf * frequency * (K1 + 1) / (
                    frequency + K1 * (1 - B + B * length / average_length))
        if score > 0 and score >= threshold:
            ranked.append({**source, "score": round(score, 6)})
    return sorted(ranked, key=lambda row: (-row["score"], row["id"]))[:limit]
