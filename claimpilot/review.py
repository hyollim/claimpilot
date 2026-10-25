"""단일 사용자 로컬 PoC용 검토 기록. 운영 인증/권한/감사 보안은 제공하지 않는다."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

STATUSES = ("미검토", "검토중", "추가자료 요청", "검토완료")
MAX_NOTE_LENGTH = 4000
MAX_CHECKLIST_ITEMS = 30


def _claim_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("청구 ID는 문자열 또는 정수여야 합니다.")
    value = str(value).strip()
    if not value or len(value) > 128:
        raise ValueError("청구 ID는 1~128자여야 합니다.")
    return value


def _checklist(value):
    if not isinstance(value, dict) or len(value) > MAX_CHECKLIST_ITEMS:
        raise ValueError("체크리스트는 30개 이하의 항목 사전이어야 합니다.")
    if any(not isinstance(k, str) or not k.strip() or len(k) > 100
           or not isinstance(v, bool) for k, v in value.items()):
        raise ValueError("체크리스트 항목명은 1~100자, 값은 bool이어야 합니다.")
    return dict(value)


class ReviewStore:
    """각 저장은 트랜잭션으로 수행한다. checklist 생략 시 기존 값을 보존한다."""

    def __init__(self, path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("지속저장을 위해 파일 경로를 지정하세요.")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS reviews (
                claim_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                note TEXT NOT NULL, checklist TEXT NOT NULL DEFAULT '{}',
                updated_at TEXT NOT NULL DEFAULT '')""")
            # 기존 최소 계약(status/note) DB도 비파괴적으로 확장한다.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(reviews)")}
            if "checklist" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN checklist TEXT NOT NULL DEFAULT '{}'")
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE reviews ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''")

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _decode(row):
        record = dict(row)
        record["checklist"] = _checklist(json.loads(record["checklist"]))
        return record

    def get(self, claim_id, checklist=None):
        """미저장 건은 미검토. optional checklist는 신규 항목의 기본값이다."""
        claim_id = _claim_id(claim_id)
        defaults = {} if checklist is None else _checklist(checklist)
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT * FROM reviews WHERE claim_id = ?", (claim_id,)).fetchone()
        record = self._decode(row) if row else {
            "claim_id": claim_id, "status": STATUSES[0], "note": "",
            "checklist": {}, "updated_at": "",
        }
        record["checklist"] = {**defaults, **record["checklist"]}
        return record

    def save(self, claim_id, status, note, checklist=None):
        claim_id = _claim_id(claim_id)
        if status not in STATUSES:
            raise ValueError("지원하지 않는 검토 상태입니다.")
        if not isinstance(note, str) or len(note) > MAX_NOTE_LENGTH:
            raise ValueError(f"메모는 {MAX_NOTE_LENGTH:,}자 이하여야 합니다.")
        serialized = None if checklist is None else json.dumps(
            _checklist(checklist), ensure_ascii=False, sort_keys=True)
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as conn, conn:
            conn.execute("""INSERT INTO reviews (claim_id, status, note, checklist, updated_at)
                VALUES (?, ?, ?, COALESCE(?, '{}'), ?)
                ON CONFLICT(claim_id) DO UPDATE SET status=excluded.status,
                note=excluded.note, checklist=COALESCE(?, reviews.checklist),
                updated_at=excluded.updated_at""",
                (claim_id, status, note, serialized, now, serialized))

    def all(self):
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT * FROM reviews ORDER BY claim_id").fetchall()
        return [self._decode(row) for row in rows]
