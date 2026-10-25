import sqlite3

import pytest

from claimpilot.review import MAX_NOTE_LENGTH, STATUSES, ReviewStore


@pytest.fixture
def store(tmp_path):
    return ReviewStore(tmp_path / "nested" / "review.sqlite3")


def test_new_review_and_defaults_do_not_write(store):
    result = store.get("C-1", {"자료 확인": False})
    assert result["status"] == "미검토"
    assert result["note"] == ""
    assert result["checklist"] == {"자료 확인": False}
    assert store.all() == []


def test_persistence_and_update(store):
    store.save("C-1", "검토중", "원문 재확인", {"자료 확인": True, "추가자료": False})
    other = ReviewStore(store.path)
    result = other.get("C-1")
    assert result["note"] == "원문 재확인"
    assert result["checklist"] == {"자료 확인": True, "추가자료": False}
    assert result["updated_at"]
    other.save("C-1", "검토완료", "기존 체크리스트 보존")
    assert store.get("C-1")["checklist"] == result["checklist"]
    store.save("C-1", "추가자료 요청", "체크리스트 교체", {})
    assert other.get("C-1")["checklist"] == {}
    assert len(store.all()) == 1


@pytest.mark.parametrize("status", STATUSES)
def test_allowed_statuses(store, status):
    store.save(1, status, "")
    assert store.get(1)["status"] == status


def test_sql_parameters_and_unicode(store):
    malicious = "x'; DROP TABLE reviews;--"
    store.save(malicious, "검토중", malicious, {"증빙 ' 확인": True})
    store.save("normal", "미검토", "정상")
    assert len(store.all()) == 2
    assert store.get(malicious)["note"] == malicious


@pytest.mark.parametrize("claim_id", ["", " ", "a" * 129, None, True, []])
def test_invalid_id(store, claim_id):
    with pytest.raises(ValueError):
        store.save(claim_id, "미검토", "")


@pytest.mark.parametrize("kwargs", [
    {"status": "지급승인"}, {"note": "x" * (MAX_NOTE_LENGTH + 1)}, {"note": None},
    {"checklist": []}, {"checklist": {"a": 1}}, {"checklist": {"": True}},
    {"checklist": {str(i): False for i in range(31)}},
])
def test_invalid_input_does_not_overwrite(store, kwargs):
    store.save("C", "검토중", "보존", {"확인": True})
    values = {"status": "미검토", "note": "", **kwargs}
    with pytest.raises(ValueError):
        store.save("C", **values)
    assert store.get("C")["note"] == "보존"


def test_note_boundary(store):
    store.save("C", "미검토", "가" * MAX_NOTE_LENGTH)
    assert len(store.get("C")["note"]) == MAX_NOTE_LENGTH


def test_legacy_schema_migration(tmp_path):
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE reviews (claim_id TEXT PRIMARY KEY, status TEXT, note TEXT)")
        conn.execute("INSERT INTO reviews VALUES (?, ?, ?)", ("old", "검토중", "유지"))
    store = ReviewStore(db)
    assert store.get("old")["checklist"] == {}
    assert store.get("old")["note"] == "유지"
    store.save("old", "검토완료", "완료", {"원문": True})
    assert store.get("old")["checklist"]["원문"] is True


def test_defaults_merge_without_overwriting_saved(store):
    store.save("C", "검토중", "", {"확인": True})
    assert store.get("C", {"확인": False, "신규": False})["checklist"] == {
        "확인": True, "신규": False}
