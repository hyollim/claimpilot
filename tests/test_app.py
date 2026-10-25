"""실제 생성 산출물로 Streamlit 화면 경로와 SQLite 복원을 확인한다."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
import claimpilot.review as review
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def app(tmp_path, monkeypatch):
    if not all((ROOT/'reports'/n).exists() for n in ['scored_claims.csv','metrics.json','explanations.json']):
        pytest.skip('실제 데이터 학습 후 UI 통합 테스트 가능')
    real=review.ReviewStore
    monkeypatch.setattr(review,'ReviewStore',lambda *args,**kwargs:real(tmp_path/'review.sqlite'))
    return AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()

def test_app_initial_and_no_label_in_review(app):
    assert not app.exception
    assert [t.label for t in app.tabs]==['청구 검토','성능 평가','공식 출처 검색']
    for frame in app.tabs[0].dataframe:
        assert not any(str(c).lower() in ['label','fraudfound_p','정답 라벨 (평가 전용)'] for c in frame.value.columns)
    # AgeOfVehicle와 AgentType이 age 부분문자열 때문에 숨겨지던 회귀 방지.
    details=app.tabs[0].dataframe[1].value
    assert '차량 연식 구간' in details['항목'].tolist()
    assert '모집 경로 구분' in details['항목'].tolist()

def test_review_form_persists(app):
    status=next(s for s in app.selectbox if s.label=='검토 상태')
    status.set_value('검토중')
    next(t for t in app.text_area if t.label=='검토 메모').set_value('자동 테스트: 원본 대조 필요')
    app.checkbox[0].check()
    next(b for b in app.button if b.label=='검토 기록 저장').click().run()
    assert not app.exception
    assert next(s for s in app.selectbox if s.label=='검토 상태').value=='검토중'
    assert app.checkbox[0].value
    app.run()
    assert next(t for t in app.text_area if t.label=='검토 메모').value=='자동 테스트: 원본 대조 필요'

def test_official_search(app):
    next(t for t in app.text_input if t.label=='검색어').set_value('과실비율')
    next(b for b in app.button if b.label=='공식 발췌 검색').click().run()
    assert not app.exception
    assert any('과실비율' in m.value for m in app.markdown)
    assert len(app.get('link_button'))>0
