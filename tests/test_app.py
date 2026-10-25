"""실제 생성 산출물로 Streamlit 화면 경로와 SQLite 복원을 확인한다."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
import claimpilot.review as review
import claimpilot.intake as intake
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def app(tmp_path, monkeypatch):
    if not all((ROOT/'reports'/n).exists() for n in ['scored_claims.csv','metrics.json','explanations.json']):
        pytest.skip('실제 데이터 학습 후 UI 통합 테스트 가능')
    real=review.ReviewStore
    monkeypatch.setattr(review,'ReviewStore',lambda *args,**kwargs:real(tmp_path/'review.sqlite'))
    real_batch=intake.BatchStore
    monkeypatch.setattr(intake,'BatchStore',lambda *args,**kwargs:real_batch(tmp_path/'intake.sqlite'))
    return AppTest.from_file(str(ROOT/'app.py'),default_timeout=20).run()

def test_app_initial_and_no_label_in_review(app):
    assert not app.exception
    assert next(r for r in app.radio if r.label=='업무 메뉴').options==['업무 현황','청구 데이터 분석','청구 상세 검토','모델 성능 및 업무 효과','공식 업무 참고자료']
    next(r for r in app.radio if r.label=='업무 메뉴').set_value('청구 상세 검토').run()
    for frame in app.dataframe:
        assert not any(str(c).lower() in ['label','fraudfound_p','정답 라벨 (평가 전용)'] for c in frame.value.columns)
    # AgeOfVehicle와 AgentType이 age 부분문자열 때문에 숨겨지던 회귀 방지.
    details=app.dataframe[1].value
    assert '차량 연식 구간' in details['항목'].tolist()
    assert '모집 경로 구분' in details['항목'].tolist()

def test_review_form_persists(app):
    next(r for r in app.radio if r.label=="업무 메뉴").set_value("청구 상세 검토").run()
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
    next(r for r in app.radio if r.label=="업무 메뉴").set_value("공식 업무 참고자료").run()
    next(t for t in app.text_input if t.label=='검색어').set_value('과실비율')
    next(b for b in app.button if b.label=='공식 발췌 검색').click().run()
    assert not app.exception
    assert any('과실비율' in m.value for m in app.markdown)
    assert len(app.get('link_button'))>0


def test_synthetic_analysis_and_review_persistence(app):
    if not (ROOT/'models/inference_bundle.joblib').exists():
        pytest.skip('추론 번들 생성 후 실제 신규 배치 통합 테스트')
    next(r for r in app.radio if r.label=='업무 메뉴').set_value('청구 데이터 분석').run()
    next(b for b in app.button if b.label=='합성 시연 6건 분석').click().run()
    assert not app.exception
    assert any('6건 분석 완료' in s.value for s in app.success)
    assert any('합성' in c.value for c in app.caption)
    next(b for b in app.button if b.label=='이 배치 상세 검토').click().run()
    assert not app.exception
    assert next(r for r in app.radio if r.label=='업무 메뉴').value=='청구 상세 검토'
    assert next(s for s in app.selectbox if s.label=='검토 자료').value!='mvp'
    assert any('DEMO-' in str(frame.value) for frame in app.dataframe)
    next(s for s in app.selectbox if s.label=='검토 상태').set_value('추가자료 요청')
    next(t for t in app.text_area if t.label=='검토 메모').set_value('합성 시연: 미관측 범주 대조')
    next(b for b in app.button if b.label=='검토 기록 저장').click().run()
    assert not app.exception
    app.run()
    assert next(s for s in app.selectbox if s.label=='검토 상태').value=='추가자료 요청'
    next(s for s in app.selectbox if s.label=='검토 자료').set_value('mvp').run()
    assert next(s for s in app.selectbox if s.label=='검토 상태').value=='미검토'


def test_dynamic_budget_does_not_use_new_labels(app):
    next(r for r in app.radio if r.label=='업무 메뉴').set_value('모델 성능 및 업무 효과').run()
    next(s for s in app.slider if s.label=='우선 검토 비율 (%)').set_value(10).run()
    assert not app.exception
    assert any('309건 검토 / 39건 포집' in m.value for m in app.markdown)


def test_filters_do_not_leak_between_datasets(app):
    if not (ROOT/'models/inference_bundle.joblib').exists():pytest.skip('추론 번들 필요')
    next(r for r in app.radio if r.label=='업무 메뉴').set_value('청구 데이터 분석').run()
    next(b for b in app.button if b.label=='합성 시연 6건 분석').click().run()
    next(b for b in app.button if b.label=='이 배치 상세 검토').click().run()
    batch=next(s for s in app.selectbox if s.label=='검토 자료').value
    next(s for s in app.selectbox if s.label=='검토 자료').set_value('mvp').run()
    next(t for t in app.text_input if t.label=='청구 ID 검색').set_value('3520').run()
    # 메뉴를 떠나지 않고 같은 화면에서 dataset만 전환한다.
    next(s for s in app.selectbox if s.label=='검토 자료').set_value(batch).run()
    assert not app.exception
    assert next(t for t in app.text_input if t.label=='청구 ID 검색').value==''
    assert next(s for s in app.selectbox if s.label=='상세 조회할 청구 ID').value.startswith('DEMO-')
    next(t for t in app.text_input if t.label=='청구 ID 검색').set_value('DEMO-001').run()
    next(s for s in app.selectbox if s.label=='검토 자료').set_value('mvp').run()
    assert next(t for t in app.text_input if t.label=='청구 ID 검색').value==''
