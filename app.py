"""ClaimPilot: 검토 순서 지원용 단일 사용자 로컬 PoC."""
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from claimpilot.review import MAX_NOTE_LENGTH, STATUSES, ReviewStore
from claimpilot.search import search_sources
from claimpilot.retrieval import search_documents
from claimpilot.modeling import FEATURES
from claimpilot.intake import BatchStore
from claimpilot.ui_phase2 import overview_screen, analysis_screen, effects_screen

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
CHECKLIST = {
    "접수 내용과 원본 자료 대조": False,
    "모델 위험요인의 실제 사실관계 확인": False,
    "추가자료 필요 여부 검토": False,
    "공식 원문과 담당자 판단 확인": False,
}
METRIC_NAMES = {
    "ap": "AP (평균 정밀도)", "precision": "정밀도", "recall": "재현율",
    "f1": "F1", "capture_at_20": "상위 20% 포집률",
    "precision_at_20": "상위 20% 정밀도", "lift_at_20": "상위 20% 리프트",
    "reviewed_count": "검토 대상 수", "captured_count": "포집 양성 수",
    "positives": "전체 양성 수", "n": "평가 표본 수",
}
DISPLAY_NAMES = {
    "claim_id": "청구 ID", "risk_score": "모델 점수", "priority_rank": "검토 순위",
    "validation_status": "입력 확인 상태",
    "review_band": "우선순위 구간", "status": "검토 상태", "note": "검토 메모",
    "Make": "차량 제조사", "AccidentArea": "사고 지역", "PolicyType": "계약 유형",
    "VehicleCategory": "차량 구분", "VehiclePrice": "차량 가격 구간",
    "Deductible": "자기부담금", "PastNumberOfClaims": "과거 청구 횟수 구간",
    "PoliceReportFiled": "경찰 신고 기록", "WitnessPresent": "목격자 기록",
    "AgentType": "모집 경로 구분", "BasePolicy": "기본 계약 유형",
    "Month": "사고 월", "WeekOfMonth": "사고 주차", "DayOfWeek": "사고 요일",
    "MonthClaimed": "청구 월", "WeekOfMonthClaimed": "청구 주차", "DayOfWeekClaimed": "청구 요일",
    "Days_Policy_Accident": "계약·사고 경과 구간", "Days_Policy_Claim": "계약·청구 경과 구간",
    "AgeOfVehicle": "차량 연식 구간", "NumberOfCars": "차량 수 구간",
}


def _sensitive(name):
    """검토 UI에 라벨/인구통계 또는 해당 파생변수가 전달되지 않도록 차단."""
    key = re.sub(r"[^a-z0-9가-힣]", "", str(name).casefold())
    # AgeOfVehicle·AgentType은 나이/성별이 아닌 업무 변수다. 부분문자열 차단 금지.
    return key in {"label", "fraudfoundp", "fraudreported", "target", "gender", "sex",
                   "age", "ageofpolicyholder", "maritalstatus", "성별", "나이", "연령", "정답"}


def _read_json(name):
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def _score(value):
    return f"{float(value):.4f}"


def _review_screen(raw, explanations, store):
    # 원본 라벨을 포함하는 프레임은 이 함수 내 표시 경로에서 즉시 제거한다.
    allowed = set(FEATURES + ["claim_id", "record_id", "risk_score", "priority_rank", "review_band", "validation_status"])
    claims = raw.loc[:, [column for column in raw.columns if column in allowed and not _sensitive(column)]].copy()
    saved = {record["claim_id"]: record for record in store.all()}
    record_keys = claims["record_id"] if "record_id" in claims else claims["claim_id"]
    claims["status"] = record_keys.map(lambda cid: saved.get(str(cid), {}).get("status", "미검토"))
    with st.sidebar:
        st.header("검토 대상 필터")
        query = st.text_input("청구 ID 검색", max_chars=128)
        bands = st.multiselect("우선순위 구간", sorted(claims["review_band"].dropna().unique()))
        statuses = st.multiselect("검토 상태", list(STATUSES))
        low, high = float(claims.risk_score.min()), float(claims.risk_score.max())
        if low < high:
            score_range = st.slider("모델 점수 범위 (확률 아님)", low, high, (low, high), format="%.4f")
        else:
            score_range = (low, high)
        display_limit = st.selectbox("목록 최대 표시 건수", [20, 50, 100, 500], index=1)
        st.caption("필터는 현재 검토 배치에만 적용됩니다. 성능 평가는 별도의 고정 test 기준입니다.")
    filtered = claims[claims.risk_score.between(*score_range)]
    if query.strip():
        filtered = filtered[filtered.claim_id.str.contains(query.strip(), regex=False, case=False)]
    if bands:
        filtered = filtered[filtered.review_band.isin(bands)]
    if statuses:
        filtered = filtered[filtered.status.isin(statuses)]
    filtered = filtered.sort_values(["priority_rank", "claim_id"])
    top = st.columns(4)
    top[0].metric("선택 배치 청구", f"{len(claims):,}건")
    top[1].metric("필터 결과", f"{len(filtered):,}건")
    top[2].metric("검토중", f"{int((claims.status == '검토중').sum()):,}건")
    top[3].metric("검토완료", f"{int((claims.status == '검토완료').sum()):,}건")
    st.subheader("검토 대기열")
    st.caption("높은 모델 점수부터 검토를 제안합니다. 점수는 보정된 사기 확률이 아니며 지급 여부를 뜻하지 않습니다.")
    view_columns = ["priority_rank", "claim_id", "risk_score", "review_band", "status"]
    st.dataframe(filtered[view_columns].head(display_limit).rename(columns=DISPLAY_NAMES),
                 hide_index=True, width="stretch", column_config={
                     "모델 점수": st.column_config.NumberColumn(format="%.4f"),
                     "검토 순위": st.column_config.NumberColumn(format="%d"),
                 })
    if filtered.empty:
        st.info("조건에 맞는 청구가 없습니다. 필터를 조정하세요.")
        return
    if len(filtered) > display_limit:
        st.caption(f"{len(filtered):,}건 중 {display_limit:,}건 표시. 상세 선택은 전체 필터 결과에서 가능합니다.")
    selected = st.selectbox("상세 조회할 청구 ID", filtered.claim_id.tolist())
    current = filtered[filtered.claim_id == selected].iloc[0]
    record_key = str(current.get("record_id", selected))
    record = store.get(record_key, CHECKLIST)
    st.divider()
    st.subheader(f"청구 상세 · {selected}")
    left, right = st.columns([1, 1])
    with left:
        st.markdown("#### 청구 자료")
        details = [{"항목": DISPLAY_NAMES.get(key, key), "값": str(value)}
                   for key, value in current.items() if key not in view_columns + ["record_id"]]
        st.write(f"검토 순위 {int(current.priority_rank):,} · 모델 점수 {_score(current.risk_score)}")
        if details:
            st.dataframe(pd.DataFrame(details), hide_index=True, width="stretch")
        else:
            st.caption("제공된 추가 청구 변수가 없습니다.")
        st.markdown("#### 모델 위험요인 · 검토 단서")
        explanation = explanations.get(str(selected), {})
        factors = [item for item in explanation.get("factors", [])
                   if not _sensitive(item.get("feature", ""))]
        if factors:
            factor_frame = pd.DataFrame(factors)
            factor_frame["contribution"] = pd.to_numeric(factor_frame["contribution"], errors="coerce")
            factor_frame = factor_frame.dropna(subset=["contribution"])
            factor_frame = factor_frame.loc[factor_frame.contribution.abs().sort_values(ascending=False).index].head(12)
            factor_frame["feature"] = factor_frame["feature"].map(lambda key: DISPLAY_NAMES.get(key, key))
            st.dataframe(factor_frame[["feature", "value", "contribution"]].rename(columns={
                "feature": "변수", "value": "관측값", "contribution": "SHAP 기여도"}),
                hide_index=True, width="stretch", column_config={
                    "SHAP 기여도": st.column_config.NumberColumn(format="%+.4f")})
            chart = factor_frame[["feature", "contribution"]].copy()
            chart["feature"] = chart["feature"].map(lambda key: DISPLAY_NAMES.get(key, key))
            st.bar_chart(chart.set_index("feature"), horizontal=True, color="#24518a", height=230)
            st.caption("절댓값이 큰 8개 SHAP raw log-odds 기여도입니다. 양수는 점수 상승, 음수는 하락 방향입니다. 전체 합계가 아니며 인과관계나 사기 증거가 아닙니다.")
        else:
            st.info("이 청구의 표시 가능한 모델 설명이 없습니다. 요인을 추정하지 않습니다.")
    with right:
        st.markdown("#### 담당자 검토 기록")
        if st.session_state.pop("review_saved", None) == record_key:
            st.success("상태·메모·체크리스트를 저장했습니다.")
        st.caption("일반적인 자료 확인용 체크리스트이며 보험사 공식 심사 규칙이 아닙니다.")
        with st.form(f"review_{record_key}"):
            status = st.selectbox("검토 상태", STATUSES, index=STATUSES.index(record["status"])
                                  if record["status"] in STATUSES else 0)
            checklist = {key: st.checkbox(key, value=value) for key, value in record["checklist"].items()}
            note = st.text_area("검토 메모", record["note"], height=170, max_chars=MAX_NOTE_LENGTH,
                                help="주민등록번호·연락처 등 실제 개인정보를 입력하지 마세요.")
            submitted = st.form_submit_button("검토 기록 저장", type="primary")
        if submitted:
            try:
                store.save(record_key, status, note, checklist)
            except (ValueError, OSError, sqlite3.Error) as error:
                st.error(f"저장하지 못했습니다: {error}")
            else:
                st.session_state["review_saved"] = record_key
                st.rerun()
        st.caption(f"최종 저장 (UTC): {record['updated_at'] or '저장 이력 없음'}")
        st.info("검토완료는 지급 승인 또는 사기 확정을 의미하지 않습니다. 최종 사기 여부와 보험금 지급 판단은 담당자가 수행합니다.")


def _evaluation_screen(raw, metrics, audit):
    st.subheader("모델 성능 평가")
    st.caption("정답 라벨은 이 평가 탭에서만 표시합니다. 검토 필터와 무관한 최종 test 표본이며 실서비스 성과가 아닙니다.")
    st.write(f"선정 모델: {metrics.get('selected_model', '미제공')}")
    threshold = metrics.get("threshold")
    if threshold is not None:
        st.write(f"검증세트 선정 임계값: {threshold:.4f} (확률 아님)")
    rows = []
    models = metrics.get("models", [])
    if isinstance(models, dict):
        models = [{"name": name, **value} for name, value in models.items()]
    for model in models:
        for split in ("validation", "test"):
            values = model.get(split, {})
            if values:
                rows.append({"모델": model.get("name", "미제공"), "분할": split,
                             **{METRIC_NAMES[key]: values[key] for key in METRIC_NAMES if key in values}})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.warning("보고서에 모델 평가 지표가 없습니다. 지표를 추정하지 않습니다.")
    st.caption("AP는 average precision입니다. 상위 20% 지표는 검토 예산 기준이며 임계값 기반 정밀도·재현율과 다릅니다. 리프트 1은 무작위 검토 기준입니다.")
    st.warning("공개 학습 데이터의 오프라인 평가입니다. 변수의 실제 수집 시점·업무 누수·외부 데이터 일반화는 운영 전 별도 검증이 필요합니다.")
    with st.expander("분할 및 데이터 점검 보고서"):
        st.json({key: metrics[key] for key in ("split",) if key in metrics})
        if audit:
            st.json({key: audit[key] for key in ("counts", "splits", "leakage") if key in audit})
        else:
            st.caption("data_audit.json 미생성. 실제 데이터 점검 결과를 추가로 확인하세요.")
    with st.expander("추가 검증 · 시간 분할 / 부트스트랩 / SHAP 일치성"):
        extra = {key: metrics[key] for key in ("temporal_stress", "bootstrap_95_fixed_predictions", "shap_max_additivity_error") if key in metrics}
        if extra:
            st.json(extra)
            st.caption("부트스트랩 구간은 고정된 예측에 대한 표본 불확실성이며 재학습 변동이나 운영 성능을 보장하지 않습니다.")
        else:
            st.caption("추가 검증 결과 미제공")
    if "label" in raw.columns:
        with st.expander("최종 test 정답 라벨 · 평가 전용"):
            st.dataframe(raw[["claim_id", "risk_score", "label"]].rename(columns={
                "claim_id": "청구 ID", "risk_score": "모델 점수", "label": "정답 라벨 (평가 전용)"}),
                hide_index=True, width="stretch")


def _sources_screen():
    st.subheader("공식 출처 검색")
    st.write("기존 발췌 8개와 신규 원문 경로 안내 3개를 검색합니다. 신규 본문·제한된 PDF는 수집하지 않았습니다. 생성형 답변이나 RAG가 아닙니다.")
    st.info("업무 참고자료 검색은 모델 위험요인 설명과 별개입니다. 검색 결과는 개별 청구의 사기·지급 판단 근거를 자동 생성하지 않습니다.")
    with st.form("sources_search"):
        query = st.text_input("검색어", placeholder="예: 사고 접수, 과실비율, 참고자료", max_chars=500)
        limit = st.selectbox("검색 결과 수", [3, 5, 10], index=1)
        method = st.selectbox("검색 방식", ["기존 가중 키워드", "BM25 후보 검색", "BM25 보수적 검색 (연구)"])
        submit = st.form_submit_button("공식 발췌 검색")
    if submit:
        if not query.strip():
            st.info("검색어를 입력하세요.")
            return
        try:
            code = "substring" if method == "기존 가중 키워드" else "bm25"
            threshold = 24.0 if method == "BM25 보수적 검색 (연구)" else 0.0
            results = search_documents(query, limit, ROOT / "resources/official_index.json", method=code, threshold=threshold)
        except (OSError, ValueError) as error:
            st.warning(f"공식 출처 파일을 읽지 못했습니다: {error}")
            return
        st.caption(f"검색 결과 {len(results)}건 · 관련도는 확률이 아닙니다.")
        if not results:
            st.info("관련 발췌 없음. 다른 검색어로 원문을 확인하세요. 답변을 추정하지 않습니다.")
        for item in results:
            with st.container(border=True):
                st.markdown(f"#### {item['title']}")
                st.write(f"발행기관: {item['organization']} · 확인일: {item['accessed_at']}")
                st.caption(f"발췌 위치: {item['section']} · 검색 관련도: {item['score']:.3f}")
                st.caption("원문 경로 안내 · 본문 미수집" if item.get("record_type") == "navigation" else "기존 공식 발췌")
                st.text(item["text"])
                st.link_button("공식 원문 확인", item["url"])
                st.caption(item["limitations"])
    st.caption("범위 밖 질의도 어휘가 겹치면 후보가 반환될 수 있습니다. 검색 결과는 답변이나 모델 점수의 근거가 아닙니다.")
    with st.expander("검색 방법 비교와 한계"):
        st.write("기존8개 발췌에 고정한 수작업 내부 평가12개에서 Recall@3은 키워드0.917, BM25(기준0)1.000이었습니다. 범위 밖 평가2개는 두 방식 모두 반환했습니다. 개발자료에서 고른 기준24는 범위 밖 반환0/2 대신 Recall@3이0.333으로 낮아졌습니다.")
        st.caption("새 경로3개를 포함한 실제 색인의 성능으로 일반화하지 않습니다. 외부 독립 질의 정답집이 아니며 평가 결과로 임계값을 다시 고르지 않았습니다. 기존 키워드 방식을 기본으로 유지합니다.")


def main():
    st.set_page_config(page_title="ClaimPilot | 청구 검토", page_icon=None, layout="wide")
    st.subheader("ClaimPilot | 청구 검토")
    st.caption("자동차보험 청구 검토 우선순위 · 담당자 업무지원 PoC")
    st.info("모델 점수는 사기 확률이나 지급 판정이 아닙니다. 최종 사기 여부와 보험금 지급 판단은 담당자가 수행합니다.")
    st.caption("단일 사용자·로컬 PoC · 운영 인증/권한/암호화/감사체계 미구현 · 실제 개인정보 입력 금지")
    if 'next_menu' in st.session_state:
        st.session_state['work_menu'] = st.session_state.pop('next_menu')
    if 'next_dataset' in st.session_state:
        st.session_state['active_dataset'] = st.session_state.pop('next_dataset')
    try:
        store = ReviewStore(ROOT / 'data/reviews.sqlite3')
        batches = BatchStore(ROOT / 'data/intake.sqlite3')
        raw = pd.read_csv(REPORTS / 'scored_claims.csv', dtype={'claim_id':str}) if (REPORTS / 'scored_claims.csv').exists() else pd.DataFrame()
        explanations = _read_json('explanations.json') if (REPORTS / 'explanations.json').exists() else {}
        metrics = _read_json('metrics.json') if (REPORTS / 'metrics.json').exists() else {}
        audit = _read_json('data_audit.json') if (REPORTS / 'data_audit.json').exists() else {}
        listing = batches.list_batches()
    except (OSError, ValueError, sqlite3.Error) as error:
        st.error(f'분석 자료 또는 검토 저장소를 열지 못했습니다: {error}')
        return
    menus=['업무 현황','청구 데이터 분석','청구 상세 검토','모델 성능 및 업무 효과','공식 업무 참고자료']
    options=['mvp']+[batch['batch_id'] for batch in listing]
    labels={'mvp':'기존 test 시연 · 정답 비노출'}
    labels.update({b['batch_id']:('합성 시연' if b['source_kind']=='synthetic' else '사용자 입력')+' · '+b['name']+' · '+b['batch_id'][:6] for b in listing})
    if st.session_state.get('active_dataset') not in options:
        st.session_state['active_dataset']='mvp'
    with st.sidebar:
        st.subheader('ClaimPilot')
        menu=st.radio('업무 메뉴',menus,key='work_menu')
        dataset=st.selectbox('검토 자료',options,format_func=lambda key:labels[key],key='active_dataset')
        percentage=st.slider('우선 검토 비율 (%)',1,100,20,step=1,key='review_percentage')
        st.caption('비율 변경은 검토 순위 구간과 고정 test의 업무 효과 분석에 적용됩니다.')
    selected_rows=raw
    selected_explanations=explanations
    origin='기존 MVP test 시연 자료 · 실제 신규 청구가 아닙니다.'
    if dataset!='mvp':
        batch=batches.load(dataset)
        selected_rows=batch['rows']; selected_explanations=batch['explanations']
        origin=('합성 시연 자료 · 실제 청구/사기 라벨 아님' if batch['source_kind']=='synthetic' else '사용자 제공 자료 · 진위/정답 미확인')+' · 배치 '+dataset
    if not selected_rows.empty:
        import math
        selected_rows=selected_rows.copy()
        selected_rows['review_band']=selected_rows.priority_rank.map(lambda rank:'우선 검토' if rank<=math.ceil(len(selected_rows)*percentage/100) else '일반 검토')
    if menu=='업무 현황':
        overview_screen(selected_rows,store,origin,DISPLAY_NAMES)
    elif menu=='청구 데이터 분석':
        analysis_screen(batches,DISPLAY_NAMES,percentage/100)
    elif menu=='청구 상세 검토':
        st.caption(origin)
        if selected_rows.empty:
            st.info('검토할 청구가 없습니다. 청구 데이터 분석 메뉴에서 CSV 또는 합성 시연을 분석하세요.')
        else:
            _review_screen(selected_rows,selected_explanations,store)
    elif menu=='모델 성능 및 업무 효과':
        effects_screen(raw,metrics,percentage/100)
        st.divider()
        if not raw.empty:_evaluation_screen(raw,metrics,audit)
    elif menu=='공식 업무 참고자료':
        _sources_screen()
    st.caption('공개 데이터 기반 검토 우선순위 실험 · 최종 사기 여부·보험금 지급 판단은 담당자 수행')


if __name__ == "__main__":
    main()
