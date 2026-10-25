"""기존 검토 화면을 유지하고 신규 입력·검토량 분석 경로를 추가한다."""
import json
import math
import sqlite3
from pathlib import Path
import pandas as pd
import streamlit as st
from claimpilot.inference import InputValidationError,read_claim_csv,load_bundle,score_claims,export_csv
from claimpilot.budget import budget_metrics,budget_curve,cumulative_capture

ROOT=Path(__file__).resolve().parents[1]


def navigate(menu,dataset=None):
    st.session_state['next_menu']=menu
    if dataset is not None:st.session_state['next_dataset']=dataset
    st.rerun()


def overview_screen(rows,store,origin,display_names):
    st.subheader('업무 현황')
    st.caption(origin)
    if rows.empty:
        st.info('조회할 청구가 없습니다. CSV를 업로드하거나 합성 시연 입력을 분석하세요.')
    else:
        saved={r['claim_id']:r for r in store.all()}
        keys=rows['record_id'] if 'record_id' in rows else rows['claim_id']
        statuses=keys.map(lambda key:saved.get(str(key),{}).get('status','미검토'))
        cols=st.columns(4)
        cols[0].metric('선택 자료',f'{len(rows):,}건')
        cols[1].metric('미검토',f'{int((statuses=="미검토").sum()):,}건')
        cols[2].metric('검토중·추가자료',f'{int(statuses.isin(["검토중","추가자료 요청"]).sum()):,}건')
        cols[3].metric('검토완료',f'{int((statuses=="검토완료").sum()):,}건')
        view=rows[['priority_rank','claim_id','risk_score','review_band']].copy()
        view['status']=statuses.to_numpy()
        st.dataframe(view.head(10).rename(columns=display_names),hide_index=True,width='stretch')
        st.caption('점수순 상위 10건. 검토완료는 지급 승인 또는 사기 확정이 아닙니다.')
    left,right=st.columns(2)
    if left.button('청구 데이터 분석으로 이동'):navigate('청구 데이터 분석')
    if right.button('청구 상세 검토로 이동',disabled=rows.empty):navigate('청구 상세 검토')


def _run_analysis(content,name,kind,batches,fraction):
    bundle_path=ROOT/'models/inference_bundle.joblib'
    if not bundle_path.exists():
        st.warning('추론 번들이 없습니다. 먼저 모델을 학습하고 추론 준비 명령을 실행하세요.')
        st.code('python scripts/download_data.py\npython scripts/train.py\npython scripts/prepare_inference.py')
        return
    try:
        result=score_claims(read_claim_csv(content),load_bundle(bundle_path),fraction)
        batches.save(result,name,kind)
    except InputValidationError as error:
        st.error('입력 오류가 있어 배치를 저장하거나 분석하지 않았습니다.')
        st.dataframe(pd.DataFrame(error.issues).rename(columns={'severity':'구분','row':'CSV 행','column':'열','message':'확인 내용'}),hide_index=True,width='stretch')
        return
    except (OSError,ValueError,KeyError,sqlite3.Error) as error:
        st.error(f'분석하지 못했습니다: {error}')
        return
    st.session_state['last_analysis_batch']=result.batch_id
    st.success(f'{len(result.rows):,}건 분석 완료. 점수·SHAP·입력 확인 사항을 별도 배치로 저장했습니다.')


def analysis_screen(batches,display_names,fraction):
    st.subheader('청구 데이터 분석')
    st.caption('CSV 입력 → 형식/범주 점검 → 저장 모델 추론 → 배치 저장 → 상세 검토')
    st.info('이 메뉴는 정답 라벨을 받지 않습니다. 신규 입력의 포집률·정밀도·사기 여부를 계산하거나 생성하지 않습니다.')
    st.write('UTF-8 CSV · 최대 5 MiB/5,000행 · claim_id와 승인된 모델 입력 19개 열')
    st.caption('동일 입력·모델 재업로드는 기존 배치의 출처와 검토 기록을 유지합니다. 검토 비율은 사이드바 설정을 따릅니다.')
    schema_path=ROOT/'resources/upload_schema.json'
    with st.expander('CSV 스키마·누락/미관측 범주 처리'):
        if schema_path.exists():
            schema=json.loads(schema_path.read_text(encoding='utf-8'))
            st.write({k:schema[k] for k in ('id_rule','numeric','categorical','missing','labels')})
            st.write('입력 열: '+', '.join(schema['columns']))
        else:st.caption('python scripts/prepare_inference.py 실행 후 스키마를 생성하세요.')
    demo=ROOT/'resources/demo_claims.csv'
    if demo.exists():
        st.download_button('합성 시연 CSV 내려받기',demo.read_bytes(),'claimpilot_synthetic_demo.csv','text/csv')
    uploaded=st.file_uploader('신규 청구 CSV',type=['csv'],help='실제 개인정보는 입력하지 마세요. FraudFound_P/label 등 정답 열을 포함할 수 없습니다.')
    kind=st.selectbox('입력 자료 출처',['사용자 제공 자료 (진위 미확인)','합성 시연 자료'])
    if st.button('CSV 분석 실행',type='primary',disabled=uploaded is None):
        _run_analysis(uploaded.getvalue(),uploaded.name,'synthetic' if kind=='합성 시연 자료' else 'user_upload',batches,fraction)
    st.divider()
    st.write('합성 시연은 코드가 만든 6건입니다. 1개 미관측 범주와 2개 누락값을 포함하며 실제 청구나 성능 평가 데이터가 아닙니다.')
    if st.button('합성 시연 6건 분석',disabled=not demo.exists()):
        _run_analysis(demo.read_bytes(),'합성 시연 6건','synthetic',batches,fraction)
    recent=st.session_state.get('last_analysis_batch')
    if recent:
        batch=batches.load(recent);rows=batch['rows'].copy()
        rows['review_band']=rows.priority_rank.map(lambda rank:'우선 검토' if rank<=math.ceil(len(rows)*fraction) else '일반 검토')
        st.markdown('#### 최근 분석 결과')
        st.caption(('합성 시연 자료' if batch['source_kind']=='synthetic' else '사용자 제공 자료, 진위 미확인')+' · '+batch['batch_id'])
        st.dataframe(rows[['priority_rank','claim_id','risk_score','validation_status','review_band']].rename(columns=display_names),hide_index=True,width='stretch')
        if batch['issues']:
            st.warning(f"자료 확인 사항 {len(batch['issues'])}개. 누락/미관측 범주가 있어도 사기 여부를 뜻하지 않습니다.")
            st.dataframe(pd.DataFrame(batch['issues']).rename(columns={'severity':'구분','row':'CSV 행','column':'열','message':'확인 내용'}),hide_index=True,width='stretch')
        st.download_button('분석 결과 CSV 내려받기',export_csv(rows),'claimpilot_scored_'+recent+'.csv','text/csv')
        if st.button('이 배치 상세 검토'):navigate('청구 상세 검토',recent)
    listing=batches.list_batches()
    if listing:
        st.markdown('#### 저장된 분석 배치')
        st.caption('누적 한도: 50개 배치 또는 JSON payload 50 MiB. 한도 도달 시 기존 기록을 자동 삭제하지 않고 추가 저장을 거부합니다.')
        st.dataframe(pd.DataFrame(listing)[['batch_id','name','source_kind','n','created_at']].rename(columns={'batch_id':'배치 ID','name':'자료명','source_kind':'자료 출처','n':'건수','created_at':'생성 시각 (UTC)'}),hide_index=True,width='stretch')


def effects_screen(raw,metrics,fraction):
    st.subheader('검토 예산별 업무 효과')
    st.caption('아래 지표는 기존 정답 라벨이 있는 test 3,084건만 사용합니다. 사이드바의 신규 배치와 무관합니다.')
    if raw.empty or 'label' not in raw:
        st.info('기존 학습·평가 산출물 생성 후 확인할 수 있습니다.');return
    rows=budget_curve(raw.label.to_numpy(),raw.risk_score.to_numpy())
    table=pd.DataFrame(rows)
    fields={'requested_fraction':'요청 검토 비율','reviewed_fraction':'실제 검토 비율','k':'검토 건수','captured_count':'포집 양성','capture_rate':'양성 포집률','precision':'상위 정밀도','random_expected_count':'무작위 기대 포집 건수','random_expected_capture':'무작위 기대 포집률','lift':'리프트'}
    st.dataframe(table[list(fields)].rename(columns=fields),hide_index=True,width='stretch')
    point=budget_metrics(raw.label.to_numpy(),raw.risk_score.to_numpy(),fraction)
    st.write(f"선택 검토 비율 {fraction:.0%}: {point['k']:,}건 검토 / {point['captured_count']}건 포집 / 포집률 {point['capture_rate']:.2%} / 정밀도 {point['precision']:.2%} / 리프트 {point['lift']:.3f}")
    curve=pd.DataFrame(cumulative_capture(raw.label.to_numpy(),raw.risk_score.to_numpy()))
    st.line_chart(curve.set_index('reviewed_fraction')[['capture_rate','random_expected_capture']].rename(columns={'capture_rate':'모델 순위 누적 포집률','random_expected_capture':'무작위 기대 포집률'}),color=['#24518a','#9ba3ae'],height=260)
    st.caption('x축은 실제 검토 비율, y축은 양성 누적 포집률. 검토 건수는 올림 처리하며 동점은 고정 입력 순서입니다. 비용·시간 절감으로 환산하지 않았습니다.')
    validation=ROOT/'reports/phase2_validation.json'
    if validation.exists():
        report=json.loads(validation.read_text(encoding='utf-8'))
        comparison=[]
        selected=next((m for m in metrics.get('models',[]) if m['name']==metrics.get('selected_model')),None)
        if selected:comparison.append({'평가':'기존 무작위 분할','선정 모델':selected['name'],**selected['test']})
        for key,label in [('group_evaluation','동일 입력 그룹 격리'),('temporal_independent','1994~95 선정 / 1996 평가')]:
            part=report[key];comparison.append({'평가':label,'선정 모델':part['selected_model'],**part['test']})
        st.markdown('#### 분할 방식별 검증')
        st.dataframe(pd.DataFrame(comparison)[['평가','선정 모델','n','positives','ap','precision','recall','capture_at_20','precision_at_20','brier']].rename(columns={'n':'평가 건수','positives':'양성 건수','ap':'AP','precision':'임계값 정밀도','recall':'임계값 재현율','capture_at_20':'상위20% 포집률','precision_at_20':'상위20% 정밀도','brier':'Brier'}),hide_index=True,width='stretch')
        st.caption('평가 표본과 분할이 달라 증감 자체를 성능 향상으로 단정하지 않습니다. 동일 입력은 동일 고객·사건이 아닙니다.')
        with st.expander('동일 입력 그룹·독립 연도·가중치 진단 원본'):
            st.json(report)
            st.caption('동일 입력 그룹은 동일 고객·사건이 아닙니다. 1996년은 이번 선정에서 격리했지만 MVP에서 이미 관측한 자료입니다.')
    calibration=ROOT/'reports/phase2_calibration.json'
    if calibration.exists():
        research=json.loads(calibration.read_text(encoding='utf-8'))
        st.markdown('#### 점수 보정 진단 · 추론 모델에 미적용')
        names={'raw_balanced':'가중치 적용 · 비보정','raw_none':'가중치 없음 · 비보정','sigmoid_balanced':'가중치 적용 · sigmoid 보정','sigmoid_none':'가중치 없음 · sigmoid 보정','constant_train_prevalence':'학습 양성률 상수'}
        rows=[{'연구 조건':names.get(m['name'],m['name']),**m['test']} for m in research['models']]
        st.dataframe(pd.DataFrame(rows)[['연구 조건','ap','brier','capture_at_20','precision_at_20']].rename(columns={'ap':'AP','brier':'Brier','capture_at_20':'상위20% 포집률','precision_at_20':'상위20% 정밀도'}),hide_index=True,width='stretch')
        st.caption('Brier는 작을수록 좋습니다. class_weight가 바꾼 학습 목표와 점수 분포를 진단했으며 신규 추론 모델은 교체하지 않았습니다.')
        with st.expander('분리된 확률 보정 연구 결과 원본'):
            st.json(research)
            st.caption('보정 실험을 이유로 운영 모델을 자동 교체하지 않았습니다. 현재 신규 추론 점수는 비보정 값입니다.')
