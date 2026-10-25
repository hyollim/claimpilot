"""저장된 로컬 파이프라인의 신규 청구 추론. 정답 라벨·자동 사기 판단은 받지 않는다."""
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from claimpilot.modeling import FEATURES

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 5000
NUMERIC = {'WeekOfMonth': (1, 5), 'WeekOfMonthClaimed': (1, 5), 'Deductible': (0, None)}
COLUMNS = ['claim_id'] + FEATURES

class InputValidationError(ValueError):
    def __init__(self, issues):
        self.issues = issues
        super().__init__('CSV 검증 실패: ' + '; '.join(i['message'] for i in issues if i['severity']=='error')[:1200])

@dataclass
class InferenceResult:
    rows: pd.DataFrame
    explanations: dict
    issues: list
    batch_id: str
    metadata: dict


def read_claim_csv(content):
    if not isinstance(content, bytes) or len(content) > MAX_BYTES:
        raise ValueError('UTF-8 CSV 파일은 5 MiB 이하여야 합니다.')
    try:
        text = content.decode('utf-8-sig')
        parsed = csv.reader(io.StringIO(text), strict=True)
        header = next(parsed, [])
        if len(set(header)) != len(header):
            raise ValueError('중복된 CSV 열 이름이 있습니다.')
        records = []
        for record in parsed:
            if not record: continue
            if len(record) != len(header):
                raise ValueError('CSV 행의 항목 수가 헤더와 다릅니다.')
            records.append(record)
            if len(records) > MAX_ROWS:
                raise ValueError('CSV는 5,000행 이하여야 합니다.')
        return pd.DataFrame(records, columns=header)
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError('UTF-8 CSV 인코딩 또는 구분 형식을 확인하세요.') from error


def validate_claims(df, categories):
    """열은 필수, 입력값 누락은 명시적 sentinel로 처리. ID 누락·잘못된 숫자는 거부한다."""
    issues=[]
    def issue(level,row,column,message):
        issues.append(dict(severity=level,row=row,column=column,message=message))
    if not isinstance(df,pd.DataFrame): raise ValueError('DataFrame 입력이 필요합니다.')
    missing=set(COLUMNS)-set(df.columns); unexpected=set(df.columns)-set(COLUMNS)
    if missing: issue('error',None,'schema','필수 열 누락: '+', '.join(sorted(missing)))
    if unexpected: issue('error',None,'schema','허용하지 않는 열(정답 라벨 포함 불가): '+', '.join(sorted(unexpected)))
    if df.columns.duplicated().any(): issue('error',None,'schema','중복 열 이름')
    if not 1<=len(df)<=MAX_ROWS: issue('error',None,'schema','입력은 1~5,000행이어야 합니다.')
    if issues: raise InputValidationError(issues)
    clean=df[COLUMNS].copy().reset_index(drop=True)
    for i,row in clean.iterrows():
        rid=row['claim_id']; rid='' if pd.isna(rid) else str(rid).strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',rid):
            issue('error',i+2,'claim_id','청구 ID는 영문/숫자로 시작하는 1~64자 영문·숫자·_.- 조합이어야 합니다.')
        clean.at[i,'claim_id']=rid
        present=0
        for column in FEATURES:
            value=row[column]
            if not pd.api.types.is_scalar(value):
                issue('error',i+2,column,'문자열 또는 단일 숫자만 입력할 수 있습니다.'); continue
            blank=pd.isna(value) or str(value).strip() in ('','__MISSING__')
            if blank:
                clean.at[i,column]='__MISSING__'
                issue('warning',i+2,column,'누락값: 모델 입력은 __MISSING__, 실제 자료 확인 필요')
                continue
            value=str(value).strip()
            if len(value)>128 or any(ord(c)<32 for c in value):
                issue('error',i+2,column,'입력값은 제어문자 없는 128자 이하여야 합니다.'); continue
            if column in NUMERIC:
                try:
                    number=float(value)
                    low,high=NUMERIC[column]
                    if not math.isfinite(number) or not number.is_integer() or number<low or (high is not None and number>high):
                        raise ValueError()
                    value=str(int(number))
                except (ValueError,OverflowError):
                    issue('error',i+2,column,'스키마 숫자 형식/범위 오류: 정수 '+('1~5' if column.startswith('Week') else '0 이상'))
                    continue
            clean.at[i,column]=value; present+=1
            if value not in categories[column]:
                issue('warning',i+2,column,'학습에서 미관측한 범주: 해당 one-hot는 0으로 처리되며 실제 자료 확인 필요')
        if present==0: issue('error',i+2,'row','모든 모델 입력이 누락된 행은 분석할 수 없습니다.')
    if clean.claim_id.duplicated().any(): issue('error',None,'claim_id','배치 내 청구 ID가 중복되었습니다.')
    if any(i['severity']=='error' for i in issues): raise InputValidationError(issues)
    return clean,issues


def load_bundle(path):
    """프로젝트에서 생성한 신뢰 가능한 파일만 로드. 사용자 모델 업로드는 지원하지 않는다."""
    bundle=joblib.load(Path(path))
    if bundle.get('schema_version')!=1 or bundle.get('features')!=FEATURES:
        raise ValueError('추론 번들 스키마가 다릅니다. prepare_inference.py를 실행하세요.')
    return bundle


def shap_explanations(X,bundle):
    model=bundle['model']
    if hasattr(model,'named_steps'):
        import shap
        encoder=model.named_steps['encode']; clf=model.named_steps['clf']
        encoded=encoder.transform(X)
        mean=np.asarray(bundle['background_mean'])
        explainer=shap.LinearExplainer(clf,(mean,np.zeros((len(mean),len(mean)))))
        values=np.asarray(explainer.shap_values(encoded))
        offsets=np.cumsum([0]+[len(cat) for cat in encoder.categories_])
        contrib=np.column_stack([values[:,offsets[i]:offsets[i+1]].sum(axis=1) for i in range(len(FEATURES))])
        bases=np.full(len(X),float(explainer.expected_value)); raw=model.decision_function(X)
    else:
        from catboost import Pool
        values=model.get_feature_importance(Pool(X,cat_features=FEATURES),type='ShapValues')
        contrib=values[:,:-1]; bases=values[:,-1]; raw=model.predict(X,prediction_type='RawFormulaVal')
    if np.max(np.abs(contrib.sum(axis=1)+bases-raw))>1e-7:
        raise ValueError('SHAP 합계 검증에 실패했습니다.')
    explanations=[]
    for i in range(len(X)):
        order=np.argsort(-np.abs(contrib[i]),kind='stable')[:8]
        explanations.append(dict(base_value=float(bases[i]),raw_score=float(raw[i]),
            other_contribution=float(contrib[i].sum()-contrib[i,order].sum()),
            factors=[dict(feature=FEATURES[j],value=str(X.iloc[i,j]),contribution=float(contrib[i,j])) for j in order]))
    return explanations


def score_claims(df,bundle,review_fraction=.2):
    if isinstance(review_fraction,bool) or not isinstance(review_fraction,(int,float)) or not 0<review_fraction<=1:
        raise ValueError('검토 비율은 0 초과 1 이하여야 합니다.')
    clean,issues=validate_claims(df,bundle['categories'])
    X=clean[FEATURES]; model=bundle['model']; scores=model.predict_proba(X)[:,1]
    if not np.isfinite(scores).all(): raise ValueError('추론 결과가 유효하지 않습니다.')
    decomposition=shap_explanations(X,bundle)
    # 배치 내용·모델 버전별로 기록 분리. 기존 평가의 숫자 ID와 충돌하지 않는다.
    digest=hashlib.sha256((bundle['model_fingerprint']+clean.to_json(orient='records',force_ascii=False)).encode()).hexdigest()[:16]
    rows=clean.copy(); rows['risk_score']=scores
    rows['record_id']=['new:'+digest+':'+cid for cid in rows.claim_id]
    rows['validation_status']=['자료 보완 필요' if any(t['row']==i+2 for t in issues) else '형식 확인' for i in range(len(rows))]
    rows=rows.sort_values('risk_score',ascending=False,kind='stable').reset_index(drop=True)
    rows['priority_rank']=np.arange(1,len(rows)+1)
    rows['review_band']=np.where(rows.priority_rank<=math.ceil(len(rows)*review_fraction),'우선 검토','일반 검토')
    explanations={cid:ex for cid,ex in zip(clean.claim_id,decomposition)}
    metadata=dict(model_fingerprint=bundle['model_fingerprint'],review_fraction=review_fraction,n=len(rows),
        score_note='비보정 모델 점수. 실제 사기 확률·증거·자동 지급 판단이 아님.',
        missing_cells=sum(t['message'].startswith('누락값') for t in issues),
        unseen_cells=sum(t['message'].startswith('학습에서') for t in issues))
    return InferenceResult(rows,explanations,issues,digest,metadata)


def export_csv(rows):
    """Excel 수식 실행 방지. 화면/모델 입력값은 바꾸지 않고 다운로드 문자열만 이스케이프한다."""
    output=rows.drop(columns=['record_id'],errors='ignore').copy()
    for column in output.select_dtypes(include=['object','string']):
        output[column]=output[column].map(lambda x:"'"+x if isinstance(x,str) and x.startswith(('=','+','-','@','\t','\r')) else x)
    return output.to_csv(index=False).encode('utf-8-sig')
