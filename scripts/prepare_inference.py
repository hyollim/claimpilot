"""선정 모델은 바꾸지 않고 학습셋 배경 평균과 입력 범주만 추론 번들에 추가한다."""
from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from claimpilot.modeling import FEATURES,prepare

def main():
    original=ROOT/'models/selected.joblib'; bundle=joblib.load(original)
    df=pd.read_csv(ROOT/'data/raw/fraud_oracle.csv',encoding='utf-8-sig')
    expected=json.loads((ROOT/'resources/data_provenance.json').read_text())['sha256']
    if hashlib.sha256((ROOT/'data/raw/fraud_oracle.csv').read_bytes()).hexdigest()!=expected:
        raise ValueError('원본 데이터 체크섬이 다릅니다. 기존 모델의 학습 배경을 대체하지 않습니다.')
    dev,_=train_test_split(np.arange(len(df)),test_size=.2,stratify=df.FraudFound_P,random_state=42)
    train,_=train_test_split(dev,test_size=.25,stratify=df.FraudFound_P.iloc[dev],random_state=42)
    X=prepare(df.iloc[train],FEATURES)
    bundle.update(schema_version=1,features=FEATURES,categories={c:sorted(X[c].unique().tolist()) for c in FEATURES},
        model_fingerprint=hashlib.sha256(original.read_bytes()).hexdigest(),training_rows=len(train),
        data_sha256=expected,background_scope='MVP 학습세트만 사용. validation/test 배경 사용 금지.')
    if hasattr(bundle['model'],'named_steps'):
        encoder=bundle['model'].named_steps['encode']
        bundle['background_mean']=np.asarray(encoder.transform(X).mean(axis=0)).ravel()
    joblib.dump(bundle,ROOT/'models/inference_bundle.joblib')
    schema={'schema_version':1,'columns':['claim_id']+FEATURES,'row_limit':5000,'max_bytes':5242880,
      'id_rule':'영문/숫자 시작 1~64자 영문·숫자·_.-',
      'numeric':{'WeekOfMonth':'1~5 정수, 누락 허용','WeekOfMonthClaimed':'1~5 정수, 누락 허용','Deductible':'0 이상 정수, 누락 허용'},
      'categorical':'나머지는 128자 이하 문자열. 미관측 범주는 경고 후 모델의 unknown 처리.',
      'missing':'누락 열은 거부. 값 누락은 __MISSING__. 모든 입력 누락 행과 ID 누락·중복 거부.',
      'labels':'정답/사기 라벨 열은 허용하지 않음','categories':bundle['categories']}
    (ROOT/'resources/upload_schema.json').write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding='utf-8')
    # 관측 범주의 조합으로 만든 합성 시연 행. 실제 청구나 라벨이 아니다.
    demo=[]
    for i in range(6):
        row={c:bundle['categories'][c][(i*3+j)%len(bundle['categories'][c])] for j,c in enumerate(FEATURES)}
        row['claim_id']=f'DEMO-{i+1:03d}';demo.append(row)
    demo[4]['Make']='DemoMotor';demo[5]['VehiclePrice']='';demo[5]['Deductible']=''
    pd.DataFrame(demo,columns=['claim_id']+FEATURES).to_csv(ROOT/'resources/demo_claims.csv',index=False)
    print('기존 모델 보존, 추론 번들 준비 완료:',len(train),'학습 배경 행 / 합성 시연 6행, 정답 라벨 없음')
if __name__=='__main__':main()
