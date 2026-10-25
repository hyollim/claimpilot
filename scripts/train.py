"""검증세트에서만 모델·임계값 선정. 최종평가는 잠금 후 1회 산출한다."""
from pathlib import Path
import sys, json, hashlib, time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from claimpilot.modeling import TARGET,EXCLUDED,feature_columns,prepare,evaluate,choose_threshold
SEED=42

def write_json(name,obj):
    (ROOT/'reports'/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def main():
    started=time.time(); (ROOT/'reports').mkdir(exist_ok=True); (ROOT/'models').mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'data/raw/fraud_oracle.csv',encoding='utf-8-sig')
    assert df[TARGET].isin([0,1]).all() and df.PolicyNumber.is_unique
    features=feature_columns(df); X=prepare(df,features); y=df[TARGET]
    dev,test=train_test_split(np.arange(len(df)),test_size=.2,stratify=y,random_state=SEED)
    train,val=train_test_split(dev,test_size=.25,stratify=y.iloc[dev],random_state=SEED)
    audit={'rows':len(df),'columns':len(df.columns),'positive_count':int(y.sum()),'prevalence':float(y.mean()),
       'null_cells':int(df.isna().sum().sum()),'duplicate_rows':int(df.duplicated().sum()),
       'duplicate_without_id':int(df.drop(columns=['PolicyNumber']).duplicated().sum()),
       'unique_policy':bool(df.PolicyNumber.is_unique),'age_zero_count':int((df.Age==0).sum()),
       'claim_month_zero_count':int((df.MonthClaimed.astype(str)=='0').sum()),
       'claim_day_zero_count':int((df.DayOfWeekClaimed.astype(str)=='0').sum()),
       'years':df.Year.value_counts().sort_index().to_dict(),
       'splits':{n:{'n':len(i),'positives':int(y.iloc[i].sum()),'prevalence':float(y.iloc[i].mean())} for n,i in [('train',train),('validation',val),('test',test)]},
       'features':features,'excluded':EXCLUDED,'split_overlap':bool(set(train)&set(val) or set(train)&set(test) or set(val)&set(test)),
       'data_sha256':hashlib.sha256((ROOT/'data/raw/fraud_oracle.csv').read_bytes()).hexdigest(),
       'leakage':'식별자·판단시점 불명 변수 제외. 나머지 변수도 실제 수집시점 미확인으로 업무 누수 완전 배제 불가.'}
    write_json('data_audit.json',audit)
    dictionary=[]
    for c in df:
        dictionary.append({'column':c,'dtype':str(df[c].dtype),'unique':int(df[c].nunique()),'missing':int(df[c].isna().sum()),
           'examples':' | '.join(map(str,df[c].dropna().unique()[:12])),
           'role':'학습 입력' if c in features else '제외: '+EXCLUDED[c]})
    pd.DataFrame(dictionary).to_csv(ROOT/'reports/data_dictionary.csv',index=False)
    eda=[]
    for c in ['BasePolicy','AccidentArea','PastNumberOfClaims','VehiclePrice','Year','Fault']:
        for value,g in df.groupby(c):
            eda.append({'feature':c,'value':str(value),'n':len(g),'positives':int(g[TARGET].sum()),'rate':float(g[TARGET].mean()),'scope':'전체 데이터 기술통계, 인과관계 아님'})
    pd.DataFrame(eda).to_csv(ROOT/'reports/eda.csv',index=False)
    print('AUDIT',json.dumps(audit,ensure_ascii=False),flush=True)
    candidates=[]
    # 상수 기준선. 실제 모델과 동일한 holdout 평가.
    baseline=float(y.iloc[train].mean())
    candidates.append(('무작위 기대값 기준선',None,np.full(len(val),baseline),{}))
    for C in [.1,1.]:
        model=Pipeline([('encode',OneHotEncoder(handle_unknown='ignore')),('clf',LogisticRegression(C=C,class_weight='balanced',max_iter=1500,random_state=SEED))])
        model.fit(X.iloc[train],y.iloc[train]); candidates.append((f'Logistic Regression C={C}',model,model.predict_proba(X.iloc[val])[:,1],{'C':C}))
    for depth in [4,6]:
        model=CatBoostClassifier(iterations=450,depth=depth,learning_rate=.05,loss_function='Logloss',eval_metric='PRAUC',random_seed=SEED,verbose=False,allow_writing_files=False,thread_count=4)
        model.fit(X.iloc[train],y.iloc[train],cat_features=features,eval_set=(X.iloc[val],y.iloc[val]),early_stopping_rounds=60)
        candidates.append((f'CatBoost depth={depth}',model,model.predict_proba(X.iloc[val])[:,1],{'depth':depth,'best_iteration':model.best_iteration_}))
    ranked=sorted(candidates[1:],key=lambda c:(evaluate(y.iloc[val],c[2])['ap'],evaluate(y.iloc[val],c[2])['capture_at_20']),reverse=True)
    selected=ranked[0]; threshold=choose_threshold(y.iloc[val],selected[2]); rows=[]
    # 이 지점 이후에만 final test 점수 조회. 후보 성능은 비교용이며 선정에는 재사용하지 않음.
    for name,model,scores,params in candidates:
        thr=choose_threshold(y.iloc[val],scores)
        test_scores=model.predict_proba(X.iloc[test])[:,1] if model is not None else np.full(len(test),baseline)
        rows.append({'name':name,'params':params,'threshold':thr,'validation':evaluate(y.iloc[val],scores,thr),'test':evaluate(y.iloc[test],test_scores,thr)})
        print('RESULT',name,rows[-1],flush=True)
    model=selected[1]; score=model.predict_proba(X.iloc[test])[:,1]
    joblib.dump({'model':model,'features':features,'threshold':threshold},ROOT/'models/selected.joblib')
    if isinstance(model,CatBoostClassifier):
        values=model.get_feature_importance(Pool(X.iloc[test],cat_features=features),type='ShapValues')
        contributions=values[:,:-1]; bases=values[:,-1]; raw=model.predict(X.iloc[test],prediction_type='RawFormulaVal')
    else:
        # 線形 모델의 one-hot SHAP는 학습 분포 기대값 기준으로 원래 변수별 합산.
        import shap
        encoder=model.named_steps['encode']; clf=model.named_steps['clf']
        background=encoder.transform(X.iloc[train]).mean(axis=0)
        explainer=shap.LinearExplainer(clf,(np.asarray(background).ravel(), np.zeros((background.shape[1],background.shape[1]))))
        encoded=encoder.transform(X.iloc[test]); sv=explainer.shap_values(encoded)
        offsets=np.cumsum([0]+[len(c) for c in encoder.categories_]); contributions=np.column_stack([np.asarray(sv)[:,offsets[i]:offsets[i+1]].sum(axis=1) for i in range(len(features))]); bases=np.repeat(float(explainer.expected_value),len(test)); raw=model.decision_function(X.iloc[test])
    residual=float(np.max(np.abs(contributions.sum(axis=1)+bases-raw)))
    explanations={}
    for j,idx in enumerate(test):
        order=np.argsort(-np.abs(contributions[j]))[:8]
        explanations[str(int(df.iloc[idx].PolicyNumber))]={'base_value':float(bases[j]),'raw_score':float(raw[j]),'factors':[{'feature':features[k],'value':str(X.iloc[idx][features[k]]),'contribution':float(contributions[j,k])} for k in order]}
    write_json('explanations.json',explanations)
    global_shap=pd.DataFrame({'feature':features,'mean_abs_shap':np.abs(contributions).mean(axis=0)}).sort_values('mean_abs_shap',ascending=False)
    global_shap.to_csv(ROOT/'reports/global_shap.csv',index=False)
    scored=df.iloc[test].copy(); scored['claim_id']=scored.PolicyNumber.astype(str); scored['risk_score']=score
    scored=scored.sort_values('risk_score',ascending=False,kind='stable'); scored['priority_rank']=np.arange(1,len(scored)+1); scored['review_band']=np.where(scored.priority_rank<=int(np.ceil(len(scored)*.2)),'우선 검토','일반 검토'); scored['label']=scored[TARGET]
    scored.to_csv(ROOT/'reports/scored_claims.csv',index=False)
    # 모델 선정에 사용하지 않는 연도 외삽 스트레스 평가. 파라미터 고정, 마지막 연도 holdout.
    latest=df.Year.max(); early=np.where(df.Year<latest)[0]; future=np.where(df.Year==latest)[0]
    a,b=train_test_split(early,test_size=.25,stratify=y.iloc[early],random_state=SEED)
    if isinstance(model,CatBoostClassifier):
        stress=CatBoostClassifier(**model.get_params()); stress.fit(X.iloc[a],y.iloc[a],cat_features=features,eval_set=(X.iloc[b],y.iloc[b]),early_stopping_rounds=60)
    else:
        from sklearn.base import clone
        stress=clone(model); stress.fit(X.iloc[a],y.iloc[a])
    st=choose_threshold(y.iloc[b],stress.predict_proba(X.iloc[b])[:,1]); stress_result=evaluate(y.iloc[future],stress.predict_proba(X.iloc[future])[:,1],st)
    # 고정 예측에서 test 행 재표집. 재학습 불확실성이나 사업장 이동을 포함하지 않음.
    rng=np.random.default_rng(SEED); samples=[]
    for _ in range(500):
        bidx=rng.integers(0,len(test),len(test)); samples.append(evaluate(y.iloc[test].to_numpy()[bidx],score[bidx],threshold))
    intervals={key:[float(np.percentile([r[key] for r in samples],2.5)),float(np.percentile([r[key] for r in samples],97.5))] for key in ['ap','capture_at_20','precision_at_20']}
    metrics={'models':rows,'selected_model':selected[0],'threshold':threshold,'split':audit['splits'],'features':features,'excluded_features':EXCLUDED,
       'seed':SEED,'selection_rule':'검증 AP 최대, 동률이면 검증 상위20% 포집률. test로 재선정하지 않음.',
       'ap_definition':'average_precision_score: 계단형 PR-AUC(AP), 사다리꼴 적분과 다름',
       'shap_max_additivity_error':residual,'bootstrap_95_fixed_predictions':intervals,'temporal_stress':{'heldout_year':int(latest),'metrics':stress_result,'train_n':len(a),'validation_n':len(b),'test_n':len(future),'limitations':'정확한 접수일 미제공·연도 의미 미검증, 연도별 이동 민감도 참고용'},
       'elapsed_seconds':round(time.time()-started,2),'score_note':'비보정 모델 점수. 실제 사기 확률 또는 자동 판단으로 사용 금지.'}
    write_json('metrics.json',metrics)
    print('SELECTED',selected[0],'SHAP ERROR',residual,'SECONDS',metrics['elapsed_seconds'],flush=True)
if __name__=='__main__': main()
