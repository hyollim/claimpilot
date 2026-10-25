import math
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from claimpilot.modeling import FEATURES
from claimpilot.inference import (InputValidationError,read_claim_csv,validate_claims,score_claims,export_csv)
from claimpilot.intake import BatchStore
from claimpilot.review import ReviewStore

@pytest.fixture
def bundle():
    X=pd.DataFrame({c:['a','b']*6 for c in FEATURES})
    for c in ['WeekOfMonth','WeekOfMonthClaimed']:X[c]=['1','2']*6
    X['Deductible']=['300','400']*6
    model=Pipeline([('encode',OneHotEncoder(handle_unknown='ignore')),('clf',LogisticRegression(C=.1))])
    model.fit(X,[0,1]*6)
    mean=np.asarray(model.named_steps['encode'].transform(X).mean(axis=0)).ravel()
    return dict(model=model,features=FEATURES,categories={c:sorted(X[c].unique()) for c in FEATURES},
                model_fingerprint='test-model',background_mean=mean,schema_version=1)

@pytest.fixture
def claims(bundle):
    rows=pd.DataFrame({c:[bundle['categories'][c][0]]*6 for c in FEATURES})
    rows.insert(0,'claim_id',[f'NEW-{i}' for i in range(6)]);return rows

def test_schema_rejects_labels_missing_and_extra(bundle,claims):
    for bad in [claims.assign(label=0),claims.assign(FraudFound_P=0),claims.drop(columns=['Make']),claims.assign(new_outcome='x')]:
        with pytest.raises(InputValidationError):validate_claims(bad,bundle['categories'])

def test_missing_unseen_and_numeric_normalization(bundle,claims):
    claims.loc[0,'Make']=None;claims.loc[1,'Make']='unknown';claims.loc[2,'WeekOfMonth']='2.0'
    clean,issues=validate_claims(claims,bundle['categories'])
    assert clean.loc[0,'Make']=='__MISSING__' and clean.loc[2,'WeekOfMonth']=='2'
    assert len(issues)==2 and all(i['severity']=='warning' for i in issues)

@pytest.mark.parametrize('column,value',[('WeekOfMonth','0'),('WeekOfMonthClaimed','6'),('Deductible','-1'),('Deductible','NaN'),('Deductible','inf'),('WeekOfMonth','1.5'),('Deductible','abc')])
def test_bad_numeric_rejected(bundle,claims,column,value):
    claims.loc[0,column]=value
    with pytest.raises(InputValidationError):validate_claims(claims,bundle['categories'])

def test_empty_all_missing_bad_and_duplicate_id(bundle,claims):
    for bad in [claims.iloc[:0],claims.assign(claim_id='same'),claims.assign(claim_id=''),claims.assign(claim_id='../unsafe')]:
        with pytest.raises(InputValidationError):validate_claims(bad,bundle['categories'])
    claims.loc[0,FEATURES]=None
    with pytest.raises(InputValidationError):validate_claims(claims,bundle['categories'])

def test_csv_safety():
    with pytest.raises(ValueError):read_claim_csv(b'a,a\n1,2\n')
    with pytest.raises(ValueError):read_claim_csv(b'a,b\n1\n')
    with pytest.raises(ValueError):read_claim_csv(b'\xff\xff')
    with pytest.raises(ValueError):read_claim_csv(b'x'* (5*1024*1024+1))

def test_inference_shap_and_budget(bundle,claims):
    result=score_claims(claims,bundle)
    assert len(result.rows)==6 and result.rows.review_band.eq('우선 검토').sum()==2
    assert not {'label','FraudFound_P'}&set(result.rows)
    assert result.rows.record_id.str.startswith('new:').all()
    for row in result.rows.itertuples():
        e=result.explanations[row.claim_id]
        raw=e['base_value']+sum(f['contribution'] for f in e['factors'])+e['other_contribution']
        assert abs(raw-e['raw_score'])<1e-8
        assert abs(1/(1+math.exp(-raw))-row.risk_score)<1e-8
    assert result.batch_id==score_claims(claims,bundle).batch_id
    changed=claims.copy();changed.loc[0,'Make']='b'
    assert result.batch_id!=score_claims(changed,bundle).batch_id

def test_unknown_still_scores_and_warns(bundle,claims):
    claims.loc[0,'Make']='never-seen'
    result=score_claims(claims,bundle)
    assert result.metadata['unseen_cells']==1 and len(result.rows)==6
    assert result.rows.iloc[0].validation_status=='자료 보완 필요'

def test_persistence_review_namespacing(bundle,claims,tmp_path):
    result=score_claims(claims,bundle);store=BatchStore(tmp_path/'intake.sqlite')
    store.save(result,'fixture.csv','synthetic');store.save(result,'second.csv','user_upload')
    assert len(store.list_batches())==1 and store.load(result.batch_id)['source_kind']=='synthetic'
    loaded=BatchStore(tmp_path/'intake.sqlite').load(result.batch_id)
    assert len(loaded['rows'])==6 and loaded['explanations']==result.explanations
    reviews=ReviewStore(tmp_path/'reviews.sqlite')
    key=result.rows.iloc[0].record_id
    reviews.save(key,'검토중','신규 배치 검토',{'자료 대조':True})
    assert reviews.get(key)['status']=='검토중'
    assert reviews.get(result.rows.iloc[0].claim_id)['status']=='미검토'
    with pytest.raises(ValueError):store.save(result,'fixture','unknown')

def test_export_formula_protection():
    result=export_csv(pd.DataFrame({'claim_id':['A'],'Make':['=HYPERLINK(1)'],'risk_score':[.2]})).decode('utf-8-sig')
    assert "'=HYPERLINK" in result

def test_real_model_new_claims():
    from pathlib import Path
    from claimpilot.inference import load_bundle
    root=Path(__file__).resolve().parents[1]
    if not (root/'models/inference_bundle.joblib').exists():pytest.skip('prepare_inference.py 후 실제 파이프라인 검증')
    bundle=load_bundle(root/'models/inference_bundle.joblib')
    result=score_claims(read_claim_csv((root/'resources/demo_claims.csv').read_bytes()),bundle)
    assert len(result.rows)==6 and result.metadata['missing_cells']==2 and result.metadata['unseen_cells']==1
    assert not {'label','FraudFound_P'}&set(result.rows)

@pytest.mark.parametrize('column,value',[('WeekOfMonth','1.0000000000000001'),('Deductible','300.00000000000000001'),('Deductible','1e10000')])
def test_decimal_precision_not_rounded(bundle,claims,column,value):
    claims.loc[0,column]=value
    with pytest.raises(InputValidationError):validate_claims(claims,bundle['categories'])

def test_nullable_and_categorical_frames(bundle,claims):
    claims['WeekOfMonth']=pd.Series([1]*6,dtype='Int64')
    claims['Make']=pd.Categorical(['a',None,'a','a','a','a'])
    clean,issues=validate_claims(claims,bundle['categories'])
    assert clean.WeekOfMonth.tolist()==['1']*6 and clean.loc[1,'Make']=='__MISSING__'
    claims['claim_id']=claims.claim_id.astype(object);claims.at[0,'claim_id']=['A','B']
    with pytest.raises(InputValidationError):validate_claims(claims,bundle['categories'])

def test_nonfinite_background_rejected(bundle,claims):
    bundle['background_mean'][0]=np.nan
    with pytest.raises(ValueError,match='비유한'):score_claims(claims,bundle)

def test_storage_capacity_never_deletes(bundle,claims,tmp_path,monkeypatch):
    import claimpilot.intake as intake
    monkeypatch.setattr(intake,'MAX_BATCHES',1)
    store=BatchStore(tmp_path/'capacity.sqlite');a=score_claims(claims,bundle)
    store.save(a,'first','synthetic');store.save(a,'duplicate','synthetic')
    claims.loc[0,'Make']='b';b=score_claims(claims,bundle)
    with pytest.raises(ValueError,match='저장 한도'):store.save(b,'second','synthetic')
    assert len(store.list_batches())==1 and store.load(a.batch_id)['name']=='first'
    monkeypatch.setattr(intake,'MAX_BATCHES',50);monkeypatch.setattr(intake,'MAX_PAYLOAD_BYTES',1)
    empty=BatchStore(tmp_path/'bytes.sqlite')
    with pytest.raises(ValueError,match='저장 한도'):empty.save(a,'oversize','synthetic')
    assert empty.list_batches()==[]
