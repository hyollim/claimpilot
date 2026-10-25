import numpy as np
import pandas as pd
import pytest
from claimpilot.modeling import FEATURES,evaluate,choose_threshold,feature_columns,prepare

def test_top20_uses_ceil_and_counts():
    r=evaluate([1,0,0,0,1,0],[.9,.8,.7,.6,.5,.4])
    assert r['reviewed_count']==2 and r['captured_count']==1 and r['capture_at_20']==.5

def test_top20_ties_stable():
    assert evaluate([1,0,0,0,0],[.1]*5)['capture_at_20']==1

def test_excluded_features_never_enter_model():
    df=pd.DataFrame({c:['value'] for c in FEATURES})
    df['FraudFound_P']=1; df['PolicyNumber']=5; df['Fault']='x'; df['Sex']='x'
    assert feature_columns(df)==FEATURES
    df['InvestigationResult']='fraud'
    with pytest.raises(ValueError): feature_columns(df)

def test_missing_feature_rejected():
    with pytest.raises(ValueError): feature_columns(pd.DataFrame({'Make':['Honda']}))

def test_encoder_ignores_validation_only_category():
    from sklearn.preprocessing import OneHotEncoder
    encoder=OneHotEncoder(handle_unknown='ignore').fit([['Honda'],['Toyota']])
    assert encoder.transform([['unseen']]).sum()==0
    assert 'unseen' not in encoder.categories_[0]

def test_missing_transformed_without_fit():
    assert prepare(pd.DataFrame({'a':[None]}),['a']).iloc[0,0]=='__MISSING__'

def test_threshold_on_validation():
    assert choose_threshold([0,1,1],[.1,.6,.9])==.6

def test_invalid_scores():
    with pytest.raises(ValueError): evaluate([1],[np.nan])
