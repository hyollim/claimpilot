import numpy as np
import pandas as pd
import pytest
from claimpilot.modeling import evaluate,choose_threshold,feature_columns,prepare

def test_top20_uses_ceil_and_counts():
    r=evaluate([1,0,0,0,1,0],[.9,.8,.7,.6,.5,.4])
    assert r['reviewed_count']==2 and r['captured_count']==1 and r['capture_at_20']==.5

def test_top20_ties_stable():
    assert evaluate([1,0,0,0,0],[.1]*5)['capture_at_20']==1

def test_excluded_features_never_enter_model():
    df=pd.DataFrame({'FraudFound_P':[1], 'PolicyNumber':[5],'Fault':['x'],'Sex':['x'],'Make':['Honda']})
    assert feature_columns(df)==['Make']

def test_missing_transformed_without_fit():
    assert prepare(pd.DataFrame({'a':[None]}),['a']).iloc[0,0]=='__MISSING__'

def test_threshold_on_validation():
    assert choose_threshold([0,1,1],[.1,.6,.9])==.6

def test_invalid_scores():
    with pytest.raises(ValueError): evaluate([1],[np.nan])
