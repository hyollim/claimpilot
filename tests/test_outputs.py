from pathlib import Path
import json
import pandas as pd
import pytest
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture
def metrics():
    p=ROOT/'reports/metrics.json'
    if not p.exists(): pytest.skip('python scripts/train.py 후 산출물 검증')
    return json.loads(p.read_text())
def test_split_and_shap(metrics):
    assert sum(v['n'] for v in metrics['split'].values())==15420
    assert metrics['shap_max_additivity_error']<1e-5
    assert metrics['selected_model']==max(metrics['models'][1:],key=lambda r:(r['validation']['ap'],r['validation']['capture_at_20']))['name']
def test_score_and_explanation_alignment(metrics):
    df=pd.read_csv(ROOT/'reports/scored_claims.csv'); exp=json.loads((ROOT/'reports/explanations.json').read_text())
    assert df.claim_id.is_unique and len(df)==metrics['split']['test']['n']
    assert set(df.claim_id.astype(str))==set(exp)
    assert df.risk_score.between(0,1).all()
    audit=json.loads((ROOT/'reports/data_audit.json').read_text()); assert not audit['split_overlap']
