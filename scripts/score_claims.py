"""CSV 추론 CLI. 입력 자료를 평가 데이터와 분리해 로컬 배치로 저장한다."""
from pathlib import Path
import argparse,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from claimpilot.inference import read_claim_csv,load_bundle,score_claims
from claimpilot.intake import BatchStore

def main():
    parser=argparse.ArgumentParser();parser.add_argument('input',type=Path);parser.add_argument('--synthetic',action='store_true');args=parser.parse_args()
    result=score_claims(read_claim_csv(args.input.read_bytes()),load_bundle(ROOT/'models/inference_bundle.joblib'))
    store=BatchStore(ROOT/'data/intake.sqlite3');store.save(result,args.input.name,'synthetic' if args.synthetic else 'user_upload')
    print(json.dumps(dict(batch_id=result.batch_id,n=len(result.rows),issues=len(result.issues),synthetic=args.synthetic,has_labels=False),ensure_ascii=False))
if __name__=='__main__':main()
