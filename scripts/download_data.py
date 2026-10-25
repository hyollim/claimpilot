"""Kaggle 공개 다운로드. 데이터 이용조건은 출처에서 확인해야 한다."""
from pathlib import Path
import hashlib, io, zipfile, json
import requests
ROOT = Path(__file__).resolve().parents[1]
URL = 'https://www.kaggle.com/api/v1/datasets/download/shivamb/vehicle-claim-fraud-detection'

def main():
    target=ROOT/'data/raw/fraud_oracle.csv'; target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        response=requests.get(URL,timeout=120); response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            # 명시적 파일만 추출. 임의 경로 압축 해제 방지.
            target.write_bytes(archive.read('fraud_oracle.csv'))
    metadata={'source':'https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection',
              'download_url':URL,'file':'fraud_oracle.csv',
              'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
              'redistribution':'원본 데이터는 저장소에 포함하지 않음. 이용조건은 Kaggle 원문 확인.'}
    (ROOT/'resources/data_provenance.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(metadata['sha256'])
if __name__=='__main__': main()
