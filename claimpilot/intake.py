"""신규 분석 배치 저장. 정답 라벨과 평가 데이터는 이 DB에 저장하지 않는다."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

MAX_BATCHES=50
MAX_PAYLOAD_BYTES=50*1024*1024

class BatchStore:
    def __init__(self,path):
        self.path=str(path); Path(self.path).parent.mkdir(parents=True,exist_ok=True)
        with closing(self._connect()) as conn,conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY, name TEXT NOT NULL, source_kind TEXT NOT NULL,
                created_at TEXT NOT NULL, rows_json TEXT NOT NULL,
                explanations_json TEXT NOT NULL, issues_json TEXT NOT NULL, metadata_json TEXT NOT NULL)''')
    def _connect(self):
        con=sqlite3.connect(self.path,timeout=10);con.row_factory=sqlite3.Row;return con
    def save(self,result,name,source_kind):
        if source_kind not in ('synthetic','user_upload'): raise ValueError('입력 출처를 명시하세요.')
        if not isinstance(name,str) or not 1<=len(name)<=200: raise ValueError('배치명은 1~200자입니다.')
        if any(c in result.rows for c in ['label','FraudFound_P','fraud_reported']): raise ValueError('정답 라벨 저장 금지')
        now=datetime.now(timezone.utc).isoformat()
        payload=(result.rows.to_json(orient='records',force_ascii=False),json.dumps(result.explanations,ensure_ascii=False),json.dumps(result.issues,ensure_ascii=False),json.dumps(result.metadata,ensure_ascii=False))
        payload_size=sum(len(p.encode('utf-8')) for p in payload)
        # 동일 내용 재업로드는 기존 배치·담당자 기록을 보존한다. 이름/출처를 바꾸어 덮어쓰지 않는다.
        with closing(self._connect()) as conn,conn:
            conn.execute('BEGIN IMMEDIATE')
            if conn.execute('SELECT 1 FROM batches WHERE batch_id=?',(result.batch_id,)).fetchone():return result.batch_id
            count,size=conn.execute('SELECT COUNT(*), COALESCE(SUM(length(CAST(rows_json AS BLOB))+length(CAST(explanations_json AS BLOB))+length(CAST(issues_json AS BLOB))+length(CAST(metadata_json AS BLOB))),0) FROM batches').fetchone()
            if count>=MAX_BATCHES or size+payload_size>MAX_PAYLOAD_BYTES:
                raise ValueError('배치 저장 한도(50개 또는 payload 50 MiB)에 도달했습니다. 기존 기록은 삭제하지 않습니다. 백업·정리는 사용자 확인 후 진행하세요.')
            conn.execute('INSERT INTO batches VALUES (?,?,?,?,?,?,?,?)',(result.batch_id,name,source_kind,now,*payload))
        return result.batch_id
    def list_batches(self):
        with closing(self._connect()) as conn:
            rows=conn.execute('SELECT batch_id,name,source_kind,created_at,metadata_json FROM batches ORDER BY created_at DESC').fetchall()
        return [{**{k:r[k] for k in ['batch_id','name','source_kind','created_at']},**json.loads(r['metadata_json'])} for r in rows]
    def load(self,batch_id):
        with closing(self._connect()) as conn:
            row=conn.execute('SELECT * FROM batches WHERE batch_id=?',(batch_id,)).fetchone()
        if row is None: raise KeyError('저장된 분석 배치를 찾을 수 없습니다.')
        rows=pd.DataFrame(json.loads(row['rows_json']));rows['claim_id']=rows['claim_id'].astype(str)
        return dict(rows=rows,explanations=json.loads(row['explanations_json']),issues=json.loads(row['issues_json']),
                    metadata=json.loads(row['metadata_json']),batch_id=row['batch_id'],name=row['name'],source_kind=row['source_kind'],created_at=row['created_at'])
