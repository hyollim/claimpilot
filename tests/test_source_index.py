import json
from pathlib import Path
from claimpilot.retrieval import search_documents
ROOT=Path(__file__).resolve().parents[1]

def test_original_excerpts_preserved_and_navigation_labeled():
    old=json.loads((ROOT/'resources/official_sources.json').read_text())
    index=json.loads((ROOT/'resources/official_index.json').read_text())
    assert index[:8]==old and len(index)==11
    assert all(s['record_type']=='navigation' and '본문' in s['limitations'] for s in index[8:])
    assert len({s['id'] for s in index})==11
    assert 'resources/official_sources_phase2_candidates.json' in (ROOT/'.gitignore').read_text()

def test_document_route_search():
    rows=search_documents('자동차사고 필요서류',5,ROOT/'resources/official_index.json',method='bm25')
    assert any(r['id']=='p2-route-kb-documents' and r['record_type']=='navigation' for r in rows)
    for r in rows:
        assert all(r.get(key) for key in ['title','organization','accessed_at','url','section','text'])
