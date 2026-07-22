from fin_risk.data.loader import Document
from fin_risk.rag.retriever import TfidfRetriever


def _doc(doc_id, company_id, date, title, content, tags=None):
    return Document(
        doc_id=doc_id, company_id=company_id, date=date, source_type="news",
        title=title, content=content, tags=tags or [],
    )


DOCS = [
    _doc("A1", "COMP_A", "2026-01-01", "评级下调公告", "评级机构将公司信用评级下调,提示流动性风险与偿债压力。"),
    _doc("A2", "COMP_A", "2026-06-01", "股东大会公告", "公司召开年度股东大会,审议年度分红方案。"),
    _doc("B1", "COMP_B", "2026-05-01", "并购公告", "公司完成对上游供应商的战略并购,拓展产业链。"),
]


def test_retrieve_ranks_semantically_relevant_doc_first():
    retriever = TfidfRetriever(DOCS, recency_half_life_days=9999)
    results = retriever.retrieve("信用评级 偿债压力", company_id="COMP_A", top_k=2, recency_boost=0.0)
    assert results[0].document.doc_id == "A1"


def test_retrieve_filters_by_company_id():
    retriever = TfidfRetriever(DOCS)
    results = retriever.retrieve("公告", company_id="COMP_B", top_k=5)
    assert all(r.document.company_id == "COMP_B" for r in results)
    assert len(results) == 1


def test_recency_boost_prefers_more_recent_document_given_similar_relevance():
    docs = [
        _doc("R1", "COMP_C", "2020-01-01", "行业动态", "行业景气度变化值得关注。"),
        _doc("R2", "COMP_C", "2026-06-30", "行业动态", "行业景气度变化值得关注。"),
    ]
    retriever = TfidfRetriever(docs, recency_half_life_days=180)
    results = retriever.retrieve("行业景气度", company_id="COMP_C", top_k=2, recency_boost=0.8, reference_date="2026-07-01")
    assert results[0].document.doc_id == "R2"


def test_empty_corpus_returns_empty_list():
    retriever = TfidfRetriever([])
    assert retriever.retrieve("任意查询") == []


def test_no_matching_company_returns_empty_list():
    retriever = TfidfRetriever(DOCS)
    assert retriever.retrieve("公告", company_id="NO_SUCH_COMPANY") == []
