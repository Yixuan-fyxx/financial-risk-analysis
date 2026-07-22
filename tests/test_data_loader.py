from fin_risk.data.loader import list_companies, load_announcements, load_company, load_corpus, load_news


def test_list_companies_finds_all_three_real_companies():
    ids = list_companies()
    assert set(ids) == {"000333", "600585", "3333HK"}


def test_load_company_parses_required_fields():
    for company_id in list_companies():
        company = load_company(company_id)
        assert company.name
        assert company.periods, f"{company_id} has no periods"
        latest = company.latest
        # Every real period must at least carry revenue and net_profit —
        # otherwise a risk assessment can't say anything meaningful at all.
        assert latest.revenue is not None
        assert latest.net_profit is not None


def test_announcements_and_news_reference_their_own_company():
    for company_id in list_companies():
        for doc in load_announcements(company_id) + load_news(company_id):
            assert doc.company_id == company_id
            assert doc.date
            assert doc.title
            assert doc.doc_id


def test_load_corpus_without_filter_returns_all_documents():
    all_docs = load_corpus()
    per_company_total = sum(
        len(load_announcements(cid)) + len(load_news(cid)) for cid in list_companies()
    )
    assert len(all_docs) == per_company_total
    assert len(all_docs) > 0
