from training.data_gen.build_sft_dataset import build_records
from training.verifier.report_verifier import is_acceptable, verify_report


def test_build_records_covers_all_three_companies():
    records = build_records(n_per_company=3, seed=0)
    company_ids = {r["meta"]["company_id"] for r in records}
    assert company_ids == {"000333", "600585", "3333HK"}
    assert len(records) == 9


def test_each_record_has_well_formed_chat_messages():
    records = build_records(n_per_company=2, seed=0)
    for rec in records:
        roles = [m["role"] for m in rec["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert all(isinstance(m["content"], str) and m["content"] for m in rec["messages"])


def test_first_variant_per_company_is_flagged_real():
    records = build_records(n_per_company=3, seed=0)
    real_flags = [r for r in records if r["meta"]["company_id"] == "000333"]
    assert sum(1 for r in real_flags if r["meta"]["is_real"]) == 1


def test_generated_assistant_reports_pass_the_verifier():
    records = build_records(n_per_company=3, seed=0)
    for rec in records:
        assistant_text = rec["messages"][-1]["content"]
        verdict = verify_report(
            assistant_text,
            evidence_count=rec["meta"]["evidence_count"],
            data_coverage=rec["meta"]["data_coverage"],
        )
        assert is_acceptable(verdict), f"{rec['meta']}: {verdict.details}"


def test_build_records_is_reproducible_given_same_seed():
    a = build_records(n_per_company=3, seed=7)
    b = build_records(n_per_company=3, seed=7)
    assert a == b
