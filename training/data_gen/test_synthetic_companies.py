from fin_risk.data.loader import load_company
from training.data_gen.synthetic_companies import (
    generate_variants,
    load_all_real_companies,
    stable_seed_offset,
)


def test_first_variant_is_always_the_unperturbed_original():
    company = load_company("600585")
    variants = generate_variants(company, n=5, magnitude=0.3, seed=1)
    assert variants[0] == company


def test_perturbed_variants_stay_within_magnitude_bound():
    company = load_company("600585")
    variants = generate_variants(company, n=10, magnitude=0.2, seed=1)
    original_latest = company.latest
    for variant in variants[1:]:
        latest = variant.latest
        for field_name in ["revenue", "net_profit", "total_assets", "total_liabilities"]:
            orig = getattr(original_latest, field_name)
            new = getattr(latest, field_name)
            if orig is None:
                assert new is None
                continue
            assert abs(new - orig) <= abs(orig) * 0.2 + 1e-6


def test_none_fields_stay_none():
    company = load_company("3333HK")
    variants = generate_variants(company, n=5, magnitude=0.3, seed=2)
    for variant in variants:
        assert variant.latest.cash is None
        assert variant.latest.operating_cash_flow is None


def test_company_identity_fields_are_preserved():
    company = load_company("000333")
    variants = generate_variants(company, n=3, magnitude=0.2, seed=3)
    for variant in variants:
        assert variant.company_id == company.company_id
        assert variant.name == company.name
        assert variant.ticker == company.ticker


def test_generation_is_reproducible_given_same_seed():
    company = load_company("000333")
    a = generate_variants(company, n=5, magnitude=0.2, seed=42)
    b = generate_variants(company, n=5, magnitude=0.2, seed=42)
    assert a == b


def test_different_seeds_produce_different_variants():
    company = load_company("000333")
    a = generate_variants(company, n=5, magnitude=0.2, seed=1)
    b = generate_variants(company, n=5, magnitude=0.2, seed=2)
    assert a[1:] != b[1:]


def test_n_zero_returns_empty_list():
    company = load_company("000333")
    assert generate_variants(company, n=0) == []


def test_stable_seed_offset_is_deterministic_across_calls():
    assert stable_seed_offset("000333") == stable_seed_offset("000333")
    assert stable_seed_offset("000333") != stable_seed_offset("600585")


def test_load_all_real_companies_returns_three_companies():
    companies = load_all_real_companies()
    assert {c.company_id for c in companies} == {"000333", "600585", "3333HK"}
