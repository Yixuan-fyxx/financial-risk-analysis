"""Synthetic financial-snapshot augmentation for SFT/agent dataset generation.

Only 3 real companies exist under data/, far too few (company, snapshot)
combinations to SFT a model on report-writing format and style. Rather than
inventing new companies with fabricated news — which this project's whole
ethos rejects (see root README's "don't fabricate numbers to fill gaps") —
this module perturbs a real company's *financial figures* within a bounded
range around their actually-disclosed values, producing counterfactual
financial snapshots. `company_id`/`name` are left untouched, so downstream
code (RAG evidence retrieval) still resolves to that company's real,
unmodified announcements/news corpus — only the numbers feeding
risk_scoring change, never any textual evidence.
"""

from __future__ import annotations

import random
from dataclasses import replace

from fin_risk.data.loader import CompanyFinancials, PeriodFinancials, list_companies, load_company

_PERTURBABLE_FIELDS = [
    "revenue", "net_profit", "total_assets", "total_liabilities", "equity",
    "current_assets", "current_liabilities", "inventory", "cash",
    "operating_cash_flow", "short_term_debt",
]


def _perturb_period(period: PeriodFinancials, rng: random.Random, magnitude: float) -> PeriodFinancials:
    updates = {}
    for field_name in _PERTURBABLE_FIELDS:
        value = getattr(period, field_name)
        if value is None:
            continue
        factor = 1 + rng.uniform(-magnitude, magnitude)
        updates[field_name] = round(value * factor, 2)
    return replace(period, **updates)


def generate_variants(
    company: CompanyFinancials, n: int, magnitude: float = 0.20, seed: int = 0
) -> list[CompanyFinancials]:
    """Returns `n` counterfactual variants of `company`.

    Variant 0 is always the unperturbed original, so every batch includes at
    least one ground-truth (not synthetic) sample. Variants 1..n-1 have each
    populated numeric field independently scaled by a factor drawn from
    `Uniform(1-magnitude, 1+magnitude)` — this does not preserve balance-sheet
    identities (e.g. equity == total_assets - total_liabilities) exactly; it's
    a deliberately simple augmentation for training report-writing style, not
    a synthetic accounting simulator.
    """
    if n <= 0:
        return []
    variants = [company]
    rng = random.Random(seed)
    for _ in range(n - 1):
        periods = [_perturb_period(p, rng, magnitude) for p in company.periods]
        variants.append(replace(company, periods=periods))
    return variants


def stable_seed_offset(company_id: str) -> int:
    """A reproducible per-company seed offset (unlike Python's randomized `hash()`)."""
    return sum(ord(c) for c in company_id)


def load_all_real_companies() -> list[CompanyFinancials]:
    return [load_company(cid) for cid in list_companies()]
