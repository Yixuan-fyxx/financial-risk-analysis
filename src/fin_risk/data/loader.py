"""Loads company financials and the RAG document corpus (announcements/news) from disk.

All financial fields are optional: real public disclosures rarely expose every
line item a textbook ratio needs, so the risk-scoring layer must degrade
gracefully (skip an indicator, don't fabricate a number) rather than assume a
clean dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fin_risk.config import ANNOUNCEMENTS_DIR, COMPANIES_DIR, NEWS_DIR


@dataclass
class PeriodFinancials:
    period: str
    source: Optional[str] = None
    revenue: Optional[float] = None
    net_profit: Optional[float] = None
    total_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    equity: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    inventory: Optional[float] = None
    cash: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    short_term_debt: Optional[float] = None

    @classmethod
    def from_dict(cls, raw: dict) -> "PeriodFinancials":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in raw.items() if k in known})


@dataclass
class CompanyFinancials:
    company_id: str
    name: str
    ticker: str
    industry: str
    unit: str
    data_note: str = ""
    periods: list[PeriodFinancials] = field(default_factory=list)

    @property
    def latest(self) -> PeriodFinancials:
        return self.periods[-1]

    @property
    def previous(self) -> Optional[PeriodFinancials]:
        return self.periods[-2] if len(self.periods) > 1 else None


@dataclass
class Document:
    doc_id: str
    company_id: str
    date: str
    source_type: str  # "announcement" | "news"
    title: str
    content: str
    source: str = ""
    url: str = ""
    tags: list[str] = field(default_factory=list)


def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_companies() -> list[str]:
    return sorted(p.stem for p in COMPANIES_DIR.glob("*.json"))


def load_company(company_id: str) -> CompanyFinancials:
    path = COMPANIES_DIR / f"{company_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No company data file for '{company_id}' at {path}")
    raw = _read_json(path)
    periods = [PeriodFinancials.from_dict(p) for p in raw.get("periods", [])]
    return CompanyFinancials(
        company_id=raw["company_id"],
        name=raw["name"],
        ticker=raw.get("ticker", ""),
        industry=raw.get("industry", ""),
        unit=raw.get("unit", ""),
        data_note=raw.get("data_note", ""),
        periods=periods,
    )


def _load_documents(directory: Path, company_id: str) -> list[Document]:
    path = directory / f"{company_id}.json"
    if not path.exists():
        return []
    raw = _read_json(path)
    return [Document(**item) for item in raw]


def load_announcements(company_id: str) -> list[Document]:
    return _load_documents(ANNOUNCEMENTS_DIR, company_id)


def load_news(company_id: str) -> list[Document]:
    return _load_documents(NEWS_DIR, company_id)


def load_corpus(company_id: Optional[str] = None) -> list[Document]:
    """Loads the full RAG document corpus, optionally filtered to one company."""
    ids = [company_id] if company_id else list_companies()
    docs: list[Document] = []
    for cid in ids:
        docs.extend(load_announcements(cid))
        docs.extend(load_news(cid))
    return docs
