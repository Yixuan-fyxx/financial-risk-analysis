"""Shared paths and constants for the fin_risk package."""

from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COMPANIES_DIR = DATA_DIR / "companies"
ANNOUNCEMENTS_DIR = DATA_DIR / "announcements"
NEWS_DIR = DATA_DIR / "news"

# Risk score is on a 0-100 scale, higher = riskier.
RISK_BANDS = [
    (0, 25, "低风险"),
    (25, 50, "中等风险"),
    (50, 75, "较高风险"),
    (75, 101, "高风险"),
]

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-5"
