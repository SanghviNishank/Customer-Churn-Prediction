"""Utility helpers for paths, hashing, logging, and metrics formatting."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_analytics")


def get_project_root() -> Path:
    """Find the root directory of the project regardless of current working directory."""
    # Start with the directory containing this file
    curr = Path(__file__).resolve().parent
    while curr != curr.parent:
        if (curr / "requirements.txt").exists() or (curr / "CustomerFinal.ipynb").exists():
            return curr
        curr = curr.parent
    # Fallback to current working directory
    return Path.cwd()


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
SUMMARIES_DIR = REPORTS_DIR / "summaries"


def compute_file_hash(path: Path | str, algorithm: str = "sha256") -> str:
    """Compute cryptographic hash of a file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    h = getattr(hashlib, algorithm)()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def format_currency(val: float | int) -> str:
    """Format numeric value as USD currency."""
    return f"${val:,.2f}"


def format_percentage(val: float) -> str:
    """Format decimal value as percentage."""
    return f"{val:.1%}"

