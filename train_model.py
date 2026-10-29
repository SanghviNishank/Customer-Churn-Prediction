"""CLI script to train and evaluate customer churn models."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is on sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.models.train import main

if __name__ == "__main__":
    raise SystemExit(main())
