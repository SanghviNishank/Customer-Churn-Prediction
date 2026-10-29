"""CLI script to train churn models and export artifacts."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.train import main

if __name__ == "__main__":
    raise SystemExit(main())

