"""Verification script to confirm that original Jupyter notebooks remain 100% unchanged."""
from __future__ import annotations

import sys
from pathlib import Path
import hashlib

# Expected hashes of the original notebook files
EXPECTED_HASHES = {
    "CustomerFinal.ipynb": "FF9873E29B47A0102B9A3A0ED8581145155B751A8664F1251B94F52AF9630ECE".lower(),
    "Customer_pratice.ipynb": "12E7232CA3674E632AB361004543083ADC84F7EEB9CAFBC6064D8B2F2DD4C8F5".lower(),
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().lower()

def main() -> int:
    root = Path(__file__).resolve().parent.parent
    notebook_paths = [
        root / "CustomerFinal.ipynb",
        root / "Customer_pratice.ipynb",
        root / "notebooks" / "original" / "CustomerFinal.ipynb",
        root / "notebooks" / "original" / "Customer_pratice.ipynb",
    ]

    all_matched = True
    print("=" * 70)
    print("NOTEBOOK INTEGRITY VERIFICATION")
    print("=" * 70)

    for p in notebook_paths:
        if not p.exists():
            print(f"[MISSING] {p}")
            all_matched = False
            continue
        
        current_hash = sha256_file(p)
        base_name = p.name
        expected_hash = EXPECTED_HASHES.get(base_name)

        if current_hash == expected_hash:
            print(f"[PASSED]  {p.relative_to(root)}")
            print(f"          SHA256: {current_hash}")
        else:
            print(f"[FAILED]  {p.relative_to(root)}")
            print(f"          Expected: {expected_hash}")
            print(f"          Actual:   {current_hash}")
            all_matched = False

    print("=" * 70)
    if all_matched:
        print("ALL NOTEBOOK HASHES MATCH EXACT ORIGINAL BASELINE! Integrity preserved.")
        return 0
    else:
        print("ERROR: Notebook hash mismatch detected!")
        return 1

if __name__ == "__main__":
    raise SystemExit(main())

