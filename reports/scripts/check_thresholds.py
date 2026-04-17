"""CI gate: assert demo cosine-similarity quality thresholds don't regress.

Reads the most recent `reports/<YYYY-MM-DD>-demo-results.json` written by
`reports/scripts/run_demo_modes.py` and fails if any mode's avg cosine
similarity falls below its committed threshold.

Thresholds are intentionally slack (below observed values) to allow for
numeric jitter across torch versions and CPU microarchitectures while still
catching real algorithmic regressions (like the 2026-04 QJL transpose bug).
"""
from __future__ import annotations

import glob
import json
import os
import sys

THRESHOLDS = {
    "2.5-bit mixed": 0.85,
    "3.5-bit mixed": 0.95,
}

def main() -> int:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pattern = os.path.join(root, "reports", "*-demo-results.json")
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        print(f"ERROR: no demo-results JSON in {pattern}")
        return 2
    latest = files[-1]
    data = json.load(open(latest))
    print(f"Checking {os.path.relpath(latest, root)}")
    fail = False
    for r in data["results"]:
        mode = r["mode"]
        threshold = THRESHOLDS.get(mode)
        if threshold is None:
            print(f"  {mode}: no threshold defined, skipping")
            continue
        cos = r["avg_cosine"]
        ok = cos >= threshold
        print(f"  {mode}: avg_cosine={cos:.4f} (>= {threshold}?) {'OK' if ok else 'FAIL'}")
        if not ok:
            fail = True
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
