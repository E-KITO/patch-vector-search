"""Does raising nprobe (IVF clusters probed per search) recover ground-truth
slides that rank poorly under the default nprobe=64?

Step 2 of the IVF-clustering-coverage hypothesis check (see Step 1:
scripts/validate_against_ground_truth.py, which established the nprobe=64
baseline in outputs/gt_validation_results.csv). This script re-runs the same
7-category ground-truth comparison at several nprobe values on the uni_v1
index only (uni_v2 not built in this repo state). nprobe=4096 equals
nlist=4096 (see lib/faiss_index.py), i.e. every IVF cluster is probed — a
pseudo-exhaustive search where cluster-coverage loss is eliminated and only
PQ quantization error remains. If best_rank for the poorly-ranked categories
(e.g. Kupffer cell proliferation, Inclusion body) improves substantially as
nprobe approaches nlist, that supports the IVF-clustering hypothesis; if it
barely moves, clustering coverage is likely not the primary cause on uni_v1.

Usage:
    .venv/bin/python3 scripts/nprobe_sweep.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from validate_against_ground_truth import default_pipelines, run_comparison

NPROBE_VALUES = [64, 256, 1024, 4096]


def main() -> None:
    pipelines = default_pipelines()

    results = []
    t_all = time.time()
    for nprobe in NPROBE_VALUES:
        print(f"\n{'=' * 70}\nnprobe={nprobe}\n{'=' * 70}")
        df = run_comparison(pipelines, nprobe=nprobe)
        df.insert(0, "nprobe", nprobe)
        results.append(df)

    combined = pd.concat(results, ignore_index=True)
    print(f"\nTOTAL: {time.time() - t_all:.1f}s")

    out_path = Path("outputs/gt_validation_nprobe_sweep.csv")
    out_path.parent.mkdir(exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
