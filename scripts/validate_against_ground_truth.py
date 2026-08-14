"""Compare candidate query-embedding pipelines against confirmed ground truth.

Before trusting any change to how a query image is embedded/searched (a new
resize mode, auto-scaling, stain normalization, tiling stride, etc.), run it
through this script rather than judging by eye. Looking visually cleaner does
NOT mean better retrieval — this is how
lib.query_embedding.embed_image_tiles_auto_scale (magnification auto-scaling)
and Macenko stain normalization were both found to usually *hurt* recall
despite producing visibly sharper query tiles (see that function's docstring
for the full comparison table this script produced).

Ground truth: data/processed_csv/single_finding_liver.csv (slide_id ->
FINDING_TYPE for slides with one confirmed pathology). For each configured
NTP-atlas category, this maps to a FINDING_TYPE, computes the full
model-similarity ranking of all 998 corpus slides via
PatchIndex.search_top_slides_multi, and reports the best/mean rank and count
of the confirmed ground-truth slides within that ranking. Lower rank is
better; found=n_gt means every ground-truth slide had at least one candidate
hit.

Usage:
    .venv/bin/python3 scripts/validate_against_ground_truth.py

    # Or import and add your own pipeline variant:
    from validate_against_ground_truth import run_comparison, PIPELINES
    PIPELINES["my_new_idea"] = lambda images: my_embed_fn(images)
    run_comparison(PIPELINES)

No Slurm/GPU-heavy job needed — this runs locally in a few minutes (~7
categories x however many PIPELINES are registered).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from lib.query_embedding import embed_image_tiles
from lib.search import PatchIndex

INDEX_EXP_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")
ATLAS_DIR = Path("data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver")
GT_CSV = Path("data/processed_csv/single_finding_liver.csv")

# NTP atlas category directory -> confirmed ground-truth FINDING_TYPE.
# Only categories with >=1 ground-truth slide inside the 998-slide corpus are
# useful here — most atlas categories have zero (a known corpus limitation,
# see patch-vector-search-project memory), so this list is intentionally
# short rather than all 25 atlas categories.
CATEGORIES = {
    "Liver, Hepatocyte - Hypertrophy - Nonneoplastic Lesion Atlas": "Hypertrophy",
    "Liver - Necrosis - Nonneoplastic Lesion Atlas": "Single cell necrosis",
    "Liver, Hepatocyte – Increased Mitosis - Nonneoplastic Lesion Atlas": "Increased mitosis",
    "Liver, Hepatocyte - Glycogen Accumulation and Depletion - Nonneoplastic Lesion Atlas": "Deposit, glycogen",
    "Liver - Extramedullary Hematopoiesis - Nonneoplastic Lesion Atlas": "Hematopoiesis, extramedullary",
    "Liver, Kupffer Cell - Hyperplasia - Nonneoplastic Lesion Atlas": "Proliferation, Kupffer cell",
    "Liver, Hepatocyte - Cytoplasmic Inclusions - Nonneoplastic Lesion Atlas": "Inclusion body, intracytoplasmic",
}

# Named pipelines to compare: category dir -> list of reference image paths -> (n_tiles, 1024) embeddings.
# "baseline" (plain tiling, no correction) is the current recommended default
# — see lib/query_embedding.py's module-level guidance. Add more entries here
# (or from a calling script, per this module's docstring) to test a new idea.
PIPELINES = {
    "baseline": lambda images: np.concatenate([embed_image_tiles(str(f), tile_size=224) for f in images], axis=0),
}


def rank_stats(ranked_df: pd.DataFrame, gt_slides: set[str]) -> dict:
    ranked_df = ranked_df.reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    hits = ranked_df[ranked_df.slide_id.isin(gt_slides)]
    return dict(
        found=len(hits),
        best_rank=int(hits["rank"].min()) if len(hits) else None,
        mean_rank=round(float(hits["rank"].mean()), 1) if len(hits) else None,
    )


def run_comparison(pipelines: dict[str, callable] = PIPELINES) -> pd.DataFrame:
    pi = PatchIndex.load(
        index_path=INDEX_EXP_DIR / "index.faiss",
        manifest_path=INDEX_EXP_DIR / "manifest.parquet",
        slide_meta_path=INDEX_EXP_DIR / "slide_meta.parquet",
        features_dir=FEATURES_DIR,
    )
    gt = pd.read_csv(GT_CSV)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    corpus_slide_ids = set(pi.slide_meta.index.astype(str))

    results = []
    t_all = time.time()
    for cat_dir_name, finding_type in CATEGORIES.items():
        cat_dir = ATLAS_DIR / cat_dir_name
        images = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png"))
        if not images:
            print(f"WARNING: no images found for {cat_dir_name!r}, skipping")
            continue
        gt_slides = set(gt.loc[gt.FINDING_TYPE == finding_type, "slide_id"]) & corpus_slide_ids
        label = cat_dir_name.split(" - ")[0][:25]
        row = {"category": label, "n_gt": len(gt_slides)}

        for name, embed_fn in pipelines.items():
            tiles = embed_fn(images)
            ranked = pi.search_top_slides_multi(tiles, k_candidates=8000, nprobe=64, top_n_slides=998)
            s = rank_stats(ranked, gt_slides)
            row[f"{name}_found"] = s["found"]
            row[f"{name}_best"] = s["best_rank"]
            row[f"{name}_mean"] = s["mean_rank"]

        results.append(row)
        print(row, flush=True)

    print(f"\nTOTAL: {time.time() - t_all:.1f}s")
    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_comparison()
    out_path = Path("outputs/gt_validation_results.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
