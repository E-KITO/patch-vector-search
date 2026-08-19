"""Step 3 of the IVF-clustering-hypothesis check: is a poorly-ranked
ground-truth slide poorly ranked because of PQ quantization error, or
because its true (exact) similarity to the query is genuinely low?

Step 2 (scripts/nprobe_sweep.py) showed that raising nprobe up to
nlist=4096 (i.e. probing every IVF cluster, eliminating cluster-coverage
loss) barely changes best_rank for the worst categories (Kupffer cell
proliferation, Inclusion body) — evidence against the IVF cluster-coverage
hypothesis. What nprobe can't rule out is PQ quantization error itself:
search_top_slides_multi's ranking always uses FAISS's PQ-approximated inner
product (see lib/search.py's docstring on search_top_slides — "no exact
re-rank since this is an aggregate ranking"), so a genuinely strong match
could still be scored artificially low by PQ compression alone, independent
of which/how many clusters were searched.

This script recomputes the *exact* (uncompressed, raw float32) cosine
similarity for the same approximate candidate pool (reusing
PatchIndex._exact_similarity, the same routine search_similar_patches
already uses for patch-level re-ranking) and re-ranks slides with it,
then compares best_rank/mean_rank against the standard PQ-approximate
ranking for each of the 7 ground-truth categories.

    Large gap (exact much better than approx) -> PQ quantization error is
    a real driver of the bad rankings, independent of nprobe/nlist coverage.
    Small/no gap -> the true model similarity is genuinely low; the
    ceiling here is the query image content (see README's "manual ROI crop"
    open item), not the ANN index.

Cost control: exact-reranking every tile of a reference image against an
8000-candidate pool is expensive (an atlas image tiles into ~hundreds of
224px tiles). Mirrors the same trade-off lib.search.PatchIndex.
search_similar_patches_multi already makes in production: rank tiles by a
cheap raw FAISS score first, and only exact-rerank the top
MAX_TILES_RERANKED of them. A real signal hiding only in a tile that scores
poorly even by the raw approximate metric would be missed by this — a
caveat to keep in mind if this diagnostic comes back negative.

Usage:
    .venv/bin/python3 scripts/exact_vs_approx_diagnostic.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from validate_against_ground_truth import ATLAS_DIR, CATEGORIES, GT_CSV, default_pipelines, rank_stats

NPROBE = 4096  # = nlist; the "probe every cluster" setting Step 2 found saturates at.
K_CANDIDATES = 8000  # matches the default_pipelines()/run_comparison() convention.
MAX_TILES_RERANKED = 8  # same cost/coverage trade-off as search_similar_patches_multi (there: 4).


def _rank_slides(combined: pd.DataFrame, similarity_col: str, slide_meta: pd.DataFrame) -> pd.DataFrame:
    """Aggregate deduplicated per-patch similarities to a slide-level ranking,
    same shape/logic as PatchIndex.search_top_slides's return value."""
    agg = combined.groupby("slide_id")[similarity_col].agg(n_hits="size", mean_similarity="mean", max_similarity="max")
    agg = agg.join(slide_meta[["total_patches"]])
    agg["n_hits_ratio"] = agg["n_hits"] / agg["total_patches"]
    return agg.sort_values("n_hits_ratio", ascending=False).reset_index()


def approx_vs_exact_ranking(pi, tiles: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For the top-scoring tiles of one reference image set, retrieve the
    approximate FAISS candidate pool and recompute exact similarity for it.

    Returns (approx_ranked, exact_ranked): slide-level rankings built from
    the *same* candidate patches, one scored with FAISS's PQ-approximate
    inner product, the other with PatchIndex._exact_similarity.
    """
    pi._set_nprobe(NPROBE)
    tiles = np.atleast_2d(tiles).astype(np.float32)

    approx_scores = pi.index.search(tiles, 1)[0][:, 0]
    top_tile_idx = np.argsort(approx_scores)[::-1][:MAX_TILES_RERANKED]

    frames = []
    for i in top_tile_idx:
        query = tiles[i]
        distances, ids = pi.index.search(query.reshape(1, -1), K_CANDIDATES)
        distances, ids = distances[0], ids[0]
        valid = ids >= 0
        distances, ids = distances[valid], ids[valid]

        candidates = pi.manifest.loc[ids, ["global_idx", "slide_id", "local_idx"]].reset_index(drop=True)
        candidates["approx_similarity"] = distances
        candidates["exact_similarity"] = pi._exact_similarity(query, candidates)
        frames.append(candidates)

    combined = pd.concat(frames, ignore_index=True)
    # Same patch can appear via more than one reranked tile — keep its best
    # score per metric, mirroring search_top_slides_multi's dedup-by-max.
    combined = combined.groupby("global_idx", as_index=False).agg(
        slide_id=("slide_id", "first"),
        approx_similarity=("approx_similarity", "max"),
        exact_similarity=("exact_similarity", "max"),
    )

    approx_ranked = _rank_slides(combined, "approx_similarity", pi.slide_meta)
    exact_ranked = _rank_slides(combined, "exact_similarity", pi.slide_meta)
    return approx_ranked, exact_ranked


def main() -> None:
    pi, embed_fn = default_pipelines()["baseline_v1"]

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
        label = cat_dir_name.split(" - ")[0][:25]

        gt_slides = set(gt.loc[gt.FINDING_TYPE == finding_type, "slide_id"]) & corpus_slide_ids
        tiles = embed_fn(images)
        approx_ranked, exact_ranked = approx_vs_exact_ranking(pi, tiles)

        approx_stats = rank_stats(approx_ranked, gt_slides)
        exact_stats = rank_stats(exact_ranked, gt_slides)

        row = {
            "category": label,
            "n_gt": len(gt_slides),
            "approx_found": approx_stats["found"],
            "approx_best": approx_stats["best_rank"],
            "approx_mean": approx_stats["mean_rank"],
            "exact_found": exact_stats["found"],
            "exact_best": exact_stats["best_rank"],
            "exact_mean": exact_stats["mean_rank"],
        }
        results.append(row)
        print(row, flush=True)

    print(f"\nTOTAL: {time.time() - t_all:.1f}s")

    df = pd.DataFrame(results)
    out_path = Path("outputs/gt_validation_exact_vs_approx.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
