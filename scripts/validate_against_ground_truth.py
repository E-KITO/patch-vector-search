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

    # Or import and add your own pipeline variant (same index, different
    # query embedding — e.g. a new resize mode or tiling stride):
    from validate_against_ground_truth import run_comparison, default_pipelines
    pipelines = default_pipelines()
    pipelines["my_new_idea"] = (pipelines["baseline_v1"][0], lambda images: my_embed_fn(images))
    run_comparison(pipelines)

    # A pipeline can also point at an entirely different PatchIndex (e.g.
    # comparing the uni_v1 corpus against a differently-encoded/normalized
    # uni_v2 corpus, not just a different query embedding on the same
    # index) — each pipeline entry is (PatchIndex, embed_fn), so just pass a
    # different PatchIndex.load(...) as the first element of the tuple.

No Slurm/GPU-heavy job needed for the default (single-index) case — runs
locally in a few minutes (~7 categories x however many pipelines are
registered). Comparing a different corpus/index first requires that index to
be built (see experiments/0001+0002, or 0004+0005 for the uni_v2 corpus).
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


def load_v1_index() -> PatchIndex:
    """The original, GT-validated-best corpus: uni_v1 (1024-dim), 224px native patches."""
    exp_dir = Path("outputs/0002_20260808_build_faiss_index/default")
    return PatchIndex.load(
        index_path=exp_dir / "index.faiss",
        manifest_path=exp_dir / "manifest.parquet",
        slide_meta_path=exp_dir / "slide_meta.parquet",
        features_dir=Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1"),
    )


def load_v2_index(exp_dir: str = "outputs/0005_20260814_build_faiss_index_v2/default") -> PatchIndex:
    """The alternate corpus: uni_v2 (1536-dim, larger ViT-giant model), 256px native
    patches, extracted from raw WSIs Macenko-normalized via `torchstain` (confirmed —
    see V2_STAIN_REFERENCE and lib/torchstain_normalize.py's docstring for the
    self-consistency test that identified this). NOT a controlled "same everything +
    normalization" comparison against v1 — encoder and patch size changed too.

    Defaults to experiments/0005 (pq_m=64, 24 dims/subvector). An exact-vs-approximate
    ranking diagnostic on the Hypertrophy category found this build's PQ was burying
    real matches hundreds of ranks too low (e.g. one GT slide's approximate rank was
    313, but its *exact* similarity ranked it ~11th among plausible candidates) — so
    experiments/0006 was built with pq_m=96 (16 dims/subvector, matching uni_v1's
    granularity) to test whether finer PQ would fix this. **It didn't**: re-running
    the full 7-category GT comparison against 0006 gave the same best_rank/mean_rank
    as 0005 within noise (e.g. Hypertrophy stayed at 27, Glycogen got slightly worse:
    107->127) despite costing more (bigger index, slower search). This means the
    buried-match problem isn't primarily PQ subvector precision — more likely IVF
    cluster coverage (nlist=4096 clusters, which nprobe controls exploration of;
    raising nprobe 64->256 only helped marginally too, 27->25). Left as an open,
    unresolved question — see patch-vector-search-project memory for the full
    diagnostic. Pass exp_dir="outputs/0006_.../default" to use the pq_m=96 build
    instead (no measured benefit, kept for reference)."""
    exp_dir = Path(exp_dir)
    return PatchIndex.load(
        index_path=exp_dir / "index.faiss",
        manifest_path=exp_dir / "manifest.parquet",
        slide_meta_path=exp_dir / "slide_meta.parquet",
        features_dir=Path("data/trident_processed_macenko/results/trident/20x_256px_0px_overlap/features_uni_v2"),
    )


# The exact reference patch used to Macenko-normalize the raw WSIs behind the uni_v2
# corpus (confirmed by the user). Must be paired with torchstain's normalizer, not
# lib.stain_normalize's — see lib/torchstain_normalize.py's docstring: a
# self-consistency test (re-embedding a known v2 corpus patch and comparing to its
# own stored vector) only reached ~0.5-0.84 cosine similarity with the from-scratch
# implementation, vs. ~0.96-0.996 with torchstain (should be ~1.0 for a perfectly
# reproduced pipeline, and IS ~1.0 for uni_v1 via the same test).
V2_STAIN_REFERENCE = Path("data/baseline/63958_x38976_y7616.png")


def _embed_v2_normalized(images) -> np.ndarray:
    from lib.torchstain_normalize import normalize_to_reference

    tiles = []
    for f in images:
        normed = normalize_to_reference(str(f), V2_STAIN_REFERENCE)
        # Tried native_tile_size=256 here (the v2 corpus's real patches were
        # cropped at 256px native then resized to 224 — see
        # lib.raw_patch.crop_patch's patch_size_level0=256 usage for this
        # corpus, and lib.query_embedding.embed_image_tiles's native_tile_size
        # param). Measured on the same 7-category GT set: worse or flat in
        # 5/7 categories (e.g. Hypertrophy 27->50, Inclusion body 14->29),
        # better in 1 (Increased mitosis 285->177). Reverted to plain
        # tile_size=224 (no native_tile_size) as the better-performing
        # default overall — atlas query images don't have a true "native
        # pixel scale" matching the corpus's level0 pixels the way real WSI
        # regions do, so this correction doesn't transfer as cleanly as it
        # does for lib.raw_patch.crop_patch's real-pixel case.
        tiles.append(embed_image_tiles(normed, tile_size=224, encoder_name="uni_v2"))
    return np.concatenate(tiles, axis=0)


def default_pipelines() -> dict[str, tuple[PatchIndex, callable]]:
    """Each pipeline is (PatchIndex, embed_fn(images) -> (n_tiles, dim) array).
    "baseline_v1" (plain tiling, no correction, uni_v1 corpus) is the current
    recommended default — see lib/query_embedding.py's module-level guidance.
    "baseline_v2" torchstain-normalizes each query image toward
    V2_STAIN_REFERENCE before tiling — the fair comparison, matching how the
    v2 corpus itself was preprocessed (raw, unnormalized v2 queries scored
    much worse due to this mismatch alone, not because uni_v2 is a worse model
    — see lib/torchstain_normalize.py's docstring).
    """
    pipelines = {
        "baseline_v1": (
            load_v1_index(),
            lambda images: np.concatenate([embed_image_tiles(str(f), tile_size=224) for f in images], axis=0),
        ),
    }
    try:
        pipelines["baseline_v2"] = (load_v2_index(), _embed_v2_normalized)
    except FileNotFoundError:
        print("NOTE: uni_v2 index not found (run experiments/0004+0005 first) — comparing baseline_v1 only")
    return pipelines


def rank_stats(ranked_df: pd.DataFrame, gt_slides: set[str]) -> dict:
    ranked_df = ranked_df.reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    hits = ranked_df[ranked_df.slide_id.isin(gt_slides)]
    return dict(
        found=len(hits),
        best_rank=int(hits["rank"].min()) if len(hits) else None,
        mean_rank=round(float(hits["rank"].mean()), 1) if len(hits) else None,
    )


def run_comparison(pipelines: dict[str, tuple[PatchIndex, callable]] | None = None) -> pd.DataFrame:
    if pipelines is None:
        pipelines = default_pipelines()

    gt = pd.read_csv(GT_CSV)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    # Ground-truth slide membership is checked per-pipeline (each may run over a
    # different corpus/index with a different slide_id set), not shared globally.

    results = []
    t_all = time.time()
    for cat_dir_name, finding_type in CATEGORIES.items():
        cat_dir = ATLAS_DIR / cat_dir_name
        images = sorted(cat_dir.glob("*.jpg")) + sorted(cat_dir.glob("*.png"))
        if not images:
            print(f"WARNING: no images found for {cat_dir_name!r}, skipping")
            continue
        label = cat_dir_name.split(" - ")[0][:25]
        row = {"category": label}

        for name, (pi, embed_fn) in pipelines.items():
            corpus_slide_ids = set(pi.slide_meta.index.astype(str))
            gt_slides = set(gt.loc[gt.FINDING_TYPE == finding_type, "slide_id"]) & corpus_slide_ids
            row[f"{name}_n_gt"] = len(gt_slides)

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
