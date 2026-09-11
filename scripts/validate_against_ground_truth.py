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
model-similarity ranking of every slide in that pipeline's corpus via
PatchIndex.search_top_slides_multi, and reports the best/mean rank and count
of the confirmed ground-truth slides within that ranking. Lower rank is
better; found=n_gt means every ground-truth slide had at least one candidate
hit.

Usage:
    .venv/bin/python3 scripts/validate_against_ground_truth.py

    # Re-run the background-removal A/B (baseline_v1 = 0018, v1_predeblank = 0002)
    # without the slow torchstain v2/macenko embeds, to a named file:
    .venv/bin/python3 scripts/validate_against_ground_truth.py \
        --pipelines baseline_v1,v1_predeblank \
        --out outputs/gt_validations/gt_validation_results_deblank_ab.csv

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
from PIL import Image

from lib.atlas_figures import query_images
from lib.query_embedding import embed_image_tiles
from lib.search import PatchIndex

ATLAS_DIR = Path("data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver")
GT_CSV = Path("data/processed_csv/single_finding_liver.csv")

# NTP atlas category directory -> confirmed ground-truth FINDING_TYPE.
# Only categories with >=1 ground-truth slide inside the corpus are
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


def load_v1_index(
    exp_dir: str = "outputs/0018_20260909_build_faiss_index_deblank/default",
) -> PatchIndex:
    """The current production index: uni_v1 (1024-dim), 224px native patches, no
    stain normalization, with the 389,959 slide-background patches (sat_frac <
    0.10 — 2.12% of the corpus) dropped from the manifest before indexing
    (experiments/0017 -> 0018, promoted to default 2026-09-09).

    Background removal left the GT best_rank / n_hits_ratio ranking neutral but
    notably improved the max_similarity ranking in the self-retrieval diagnostic
    (granular eosinophilic sim_best 20 -> 2). See the "背景パッチ" section of the
    README, scripts/measure_corpus_blankness.py, and the job 10491 threshold
    audit.

    The pre-2026-09-09 index (background unfiltered) is
    outputs/0002_20260808_build_faiss_index/default — pass exp_dir= to use it,
    or the "v1_predeblank" pipeline (default_pipelines(), off by default). The
    one-off A/B that promoted 0018 is
    outputs/gt_validations/gt_validation_results_2026-09-09_deblank_ab_job10495.csv.
    """
    exp_dir = Path(exp_dir)
    return PatchIndex.load(
        index_path=exp_dir / "index.faiss",
        manifest_path=exp_dir / "manifest.parquet",
        slide_meta_path=exp_dir / "slide_meta.parquet",
        features_dir=Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1"),
    )


def load_v1_predeblank_index() -> PatchIndex:
    """The pre-2026-09-09 uni_v1 index, before background removal (experiments/0002,
    18,368,337 patches incl. ~390k slide background). Kept for reference — the
    A/B that replaced it is documented in load_v1_index's docstring.
    """
    return load_v1_index(exp_dir="outputs/0002_20260808_build_faiss_index/default")


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


def load_v1_macenko_index(
    exp_dir: str = "outputs/0012_20260901_build_faiss_index_macenko_v1/default",
) -> PatchIndex:
    """The controlled Macenko comparison: uni_v1 (1024-dim), 224px native patches,
    identical geometry to load_v1_index's corpus — the ONLY thing changed is that
    the raw WSIs were Macenko stain-normalized (toward V1_MACENKO_STAIN_REFERENCE,
    torchstain 1.4.1 NumpyMacenkoNormalizer + default params) before patch
    extraction, on both the corpus and query side.

    This is the confound-free re-test of the "does stain normalization help
    retrieval" question. Every earlier negative result had a confound: the uni_v2
    Macenko corpus also changed encoder + patch size (and, per a wsi_preprocess
    git-history check, embedded stain-norm-failure patches as raw pixels with no
    record — a partially-contaminated corpus); the 2026-08-20 query-side torchstain
    diagnostic normalized queries against an *un-normalized* corpus (pipeline
    mismatch). See experiments/0010's config.yml. If this does not beat
    baseline_v1 on the 7-category GT comparison, the line is shelved like uni_v2.

    Built by: experiments/0010 (manifest; stain_norm_failures exclusion disabled —
    the wsi_preprocess sample audit found the Macenko failures are ~0.8% and
    background-only, so raw-passthrough of those patches is harmless, see 0010's
    config.yml) -> experiments/0012_20260901_build_faiss_index_macenko_v1 (index,
    byte-for-byte copy of 0002's hyperparameters). Pass exp_dir= if that
    experiment's number differs from the default above.
    """
    exp_dir = Path(exp_dir)
    return PatchIndex.load(
        index_path=exp_dir / "index.faiss",
        manifest_path=exp_dir / "manifest.parquet",
        slide_meta_path=exp_dir / "slide_meta.parquet",
        features_dir=Path("data/trident_processed_uni_v1_macenko/20x_224px_0px_overlap/features_uni_v1_macenko"),
    )


# The exact reference patch used to Macenko-normalize the raw WSIs behind the uni_v2
# corpus (confirmed by the user). Must be paired with torchstain's normalizer, not
# lib.stain_normalize's — see lib/torchstain_normalize.py's docstring: a
# self-consistency test (re-embedding a known v2 corpus patch and comparing to its
# own stored vector) only reached ~0.5-0.84 cosine similarity with the from-scratch
# implementation, vs. ~0.96-0.996 with torchstain (should be ~1.0 for a perfectly
# reproduced pipeline, and IS ~1.0 for uni_v1 via the same test).
V2_STAIN_REFERENCE = Path("data/baseline/63958_x38976_y7616.png")

# Same reference patch, used for the controlled uni_v1 Macenko corpus (load_v1_macenko_index).
# The corpus side (wsi_preprocess) uses torchstain 1.4.1's NumpyMacenkoNormalizer with
# default params on each 224px patch; this query side must match (numpy backend, defaults).
V1_MACENKO_STAIN_REFERENCE = Path("data/baseline/63958_x38976_y7616.png")


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


def _embed_v1_macenko_normalized(images) -> np.ndarray:
    """Query side of the controlled uni_v1 Macenko comparison: tile each query
    image into 224px crops, Macenko-normalize *each tile independently* toward
    V1_MACENKO_STAIN_REFERENCE, then uni_v1-encode.

    Granularity: this matches the corpus side, which normalizes each 224px patch
    on its own (wsi_preprocess: crop 224px patch -> torchstain NumpyMacenkoNormalizer
    with default params -> encoder; validated by the job-9559 self-consistency
    test, cos_self == 1.0). An earlier version of this function normalized the
    *whole* atlas figure once and then tiled — torchstain estimates the stain
    vectors and the concentration-rescaling factor from the whole input, so with
    a full figure (white margins, arrow ink, labels, mixed-magnification tissue)
    each query tile got a different colour transform than the matching corpus
    patch did. That whole-image run (job 9643) had v1_macenko losing to
    baseline_v1 on 5/7 categories; this per-tile version removes that last
    query/corpus pipeline mismatch.

    Degenerate tiles (near-blank crops where Macenko stain estimation fails) are
    passed through unnormalized, mirroring the corpus side, where the audit found
    the ~0.8% stain-norm failures were background-only and were embedded raw.
    """
    from lib.torchstain_normalize import normalize_to_reference

    def _norm_tile(tile: Image.Image) -> Image.Image:
        try:
            return normalize_to_reference(tile, V1_MACENKO_STAIN_REFERENCE)
        except Exception:
            return tile

    tiles = []
    for f in images:
        tiles.append(
            embed_image_tiles(
                str(f), tile_size=224, encoder_name="uni_v1", tile_transform=_norm_tile
            )
        )
    return np.concatenate(tiles, axis=0)


def default_pipelines(only: set[str] | None = None) -> dict[str, tuple[PatchIndex, callable]]:
    """Each pipeline is (PatchIndex, embed_fn(images) -> (n_tiles, dim) array).

    `only` restricts which pipelines are *constructed* (not just returned), so a
    focused A/B — e.g. only={"baseline_v1", "v1_predeblank"} — does not pay to
    load the v2 / macenko indexes and their manifests into memory.

    "baseline_v1" (plain tiling, no correction, uni_v1 corpus) is the current
    recommended default — the background-filtered index (experiments/0018), see
    load_v1_index and lib/query_embedding.py's module-level guidance.
    "v1_predeblank" is the pre-2026-09-09 index (experiments/0002, background
    unfiltered). Off by default — pass only={..., "v1_predeblank"} to re-run the
    background-removal A/B.
    "baseline_v2" torchstain-normalizes each query image toward
    V2_STAIN_REFERENCE before tiling — the fair comparison, matching how the
    v2 corpus itself was preprocessed (raw, unnormalized v2 queries scored
    much worse due to this mismatch alone, not because uni_v2 is a worse model
    — see lib/torchstain_normalize.py's docstring).
    "v1_macenko" is the confound-free stain-normalization re-test: same uni_v1
    encoder, same 224px geometry as baseline_v1, the ONLY change being Macenko
    normalization on both corpus and query side (see load_v1_macenko_index).
    Only appears once experiments/0010 + its build_faiss_index have been run.
    """
    _plain_tiling = lambda images: np.concatenate(
        [embed_image_tiles(str(f), tile_size=224) for f in images], axis=0
    )
    _want = (lambda name: only is None or name in only)

    pipelines: dict[str, tuple[PatchIndex, callable]] = {}
    if _want("baseline_v1"):
        pipelines["baseline_v1"] = (load_v1_index(), _plain_tiling)
    if only is not None and "v1_predeblank" in only:
        pipelines["v1_predeblank"] = (load_v1_predeblank_index(), _plain_tiling)
    if _want("baseline_v2"):
        try:
            pipelines["baseline_v2"] = (load_v2_index(), _embed_v2_normalized)
        except (FileNotFoundError, RuntimeError):
            # faiss.read_index raises RuntimeError (not FileNotFoundError) for a missing file.
            print("NOTE: uni_v2 index not found (run experiments/0004+0005 first) — skipping baseline_v2")
    if _want("v1_macenko"):
        try:
            pipelines["v1_macenko"] = (load_v1_macenko_index(), _embed_v1_macenko_normalized)
        except (FileNotFoundError, RuntimeError):
            print("NOTE: uni_v1 Macenko index not found (run experiments/0010 + its build_faiss_index) — skipping v1_macenko")
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


def run_comparison(
    pipelines: dict[str, tuple[PatchIndex, callable]] | None = None,
    nprobe: int = 64,
) -> pd.DataFrame:
    """Args:
        pipelines: see default_pipelines().
        nprobe: Number of IVF clusters to probe, forwarded to
            search_top_slides_multi. Default (64) matches the original
            fixed value this function always used. Raising it (up to
            nlist=4096, i.e. every cluster) is how to test whether IVF
            cluster coverage — rather than PQ quantization error — is
            responsible for a ground-truth slide ranking far worse than its
            true similarity would suggest (see load_v2_index's docstring for
            the diagnostic that first surfaced this on the uni_v2 corpus).
    """
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
        # query_images() drops the atlas's own "Normal liver ... for comparison"
        # control figures (is_normal_control=yes in nnl_liver_atlas_figures.csv).
        # For the 7 GT categories this only affects Hypertrophy (2 of its 9
        # atlas figures are normal-liver controls).
        all_imgs = query_images(cat_dir, exclude_normal_control=False)
        images = query_images(cat_dir)
        if not images:
            print(f"WARNING: no images found for {cat_dir_name!r}, skipping")
            continue
        if len(images) != len(all_imgs):
            print(f"  {cat_dir_name}: excluded {len(all_imgs) - len(images)} normal-control figure(s)")
        label = cat_dir_name.split(" - ")[0][:25]
        row = {"category": label}

        for name, (pi, embed_fn) in pipelines.items():
            corpus_slide_ids = set(pi.slide_meta.index.astype(str))
            gt_slides = set(gt.loc[gt.FINDING_TYPE == finding_type, "slide_id"]) & corpus_slide_ids
            row[f"{name}_n_gt"] = len(gt_slides)

            tiles = embed_fn(images)
            # Rank every slide in this pipeline's corpus — the literal 998 this
            # used to hard-code was a stale mid-Aug snapshot of the uni_v1
            # feature count (the built corpus is 1000; see git blame). With a
            # smaller cap the lowest-ranked slides never receive a rank, which
            # skews mean_rank and makes baseline_v1 vs a differently-sized
            # corpus (e.g. the Macenko rebuild) not strictly comparable.
            ranked = pi.search_top_slides_multi(
                tiles, k_candidates=8000, nprobe=nprobe, top_n_slides=len(corpus_slide_ids)
            )
            s = rank_stats(ranked, gt_slides)
            row[f"{name}_found"] = s["found"]
            row[f"{name}_best"] = s["best_rank"]
            row[f"{name}_mean"] = s["mean_rank"]

        results.append(row)
        print(row, flush=True)

    print(f"\nTOTAL: {time.time() - t_all:.1f}s")
    return pd.DataFrame(results)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--pipelines",
        default=None,
        help="comma-separated subset of default_pipelines() to run "
        "(e.g. 'baseline_v1,v1_predeblank' for the background-removal A/B without "
        "the slow torchstain v2/macenko embeds). Default: all available.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/gt_validation_results.csv"),
        help="output CSV (default: %(default)s). Back up the existing file first "
        "if it holds a baseline you want to keep — this overwrites it.",
    )
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="Run a single 'baseline_v1'-style pipeline (plain tiling, no "
        "correction) against this FAISS index run_dir instead of the default "
        "pipeline set. Use for an A/B against an alternative corpus index "
        "(e.g. experiments/0025's whitened index).",
    )
    args = ap.parse_args()

    if args.index_dir:
        _plain = lambda images: np.concatenate(
            [embed_image_tiles(str(f), tile_size=224) for f in images], axis=0
        )
        pipelines = {"index": (load_v1_index(str(args.index_dir)), _plain)}
    elif args.pipelines:
        want = [p.strip() for p in args.pipelines.split(",") if p.strip()]
        pipelines = default_pipelines(only=set(want))
        missing = [p for p in want if p not in pipelines]
        if missing:
            raise SystemExit(
                f"requested pipeline(s) not available: {missing} "
                f"(have: {sorted(pipelines)})"
            )
        pipelines = {p: pipelines[p] for p in want}
    else:
        pipelines = default_pipelines()

    df = run_comparison(pipelines, nprobe=args.nprobe)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
