"""Measure how similar slide-background patches look to each finding's query,
to decide whether the background contamination seen in experiments/0015 is a
property of the embedding or a side effect of the finding having few genuine
matches in the corpus.

Motivation: job 9962's "Degeneration, granular, eosinophilic" deliver run came
back 93% background — 48 of 69 crops rejected by the blank filter, and 16 of
the surviving 21 were near-blank anyway, at similarity 0.908..0.927, the same
band as its tissue hits. The corpus itself is only ~2% blank (measured by the
random controls in job 9975), and the seed slides are not tissue-sparse (0% of
34 sampled patches were blank), so the search is enriching background ~50x
rather than merely reflecting a dirty corpus. Two explanations survive:

  (a) UNI v1 places this finding's morphology close to the background cluster,
      i.e. an embedding problem, which would justify testing uni_v2 for it.
  (b) The corpus holds few patches genuinely similar to this finding, so its
      tissue hits top out at 0.927 and background — which sits at a moderate
      similarity to everything — floats up by default. That is a property of
      the data, and re-embedding would not fix it.

These make opposite predictions, which is what this measures. Sample patches
from the corpus at random, crop them, split them into background and tissue,
then score both groups against each finding's seed query (the same query
vectors experiments/0015 builds: patch features read straight from the seed
slides' h5, no encoder). If background scores about the same against every
finding, (b) holds and granular eosinophilic is simply short of real matches.
If background scores distinctly higher against granular eosinophilic than
against the findings that work, (a) holds.

Blankness is recorded as raw measurements (mean intensity, standard deviation,
saturated-pixel fraction) rather than a single verdict, because the current
lib.query_embedding._is_blank_tile threshold demonstrably passes patches that
are background to the eye — the 16 near-blank survivors above. Deciding a
better threshold is a second use of this output.

Output:
  outputs/gt_validations/blank_patch_similarity_patches.csv  per sampled patch
  outputs/gt_validations/blank_patch_similarity_summary.csv  per finding
"""
from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CORPUS_INDEX_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")
GT_CSV = Path("data/processed_csv/single_finding_liver.csv")
OUT_DIR = Path("outputs/gt_validations")

SEED = 42
# Patches sampled from the corpus and cropped. The corpus is ~2% blank, so this
# yields roughly 40 background patches — enough to compare distributions, and
# cropping is the slow part (openslide reads dominate).
N_SAMPLE = 2000
# Query construction, matching experiments/0015 so the numbers are comparable.
N_QUERY_PATCHES_PER_SEED_SLIDE = 120
MAX_SEED_SLIDES = 6

# The three findings 0015 has run through deliver, plus hypertrophy for
# reference. Their seed slides are every corpus slide with that confirmed
# single finding, exactly as experiments/0015 picks them in deliver mode.
FINDINGS = (
    "Deposit, glycogen",              # works (91% blind control, 3% blank)
    "Ground glass appearance",        # works (91% blind control, 0% blank)
    "Degeneration, granular, eosinophilic",  # fails (93% background)
    "Hypertrophy",                    # fails differently (relative-reference finding)
)

# Saturation above this counts a pixel as stained tissue. Provisional — the
# per-patch CSV carries the raw fraction so the cut can be revisited.
SAT_THRESHOLD = 0.10


def blankness_metrics(img: Image.Image) -> dict:
    """Raw measurements, no verdict. `sat_frac` is the fraction of pixels with
    any real colour — background is bright *and* unsaturated, whereas pale
    tissue is bright but still stained, which is what mean+std alone misses."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    return {
        "mean_intensity": float(rgb.mean()),
        "std_intensity": float(rgb.std()),
        "sat_frac": float((hsv[..., 1] > SAT_THRESHOLD).mean()),
    }


def load_features_for(rows: pd.DataFrame) -> np.ndarray:
    """Read each sampled patch's stored feature vector, one h5 open per slide.
    manifest.local_idx is the row of that slide's features array."""
    vecs: list[np.ndarray] = [None] * len(rows)  # type: ignore[list-item]
    positions = {i: p for p, i in enumerate(rows.index)}
    for sid, g in rows.groupby("slide_id", sort=False):
        with h5py.File(FEATURES_DIR / f"{sid}.h5", "r") as f:
            idx = np.sort(g["local_idx"].to_numpy().astype(int))
            feats = f["features"][idx].astype(np.float32)
        order = {li: k for k, li in enumerate(idx)}
        for row_idx, li in zip(g.index, g["local_idx"].astype(int)):
            vecs[positions[row_idx]] = feats[order[li]]
    out = np.vstack(vecs)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return out / norms


def seed_query_vecs(slide_ids: list[str], rng: np.random.Generator) -> np.ndarray:
    """experiments/0015's load_slide_query_vecs, duplicated so this diagnostic
    stays independent of the experiment directory (a frozen record)."""
    parts = []
    for sid in slide_ids:
        with h5py.File(FEATURES_DIR / f"{sid}.h5", "r") as f:
            n = f["features"].shape[0]
            if n <= N_QUERY_PATCHES_PER_SEED_SLIDE:
                v = f["features"][:].astype(np.float32)
            else:
                sel = np.sort(rng.choice(n, size=N_QUERY_PATCHES_PER_SEED_SLIDE, replace=False))
                v = f["features"][sel].astype(np.float32)
        parts.append(v)
    vecs = np.concatenate(parts, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main() -> None:
    from lib.query_embedding import _is_blank_tile
    from lib.raw_patch import crop_patch

    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(CORPUS_INDEX_DIR / "manifest.parquet")
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    slide_meta = pd.read_parquet(CORPUS_INDEX_DIR / "slide_meta.parquet").set_index("slide_id")
    slide_meta.index = slide_meta.index.astype(str)
    corpus_slides = set(slide_meta.index)
    print(f"corpus: {len(manifest)} patches, {len(corpus_slides)} slides")

    sampled = manifest.iloc[np.sort(rng.choice(len(manifest), size=N_SAMPLE, replace=False))].copy()

    print(f"cropping {len(sampled)} sampled patches ...")
    metrics = []
    for row in sampled.itertuples():
        psl = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, RAW_SLIDE_DIR, psl)
        m = blankness_metrics(patch)
        m["is_blank_current"] = bool(_is_blank_tile(patch))
        metrics.append(m)
    sampled = pd.concat([sampled, pd.DataFrame(metrics, index=sampled.index)], axis=1)

    # Background by the measurement this diagnostic trusts: bright and unstained.
    sampled["is_background"] = (sampled["sat_frac"] < 0.10) & (sampled["mean_intensity"] > 215)
    n_bg = int(sampled["is_background"].sum())
    n_cur = int(sampled["is_blank_current"].sum())
    print(
        f"background: {n_bg}/{len(sampled)} ({100*n_bg/len(sampled):.1f}%) by sat_frac; "
        f"{n_cur} ({100*n_cur/len(sampled):.1f}%) by the current _is_blank_tile"
    )

    print("loading features for sampled patches ...")
    feats = load_features_for(sampled)

    gt = pd.read_csv(GT_CSV)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)

    per_patch = sampled.reset_index(drop=True)
    rows = []
    for finding in FINDINGS:
        seed_slides = sorted(set(gt.loc[gt["FINDING_TYPE"] == finding, "slide_id"]) & corpus_slides)
        if not seed_slides:
            print(f"  {finding!r}: no corpus slides, skipped")
            continue
        # Deterministic first-N rather than 0015's shuffled pick. For the three
        # findings with <= MAX_SEED_SLIDES corpus slides (glycogen 6, ground
        # glass 4, granular eosinophilic 5) this is the identical seed set;
        # Hypertrophy has 25, so its query here differs from job 9932's and is
        # only a rough reference point.
        seed_slides = seed_slides[:MAX_SEED_SLIDES]
        q = seed_query_vecs(seed_slides, np.random.default_rng(SEED))
        # Each patch's best similarity over the query set — what the search ranks by.
        sims = (feats @ q.T).max(axis=1)
        col = f"maxsim__{finding}"
        per_patch[col] = sims

        bg, tis = sims[per_patch["is_background"].to_numpy()], sims[~per_patch["is_background"].to_numpy()]
        rows.append({
            "finding": finding,
            "n_seed_slides": len(seed_slides),
            "n_query_vecs": int(q.shape[0]),
            "n_background": len(bg),
            "n_tissue": len(tis),
            "bg_median": round(float(np.median(bg)), 4) if len(bg) else None,
            "bg_p95": round(float(np.percentile(bg, 95)), 4) if len(bg) else None,
            "bg_max": round(float(bg.max()), 4) if len(bg) else None,
            "tissue_median": round(float(np.median(tis)), 4),
            "tissue_p95": round(float(np.percentile(tis, 95)), 4),
            "tissue_max": round(float(tis.max()), 4),
            # >0 means background outscores the 95th percentile of tissue, i.e.
            # background wins slots in a top-k the tissue cannot defend.
            "bg_p95_minus_tissue_p95": round(float(np.percentile(bg, 95) - np.percentile(tis, 95)), 4)
            if len(bg) else None,
        })
        print(
            f"  {finding!r}: bg median {rows[-1]['bg_median']} p95 {rows[-1]['bg_p95']} | "
            f"tissue median {rows[-1]['tissue_median']} p95 {rows[-1]['tissue_p95']}"
        )

    per_patch.to_csv(OUT_DIR / "blank_patch_similarity_patches.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "blank_patch_similarity_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nWrote {OUT_DIR}/blank_patch_similarity_{{patches,summary}}.csv")


if __name__ == "__main__":
    main()
