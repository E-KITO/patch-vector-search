"""Find candidate "most average" patches for use as a stain-normalization
reference/target patch.

Approach: sample a subset of slides' UNI feature vectors, L2-normalize them,
compute the mean direction (centroid) over the sample, then rank patches by
cosine similarity to that centroid. The closest patches are the ones whose
UNI embedding is most "typical" of the corpus. Prior analysis in this repo
(wsi-ad-batch-effect-finding) found that slide/staining differences are a
dominant axis of variation in UNI feature space for this dataset, so
proximity to the embedding centroid is a reasonable proxy for "typical
staining" — not a rigorous color-space definition, but a cheap way to avoid
outlier/atypical patches (background scraps, artifacts, unusually dark or
pale staining) without hand-picking one.

Candidates are capped per-slide so the shortlist isn't dominated by one WSI,
then cropped from data/raw_wsi at full resolution via lib.raw_patch.

This is a one-time setup utility, already run — its output
(outputs/average_patch_candidates/, esp. candidate 63958_x38976_y7616.png,
the top-ranked "most typical" patch) is reused as the standard
--stain_reference / stain_reference= target across this project rather than
being re-picked per query. Only re-run this if the corpus itself changes.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from lib.raw_patch import crop_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/trident_processed/20x_224px_0px_overlap/features_uni_v1")
RAW_SLIDE_DIR = Path("data/raw_wsi")
SLIDE_META_PATH = Path("outputs/0001_20260808_build_patch_manifest/default/slide_meta.parquet")


def sample_slide_features(n_slides: int, seed: int) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Read *all* patches from a random subset of slides (contiguous h5 reads).

    Returns (features (N,1024) float32, slide_id per row, coords (N,2) int64).
    """
    h5_paths = sorted(FEATURES_DIR.glob("*.h5"))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(h5_paths), size=min(n_slides, len(h5_paths)), replace=False)

    feats, slide_ids, coords = [], [], []
    for i, idx in enumerate(chosen):
        h5_path = h5_paths[idx]
        slide_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            feats.append(f["features"][:])
            coords.append(f["coords"][:])
        slide_ids.extend([slide_id] * feats[-1].shape[0])
        logger.info("[%d/%d] %s: %d patches", i + 1, len(chosen), slide_id, feats[-1].shape[0])

    return np.concatenate(feats, axis=0), slide_ids, np.concatenate(coords, axis=0)


def pick_candidates(
    X: np.ndarray, slide_ids: list[str], coords: np.ndarray, top_k: int, max_per_slide: int
) -> pd.DataFrame:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    centroid = Xn.mean(axis=0)
    centroid /= np.linalg.norm(centroid)
    sims = Xn @ centroid

    order = np.argsort(-sims)
    per_slide_count: dict[str, int] = {}
    rows = []
    for rank_pos in order:
        sid = slide_ids[rank_pos]
        if per_slide_count.get(sid, 0) >= max_per_slide:
            continue
        per_slide_count[sid] = per_slide_count.get(sid, 0) + 1
        rows.append(
            {
                "slide_id": sid,
                "coord_x": int(coords[rank_pos, 0]),
                "coord_y": int(coords[rank_pos, 1]),
                "cosine_sim_to_centroid": float(sims[rank_pos]),
            }
        )
        if len(rows) >= top_k:
            break
    return pd.DataFrame(rows)


def crop_candidates(candidates: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    slide_meta = pd.read_parquet(SLIDE_META_PATH).set_index("slide_id")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths, mean_rgbs = [], []
    for row in candidates.itertuples():
        patch_size_level0 = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        img = crop_patch(
            row.slide_id, row.coord_x, row.coord_y, RAW_SLIDE_DIR, patch_size_level0
        )
        out_path = out_dir / f"{row.slide_id}_x{row.coord_x}_y{row.coord_y}.png"
        img.save(out_path)
        paths.append(str(out_path))
        mean_rgbs.append(np.array(img).reshape(-1, 3).mean(axis=0).tolist())

    candidates = candidates.copy()
    candidates["image_path"] = paths
    candidates["mean_rgb"] = mean_rgbs
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_slides", type=int, default=40, help="Slides to pool for the centroid estimate")
    parser.add_argument("--top_k", type=int, default=12, help="Number of candidate patches to output")
    parser.add_argument("--max_per_slide", type=int, default=2, help="Cap candidates from any one slide")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/average_patch_candidates"))
    args = parser.parse_args()

    logger.info("Sampling features from %d slides...", args.n_slides)
    X, slide_ids, coords = sample_slide_features(args.n_slides, args.seed)
    logger.info("Pooled %d patch vectors from %d slides", X.shape[0], len(set(slide_ids)))

    candidates = pick_candidates(X, slide_ids, coords, args.top_k, args.max_per_slide)
    logger.info("Selected %d candidates (cosine sim range %.4f - %.4f)",
                len(candidates), candidates["cosine_sim_to_centroid"].min(),
                candidates["cosine_sim_to_centroid"].max())

    candidates = crop_candidates(candidates, args.out_dir)
    candidates.to_json(args.out_dir / "candidates.json", orient="records", indent=2)
    print(candidates.drop(columns="mean_rgb").to_string(index=False))
    logger.info("Wrote %d candidate PNGs + candidates.json to %s", len(candidates), args.out_dir)


if __name__ == "__main__":
    main()
