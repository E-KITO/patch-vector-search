"""Build a global manifest over all per-slide UNI feature h5 files.

Scans data/trident_processed/20x_224px_0px_overlap/features_uni_v1/*.h5 once
and produces the artifacts lib/faiss_index.py and lib/search.py build on:

- manifest.parquet: one row per patch [slide_id, local_idx, coord_x, coord_y, global_idx]
- slide_meta.parquet: one row per slide [slide_id, total_patches, level0_width,
  level0_height, patch_size_level0]
- training_sample.npy: a random subset of raw (unnormalized) float32 feature
  vectors, for fitting the FAISS index in lib.faiss_index.build_faiss_index
"""
from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_patch_manifest(
    features_dir: str | Path,
    manifest_path: str | Path,
    slide_meta_path: str | Path,
    training_sample_path: str | Path,
    train_sample_size: int = 500_000,
    seed: int = 42,
) -> None:
    """Scan every {slide_id}.h5 under features_dir once and write the manifest artifacts.

    Reads only `coords` (+ attrs) for slides not selected into the training
    sample; for slides selected into the training sample, `features` is also
    read from the same open file handle so the corpus is only traversed
    once, not twice. Slide order is randomized before scanning so the
    training sample isn't biased toward whichever slides sort first.

    Args:
        features_dir: Directory containing {slide_id}.h5 files, each with
            `features` (N,1024) float32 and `coords` (N,2) int64 datasets.
        manifest_path: Output path for the per-patch manifest (parquet).
        slide_meta_path: Output path for the per-slide metadata (parquet).
        training_sample_path: Output path for the sampled training vectors (.npy).
        train_sample_size: Target number of vectors to collect for FAISS index training.
        seed: RNG seed controlling slide scan order and the final subsample.
    """
    features_dir = Path(features_dir)
    h5_paths = sorted(features_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {features_dir}")

    rng = np.random.default_rng(seed)
    h5_paths = [h5_paths[i] for i in rng.permutation(len(h5_paths))]

    manifest_chunks: list[pd.DataFrame] = []
    slide_meta_rows: list[dict] = []
    training_chunks: list[np.ndarray] = []
    training_total = 0
    global_idx = 0

    for i, h5_path in enumerate(h5_paths):
        slide_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            attrs = f["coords"].attrs
            n_patches = coords.shape[0]

            need_training = training_total < train_sample_size
            if need_training:
                features = f["features"][:]
                training_chunks.append(features)
                training_total += features.shape[0]

            slide_meta_rows.append(
                {
                    "slide_id": slide_id,
                    "total_patches": n_patches,
                    "level0_width": int(attrs["level0_width"]),
                    "level0_height": int(attrs["level0_height"]),
                    "patch_size_level0": float(attrs["patch_size_level0"]),
                }
            )

            manifest_chunks.append(
                pd.DataFrame(
                    {
                        "slide_id": slide_id,
                        "local_idx": np.arange(n_patches, dtype=np.int32),
                        "coord_x": coords[:, 0].astype(np.int32),
                        "coord_y": coords[:, 1].astype(np.int32),
                        "global_idx": np.arange(global_idx, global_idx + n_patches, dtype=np.int64),
                    }
                )
            )
            global_idx += n_patches

        logger.info(
            "[%d/%d] %s: %d patches%s",
            i + 1,
            len(h5_paths),
            slide_id,
            n_patches,
            " (sampled for training)" if need_training else "",
        )

    manifest = pd.concat(manifest_chunks, ignore_index=True)
    manifest.to_parquet(manifest_path, index=False)

    slide_meta = pd.DataFrame(slide_meta_rows)
    slide_meta.to_parquet(slide_meta_path, index=False)

    training_sample = np.concatenate(training_chunks, axis=0)
    if training_sample.shape[0] > train_sample_size:
        idx = rng.choice(training_sample.shape[0], size=train_sample_size, replace=False)
        training_sample = training_sample[idx]
    np.save(training_sample_path, training_sample)

    logger.info("manifest: %d patches across %d slides", len(manifest), len(slide_meta))
    logger.info("training sample: %s", training_sample.shape)
