"""Build a global manifest over all per-slide UNI feature h5 files.

Scans data/trident_processed/20x_224px_0px_overlap/features_uni_v1/*.h5 once
and produces the artifacts lib/faiss_index.py and lib/search.py build on:

- manifest.parquet: one row per patch [slide_id, local_idx, coord_x, coord_y, global_idx]
- slide_meta.parquet: one row per slide [slide_id, total_patches, total_patches_raw,
  n_stain_norm_failed, level0_width, level0_height, patch_size_level0]
- training_sample.npy: a random subset of raw (unnormalized) float32 feature
  vectors, for fitting the FAISS index in lib.faiss_index.build_faiss_index

`local_idx` is always the row index into the slide's h5 `features`/`coords`
datasets. When stain_norm_failures_path is given (the Macenko corpus — see
experiments/0010), the patches that could not be Macenko-normalized are
excluded from the manifest, so `local_idx` is no longer contiguous within a
slide: it lists only the kept rows. Both consumers (lib.faiss_index,
lib.search) index the h5 by `local_idx` rather than assuming a 1:1
manifest/h5 mapping.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_stain_norm_exclusions(
    stain_norm_failures_path: str | Path,
) -> dict[str, np.ndarray]:
    """Read a stain_norm_failures.json (see scripts/audit_stain_norm_failures.py
    in the wsi_preprocess pipeline) and return {slide_id: sorted unique row
    indices to exclude}.

    Raises if the recorded overall failure rate exceeds the recorded
    threshold — a high rate means the corpus build itself is suspect and
    should be investigated before silently dropping a large fraction of it.
    """
    with open(stain_norm_failures_path) as fh:
        payload = json.load(fh)

    fail_rate = payload.get("fail_rate")
    threshold = payload.get("fail_threshold")
    if fail_rate is not None and threshold is not None and fail_rate > threshold:
        raise ValueError(
            f"stain-norm failure rate {fail_rate:.4%} exceeds threshold "
            f"{threshold:.4%} ({payload.get('n_failed')}/"
            f"{payload.get('n_patches_total')} patches) — investigate the "
            "corpus build before excluding failures from the manifest"
        )

    by_slide: dict[str, list[int]] = {}
    for rec in payload.get("failures", []):
        by_slide.setdefault(str(rec["slide_id"]), []).append(int(rec["row_index"]))
    return {
        sid: np.array(sorted(set(rows)), dtype=np.int64)
        for sid, rows in by_slide.items()
    }


def build_patch_manifest(
    features_dir: str | Path,
    manifest_path: str | Path,
    slide_meta_path: str | Path,
    training_sample_path: str | Path,
    train_sample_size: int = 500_000,
    seed: int = 42,
    stain_norm_failures_path: str | Path | None = None,
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
        stain_norm_failures_path: Optional stain_norm_failures.json from the
            Macenko corpus audit pass (scripts/audit_stain_norm_failures.py in
            the wsi_preprocess pipeline). When given, the listed patches — the
            ones the audit found could not be Macenko-normalized and which the
            corpus therefore embedded as raw, un-normalized pixels — are
            excluded from the manifest, the slide_meta patch counts and the
            training sample, so the resulting corpus is 100% stain-normalized.
            Raises if the recorded failure rate exceeds its recorded threshold.
    """
    features_dir = Path(features_dir)
    h5_paths = sorted(features_dir.glob("*.h5"))
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {features_dir}")

    exclusions: dict[str, np.ndarray] = {}
    if stain_norm_failures_path is not None:
        exclusions = _load_stain_norm_exclusions(stain_norm_failures_path)
        logger.info(
            "stain-norm exclusions: %d patches across %d slides",
            sum(len(v) for v in exclusions.values()),
            len(exclusions),
        )

    rng = np.random.default_rng(seed)
    h5_paths = [h5_paths[i] for i in rng.permutation(len(h5_paths))]

    manifest_chunks: list[pd.DataFrame] = []
    slide_meta_rows: list[dict] = []
    training_chunks: list[np.ndarray] = []
    training_total = 0
    global_idx = 0
    n_excluded_total = 0

    for i, h5_path in enumerate(h5_paths):
        slide_id = h5_path.stem
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
            attrs = f["coords"].attrs
            n_patches_raw = coords.shape[0]

            # local_idx = true h5 row index of every patch we keep. With a
            # stain-norm failure list this is a strict subset of range(N).
            keep = np.ones(n_patches_raw, dtype=bool)
            drop_rows = exclusions.get(slide_id)
            if drop_rows is not None and len(drop_rows):
                in_range = drop_rows[(drop_rows >= 0) & (drop_rows < n_patches_raw)]
                if len(in_range) != len(drop_rows):
                    raise ValueError(
                        f"{slide_id}: stain-norm failure list has row indices "
                        f"outside [0, {n_patches_raw}) — failure json and this "
                        "h5 disagree on the slide's patch count"
                    )
                keep[in_range] = False
            local_idx = np.nonzero(keep)[0].astype(np.int32)
            n_kept = local_idx.shape[0]
            n_excluded_total += n_patches_raw - n_kept

            need_training = training_total < train_sample_size
            if need_training:
                features = f["features"][:]
                if n_kept != n_patches_raw:
                    features = features[keep]
                training_chunks.append(features)
                training_total += features.shape[0]

            slide_meta_rows.append(
                {
                    "slide_id": slide_id,
                    "total_patches": n_kept,
                    "total_patches_raw": n_patches_raw,
                    "n_stain_norm_failed": n_patches_raw - n_kept,
                    "level0_width": int(attrs["level0_width"]),
                    "level0_height": int(attrs["level0_height"]),
                    "patch_size_level0": float(attrs["patch_size_level0"]),
                }
            )

            coords_kept = coords[local_idx]
            manifest_chunks.append(
                pd.DataFrame(
                    {
                        "slide_id": slide_id,
                        "local_idx": local_idx,
                        "coord_x": coords_kept[:, 0].astype(np.int32),
                        "coord_y": coords_kept[:, 1].astype(np.int32),
                        "global_idx": np.arange(global_idx, global_idx + n_kept, dtype=np.int64),
                    }
                )
            )
            global_idx += n_kept

        logger.info(
            "[%d/%d] %s: %d patches%s%s",
            i + 1,
            len(h5_paths),
            slide_id,
            n_kept,
            f" (-{n_patches_raw - n_kept} stain-norm failures)" if n_kept != n_patches_raw else "",
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
    if stain_norm_failures_path is not None:
        logger.info("stain-norm failures excluded: %d patches", n_excluded_total)
    logger.info("training sample: %s", training_sample.shape)
