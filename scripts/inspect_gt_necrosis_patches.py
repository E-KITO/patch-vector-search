"""One-off: render a random sample of real-resolution patches from the 4
ground-truth "Single cell necrosis" slides (41484, 58720, 58787, 59195;
see data/processed_csv/single_finding_liver.csv, all graded minimal/slight)
that the necrosis-crop search query (see
experiments/0009_20260820_..._tile_score_heatmap's
query__necrosis_imgi11_crop_x150_y400 run) failed to retrieve.

Purpose: check by eye whether these labels look like the confluent/zonal
necrosis depicted in the NNL atlas query image (data/query/
necrosis_imgi11_crop_x150_y400.jpg), or whether -- as the "Single cell
necrosis" name and minimal/slight grade suggest -- they instead look like
scattered individual necrotic hepatocytes among otherwise normal-looking
tissue. If the latter, the poor retrieval may reflect a real content
mismatch between the query and this GT category's actual appearance,
rather than a search/embedding deficiency.

No GPU/UNI/torch/FAISS needed -- only manifest.parquet + slide_meta.parquet
(patch coordinates/sizes) and lib.raw_patch.crop_patch (openslide). Not a
Slurm job; run directly from the project root.

Usage:
    .venv/bin/python3 scripts/inspect_gt_necrosis_patches.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.raw_patch import crop_patch

MANIFEST_PATH = Path("outputs/0002_20260808_build_faiss_index/default/manifest.parquet")
SLIDE_META_PATH = Path("outputs/0002_20260808_build_faiss_index/default/slide_meta.parquet")
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")
OUT_DIR = Path("outputs/gt_necrosis_patch_gallery")

# Confirmed GT "Single cell necrosis" slides referenced in this project's
# tile-selection-bias investigation (see README.md, 2026-08-20).
GT_NECROSIS_SLIDES = ["41484", "58720", "58787", "59195"]
N_PATCHES_PER_SLIDE = 16
N_COLS = 4
SEED = 42


def main() -> None:
    manifest = pd.read_parquet(MANIFEST_PATH)
    slide_meta = pd.read_parquet(SLIDE_META_PATH).set_index("slide_id")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for slide_id in GT_NECROSIS_SLIDES:
        rows = manifest[manifest["slide_id"] == slide_id]
        if rows.empty:
            print(f"WARNING: no manifest rows found for slide_id={slide_id}, skipping")
            continue
        n = min(N_PATCHES_PER_SLIDE, len(rows))
        sample = rows.iloc[rng.choice(len(rows), size=n, replace=False)]
        patch_size_level0 = int(slide_meta.loc[slide_id, "patch_size_level0"])

        n_rows = -(-n // N_COLS)
        fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 2.2, n_rows * 2.4), squeeze=False)
        axes = axes.flatten()
        for ax, row in zip(axes, sample.itertuples()):
            image = crop_patch(slide_id, row.coord_x, row.coord_y, RAW_SLIDE_DIR, patch_size_level0)
            ax.imshow(image)
            ax.set_title(f"x={row.coord_x},y={row.coord_y}", fontsize=8)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"slide {slide_id} — {n} random patches (GT: Single cell necrosis)")
        fig.tight_layout()
        out_path = OUT_DIR / f"{slide_id}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path} ({n} patches out of {len(rows)} total tiles on this slide)")


if __name__ == "__main__":
    main()
