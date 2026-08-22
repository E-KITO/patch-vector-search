"""One-time build of lib.mpp_estimation's MPP auto-scale reference centroids
(2026-08-22, introducing --auto_scale despite the prior negative GT result —
see README's "MPP補正の導入" decision). Persists the output to
data/scale_centroids.npz so every --auto_scale run doesn't have to rebuild it
(openslide reads + ~150 UNI forward passes) from scratch.

Only re-run this if the corpus (raw WSIs / slide_meta) changes.
"""
from __future__ import annotations

import logging
from pathlib import Path

from lib.mpp_estimation import build_scale_reference_centroids, save_scale_centroids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WSI_DIR = PROJECT_ROOT / "data/moo_collected_tggate_wsi/raw_wsi"
SLIDE_META_PATH = PROJECT_ROOT / "outputs/0002_20260808_build_faiss_index/default/slide_meta.parquet"
OUT_PATH = PROJECT_ROOT / "data/scale_centroids.npz"


def main() -> None:
    logger.info(f"raw_wsi_dir={RAW_WSI_DIR}")
    logger.info(f"slide_meta_path={SLIDE_META_PATH}")
    centroids = build_scale_reference_centroids(RAW_WSI_DIR, SLIDE_META_PATH)
    save_scale_centroids(centroids, OUT_PATH)
    logger.info(f"saved scales={sorted(centroids.keys())} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
