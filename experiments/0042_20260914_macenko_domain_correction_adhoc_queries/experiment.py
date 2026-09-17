"""experiments/0042: query_001〜003(NNLアトラス外の単発参照タイル)で
Macenko+線形補正併用の検索性能を目視確認する。

experiments/0041はNNLアトラス25所見(図版まるごと)を対象にMacenko単独と
Macenko+線形補正併用を比較した。この実験は同じ考え方を、atlasのCATEGORIES/
所見ラベルに属さない単発の参照画像(data/query/query_001〜003.png、
experiments/0003以来使われてきたアドホックなクエリタイル)に適用し、目視で
検索性能を確認する。所見ラベルが無いため lib.finding_routing の所見別
ルーティングは経由せず、3腕(baseline/macenko/macenko_domain_corrected)を
常に並列実行する。

domain_shift は experiments/0041 が計算したmacenko空間のものをそのまま再利用
する(再計算しない)。alpha=0.25も experiments/0040 のスイープ結果を踏襲。

出力(outputs/0042_.../default/):
  <query_slug>/query__{baseline,macenko,macenko_domain_corrected}/...  検索3腕
  summary.csv   3クエリ×3腕のギャラリー数比較
"""
from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=str, default="config.yml")
    return ap.parse_args()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.ovr_scoring import run_query_arm
    from lib.torchstain_normalize import normalize_to_reference
    from lib.search import PatchIndex
    from lib import finding_routing

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    alpha = float(config["alpha"])

    baseline_index_dir = project_root / config["baseline_index_dir"]
    baseline_features_dir = project_root / config["baseline_features_dir"]
    baseline_index = PatchIndex.load(
        index_path=baseline_index_dir / "index.faiss",
        manifest_path=baseline_index_dir / "manifest.parquet",
        slide_meta_path=baseline_index_dir / "slide_meta.parquet",
        features_dir=baseline_features_dir,
    )

    macenko_index_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    macenko_features_dir = project_root / finding_routing.MACENKO_FEATURES_DIR
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE
    macenko_index = PatchIndex.load(
        index_path=macenko_index_dir / "index.faiss",
        manifest_path=macenko_index_dir / "manifest.parquet",
        slide_meta_path=macenko_index_dir / "slide_meta.parquet",
        features_dir=macenko_features_dir,
    )

    domain_shift = np.load(project_root / config["domain_shift_macenko_path"])
    logger.info(f"loaded domain_shift_macenko (norm={np.linalg.norm(domain_shift):.4f}), alpha={alpha}")

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    params = {
        "raw_slide_dir": project_root / config["raw_slide_dir"],
        "thumbnails_dir": project_root / config["baseline_thumbnails_dir"],
        "nprobe": int(config.get("nprobe", 32)),
        "rerank_pool": int(config.get("rerank_pool", 200)),
        "max_tiles_reranked": config.get("max_tiles_reranked", None),
        "k_candidates": int(config.get("k_candidates", 8000)),
        "top_n_slides": int(config.get("top_n_slides", 20)),
        "top_n_slides_to_plot": int(config.get("top_n_slides_to_plot", 3)),
    }

    rows = []
    for query_rel in config["query_images"]:
        query_path = project_root / query_rel
        query_slug = _slug(Path(query_rel).stem)
        logger.info(f"===== {query_rel} =====")
        query_dir = run_dir / query_slug
        query_dir.mkdir(exist_ok=True)

        baseline_vecs = embed_image_tiles(str(query_path), tile_size=tile_size)
        macenko_vecs = embed_image_tiles(
            str(query_path), tile_size=tile_size, tile_transform=_macenko_tile_transform
        )
        corrected_vecs = _l2_normalize(macenko_vecs - alpha * domain_shift)

        arms = {
            "baseline": (baseline_index, baseline_vecs),
            "macenko": (macenko_index, macenko_vecs),
            "macenko_domain_corrected": (macenko_index, corrected_vecs),
        }
        arm_results = {}
        for arm_label, (patch_index, vecs) in arms.items():
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=query_dir,
                params=params, logger=logger,
            )
        logger.info(f"[{query_rel}] galleries: baseline={arm_results['baseline']['n_galleries']}, "
                    f"macenko={arm_results['macenko']['n_galleries']}, "
                    f"macenko_domain_corrected={arm_results['macenko_domain_corrected']['n_galleries']}")

        rows.append({
            "query_image": query_rel,
            "n_galleries_baseline": arm_results["baseline"]["n_galleries"],
            "n_galleries_macenko": arm_results["macenko"]["n_galleries"],
            "n_galleries_macenko_domain_corrected": arm_results["macenko_domain_corrected"]["n_galleries"],
            "top_slides_baseline": arm_results["baseline"]["top_slide_ids"][:5],
            "top_slides_macenko_domain_corrected": arm_results["macenko_domain_corrected"]["top_slide_ids"][:5],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
