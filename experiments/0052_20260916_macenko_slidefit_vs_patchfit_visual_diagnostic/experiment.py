"""experiments/0052: macenko_patchfit(既存, experiments/0033) vs macenko_slidefit
(新規, experiments/0050)を、GT対応7所見全部について目視ギャラリーで比較する。

experiments/0051のGT best_rank比較は所見によって改善/悪化/found減少が混在した
(README本文・summary.json参照)。found/best_rankの数値だけでは「返ってきた
パッチが本当にその所見らしく見えるか」が分からない(README「experiments/0037」
「0041」「0045」の教訓)ため、この実験は両索引のギャラリー(サムネイル+パッチ
コンタクトシート)を並べて目視診断できる形で出力するところまでを行う——採否
判断はこの出力を見てから別途行う。

クエリ側embed関数は両アームで完全に同一(experiments/0051と同じ
_macenko_tile_embed) — コーパス側の染色ベクトル推定方法だけが違うよう統制
している。alphaスイープは無い(経路の違いそのものを見るのが目的)。

出力(outputs/0052_.../default/):
  <finding_slug>/query__{macenko_patchfit,macenko_slidefit}/...  各所見・各アーム
    のtop_slides.csv・thumbnail_plots・patch_gallery
  summary.csv   所見ごとのn_galleries比較とtop_slide_id一致状況
"""
from __future__ import annotations

import json
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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.search import PatchIndex
    from lib import finding_routing
    from lib.ovr_scoring import resolve_atlas_images, run_query_arm

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    target_findings = list(config["target_findings"])
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _macenko_tile_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size, tile_transform=_macenko_tile_transform)
             for f in images],
            axis=0,
        )

    # --- 既存(パッチ単位fit)索引: lib.finding_routingが本番で使っているものと同一 ---
    patchfit_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    patchfit_index = PatchIndex.load(
        index_path=patchfit_dir / "index.faiss",
        manifest_path=patchfit_dir / "manifest.parquet",
        slide_meta_path=patchfit_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.MACENKO_FEATURES_DIR,
    )

    # --- 新規(スライド単位fit)索引: experiments/0050 ---
    slidefit_dir = project_root / config["macenko_slidefit_index_dir"]
    slidefit_index = PatchIndex.load(
        index_path=slidefit_dir / "index.faiss",
        manifest_path=slidefit_dir / "manifest.parquet",
        slide_meta_path=slidefit_dir / "slide_meta.parquet",
        features_dir=project_root / config["macenko_slidefit_features_dir"],
    )

    arms = {
        "macenko_patchfit": patchfit_index,
        "macenko_slidefit": slidefit_index,
    }

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
    for finding in target_findings:
        images = resolve_atlas_images(project_root, config, finding)
        if not images:
            logger.warning(f"no atlas images resolved for {finding!r}, skipping")
            continue

        finding_dir = run_dir / _slug(finding)
        finding_dir.mkdir(exist_ok=True)

        # embed関数は両アームで完全に同一なので1回だけ計算して使い回す。
        query_vecs = _macenko_tile_embed(images)

        arm_results = {}
        for arm_label, patch_index in arms.items():
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=query_vecs, patch_index=patch_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                params=params, logger=logger,
            )
        logger.info(
            f"[{finding}] macenko_patchfit galleries: {arm_results['macenko_patchfit']['n_galleries']}, "
            f"macenko_slidefit galleries: {arm_results['macenko_slidefit']['n_galleries']}"
        )

        top5_patchfit = arm_results["macenko_patchfit"]["top_slide_ids"][:5]
        top5_slidefit = arm_results["macenko_slidefit"]["top_slide_ids"][:5]
        rows.append({
            "finding": finding,
            "n_galleries_macenko_patchfit": arm_results["macenko_patchfit"]["n_galleries"],
            "n_galleries_macenko_slidefit": arm_results["macenko_slidefit"]["n_galleries"],
            "top5_macenko_patchfit": top5_patchfit,
            "top5_macenko_slidefit": top5_slidefit,
            "top_slides_changed": top5_patchfit != top5_slidefit,
            "top5_overlap_count": len(set(top5_patchfit) & set(top5_slidefit)),
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
