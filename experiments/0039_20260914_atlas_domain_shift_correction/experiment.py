"""experiments/0039: atlas→corpus ドメインギャップの線形補正(平均シフト)を
GT best_rank + 目視の両方で検証する。

背景: experiments/0038で、所見ラベルを無視した二値分類器がatlas図版タイル
(25所見・4891枚)とコーパスのランダムパッチを LOFO/LOIO AUROC=1.0 で完璧に
分離でき、かつコーパスGTパッチとatlas図版のドメインスコア差が所見によらず
ほぼ一定(10.0〜11.7)だった。これは単一の支配的な「atlasらしさ」方向による
線形なドメインギャップだという仮説を示唆する。

この実験は最も単純な線形補正を試す: atlas図版タイルの平均埋め込みからコーパス
ランダムパッチの平均埋め込みを引いた domain_shift ベクトルを、個々のatlas
クエリ埋め込みから差し引いてL2再正規化してから検索する。索引・コーパス側は
一切変更しない(baseline 0018のまま)——Macenko/whitenと違い別コーパスの
再構築が不要な、クエリ側後処理のみの補正。

検証は2種類:
  1. GT best_rank比較(scripts.validate_against_ground_truth.run_comparison
     を "domain_corrected" パイプラインを追加して呼び出すだけ、baseline_v1と
     同じ索引・同じ評価ロジック)。
  2. コーパスGT対応7所見の目視ギャラリー診断(lib.ovr_scoring.run_query_arm、
     experiments/0013・0030・0034と同じ枠組み)——GT数値だけで採否判断しない
     (README「experiments/0037」以降の教訓)。

出力(outputs/0039_.../default/):
  domain_shift.npy            atlas平均 - コーパス平均のシフトベクトル(1024次元)
  gt_comparison.csv           baseline_v1 vs domain_corrected の GT best_rank比較
  <finding_slug>/query__{baseline,domain_corrected}/...  検索2腕(目視診断)
  summary.csv                 7所見横並びのギャラリー数比較
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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.ovr_scoring import (
        list_atlas_folders, embed_atlas_images_as_positives, sample_negative_pool, run_query_arm,
    )
    from lib.atlas_figures import query_images
    from scripts.validate_against_ground_truth import CATEGORIES, default_pipelines, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    rng = np.random.default_rng(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    features_dir = project_root / config["features_dir"]
    index_dir = project_root / config["index_dir"]
    tile_size = int(config.get("tile_size", 224))
    alpha = float(config["alpha"])

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))

    # --- 1. domain_shift = atlas図版タイル(25所見プール)の平均 - コーパス
    #        ランダムパッチの平均(experiments/0038と同一レシピ) ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    pos_vecs_list = []
    for folder_name, images in folders:
        vecs, _ = embed_atlas_images_as_positives(images, tile_size)
        pos_vecs_list.append(vecs)
    pos_vecs = np.concatenate(pos_vecs_list, axis=0)
    logger.info(f"pooled atlas tiles: {pos_vecs.shape[0]} from {len(folders)} findings")

    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"], exclude_slides=set(), rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slide_ids))} slides")

    domain_shift = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    np.save(run_dir / "domain_shift.npy", domain_shift)
    logger.info(f"domain_shift norm: {np.linalg.norm(domain_shift):.4f} (alpha={alpha})")

    def _corrected_embed_fn(images) -> np.ndarray:
        tiles = np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
        )
        corrected = tiles - alpha * domain_shift
        norms = np.linalg.norm(corrected, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        return corrected / norms

    # --- 2. GT best_rank比較(baseline_v1 vs domain_corrected、既存の評価
    #        ハーネストをそのまま使う) ---
    pipelines = default_pipelines(only={"baseline_v1"})
    baseline_index, baseline_embed_fn = pipelines["baseline_v1"]
    pipelines["domain_corrected"] = (baseline_index, _corrected_embed_fn)

    gt_df = run_comparison(pipelines)
    gt_df.to_csv(run_dir / "gt_comparison.csv", index=False)
    logger.info(f"GT comparison:\n{gt_df.to_string()}")

    # --- 3. 目視ギャラリー診断(コーパスGT対応7所見、baseline vs domain_corrected) ---
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
    for folder_name, finding_type in CATEGORIES.items():
        cat_dir = atlas_root / folder_name
        images = query_images(cat_dir, atlas_csv=atlas_csv)
        if not images:
            logger.warning(f"no images for {folder_name!r}, skipping visual diagnostic")
            continue
        logger.info(f"===== {folder_name} ({finding_type}) =====")

        finding_dir = run_dir / _slug(folder_name)
        finding_dir.mkdir(exist_ok=True)

        baseline_vecs = np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
        )
        corrected = baseline_vecs - alpha * domain_shift
        norms = np.linalg.norm(corrected, axis=-1, keepdims=True)
        norms[norms == 0] = 1.0
        corrected_vecs = corrected / norms

        arm_results = {}
        for arm_label, vecs in (("baseline", baseline_vecs), ("domain_corrected", corrected_vecs)):
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=vecs, patch_index=baseline_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                params=params, logger=logger,
            )
        logger.info(f"[{folder_name}] baseline galleries: {arm_results['baseline']['n_galleries']}, "
                    f"domain_corrected galleries: {arm_results['domain_corrected']['n_galleries']}")

        rows.append({
            "finding": finding_type, "atlas_folder": folder_name,
            "n_galleries_baseline": arm_results["baseline"]["n_galleries"],
            "n_galleries_domain_corrected": arm_results["domain_corrected"]["n_galleries"],
            "top_slides_changed": arm_results["baseline"]["top_slide_ids"][:5]
            != arm_results["domain_corrected"]["top_slide_ids"][:5],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
