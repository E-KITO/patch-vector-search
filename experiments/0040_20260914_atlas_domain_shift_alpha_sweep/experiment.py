"""experiments/0040: atlas→corpus ドメインシフト補正の alpha スイープ。

experiments/0039(alpha=1.0)は否定的だった: GT best_rankが7所見中5所見で
悪化、うち2所見(Proliferation, Kupffer cell / Inclusion body)はGTスライドが
候補プールから完全に消失した。一方でギャラリー数は軒並み増えており、補正が
強すぎて正解ではない別のスライドに自信を持って飛びついている可能性が疑われる。

この実験は experiments/0039 が保存した domain_shift.npy をそのまま再利用し
(同じRNG状態・同じ負例サンプルで比較するため再計算しない)、補正の強さ
(alpha)を 0.25 / 0.5 / 0.75 でスイープしてGT best_rankが最も良いalphaを
探す。最良alphaについてのみ、コーパスGT対応7所見の目視ギャラリー診断も追加
実行する(GT数値だけで採否判断しない、README「experiments/0037」以降の教訓)。

出力(outputs/0040_.../default/):
  gt_comparison_alpha_sweep.csv   baseline_v1 vs 各alphaのGT best_rank比較
  best_alpha.json                  選ばれたalphaと選定根拠
  <finding_slug>/query__{baseline,domain_corrected}/...  最良alphaの検索2腕(目視診断)
  summary.csv                      最良alphaのギャラリー数比較
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


def _pick_best_alpha(gt_df: pd.DataFrame, alphas: list[float], logger) -> tuple[float, dict]:
    """アルファごとに (found が baseline を下回った所見数, best_rank悪化幅の合計)
    を計算し、辞書式順序(regressionが少ない方優先、次にrank悪化が小さい方優先)
    で最良のalphaを選ぶ。found割れ(GTスライドが候補プールから消失)を最優先で
    避ける——experiments/0039でalpha=1.0がこれを2所見で起こしたため。
    """
    scored = []
    for alpha in alphas:
        n_regressed = 0
        rank_delta_sum = 0
        for _, row in gt_df.iterrows():
            base_found, base_best = row["baseline_v1_found"], row["baseline_v1_best"]
            corr_found = row[f"alpha{alpha}_found"]
            corr_best = row[f"alpha{alpha}_best"]
            if pd.isna(corr_found) or corr_found < base_found:
                n_regressed += 1
                continue
            if pd.notna(base_best) and pd.notna(corr_best):
                rank_delta_sum += corr_best - base_best
        scored.append((alpha, n_regressed, rank_delta_sum))
        logger.info(f"  alpha={alpha}: n_regressed={n_regressed}, rank_delta_sum={rank_delta_sum:.1f}")
    best = min(scored, key=lambda t: (t[1], t[2]))
    return best[0], {"n_regressed": best[1], "rank_delta_sum": best[2]}


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.ovr_scoring import run_query_arm
    from lib.atlas_figures import query_images
    from scripts.validate_against_ground_truth import CATEGORIES, default_pipelines, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    alphas = list(config["alphas"])

    domain_shift = np.load(project_root / config["domain_shift_path"])
    logger.info(f"loaded domain_shift (norm={np.linalg.norm(domain_shift):.4f}) from "
                f"{config['domain_shift_path']}")

    def _make_corrected_embed_fn(alpha: float):
        def _fn(images) -> np.ndarray:
            tiles = np.concatenate(
                [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
            )
            corrected = tiles - alpha * domain_shift
            norms = np.linalg.norm(corrected, axis=-1, keepdims=True)
            norms[norms == 0] = 1.0
            return corrected / norms
        return _fn

    # --- 1. GT best_rank比較: baseline_v1 + 各alphaを一度のrun_comparisonで ---
    pipelines = default_pipelines(only={"baseline_v1"})
    baseline_index, _ = pipelines["baseline_v1"]
    for alpha in alphas:
        pipelines[f"alpha{alpha}"] = (baseline_index, _make_corrected_embed_fn(alpha))

    gt_df = run_comparison(pipelines)
    gt_df.to_csv(run_dir / "gt_comparison_alpha_sweep.csv", index=False)
    logger.info(f"GT comparison:\n{gt_df.to_string()}")

    best_alpha, reason = _pick_best_alpha(gt_df, alphas, logger)
    (run_dir / "best_alpha.json").write_text(json.dumps({"best_alpha": best_alpha, **reason}, indent=2))
    logger.info(f"best_alpha={best_alpha} ({reason})")

    # --- 2. 最良alphaについてのみ目視ギャラリー診断 ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
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
        logger.info(f"===== {folder_name} ({finding_type}) — best_alpha={best_alpha} =====")

        finding_dir = run_dir / _slug(folder_name)
        finding_dir.mkdir(exist_ok=True)

        baseline_vecs = np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
        )
        corrected = baseline_vecs - best_alpha * domain_shift
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
                    f"domain_corrected(alpha={best_alpha}) galleries: {arm_results['domain_corrected']['n_galleries']}")

        rows.append({
            "finding": finding_type, "atlas_folder": folder_name, "best_alpha": best_alpha,
            "n_galleries_baseline": arm_results["baseline"]["n_galleries"],
            "n_galleries_domain_corrected": arm_results["domain_corrected"]["n_galleries"],
            "top_slides_changed": arm_results["baseline"]["top_slide_ids"][:5]
            != arm_results["domain_corrected"]["top_slide_ids"][:5],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary (best_alpha={best_alpha}):\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
