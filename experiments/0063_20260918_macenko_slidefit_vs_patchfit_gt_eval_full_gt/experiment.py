"""experiments/0051: Macenko染色ベクトル推定「パッチ単位fit vs スライド単位fit」の
GT best_rank比較 — experiments/0043・0044の事前検証を受けてwsi_preprocess側に
依頼した全コーパス再構築(README「wsi_preprocess連携」)の効果を実測する、この
変更の最終判定。

既存のパッチ単位fit索引(experiments/0033、lib.finding_routingが本番で使用中)
と、新規のスライド単位fit索引(experiments/0050)を、scripts.
validate_against_ground_truth.run_comparisonの枠組みでGT対応7所見全部について
比較する。クエリ側embed関数は両アームで完全に同一
(lib.finding_routing.MACENKO_STAIN_REFERENCEへのtorchstain正規化、
experiments/0045の_macenko_tile_embedと同一実装) — コーパス側の染色ベクトル
推定方法だけが違うよう統制している。

前提: experiments/0049(manifest)・0050(索引)が完了していること。実行前に
wsi_preprocess側からfeatures_uni_v1_macenko_slidefit配下が1000/1000枚で
揃った連絡を受けていること(README「wsi_preprocess連携」参照)。

出力(outputs/0051_.../default/):
  gt_comparison.csv   所見ごとのfound/best_rank/mean_rank比較
                       (macenko_patchfit vs macenko_slidefit)
  summary.json         found減少所見の有無、best_rank改善/悪化所見の集計と
                        総合判定(verdict)
"""
from __future__ import annotations

import json
import logging
import os
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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
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

    logger.info(f"patchfit index: {patchfit_dir} (n_vectors={patchfit_index.index.ntotal})")
    logger.info(f"slidefit index: {slidefit_dir} (n_vectors={slidefit_index.index.ntotal})")

    pipelines = {
        "macenko_patchfit": (patchfit_index, _macenko_tile_embed),
        "macenko_slidefit": (slidefit_index, _macenko_tile_embed),
    }

    gt_csv = project_root / config["gt_csv"]
    gt_df = run_comparison(pipelines, gt_csv=gt_csv)
    assert len(gt_df) == len(CATEGORIES), (
        f"expected {len(CATEGORIES)} category rows, got {len(gt_df)} "
        "(an atlas category folder produced no images?)"
    )
    gt_df["finding_type"] = list(CATEGORIES.values())
    gt_df.to_csv(run_dir / "gt_comparison.csv", index=False)
    logger.info(f"GT comparison:\n{gt_df.to_string()}")

    # --- 判定: found減少(regressed)最優先、その上でbest_rank改善/悪化を集計 ---
    regressed, improved, worsened, unchanged = [], [], [], []
    for _, row in gt_df.iterrows():
        finding = row["finding_type"]
        base_found, base_best = row["macenko_patchfit_found"], row["macenko_patchfit_best"]
        new_found, new_best = row["macenko_slidefit_found"], row["macenko_slidefit_best"]
        if new_found < base_found:
            regressed.append(finding)
        elif pd.isna(base_best) or pd.isna(new_best):
            unchanged.append(finding)
        elif new_best < base_best:
            improved.append(finding)
        elif new_best > base_best:
            worsened.append(finding)
        else:
            unchanged.append(finding)

    if regressed:
        verdict = "REGRESSED (GTスライドが候補プールから消失した所見あり、slidefitへの切替は非推奨)"
    elif len(improved) > len(worsened):
        verdict = f"NET_IMPROVEMENT ({len(improved)}所見改善 / {len(worsened)}所見悪化 / {len(unchanged)}所見不変)"
    else:
        verdict = f"NET_NEUTRAL_OR_WORSE ({len(improved)}所見改善 / {len(worsened)}所見悪化 / {len(unchanged)}所見不変)"

    summary = {
        "regressed_findings": regressed,
        "improved_findings": improved,
        "worsened_findings": worsened,
        "unchanged_findings": unchanged,
        "verdict": verdict,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary: {summary}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
