"""experiments/0041: Macenko + 線形ドメイン補正を「両方とも」全25所見に適用し、
目視で改善を確認する。

これまでの検証は、macenko(per-tile染色正規化 + 別コーパス索引、
lib.finding_routing.MACENKO_FINDINGS)と線形ドメイン補正(atlas図版平均 -
コーパス平均のシフト、experiments/0039・0040)をそれぞれ独立に、plain
(baseline)コーパス・埋め込み空間を基準に検証したものだった。両方を重ねた
場合の効果、およびGTが無い18所見での見え方は未評価だった。

この実験は macenko 空間(macenko per-tile正規化クエリ + macenko索引)を基準に
domain_shift を独立に再計算し(macenko空間はplain空間と埋め込み分布が異なる
ため、experiments/0038/0039のdomain_shiftをそのまま流用せず交絡を避ける)、
NNLアトラス全25所見に「macenko単独」vs「macenko+線形補正併用」の2腕で目視
ギャラリー診断を実行する。GT対応7所見については
scripts.validate_against_ground_truth.run_comparison によるGT best_rank比較も
追加で行う(既存ハーネストの再利用でほぼ無料)。

alpha は experiments/0040(plain空間のスイープ)の結果である0.25を流用する
——macenko空間で独立に再チューニングはしていない点に注意。

出力(outputs/0041_.../default/):
  domain_shift_macenko.npy      macenko空間でのdomain_shiftベクトル
  gt_comparison.csv             GT対応7所見のmacenko vs macenko+linear比較
  <finding_slug>/query__{macenko,macenko_domain_corrected}/...  検索2腕
  summary.csv                   全25所見横並びのギャラリー数比較
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
    from lib.ovr_scoring import list_atlas_folders, sample_negative_pool, run_query_arm
    from lib.torchstain_normalize import normalize_to_reference
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    rng = np.random.default_rng(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    alpha = float(config["alpha"])

    macenko_index_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    macenko_features_dir = project_root / finding_routing.MACENKO_FEATURES_DIR
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    patch_index = PatchIndex.load(
        index_path=macenko_index_dir / "index.faiss",
        manifest_path=macenko_index_dir / "manifest.parquet",
        slide_meta_path=macenko_index_dir / "slide_meta.parquet",
        features_dir=macenko_features_dir,
    )
    manifest = patch_index.manifest.assign(slide_id=patch_index.manifest["slide_id"].astype(str))

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _macenko_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size, tile_transform=_macenko_tile_transform)
             for f in images],
            axis=0,
        )

    # --- 1. atlas 25所見全ての図版タイルをmacenko空間で埋め込み、domain_shiftを
    #        計算する(negativeもmacenkoコーパスからサンプル) ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    folder_vecs: dict[str, np.ndarray] = {}
    for folder_name, images in folders:
        folder_vecs[folder_name] = _macenko_embed(images)
        logger.info(f"  {folder_name}: {len(images)} images -> {folder_vecs[folder_name].shape[0]} tiles")

    pos_vecs = np.concatenate(list(folder_vecs.values()), axis=0)
    logger.info(f"pooled macenko-space atlas tiles: {pos_vecs.shape[0]} from {len(folders)} findings")

    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, macenko_features_dir, n=config["n_negative_patches"], exclude_slides=set(), rng=rng,
    )
    logger.info(f"macenko-space negative patches sampled: {neg_vecs.shape[0]} from "
                f"{len(set(neg_slide_ids))} slides")

    domain_shift = pos_vecs.mean(axis=0) - neg_vecs.mean(axis=0)
    np.save(run_dir / "domain_shift_macenko.npy", domain_shift)
    logger.info(f"domain_shift_macenko norm: {np.linalg.norm(domain_shift):.4f} (alpha={alpha})")

    # --- 2. 全25所見で macenko 単独 vs macenko+線形補正併用 の目視ギャラリー診断 ---
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
    for folder_name, images in folders:
        finding_dir = run_dir / _slug(folder_name)
        finding_dir.mkdir(exist_ok=True)

        macenko_vecs = folder_vecs[folder_name]
        corrected_vecs = _l2_normalize(macenko_vecs - alpha * domain_shift)

        arm_results = {}
        for arm_label, vecs in (("macenko", macenko_vecs), ("macenko_domain_corrected", corrected_vecs)):
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                params=params, logger=logger,
            )
        logger.info(f"[{folder_name}] macenko galleries: {arm_results['macenko']['n_galleries']}, "
                    f"macenko_domain_corrected galleries: {arm_results['macenko_domain_corrected']['n_galleries']}")

        rows.append({
            "atlas_folder": folder_name,
            "corpus_finding_type": CATEGORIES.get(folder_name),
            "n_galleries_macenko": arm_results["macenko"]["n_galleries"],
            "n_galleries_macenko_domain_corrected": arm_results["macenko_domain_corrected"]["n_galleries"],
            "top_slides_changed": arm_results["macenko"]["top_slide_ids"][:5]
            != arm_results["macenko_domain_corrected"]["top_slide_ids"][:5],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    # --- 3. GT対応7所見について、GT best_rank比較も追加(既存ハーネスト再利用) ---
    def _corrected_embed(images) -> np.ndarray:
        return _l2_normalize(_macenko_embed(images) - alpha * domain_shift)

    gt_pipelines = {
        "macenko": (patch_index, _macenko_embed),
        "macenko_domain_corrected": (patch_index, _corrected_embed),
    }
    gt_df = run_comparison(gt_pipelines)
    gt_df.to_csv(run_dir / "gt_comparison.csv", index=False)
    logger.info(f"GT comparison:\n{gt_df.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
