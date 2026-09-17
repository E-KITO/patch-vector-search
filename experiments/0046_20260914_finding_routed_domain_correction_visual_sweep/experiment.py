"""experiments/0046: GTの無い18所見について、finding_routingの実ルーティング
空間で複数alpha候補のギャラリーを生成する(目視によるalpha個別選定の準備)。

experiments/0045はGT対応7所見について、所見ごとに既にルーティングされている
空間(plain/whiten/macenko)でalphaをGT best_rankで細かくチューニングする。
残り18所見(コーパスGTが無い)は数値評価ができないため、目視でしかalphaを
判断できない(README「評価に使ったデータとその限界」の19カテゴリと同じ制約)。

この実験は18所見それぞれについて、config.alphasの各値でギャラリーを生成する
ところまでを行う——alpha候補自体の採否判断(目視)はこの実験の出力を見てから
別途行う。alpha候補はexperiments/0045の定量グリッドと同一(0.0〜0.5を0.05刻み、
11点、ユーザー指示で0046側も精度を合わせた)で、0.0が補正なし相当を兼ねる。

対象所見の空間振り分けはlib.finding_routingと同じ基準で決める
(finding_type -> MACENKO_FINDINGS/WHITEN_FINDINGS/それ以外=baseline)。
コーパスGTが無い18所見のうちfinding_typeが分かっているのは
"Liver - Fatty Change"(config.extra_atlas_folders経由、macenko空間)のみで、
残り17所見はfinding_type自体が不明(CATEGORIESに未登録)なため既定の
baseline(plain空間)で扱う。domain_shiftはexperiments/0045と同じ成果物
(plain: experiments/0039、macenko: experiments/0041)を再利用し、再計算しない。

出力(outputs/0046_.../default/):
  target_findings.csv                     対象18所見の空間振り分け一覧
  <finding_slug>/query__alpha<a>/...      alpha候補ごとのギャラリー(a=0.0が補正なし相当)
  summary.csv                              所見・alpha候補ごとのギャラリー数比較
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


def _family_for_finding(finding_type: str | None, macenko_findings: frozenset, whiten_findings: frozenset) -> str:
    """lib.finding_routingと同じ基準で空間を決める。finding_typeが不明
    (CATEGORIES/extra_atlas_foldersに未登録)な所見は既定のbaselineへ倒す。"""
    if finding_type in macenko_findings:
        return "macenko"
    if finding_type in whiten_findings:
        return "whiten"
    return "plain"


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import list_atlas_folders, run_query_arm
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES

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

    domain_shift_plain = np.load(project_root / config["domain_shift_plain_path"])
    domain_shift_macenko = np.load(project_root / config["domain_shift_macenko_path"])
    logger.info(f"domain_shift_plain norm={np.linalg.norm(domain_shift_plain):.4f}, "
                f"domain_shift_macenko norm={np.linalg.norm(domain_shift_macenko):.4f}")

    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    def _plain_tile_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
        )

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

    # --- 索引ロード(3空間、experiments/0045・lib.finding_routingと同一パラメータ) ---
    baseline_dir = project_root / finding_routing.BASELINE_INDEX_DIR
    whiten_dir = project_root / finding_routing.WHITEN_INDEX_DIR
    macenko_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    baseline_index = PatchIndex.load(
        index_path=baseline_dir / "index.faiss",
        manifest_path=baseline_dir / "manifest.parquet",
        slide_meta_path=baseline_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
    )
    whiten_index = PatchIndex.load(
        index_path=whiten_dir / "index.faiss",
        manifest_path=whiten_dir / "manifest.parquet",
        slide_meta_path=whiten_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
        transform_path=project_root / finding_routing.WHITEN_TRANSFORM_PATH,
    )
    macenko_index = PatchIndex.load(
        index_path=macenko_dir / "index.faiss",
        manifest_path=macenko_dir / "manifest.parquet",
        slide_meta_path=macenko_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.MACENKO_FEATURES_DIR,
    )

    families = {
        "plain": (baseline_index, _plain_tile_embed, domain_shift_plain),
        "whiten": (whiten_index, _plain_tile_embed, domain_shift_plain),
        "macenko": (macenko_index, _macenko_tile_embed, domain_shift_macenko),
    }

    # --- 対象18所見を解決(25所見からGT対応7所見を除いたもの) ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folder_to_finding = dict(CATEGORIES)
    folder_to_finding.update(config.get("extra_atlas_folders") or {})
    gt_folders = set(CATEGORIES.keys())

    all_folders = list_atlas_folders(atlas_root, atlas_csv)
    targets = [(name, imgs) for name, imgs in all_folders if name not in gt_folders]
    logger.info(f"atlas folders: {len(all_folders)} total, {len(gt_folders)} GT-covered "
                f"(handled by experiments/0045), {len(targets)} target for this sweep")

    target_rows = []
    for folder_name, images in targets:
        finding_type = folder_to_finding.get(folder_name)
        family_name = _family_for_finding(
            finding_type, finding_routing.MACENKO_FINDINGS, finding_routing.WHITEN_FINDINGS
        )
        target_rows.append({
            "atlas_folder": folder_name, "finding_type": finding_type,
            "family": family_name, "n_images": len(images),
        })
    pd.DataFrame(target_rows).to_csv(run_dir / "target_findings.csv", index=False)
    logger.info(f"target findings:\n{pd.DataFrame(target_rows).to_string()}")

    assert 0.0 in alphas, "alphasに0.0(補正なし相当)を含めること — summaryのtop_slides基準として使う"

    # --- 所見ごとに各alpha候補のギャラリーを生成 ---
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

    summary_rows = []
    for folder_name, images in targets:
        finding_type = folder_to_finding.get(folder_name)
        family_name = _family_for_finding(
            finding_type, finding_routing.MACENKO_FINDINGS, finding_routing.WHITEN_FINDINGS
        )
        patch_index, base_embed_fn, domain_shift = families[family_name]

        finding_dir = run_dir / _slug(folder_name)
        finding_dir.mkdir(exist_ok=True)

        base_vecs = base_embed_fn(images)  # alphaに依存しない、所見あたり1回だけ

        arm_vecs = {f"alpha{alpha}": _l2_normalize(base_vecs - alpha * domain_shift) for alpha in alphas}

        arm_results = {}
        for arm_label, vecs in arm_vecs.items():
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                params=params, logger=logger,
            )
        gallery_counts = {k: v["n_galleries"] for k, v in arm_results.items()}
        logger.info(f"[{folder_name}] (family={family_name}) galleries: {gallery_counts}")

        row = {
            "atlas_folder": folder_name, "finding_type": finding_type, "family": family_name,
            "top_slides_alpha0.0": arm_results["alpha0.0"]["top_slide_ids"][:5],
        }
        for arm_label, res in arm_results.items():
            row[f"n_galleries_{arm_label}"] = res["n_galleries"]
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
