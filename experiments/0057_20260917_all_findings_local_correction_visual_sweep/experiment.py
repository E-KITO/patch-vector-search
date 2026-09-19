"""experiments/0057(フェーズB): GTの無い18所見について、局所k近傍補正を
3空間(plain/whiten/macenko)×alpha2点で実ギャラリー生成し目視判断する。

experiments/0056(フェーズA)で、局所k近傍補正の効果を画像生成なしの数値指標
(top3_mean_max_similarity)でスクリーニングしようとしたが、alphaを上げるほど
全所見・全空間で一様に類似度が単調上昇するという結果になった。これは局所補正の
仕組み上ほぼ同語反復的なアーティファクト(クエリを自分自身が引いた近傍corpus
パッチの重心に寄せているだけなので、その近傍への類似度が上がるのは当然)で
あり、所見が見えやすくなったことの裏付けにはならないと判明した
(experiments/0039のalpha=1.0ギャラリー急増が見せかけだったのと同種の失敗
モード)。

数値だけでの自動選別は使えないと分かったため、experiments/0046の方針(18所見を
粗い格子で網羅し、実際に目視で判断する)に戻す。ただし0046はfinding_routingが
決めた単一空間だけを見ていたのに対し、この実験は3空間全部を対象所見ごとに
生成し、「現行の既定(plain)以外の空間の方が良く見えるか」も比較できるように
する。alphaは0.0(補正なし)と0.3(experiments/0055でGT対応7所見の多くが改善
を示した中程度の強さ)の2点に絞り、18所見×3空間×2alpha=108アームに抑える。

atlasプール・corpusプール・局所補正のロジックはexperiments/0055・0056と同一。

出力(outputs/0057_.../default/):
  target_findings.csv
  <finding_slug>/query__<space>_alpha<a>/...
  summary.csv
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


def _build_atlas_pool(folders, embed_one_fn, logger) -> tuple[np.ndarray, np.ndarray]:
    vecs, ids = [], []
    for folder_name, images in folders:
        for img_path in images:
            tile_vecs = embed_one_fn(img_path)
            vecs.append(tile_vecs)
            ids.extend([img_path] * len(tile_vecs))
        logger.info(f"  atlas pool: {folder_name} done")
    return np.concatenate(vecs, axis=0), np.array(ids)


def _embed_with_ids(images: list[str], embed_one_fn) -> tuple[np.ndarray, np.ndarray]:
    vecs, ids = [], []
    for img_path in images:
        tile_vecs = embed_one_fn(img_path)
        vecs.append(tile_vecs)
        ids.extend([img_path] * len(tile_vecs))
    return np.concatenate(vecs, axis=0), np.array(ids)


def _local_shift(
    tile_vecs: np.ndarray, tile_ids: np.ndarray,
    atlas_pool_vecs: np.ndarray, atlas_pool_ids: np.ndarray,
    corpus_pool_vecs: np.ndarray, k: int,
) -> np.ndarray:
    sims_atlas = tile_vecs @ atlas_pool_vecs.T
    same_image = tile_ids[:, None] == atlas_pool_ids[None, :]
    sims_atlas = np.where(same_image, -np.inf, sims_atlas)
    top_atlas = np.argpartition(-sims_atlas, k, axis=1)[:, :k]
    atlas_local_mean = atlas_pool_vecs[top_atlas].mean(axis=1)

    sims_corpus = tile_vecs @ corpus_pool_vecs.T
    top_corpus = np.argpartition(-sims_corpus, k, axis=1)[:, :k]
    corpus_local_mean = corpus_pool_vecs[top_corpus].mean(axis=1)

    return atlas_local_mean - corpus_local_mean


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import list_atlas_folders, sample_negative_pool, run_query_arm
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES

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
    k = int(config["k_neighbors"])
    n_corpus_pool = int(config["n_corpus_pool"])
    alphas = list(config["alphas"])
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _embed_one_plain(img_path: str) -> np.ndarray:
        return embed_image_tiles(str(img_path), tile_size=tile_size)

    def _embed_one_macenko(img_path: str) -> np.ndarray:
        return embed_image_tiles(str(img_path), tile_size=tile_size, tile_transform=_macenko_tile_transform)

    # --- atlasプール(全25所見・由来画像id付き)をplain/macenko両空間で構築 ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    atlas_pool_plain_vecs, atlas_pool_plain_ids = _build_atlas_pool(folders, _embed_one_plain, logger)
    atlas_pool_macenko_vecs, atlas_pool_macenko_ids = _build_atlas_pool(folders, _embed_one_macenko, logger)
    logger.info(f"atlas pool: plain {atlas_pool_plain_vecs.shape}, macenko {atlas_pool_macenko_vecs.shape}")

    # --- 索引ロード(3空間)とcorpusプール ---
    baseline_dir = project_root / finding_routing.BASELINE_INDEX_DIR
    whiten_dir = project_root / finding_routing.WHITEN_INDEX_DIR
    macenko_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    baseline_index = PatchIndex.load(
        index_path=baseline_dir / "index.faiss", manifest_path=baseline_dir / "manifest.parquet",
        slide_meta_path=baseline_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
    )
    whiten_index = PatchIndex.load(
        index_path=whiten_dir / "index.faiss", manifest_path=whiten_dir / "manifest.parquet",
        slide_meta_path=whiten_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
        transform_path=project_root / finding_routing.WHITEN_TRANSFORM_PATH,
    )
    macenko_index = PatchIndex.load(
        index_path=macenko_dir / "index.faiss", manifest_path=macenko_dir / "manifest.parquet",
        slide_meta_path=macenko_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.MACENKO_FEATURES_DIR,
    )

    plain_manifest = baseline_index.manifest.assign(slide_id=baseline_index.manifest["slide_id"].astype(str))
    macenko_manifest = macenko_index.manifest.assign(slide_id=macenko_index.manifest["slide_id"].astype(str))

    corpus_pool_plain_vecs, _ = sample_negative_pool(
        plain_manifest, project_root / finding_routing.PLAIN_FEATURES_DIR,
        n=n_corpus_pool, exclude_slides=set(), rng=rng,
    )
    corpus_pool_macenko_vecs, _ = sample_negative_pool(
        macenko_manifest, project_root / finding_routing.MACENKO_FEATURES_DIR,
        n=n_corpus_pool, exclude_slides=set(), rng=rng,
    )
    logger.info(f"corpus pool: plain {corpus_pool_plain_vecs.shape}, macenko {corpus_pool_macenko_vecs.shape}")

    spaces = {
        "plain": (baseline_index, _embed_one_plain, atlas_pool_plain_vecs, atlas_pool_plain_ids, corpus_pool_plain_vecs),
        "whiten": (whiten_index, _embed_one_plain, atlas_pool_plain_vecs, atlas_pool_plain_ids, corpus_pool_plain_vecs),
        "macenko": (macenko_index, _embed_one_macenko, atlas_pool_macenko_vecs, atlas_pool_macenko_ids, corpus_pool_macenko_vecs),
    }

    # --- 対象18所見を解決(25所見からGT対応7所見を除いたもの) ---
    gt_folders = set(CATEGORIES.keys())
    all_folders = folders
    targets = [(name, imgs) for name, imgs in all_folders if name not in gt_folders]
    logger.info(f"atlas folders: {len(all_folders)} total, {len(gt_folders)} GT-covered "
                f"(experiments/0055で検証済み), {len(targets)} target for this sweep")
    pd.DataFrame([{"atlas_folder": n, "n_images": len(i)} for n, i in targets]).to_csv(
        run_dir / "target_findings.csv", index=False
    )

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
        finding_dir = run_dir / _slug(folder_name)
        finding_dir.mkdir(exist_ok=True)
        row = {"atlas_folder": folder_name}

        for space_name, (patch_index, embed_one_fn, pool_vecs, pool_ids, corpus_vecs) in spaces.items():
            vecs, ids = _embed_with_ids(images, embed_one_fn)
            shift = _local_shift(vecs, ids, pool_vecs, pool_ids, corpus_vecs, k)

            for alpha in alphas:
                arm_label = f"{space_name}_alpha{alpha}"
                corrected = _l2_normalize(vecs - alpha * shift)
                res = run_query_arm(
                    arm_label=arm_label, query_vecs=corrected, patch_index=patch_index,
                    thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                    params=params, logger=logger,
                )
                row[f"n_galleries_{arm_label}"] = res["n_galleries"]
                row[f"top_slides_{arm_label}"] = res["top_slide_ids"][:5]

        logger.info(f"[{folder_name}] done: {row}")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
