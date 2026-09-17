"""experiments/0056(フェーズA): 全25所見×3空間(plain/whiten/macenko)×
局所k近傍補正alphaグリッドの軽量数値スイープ(画像生成なし)。

experiments/0055でGT対応7所見について「現行の空間ルーティング+局所補正」と
「他の空間+局所補正」を比較したところ、7所見中2所見(Deposit, glycogen /
Proliferation, Kupffer cell)で現行ルーティングが最適でなくなっていた。GT対応
7所見にとどまらず、NNLアトラス全25所見についても同じ「空間×alpha」の組み
合わせを見たいが、GTが無いため判断材料は見た目(目視)にせざるを得ない、
という依頼を受けた。

25所見×3空間×13alpha=975通りを、目視前の一次フィルタとして軽量な数値指標
(画像生成なし、PatchIndex.search_top_slides_multiのFAISS IVF検索のみ)で
スクリーニングする。実際に画像を目視するフェーズBは、この結果を見て絞り込んだ
候補だけに限定する(別実験)。

atlasプール・corpusプール・局所補正のロジックはexperiments/0055と同一。

出力(outputs/0056_.../default/):
  all_findings_space_alpha_sweep.csv   975行(所見×空間×alpha)の数値指標
"""
from __future__ import annotations

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


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import list_atlas_folders, sample_negative_pool
    from lib.search import PatchIndex
    from lib import finding_routing

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
    nprobe = int(config["nprobe"])
    k_candidates = int(config["k_candidates"])
    top_n_slides = int(config["top_n_slides"])
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

    rows = []
    for folder_name, images in folders:
        for space_name, (patch_index, embed_one_fn, pool_vecs, pool_ids, corpus_vecs) in spaces.items():
            vecs, ids = _embed_with_ids(images, embed_one_fn)
            shift = _local_shift(vecs, ids, pool_vecs, pool_ids, corpus_vecs, k)

            baseline_top5: list[str] | None = None
            for alpha in alphas:
                corrected = _l2_normalize(vecs - alpha * shift)
                top_slides = patch_index.search_top_slides_multi(
                    corrected, k_candidates=k_candidates, nprobe=nprobe, top_n_slides=top_n_slides,
                )
                top5 = top_slides["slide_id"].astype(str).head(5).tolist()
                if alpha == 0.0:
                    baseline_top5 = top5
                rows.append({
                    "finding": folder_name, "space": space_name, "alpha": alpha,
                    "n_tiles": int(vecs.shape[0]),
                    "top1_slide_id": top_slides.iloc[0]["slide_id"] if len(top_slides) else None,
                    "top1_max_similarity": float(top_slides.iloc[0]["max_similarity"]) if len(top_slides) else None,
                    "top1_n_hits_ratio": float(top_slides.iloc[0]["n_hits_ratio"]) if len(top_slides) else None,
                    "top3_mean_max_similarity": float(top_slides["max_similarity"].head(3).mean()) if len(top_slides) else None,
                    "top5_slide_ids": ",".join(top5),
                    "jaccard_top5_vs_alpha0": _jaccard(top5, baseline_top5) if baseline_top5 is not None else 1.0,
                })
        logger.info(f"{folder_name}: done (3 spaces x {len(alphas)} alphas)")

    sweep_df = pd.DataFrame(rows)
    sweep_df.to_csv(run_dir / "all_findings_space_alpha_sweep.csv", index=False)
    logger.info(f"sweep rows: {len(sweep_df)}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
