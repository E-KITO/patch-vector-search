"""experiments/0048: deliverモード候補プール構造の一括分析
(README「次の一手」#4: deliverモードの候補枯れ対策)。

experiments/0015/0019のdeliverモードは、glycogen(137/150)・ground
glass(113/150)ともtarget 150枚に未達だった。未達の中身は所見ごとに違う——
glycogenは候補プール自体が薄い構造的な頭打ち、ground glassは「多様性が
尽きた」形(GT4枚の稀少所見としては妥当な頭打ちで失敗ではない)。target=150
という一律の目標値が所見ごとに現実的かを判断するため、GT対応7所見全部について
候補プールの段階別カウントを一括で集計する。

実際のパッチ画像は切り出さない(raw WSIクロップ・コンタクトシート生成を
スキップ)——`lib.patch_set.curate_candidates`が返す診断値だけを見る、
検索(faiss-cpu)のみのCPU-onlyジョブ。シード側のクエリ埋め込みはGTスライド
自身の既存h5特徴量を直読みするだけでUNIエンコーダは使わない
(experiments/0015のload_slide_query_vecsと同一)。

出力(outputs/0048_.../default/):
  summary.csv   所見ごとの候補プール段階別カウント・現実的目標枚数の見積もり
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import h5py
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


def load_slide_query_vecs(
    features_dir: Path, slide_ids: list[str], n_per_slide: int, rng: np.random.Generator
) -> np.ndarray:
    """seedスライド自身のパッチ特徴量をh5から直読みしてL2正規化(UNIエンコーダ
    不使用、experiments/0015・0019のload_slide_query_vecsと同一)。"""
    parts = []
    for sid in slide_ids:
        with h5py.File(features_dir / f"{sid}.h5", "r") as f:
            n = f["features"].shape[0]
            if n <= n_per_slide:
                v = f["features"][:].astype(np.float32)
            else:
                rows = np.sort(rng.choice(n, size=n_per_slide, replace=False))
                v = f["features"][rows].astype(np.float32)
        parts.append(v)
    vecs = np.concatenate(parts, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.patch_set import curate_candidates
    from lib.search import PatchIndex

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    seed = int(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    index_exp_dir = project_root / config["index_exp_dir"]
    features_dir = project_root / config["features_dir"]
    gt_csv = project_root / config["gt_csv"]

    max_seed_slides = int(config["max_seed_slides"])
    n_query_patches_per_seed_slide = int(config["n_query_patches_per_seed_slide"])
    nprobe = int(config["nprobe"])
    k_candidate_patches = int(config["k_candidate_patches"])
    rerank_pool = int(config["rerank_pool"])
    max_tiles_reranked = config.get("max_tiles_reranked", None)
    sim_floor = float(config["sim_floor"])
    nms_radius_patches = float(config["nms_radius_patches"])
    max_per_slide = int(config["max_per_slide"])
    target_n_patches = int(config["target_n_patches"])

    pi = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )
    corpus_slides = set(pi.slide_meta.index.astype(str))
    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)

    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)

    summary_rows = []
    for target_finding in config["target_findings"]:
        rng = np.random.default_rng(seed)  # 所見ごとに同じ乱数系列から独立に開始
        logger.info(f"===== {target_finding} =====")

        finding_slides = sorted(
            set(gt.loc[gt["FINDING_TYPE"] == target_finding, "slide_id"]) & corpus_slides
        )
        if len(finding_slides) < 2:
            logger.warning(f"{target_finding!r}: only {len(finding_slides)} corpus GT slide(s) — skipping")
            continue

        # deliver モード: GTスライド全体をseedにして全て除外(experiments/0015・0019と同一)
        seed_slides = sorted(finding_slides)[:max_seed_slides]
        exclude_slides = set(finding_slides)

        query_vecs = load_slide_query_vecs(
            features_dir, seed_slides, n_query_patches_per_seed_slide, rng
        )
        logger.info(f"GT slides: {len(finding_slides)}, seed slides (capped at {max_seed_slides}): "
                    f"{len(seed_slides)}, query vectors: {query_vecs.shape[0]}")

        candidates = pi.search_similar_patches_multi(
            query_vecs, k=k_candidate_patches, nprobe=nprobe,
            rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked,
        )
        candidates["slide_id"] = candidates["slide_id"].astype(str)
        n_raw = len(candidates)
        candidates = candidates[~candidates["slide_id"].isin(exclude_slides)]
        n_after_exclude = len(candidates)
        seed_occupancy = round(1.0 - n_after_exclude / n_raw, 3) if n_raw else None
        logger.info(f"candidates: {n_raw} raw -> {n_after_exclude} after excluding "
                    f"{len(exclude_slides)} seed slides (seed occupancy {seed_occupancy})")

        _, curate_stats = curate_candidates(
            candidates, slide_meta,
            sim_floor=sim_floor, nms_radius_patches=nms_radius_patches,
            max_per_slide=max_per_slide, target_n_patches=target_n_patches,
        )
        logger.info(f"curation stats: {curate_stats}")

        # n_after_nms_and_cap はround-robin切り詰め前の真の上限(round-robinは
        # スライド間の配分を変えるだけで候補を増やさない) — これが
        # target_n_patchesに届かない場合、それが構造的な現実的上限になる。
        realistic_target = curate_stats["n_after_nms_and_cap"]

        summary_rows.append({
            "target_finding": target_finding,
            "n_gt_slides": len(finding_slides),
            "n_seed_slides": len(seed_slides),
            "n_raw_candidates": n_raw,
            "n_after_exclude_seed": n_after_exclude,
            "seed_occupancy_frac": seed_occupancy,
            "n_after_sim_floor": curate_stats["n_after_sim_floor"],
            "n_after_nms_and_cap": curate_stats["n_after_nms_and_cap"],
            "n_slides_after_nms": curate_stats["n_slides_after_nms"],
            "n_slides_at_max_per_slide": curate_stats["n_slides_at_max_per_slide"],
            "n_after_round_robin": curate_stats["n_after_round_robin"],
            "target_n_patches_configured": target_n_patches,
            "realistic_target_estimate": realistic_target,
            "meets_configured_target": bool(realistic_target >= target_n_patches),
            # n_slides_at_max_per_slide が多い(=多くのスライドが上限15に到達)なら
            # 「探索を深くすれば増える」余地あり、0に近ければ「スライドあたりの
            # 良い候補自体が少ない」ことが上限を決めている(README「job 9962」の
            # 解釈基準と同一)。
            "cap_limited": bool(
                curate_stats["n_slides_after_nms"] > 0
                and curate_stats["n_slides_at_max_per_slide"] / curate_stats["n_slides_after_nms"] >= 0.5
            ),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
