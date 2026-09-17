"""experiments/0047: 空間的な「周囲組織との差分」文脈ベクトルで、GT対応7所見の
LOSO AUROC(判別力)がplain単一パッチ埋め込みより改善するか検証する。

原理: 候補パッチ自身の埋め込みから、同じスライド内で空間的に隣接する
(3x3グリッドの残り8マス)パッチ埋め込みの平均を引いた差分(contrast)ベクトルを
作る。「周囲組織と比べてどれだけ違うか」を直接ベクトル化するもので、
Hypertrophyのような周囲の正常組織との対比で判断される所見(README「所見の
3クラス分け」)に直接対応する仮説。

対象をGT対応7所見全部に広げているのは、所見によって効果の向きが逆転しうる
という予想のため——glycogen等の「パッチ内で完結するテクスチャ」所見は病変が
広範囲に連続することが多く、その場合「周囲」も同じ病変を含む組織になるので、
差分を取ると本物のシグナルまで消えて悪化する懸念がある。plain vs contrastの
差が所見の性質(局在性か拡散性か)と対応するかを見るのが本実験の主眼。

隣接パッチの座標探索: manifestのcoord_x/coord_y(level-0ピクセル座標)から
スライドごとにグリッド間隔を推定する(ネイティブ倍率がスライドごとに
異なりうるため224pxを決め打ちしない)。隣接が1枚も無い候補パッチ(組織端・
背景除去済み)はスキップする。

新規のUNI埋め込みは一切行わない——既存コーパス(baseline索引0018)のh5特徴量を
読むだけのCPU-onlyジョブ。サンプリング設定はexperiments/0035・0036と揃えた。

出力(outputs/0047_.../default/):
  loso_auroc_<finding_slug>_{plain,contrast}.csv   所見・表現ごとのLOSO AUROC(study別)
  summary.csv                                        所見ごとのplain/contrast比較サマリ
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
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _infer_grid_step(coords: np.ndarray) -> int:
    """coord_x(またはcoord_y)のユニーク値間の最小正の差分を、このスライドの
    パッチグリッド間隔(level-0ピクセル)とみなす。ネイティブ倍率がスライドに
    よって異なりうるため224pxを決め打ちしない。"""
    uniq = np.unique(coords)
    if len(uniq) < 2:
        raise ValueError("cannot infer grid step from a single coordinate")
    return int(np.diff(uniq).min())


def _context_vectors_for_rows(
    rows_df: pd.DataFrame,
    manifest_by_slide: dict[str, pd.DataFrame],
    features_dir: Path,
    logger: logging.Logger,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rows_df(サンプル済み候補パッチ: slide_id/local_idx/coord_x/coord_y)の
    各行について、同じスライドの空間的に隣接する(3x3グリッドの残り8マス)
    パッチ埋め込みの平均を求め、中心埋め込み(plain)と差分(contrast)を返す。

    manifest_by_slideはコーパス全体をslide_idでgroupby済みの辞書(近傍探索は
    rows_dfでサンプルされていない全パッチの中から行う——背景除去以外の理由で
    近傍が間引かれないようにするため)。

    Returns:
        (center_vecs [N,dim] L2正規化済み, contrast_vecs [N,dim] L2正規化済み,
         n_neighbors_per_row [N], slide_id_per_row [N])
    """
    from lib.ovr_scoring import _read_patch_vectors

    center_vecs, contrast_vecs, n_neighbors_list, slides_out = [], [], [], []
    n_skipped_no_neighbor = 0

    for slide_id, slide_rows in rows_df.groupby("slide_id"):
        slide_full = manifest_by_slide[slide_id]
        idx_to_coord = dict(zip(
            slide_full["local_idx"].to_numpy().tolist(),
            zip(slide_full["coord_x"].to_numpy().tolist(), slide_full["coord_y"].to_numpy().tolist()),
        ))
        coord_to_idx = {v: k for k, v in idx_to_coord.items()}
        try:
            step_x = _infer_grid_step(slide_full["coord_x"].to_numpy())
            step_y = _infer_grid_step(slide_full["coord_y"].to_numpy())
        except ValueError:
            continue  # このスライドはパッチ1枚しか無い(想定外だが安全にスキップ)

        local_idx = slide_rows["local_idx"].to_numpy().tolist()
        neighbor_map: dict[int, list[int]] = {}
        needed: set[int] = set()
        for li in local_idx:
            cx, cy = idx_to_coord[li]
            neighbors = [
                coord_to_idx[(cx + dx, cy + dy)]
                for dx in (-step_x, 0, step_x) for dy in (-step_y, 0, step_y)
                if not (dx == 0 and dy == 0) and (cx + dx, cy + dy) in coord_to_idx
            ]
            neighbor_map[li] = neighbors
            needed.update(neighbors)
            needed.add(li)

        needed_arr = np.array(sorted(needed), dtype=np.int64)
        raw_vecs = _l2_normalize(_read_patch_vectors(slide_id, needed_arr, features_dir))
        vec_by_idx = dict(zip(needed_arr.tolist(), raw_vecs))

        for li in local_idx:
            neighbors = neighbor_map[li]
            if not neighbors:
                n_skipped_no_neighbor += 1
                continue
            center = vec_by_idx[li]
            neighbor_mean = np.mean([vec_by_idx[n] for n in neighbors], axis=0)
            center_vecs.append(center)
            contrast_vecs.append(center - neighbor_mean)
            n_neighbors_list.append(len(neighbors))
            slides_out.append(slide_id)

    logger.info(
        f"[{label}] {len(center_vecs)} patches usable (>=1 neighbor), "
        f"{n_skipped_no_neighbor} skipped (0 neighbors), "
        f"mean n_neighbors="
        f"{(np.mean(n_neighbors_list) if n_neighbors_list else float('nan')):.2f}"
    )

    return (
        np.asarray(center_vecs, dtype=np.float32),
        _l2_normalize(np.asarray(contrast_vecs, dtype=np.float32)),
        np.asarray(n_neighbors_list),
        np.asarray(slides_out),
    )


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.ovr_scoring import load_gt_slides_for_finding, loso_auroc_by_study

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    seed = int(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    features_dir = project_root / config["features_dir"]
    gt_csv = project_root / config["gt_csv"]

    manifest = pd.read_parquet(project_root / config["manifest_path"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    logger.info(f"manifest loaded: {len(manifest)} patches, {manifest['slide_id'].nunique()} slides")

    logger.info("grouping manifest by slide_id (used for neighbor lookup across all findings)...")
    manifest_by_slide = {sid: g for sid, g in manifest.groupby("slide_id")}
    corpus_slides = set(manifest_by_slide.keys())

    summary_rows = []
    for target_finding in config["target_findings"]:
        rng = np.random.default_rng(seed)  # 所見ごとに同じ乱数系列から独立に開始(0035の流儀)
        logger.info(f"===== {target_finding} =====")

        gt = load_gt_slides_for_finding(target_finding, corpus_slides, gt_csv)
        if gt.empty:
            logger.warning(f"no corpus GT slides for {target_finding!r} — skipping")
            continue
        gt_slide_ids = set(gt["slide_id"])
        exp_id_of = dict(zip(gt["slide_id"], gt["EXP_ID"].astype(str)))
        n_exp_ids = gt["EXP_ID"].nunique()
        logger.info(f"GT slides: {len(gt)} (n_exp_ids={n_exp_ids}, "
                    f"n_compounds={gt['COMPOUND_NAME'].nunique()})")
        if n_exp_ids < 2:
            logger.warning(f"only {n_exp_ids} distinct EXP_ID — cannot LOSO, skipping "
                            "(same caveat as lib.ovr_scoring.train_and_evaluate_classifier)")
            continue

        # --- 正例: GTスライドから所見あたり最大max_positive_patches_per_slide ---
        pos_row_chunks = []
        for slide_id in sorted(gt_slide_ids):
            g = manifest_by_slide.get(slide_id)
            if g is None:
                continue
            local_idx = g["local_idx"].to_numpy()
            if len(local_idx) > config["max_positive_patches_per_slide"]:
                local_idx = rng.choice(local_idx, size=config["max_positive_patches_per_slide"], replace=False)
                pos_row_chunks.append(g[g["local_idx"].isin(local_idx)])
            else:
                pos_row_chunks.append(g)
        pos_rows = pd.concat(pos_row_chunks, ignore_index=True)
        pos_center, pos_contrast, pos_n_neighbors, pos_slide_ids = _context_vectors_for_rows(
            pos_rows, manifest_by_slide, features_dir, logger, label=f"{target_finding}/positive"
        )

        # --- 負例: コーパス全体(このGTスライド除く)からn_negative_patches ---
        neg_pool = manifest[~manifest["slide_id"].isin(gt_slide_ids)]
        n_neg = min(config["n_negative_patches"], len(neg_pool))
        neg_rows = neg_pool.iloc[np.sort(rng.choice(len(neg_pool), size=n_neg, replace=False))]
        neg_center, neg_contrast, neg_n_neighbors, _ = _context_vectors_for_rows(
            neg_rows, manifest_by_slide, features_dir, logger, label=f"{target_finding}/negative"
        )

        if len(pos_center) == 0 or len(neg_center) == 0:
            logger.warning(f"{target_finding!r}: 0 usable positive or negative patches — skipping")
            continue

        # --- train/test分割(負例、plain/contrastで同一インデックスを使い回す) ---
        perm = rng.permutation(len(neg_center))
        n_test = int(round(len(neg_center) * config["neg_test_fraction"]))
        test_idx, train_idx = perm[:n_test], perm[n_test:]

        finding_result = {
            "target_finding": target_finding,
            "n_positive_slides": int(gt["slide_id"].nunique()),
            "n_exp_ids": int(n_exp_ids),
            "n_positive_patches_usable": int(len(pos_center)),
            "n_negative_patches_usable": int(len(neg_center)),
            "mean_neighbors_positive": float(np.mean(pos_n_neighbors)),
            "mean_neighbors_negative": float(np.mean(neg_n_neighbors)),
        }
        for label, pos_vecs, neg_vecs in (
            ("plain", pos_center, neg_center),
            ("contrast", pos_contrast, neg_contrast),
        ):
            neg_train, neg_test = neg_vecs[train_idx], neg_vecs[test_idx]
            loso_df = loso_auroc_by_study(
                pos_vecs, pos_slide_ids, exp_id_of, neg_train, neg_test,
                C=config["classifier_C"], seed=seed,
            )
            loso_df.to_csv(run_dir / f"loso_auroc_{_slug(target_finding)}_{label}.csv", index=False)
            if len(loso_df):
                logger.info(f"[{target_finding}/{label}] LOSO AUROC median={loso_df['auroc'].median():.3f}, "
                            f"min={loso_df['auroc'].min():.3f}, max={loso_df['auroc'].max():.3f}, "
                            f"n_folds={len(loso_df)}")
            finding_result[f"{label}_median_auroc"] = float(loso_df["auroc"].median()) if len(loso_df) else None
            finding_result[f"{label}_min_auroc"] = float(loso_df["auroc"].min()) if len(loso_df) else None
            finding_result[f"{label}_max_auroc"] = float(loso_df["auroc"].max()) if len(loso_df) else None
            finding_result[f"{label}_n_folds"] = int(len(loso_df))

        summary_rows.append(finding_result)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
