"""experiments/0038: Atlas図版とTG-GATEsコーパスの間の「ドメインギャップ」を、
所見ラベルを無視した二値分類器(atlas由来 vs corpus由来)で直接定量化する。

背景: experiments/0037で、atlas図版のタイルを正例・コーパスのランダムパッチを
負例にした所見別OvR分類器のLOIO(leave-one-image-out)AUROCが、Hypertrophyを
含む25所見全てでほぼ1.0に飽和した。これは「所見を学習できている」からではなく、
正例(atlas由来)と負例(corpus由来)がそもそも画像ドメインとして分離しやすい
(スキャナ・染色・解像度・図版特有のノイズ等)せいではないか、という仮説が
浮上した(README「experiments/0037」参照)。

この実験はその仮説を直接検証する: 所見ラベルを完全に無視し、NNLアトラス25所見
全ての図版タイルを「atlas」クラス、コーパスからのランダムパッチを「corpus」
クラスとして単一のロジスティック回帰分類器(lib.ovr_scoring.train_logreg を
そのまま流用)を学習する。

検証は2種類:
  - LOFO(leave-one-finding-out、25fold): 24所見の図版で学習し、1所見の図版
    (未見のfindingかつ未見の視覚的スタイル)を「atlasかcorpusか」判定できるか。
    所見に依存しない汎用的な「atlasらしさ」信号が存在するかを直接測る
    (lib.ovr_scoring.loio_auroc_by_image を、グループ単位を画像でなく所見
    フォルダにして再利用——同関数はグループ化キーに依存しない実装のため
    そのまま使い回せる)。
  - LOIO(leave-one-image-out、全91図版プール): 参考として、所見の区別なく
    91図版全体でのLOIOも計算する(experiments/0037と同じ枠組みを所見横断に
    展開)。

加えて、コーパスGTが確定している所見については、その所見の**コーパス内GT
スライドのパッチ**(atlas図版ではなく、本物のTG-GATEs由来の陽性パッチ)を
このドメイン分類器でスコアし、atlas図版の同じ所見のスコア分布と比較する。
もしコーパスGTパッチが「corpus寄り」のスコアに収まる一方でatlas図版が
「atlas寄り」に張り付くなら、検索結果を歪めているのは所見の内容ではなく
画像ドメインそのものだという直接的な証拠になる。

出力(outputs/0038_.../default/):
  lofo_auroc_by_finding.csv    所見ごとのLOFO AUROC(25fold)
  loio_auroc_by_image.csv      図版ごとのLOIO AUROC(91fold)
  domain_scores_by_finding.csv 所見ごとのatlas図版タイルのドメインスコア分布
  gt_patch_domain_scores.csv   コーパスGT所見の、GTパッチのドメインスコア分布
  summary.json                 headline指標のまとめ
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
    from lib.ovr_scoring import (
        list_atlas_folders, embed_atlas_images_as_positives, sample_negative_pool,
        train_logreg, loio_auroc_by_image, load_gt_slides_for_finding, sample_patch_vectors,
    )
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

    features_dir = project_root / config["features_dir"]
    index_dir = project_root / config["index_dir"]
    tile_size = int(config.get("tile_size", 224))

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))
    corpus_slides = set(manifest["slide_id"].unique())

    # --- 1. atlas 25所見全ての図版タイルを埋め込み、所見ラベルを無視して1クラス
    #        (atlas)にプールする ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    pos_vecs_list, image_ids_list, folder_ids_list = [], [], []
    for folder_name, images in folders:
        vecs, image_ids = embed_atlas_images_as_positives(images, tile_size)
        pos_vecs_list.append(vecs)
        image_ids_list.append(image_ids)
        folder_ids_list.append(np.array([folder_name] * len(vecs)))
        logger.info(f"  {folder_name}: {len(images)} images -> {len(vecs)} tiles")

    pos_vecs = np.concatenate(pos_vecs_list, axis=0)
    pos_image_ids = np.concatenate(image_ids_list, axis=0)
    pos_folder_ids = np.concatenate(folder_ids_list, axis=0)
    logger.info(f"pooled atlas tiles: {pos_vecs.shape[0]} from {len(folders)} findings, "
                f"{len(set(pos_image_ids.tolist()))} images")

    # --- 2. コーパス全体からランダムに negative(corpus)クラスをサンプル ---
    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"],
        exclude_slides=set(), rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slide_ids))} slides")

    perm = rng.permutation(len(neg_vecs))
    n_test = int(round(len(neg_vecs) * config["neg_test_fraction"]))
    neg_test_vecs = neg_vecs[perm[:n_test]]
    neg_train_vecs = neg_vecs[perm[n_test:]]

    # --- 3. LOFO(所見横断の汎用ドメイン信号があるか)と LOIO(図版横断の
    #        自己一貫性)を計算。loio_auroc_by_image はグループ化キーに依存しない
    #        実装なので、フォルダ名を渡すだけでLOFOになる ---
    C = config["classifier_C"]
    seed = config.get("seed", 42)

    lofo_df = loio_auroc_by_image(pos_vecs, pos_folder_ids, neg_train_vecs, neg_test_vecs, C, seed)
    lofo_df = lofo_df.rename(columns={"held_out_image": "held_out_finding"})
    lofo_df.to_csv(run_dir / "lofo_auroc_by_finding.csv", index=False)
    logger.info(f"LOFO AUROC (median={lofo_df['auroc'].median():.3f}, "
                f"min={lofo_df['auroc'].min():.3f}, max={lofo_df['auroc'].max():.3f}, "
                f"n_folds={len(lofo_df)}):\n{lofo_df.to_string()}")

    loio_df = loio_auroc_by_image(pos_vecs, pos_image_ids, neg_train_vecs, neg_test_vecs, C, seed)
    loio_df.to_csv(run_dir / "loio_auroc_by_image.csv", index=False)
    logger.info(f"LOIO AUROC (median={loio_df['auroc'].median():.3f}, "
                f"min={loio_df['auroc'].min():.3f}, max={loio_df['auroc'].max():.3f}, "
                f"n_folds={len(loio_df)})")

    # --- 4. 全データで最終分類器を学習し、所見ごとのドメインスコア分布と、
    #        コーパスGTパッチのドメインスコア分布を比較する ---
    final_clf = train_logreg(pos_vecs, np.concatenate([neg_train_vecs, neg_test_vecs]), C=C, seed=seed)

    neg_scores = final_clf.decision_function(neg_vecs)
    logger.info(f"negative(corpus) pool domain score: mean={neg_scores.mean():.3f}, "
                f"median={np.median(neg_scores):.3f}")

    domain_rows = []
    for folder_name, images in folders:
        mask = pos_folder_ids == folder_name
        scores = final_clf.decision_function(pos_vecs[mask])
        domain_rows.append({
            "finding": folder_name, "n_tiles": int(mask.sum()),
            "domain_score_mean": float(scores.mean()), "domain_score_median": float(np.median(scores)),
            "domain_score_min": float(scores.min()), "domain_score_max": float(scores.max()),
            "frac_scored_as_corpus": float((scores < 0).mean()),
        })
    domain_df = pd.DataFrame(domain_rows).sort_values("domain_score_mean", ascending=False)
    domain_df.to_csv(run_dir / "domain_scores_by_finding.csv", index=False)
    logger.info(f"atlas domain scores by finding:\n{domain_df.to_string()}")

    gt_rows = []
    for folder_name, finding_type in CATEGORIES.items():
        if not any(f == folder_name for f, _ in folders):
            continue
        gt = load_gt_slides_for_finding(finding_type, corpus_slides, project_root / config["gt_csv"])
        if gt.empty:
            continue
        gt_vecs, gt_slide_ids = sample_patch_vectors(
            gt["slide_id"].tolist(), manifest, features_dir,
            max_per_slide=config["max_positive_patches_per_slide"], rng=rng,
        )
        if gt_vecs.shape[0] == 0:
            continue
        gt_scores = final_clf.decision_function(gt_vecs)
        atlas_scores = final_clf.decision_function(pos_vecs[pos_folder_ids == folder_name])
        gt_rows.append({
            "finding": finding_type, "atlas_folder": folder_name,
            "n_gt_slides": int(gt["slide_id"].nunique()), "n_gt_patches": int(gt_vecs.shape[0]),
            "gt_domain_score_mean": float(gt_scores.mean()), "gt_domain_score_median": float(np.median(gt_scores)),
            "atlas_domain_score_mean": float(atlas_scores.mean()),
            "gap_atlas_minus_gt": float(atlas_scores.mean() - gt_scores.mean()),
        })
        logger.info(f"  GT patches for {finding_type!r}: domain score mean={gt_scores.mean():.3f} "
                    f"vs atlas mean={atlas_scores.mean():.3f}")
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(run_dir / "gt_patch_domain_scores.csv", index=False)

    summary = {
        "n_atlas_findings": len(folders),
        "n_atlas_images": len(set(pos_image_ids.tolist())),
        "n_atlas_tiles": int(pos_vecs.shape[0]),
        "n_negative_patches": int(neg_vecs.shape[0]),
        "lofo_median_auroc": float(lofo_df["auroc"].median()),
        "lofo_min_auroc": float(lofo_df["auroc"].min()),
        "lofo_max_auroc": float(lofo_df["auroc"].max()),
        "lofo_n_folds": int(len(lofo_df)),
        "loio_median_auroc": float(loio_df["auroc"].median()),
        "loio_min_auroc": float(loio_df["auroc"].min()),
        "loio_max_auroc": float(loio_df["auroc"].max()),
        "loio_n_folds": int(len(loio_df)),
        "negative_pool_score_mean": float(neg_scores.mean()),
        "n_gt_findings_compared": len(gt_rows),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary:\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
