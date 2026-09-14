"""experiments/0037: OvRタイル事前重み付けを、コーパスGTの無い所見も含む
NNLアトラス全25所見に展開する。

experiments/0035・0036は、コーパス内GT正例スライド(data/processed_csv/
single_finding_liver.csv)のパッチを正例にOvR分類器を学習していたため、GTが
確定している7所見にしか適用できなかった。NNLアトラス25所見の残り18所見には
コーパスGTが1件も無く、同じ枠組みが使えない。

この実験は正例の出所を「そのatlas図版自体のタイル」に変えることで(
lib.ovr_scoring.train_and_evaluate_classifier_from_atlas)、コーパスGTの有無に
関係なく25所見全部に同じ枠組みを適用する。検証もstudy単位のleave-one-study-out
(LOSO)から、atlas図版単位のleave-one-image-out(LOIO)に変更——「同じ所見の
他の図版から学習して未見の図版を当てられるか」という自己一貫性チェックであり、
コーパス上の実際の正解に紐づいた検証ではない点に注意(lib/ovr_scoring.pyの
loio_auroc_by_imageのdocstring、README「評価に使ったデータとその限界」参照)。
コーパスGTがある7所見については、0036のLOSO結果とこのLOIO結果を見比べることで、
「atlas図版を正例にする」という新しい手法自体の信頼性も間接的に確認できる。

出力(outputs/0037_.../default/):
  <finding_slug>/classifier/loio_auroc.csv   図版単位LOIO AUROC(図版2枚以上のみ)
  <finding_slug>/tile_score_heatmap/*.png    分類器スコアのタイルヒートマップ
  <finding_slug>/query__{unfiltered,ovr_filtered}/...  検索2腕
  summary.csv                                 25所見横並びの比較表(本命の出力)
"""

import json
import logging
import os
import re
import sys
import traceback
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


def process_folder(
    *, folder_name: str, images: list[str], project_root: Path, config: dict,
    manifest_str, corpus_slides: set[str], patch_index, run_dir: Path, params: dict, logger,
) -> dict:
    """1所見(=atlasフォルダ)分の分類器学習+目視診断+検索2腕を実行し、
    サマリ行(dict)を返す(experiments/0036のprocess_findingのatlas正例版)。"""
    from lib.ovr_scoring import (
        embed_tile_crops, load_gt_slides_for_finding, plot_classifier_tile_heatmap,
        run_query_arm, tile_grid_crops, train_and_evaluate_classifier_from_atlas,
    )
    from scripts.validate_against_ground_truth import CATEGORIES
    from PIL import Image as PILImage

    finding_dir = run_dir / _slug(folder_name)
    finding_dir.mkdir(exist_ok=True)

    # フォルダ名に対応するコーパスfinding_typeが分かっていれば(CATEGORIES)、
    # その既知GTスライドを負例サンプリングから除外する(本物が負例に混入するのを
    # 避ける)。対応が無い18所見はexclude_slidesが空集合になるだけで、処理は
    # 変わらず進む。
    corpus_finding_type = CATEGORIES.get(folder_name)
    exclude_slides: set[str] = set()
    if corpus_finding_type:
        gt = load_gt_slides_for_finding(corpus_finding_type, corpus_slides, project_root / config["gt_csv"])
        exclude_slides = set(gt["slide_id"])

    clf, loio_df, diagnostics = train_and_evaluate_classifier_from_atlas(
        project_root, config, manifest_str, images, folder_name, exclude_slides, logger,
    )
    diagnostics["corpus_finding_type"] = corpus_finding_type

    clf_dir = finding_dir / "classifier"
    clf_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(clf, clf_dir / "model.joblib")
    if loio_df is not None:
        loio_df.to_csv(clf_dir / "loio_auroc.csv", index=False)

    heatmap_dir = finding_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    tile_size = int(config.get("tile_size", 224))
    keep_frac = float(config["keep_frac"])

    all_kept_vecs, all_filtered_vecs, per_image_counts = [], [], []
    for image_path in images:
        pil_image = PILImage.open(image_path).convert("RGB")
        pil_image, origins, crops, n_blank = tile_grid_crops(pil_image, tile_size, tile_size)
        vecs = embed_tile_crops(crops, tile_size)
        scores = clf.decision_function(vecs)

        fig = plot_classifier_tile_heatmap(image_path, origins, scores, tile_size)
        fig.savefig(heatmap_dir / f"{_slug(Path(image_path).stem)}.png", dpi=150)
        import matplotlib.pyplot as plt
        plt.close(fig)

        n_keep = max(1, int(round(len(scores) * keep_frac)))
        keep_idx = np.argsort(scores)[::-1][:n_keep]
        all_kept_vecs.append(vecs)
        all_filtered_vecs.append(vecs[keep_idx])
        per_image_counts.append({
            "image": image_path, "n_tiles": len(scores), "n_kept": int(n_keep),
            "n_blank_excluded": n_blank,
            "score_min": float(scores.min()), "score_max": float(scores.max()),
        })
        logger.info(f"  {image_path}: {len(scores)} tiles, keeping top {n_keep} "
                    f"(score range {scores.min():.3f}..{scores.max():.3f})")

    unfiltered_vecs = np.concatenate(all_kept_vecs, axis=0)
    filtered_vecs = np.concatenate(all_filtered_vecs, axis=0)

    arm_results = {}
    for arm_label, vecs in (("unfiltered", unfiltered_vecs), ("ovr_filtered", filtered_vecs)):
        arm_results[arm_label] = run_query_arm(
            arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
            thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
            params=params, logger=logger,
        )

    finding_results = {
        "target_finding": folder_name, "keep_frac": keep_frac,
        "classifier_diagnostics": diagnostics, "per_image": per_image_counts,
        "arms": arm_results,
    }
    (finding_dir / "results.json").write_text(json.dumps(finding_results, indent=2, ensure_ascii=False))
    logger.info(f"[{folder_name}] unfiltered galleries: {arm_results['unfiltered']['n_galleries']}, "
                f"ovr_filtered galleries: {arm_results['ovr_filtered']['n_galleries']}")

    return {
        "target_finding": folder_name, "status": "ok",
        **diagnostics,
        "n_atlas_images_used": len(images),
        "n_tiles_unfiltered": arm_results["unfiltered"]["n_tiles"],
        "n_tiles_ovr_filtered": arm_results["ovr_filtered"]["n_tiles"],
        "n_galleries_unfiltered": arm_results["unfiltered"]["n_galleries"],
        "n_galleries_ovr_filtered": arm_results["ovr_filtered"]["n_galleries"],
        "top_slides_changed": arm_results["unfiltered"]["top_slide_ids"][:5]
        != arm_results["ovr_filtered"]["top_slide_ids"][:5],
    }


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.search import PatchIndex
    from lib.ovr_scoring import list_atlas_folders

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)} -> {[f for f, _ in folders]}")
    if not folders:
        raise SystemExit(f"atlas root {atlas_root} has no subfolder with images")

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, n_folders=len(folders))

    index_dir = project_root / config["index_dir"]
    patch_index = PatchIndex.load(
        index_path=index_dir / "index.faiss",
        manifest_path=index_dir / "manifest.parquet",
        slide_meta_path=index_dir / "slide_meta.parquet",
        features_dir=project_root / config["features_dir"],
    )
    manifest_str = patch_index.manifest.assign(slide_id=patch_index.manifest["slide_id"].astype(str))
    corpus_slides = set(manifest_str["slide_id"].unique())

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
    n_ok = n_fail = 0
    for folder_name, images in folders:
        logger.info(f"===== {folder_name} ({len(images)} images) =====")
        try:
            row = process_folder(
                folder_name=folder_name, images=images, project_root=project_root, config=config,
                manifest_str=manifest_str, corpus_slides=corpus_slides, patch_index=patch_index,
                run_dir=run_dir, params=params, logger=logger,
            )
        except Exception:
            traceback.print_exc()
            logger.error(f"!! folder FAILED: {folder_name}")
            row = {"target_finding": folder_name, "status": "failed"}
            n_fail += 1
        else:
            n_ok += 1
        rows.append(row)
        print(f"[{n_ok + n_fail}/{len(folders)}] {folder_name}: {row['status']}")

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary.to_string()}")

    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    logger.info(f"Done: {n_ok} ok, {n_fail} failed, of {len(folders)}.")

    complete_run(run_dir)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
