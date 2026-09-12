"""experiments/0036: OvR(One-vs-Rest)タイル事前重み付けを複数所見で再試行。

experiments/0035(Hypertrophy単体)で、①study単位LOSO AUROCの分散が異常に大きく
(0.10〜0.92)分類器がバッチを学習している疑いが強い、②タイル事前フィルタを掛けても
パッチギャラリーは0枚のままだった、という否定的な結果が出た。これがHypertrophyという
所見固有の限界(相対的・びまん性でパッチ単位の識別特徴を持たない、README「所見の3クラス
分け」)なのか、OvR手法そのものの限界なのかは1所見だけでは切り分けられない。

この実験は同じ仕組み(lib.ovr_scoring)を、atlas図版が存在する所見
(scripts.validate_against_ground_truth.CATEGORIES、config.target_findings)全部に
展開し、所見ごとにLOSO AUROCの安定性とギャラリー生成数を横並びで比較する。1所見の
GT不足・atlas図版欠如・想定外エラーが他所見の実行を止めないよう、所見ごとに
try/exceptする(experiments/0034のfindingsループと同じ構造)。

出力(outputs/0036_.../default/):
  <finding_slug>/classifier/...             所見ごとの分類器・LOSO AUROC
  <finding_slug>/tile_score_heatmap/*.png   所見ごとのタイルヒートマップ
  <finding_slug>/query__{unfiltered,ovr_filtered}/...  所見ごとの検索腕
  summary.csv                                全所見横並びの比較表(本命の出力)
  results.json
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


def process_finding(
    *, target_finding: str, project_root: Path, config: dict, manifest_str, patch_index,
    run_dir: Path, params: dict, logger,
) -> dict:
    """1所見分の分類器学習+目視診断+検索2腕を実行し、サマリ行(dict)を返す。

    GT不足・atlas図版欠如は例外を投げず status="skipped" を返す(想定内)。
    それ以外の失敗はここでは握りつぶさず呼び出し側(main のループ)の
    try/except に委ねる——本当の想定外バグを skipped と区別するため。
    """
    from lib.ovr_scoring import (
        embed_tile_crops, plot_classifier_tile_heatmap, resolve_atlas_images,
        run_query_arm, tile_grid_crops, train_and_evaluate_classifier,
    )
    from PIL import Image as PILImage

    finding_dir = run_dir / _slug(target_finding)
    finding_dir.mkdir(exist_ok=True)

    clf, loso_df, diagnostics = train_and_evaluate_classifier(
        project_root, config, manifest_str, target_finding, logger,
    )
    if clf is None:
        return {"target_finding": target_finding, "status": "skipped",
                "skip_reason": "no corpus GT slides", **diagnostics}

    clf_dir = finding_dir / "classifier"
    clf_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(clf, clf_dir / "model.joblib")
    if loso_df is not None:
        loso_df.to_csv(clf_dir / "loso_auroc.csv", index=False)

    images = resolve_atlas_images(project_root, config, target_finding)
    logger.info(f"atlas images for {target_finding!r}: {len(images)}")
    if not images:
        return {"target_finding": target_finding, "status": "skipped",
                "skip_reason": "no atlas images resolved", **diagnostics}

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
        "target_finding": target_finding, "keep_frac": keep_frac,
        "classifier_diagnostics": diagnostics, "per_image": per_image_counts,
        "arms": arm_results,
    }
    (finding_dir / "results.json").write_text(json.dumps(finding_results, indent=2, ensure_ascii=False))
    logger.info(f"[{target_finding}] unfiltered galleries: {arm_results['unfiltered']['n_galleries']}, "
                f"ovr_filtered galleries: {arm_results['ovr_filtered']['n_galleries']}")

    return {
        "target_finding": target_finding, "status": "ok",
        **diagnostics,
        "n_atlas_images": len(images),
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

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    target_findings = config["target_findings"]

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key,
                        n_target_findings=len(target_findings))
    logger.info(f"target_findings ({len(target_findings)}): {target_findings}")

    index_dir = project_root / config["index_dir"]
    patch_index = PatchIndex.load(
        index_path=index_dir / "index.faiss",
        manifest_path=index_dir / "manifest.parquet",
        slide_meta_path=index_dir / "slide_meta.parquet",
        features_dir=project_root / config["features_dir"],
    )
    # scripts/random_patch_baseline.py / self_retrieval_diagnostic.py と同じ防御的
    # キャスト(experiments/0035参照)。全所見で使い回すので一度だけ計算する。
    manifest_str = patch_index.manifest.assign(slide_id=patch_index.manifest["slide_id"].astype(str))

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
    n_ok = n_skip = n_fail = 0
    for target_finding in target_findings:
        logger.info(f"===== {target_finding} =====")
        try:
            row = process_finding(
                target_finding=target_finding, project_root=project_root, config=config,
                manifest_str=manifest_str, patch_index=patch_index, run_dir=run_dir,
                params=params, logger=logger,
            )
        except Exception:
            traceback.print_exc()
            logger.error(f"!! finding FAILED: {target_finding}")
            row = {"target_finding": target_finding, "status": "failed"}
            n_fail += 1
        else:
            n_ok += row["status"] == "ok"
            n_skip += row["status"] == "skipped"
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary.to_string()}")

    (run_dir / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str))
    logger.info(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed, of {len(target_findings)}.")

    complete_run(run_dir)
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
