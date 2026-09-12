"""experiments/0035: OvR(One-vs-Rest)線形分類器によるタイル事前重み付けのプローブ。

背景(README「検索結果の可視化改善とタイル選択バイアスの発見」/「タイル選択バイアス」):
クエリ画像を全タイル分割して検索する際、近似FAISSスコアはコーパス内の出現頻度に
引っ張られ、ありふれた正常組織のタイルが珍しい所見のタイルより高スコアになる。
experiments/0011でmax_tiles_reranked=nullにして厳密re-rank側の選抜は無くしたが、
スライド集計(lib.search.PatchIndex.search_top_slides_multi)は依然として全タイルの
近似スコアをそのまま使っており、experiments/0034ではHypertrophyでbaseline/macenko
両方ともパッチギャラリーが0枚になった(GT best_rankの改善がパッチ単位の見え方を
保証しない実例)。

この実験は、GT正例(所見の確定スライド)vsコーパス全体ランダム負例で学習した軽量な
ロジスティック回帰(lib.ovr_scoring)でクエリタイルを事前スコアリングし、スコア上位
keep_frac割合のタイルだけを検索に回す("ovr_filtered"腕)ことで、near-FAISSスコアの
頻度バイアスを迂回できるかを見る。索引・埋め込みは常にbaseline(0018)に固定し、
「フィルタなし(unfiltered)」腕と直接比較する——finding_routingのwhiten/macenko軸とは
独立に、タイル選択という別の軸だけを検証する。

対象所見はHypertrophy固定(config.target_finding、コーパス内GTが25枚と最も潤沢、かつ
experiments/0034で最も強くタイル選択バイアスが疑われた所見)。

出力(outputs/0035_.../default/):
  classifier/loso_auroc.csv          study単位leave-one-study-out AUROC
  classifier/model.joblib            最終分類器(全正例+全負例で学習)
  tile_score_heatmap/<slug>.png      分類器スコアのタイルヒートマップ(所見ごとの図版)
  query__unfiltered/top_slides.csv, patch_gallery/*.png   フィルタなし腕
  query__ovr_filtered/top_slides.csv, patch_gallery/*.png タイル事前重み付け腕
  results.json                       AUROCサマリ・ギャラリー枚数などの比較サマリ
"""

import json
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
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


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.search import PatchIndex
    from lib.ovr_scoring import (
        embed_tile_crops, plot_classifier_tile_heatmap, resolve_atlas_images,
        run_query_arm, tile_grid_crops, train_and_evaluate_classifier,
    )

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    target_finding = config["target_finding"]

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, target_finding=target_finding)
    logger.info(f"target_finding={target_finding!r}")

    index_dir = project_root / config["index_dir"]
    patch_index = PatchIndex.load(
        index_path=index_dir / "index.faiss",
        manifest_path=index_dir / "manifest.parquet",
        slide_meta_path=index_dir / "slide_meta.parquet",
        features_dir=project_root / config["features_dir"],
    )

    # ── 1. OvR分類器の学習・batch-confound検証 ──────────────────────────────
    # scripts/random_patch_baseline.py / self_retrieval_diagnostic.py と同じ防御的
    # キャスト: manifest.parquetのslide_id列の実dtypeに依存せず、GT csv由来の
    # 文字列slide_idと確実に突き合わせられるようにする(patch_index.manifest本体は
    # 検索側の他メソッドが使うので、ここではコピーにキャストする)。
    manifest_str = patch_index.manifest.assign(slide_id=patch_index.manifest["slide_id"].astype(str))
    clf, loso_df, diagnostics = train_and_evaluate_classifier(
        project_root, config, manifest_str, target_finding, logger,
    )
    if clf is None:
        raise SystemExit(f"no corpus GT slides found for finding_type={target_finding!r}")
    clf_dir = run_dir / "classifier"
    clf_dir.mkdir(exist_ok=True)
    import joblib
    joblib.dump(clf, clf_dir / "model.joblib")
    if loso_df is not None:
        loso_df.to_csv(clf_dir / "loso_auroc.csv", index=False)

    # ── 2. atlas 図版を全タイル埋め込み + 分類器スコアのヒートマップ ──────────
    images = resolve_atlas_images(project_root, config, target_finding)
    logger.info(f"atlas images for {target_finding!r}: {len(images)}")
    if not images:
        raise SystemExit(f"no atlas images resolved for finding_type={target_finding!r}")

    heatmap_dir = run_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)

    tile_size = int(config.get("tile_size", 224))
    keep_frac = float(config["keep_frac"])

    all_kept_vecs = []      # unfiltered腕用(全タイル)
    all_filtered_vecs = []  # ovr_filtered腕用(スコア上位keep_fracのみ)
    per_image_counts = []

    from PIL import Image as PILImage

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
    logger.info(f"unfiltered total tiles: {unfiltered_vecs.shape[0]}, "
                f"ovr_filtered total tiles: {filtered_vecs.shape[0]}")

    # ── 3. 検索: unfiltered(=フィルタなし、baseline相当) vs ovr_filtered ────
    thumbnails_dir = project_root / config["baseline_thumbnails_dir"]
    params = {
        "raw_slide_dir": project_root / config["raw_slide_dir"],
        "nprobe": int(config.get("nprobe", 32)),
        "rerank_pool": int(config.get("rerank_pool", 200)),
        "max_tiles_reranked": config.get("max_tiles_reranked", None),
        "k_candidates": int(config.get("k_candidates", 8000)),
        "top_n_slides": int(config.get("top_n_slides", 20)),
        "top_n_slides_to_plot": int(config.get("top_n_slides_to_plot", 3)),
    }

    arm_results = {}
    for arm_label, vecs in (("unfiltered", unfiltered_vecs), ("ovr_filtered", filtered_vecs)):
        arm_results[arm_label] = run_query_arm(
            arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
            thumbnails_dir=thumbnails_dir, run_dir=run_dir, params=params, logger=logger,
        )

    # ── 4. まとめ ────────────────────────────────────────────────────────────
    results = {
        "target_finding": target_finding,
        "keep_frac": keep_frac,
        "classifier_diagnostics": diagnostics,
        "per_image": per_image_counts,
        "arms": arm_results,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"unfiltered galleries: {arm_results['unfiltered']['n_galleries']}, "
                f"ovr_filtered galleries: {arm_results['ovr_filtered']['n_galleries']}")

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
