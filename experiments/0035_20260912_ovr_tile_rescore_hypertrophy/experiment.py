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


def resolve_atlas_images(project_root: Path, config: dict, target_finding: str) -> list[str]:
    """target_finding にマッピングされる atlas 図版フォルダの画像パスをすべて返す
    (experiments/0034 と同じ CATEGORIES 経由の解決)。"""
    sys.path.insert(0, str(project_root))
    from lib.atlas_figures import query_images
    from scripts.validate_against_ground_truth import CATEGORIES

    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]

    images: list[str] = []
    for folder_name, finding_type in CATEGORIES.items():
        if finding_type != target_finding:
            continue
        cat_dir = atlas_root / folder_name
        images.extend(str(p) for p in query_images(cat_dir, atlas_csv=atlas_csv))
    return images


def train_and_evaluate_classifier(project_root, config, manifest, logger):
    """GT正例 vs コーパス全体ランダム負例で分類器を学習し、study単位LOSO AUROCで
    「所見」と「化合物・studyというバッチ」のどちらを学習しているかを確認する。

    Returns:
        (final_classifier, loso_df, diagnostics: dict)
    """
    from lib.ovr_scoring import (
        load_gt_slides_for_finding, sample_negative_pool, sample_patch_vectors,
        train_logreg, loso_auroc_by_study,
    )

    rng = np.random.default_rng(config.get("seed", 42))
    features_dir = project_root / config["features_dir"]
    target_finding = config["target_finding"]

    corpus_slides = set(manifest["slide_id"].astype(str).unique())
    gt = load_gt_slides_for_finding(target_finding, corpus_slides, project_root / config["gt_csv"])
    if gt.empty:
        raise SystemExit(f"no corpus GT slides found for finding_type={target_finding!r}")
    logger.info(f"GT slides for {target_finding!r}: {len(gt)} "
                f"(n_compounds={gt['COMPOUND_NAME'].nunique()}, n_exp_ids={gt['EXP_ID'].nunique()})")

    pos_vecs, pos_slide_ids = sample_patch_vectors(
        gt["slide_id"].tolist(), manifest, features_dir,
        max_per_slide=config["max_positive_patches_per_slide"], rng=rng,
    )
    logger.info(f"positive patches sampled: {pos_vecs.shape[0]} from {gt['slide_id'].nunique()} slides")

    neg_vecs, neg_slide_ids = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"],
        exclude_slides=set(gt["slide_id"]), rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slide_ids))} slides")

    # 負例をLOSO専用のtrain/testに一度だけ分割(train/test漏洩を避けるため全フォールドで
    # 使い回す。lib.ovr_scoring.loso_auroc_by_study のdocstring参照)。
    perm = rng.permutation(len(neg_vecs))
    n_test = int(round(len(neg_vecs) * config["neg_test_fraction"]))
    neg_test_vecs = neg_vecs[perm[:n_test]]
    neg_train_vecs = neg_vecs[perm[n_test:]]

    exp_id_of = dict(zip(gt["slide_id"], gt["EXP_ID"].astype(str)))
    n_exp_ids = gt["EXP_ID"].nunique()
    if n_exp_ids >= 2:
        loso_df = loso_auroc_by_study(
            pos_vecs, pos_slide_ids, exp_id_of, neg_train_vecs, neg_test_vecs,
            C=config["classifier_C"], seed=config.get("seed", 42),
        )
        logger.info(f"LOSO AUROC (median={loso_df['auroc'].median():.3f}, "
                    f"n_folds={len(loso_df)}):\n{loso_df}")
    else:
        loso_df = None
        logger.warning(
            f"only {n_exp_ids} distinct EXP_ID among {target_finding!r} GT slides — "
            "cannot leave-one-study-out; AUROC would be indistinguishable from "
            "memorizing that single study (see README self_retrieval_diagnostic "
            "n_exp_ids caveat). Skipping LOSO."
        )

    final_clf = train_logreg(pos_vecs, np.concatenate([neg_train_vecs, neg_test_vecs]),
                              C=config["classifier_C"], seed=config.get("seed", 42))

    diagnostics = {
        "target_finding": target_finding,
        "n_positive_patches": int(pos_vecs.shape[0]),
        "n_positive_slides": int(gt["slide_id"].nunique()),
        "n_compounds": int(gt["COMPOUND_NAME"].nunique()),
        "n_exp_ids": int(n_exp_ids),
        "n_negative_patches": int(neg_vecs.shape[0]),
        "n_negative_slides": int(len(set(neg_slide_ids))),
        "loso_median_auroc": float(loso_df["auroc"].median()) if loso_df is not None and len(loso_df) else None,
        "loso_n_folds": int(len(loso_df)) if loso_df is not None else 0,
    }
    return final_clf, loso_df, diagnostics


def _tile_grid_crops(pil_image, tile_size: int, crop_size: int):
    """lib.query_embedding.embed_image_tiles / lib.visualize.plot_query_tile_scores
    と同一のグリッド分割・空白タイル除外ロジック(座標付きで返す必要があるため、
    plot_query_tile_scores と同じ理由で複製する——そちらのdocstring参照)。"""
    from lib.query_embedding import _is_blank_tile
    from PIL import Image

    w, h = pil_image.size
    scale = max(crop_size / w, crop_size / h, 1.0)
    if scale > 1.0:
        pil_image = pil_image.resize(
            (max(crop_size, round(w * scale)), max(crop_size, round(h * scale))), Image.LANCZOS
        )
        w, h = pil_image.size

    xs = sorted(set(list(range(0, w - crop_size + 1, crop_size)) + [w - crop_size]))
    ys = sorted(set(list(range(0, h - crop_size + 1, crop_size)) + [h - crop_size]))
    origins = [(x, y) for y in ys for x in xs]
    crops = [pil_image.crop((x, y, x + crop_size, y + crop_size)) for x, y in origins]

    blank = [_is_blank_tile(c) for c in crops]
    if all(blank):
        blank = [False] * len(blank)

    kept_origins = [o for o, b in zip(origins, blank) if not b]
    kept_crops = [c for c, b in zip(crops, blank) if not b]
    if crop_size != tile_size:
        kept_crops = [c.resize((tile_size, tile_size), Image.LANCZOS) for c in kept_crops]
    return pil_image, kept_origins, kept_crops, sum(blank)


def embed_crops(crops, tile_size: int) -> np.ndarray:
    import torch
    from torchvision import transforms
    from lib.query_embedding import _load_encoder, _normalize_rows

    encoder, device, dtype = _load_encoder(None)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    batch = torch.stack([to_tensor(c) for c in crops]).to(device=device, dtype=dtype)
    with torch.no_grad():
        embeddings = encoder(batch).float().cpu().numpy().astype(np.float32)
    return _normalize_rows(embeddings)


def plot_classifier_tile_heatmap(image_path: str, origins, crops, scores: np.ndarray, tile_size: int):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize
    from lib.query_embedding import _load_rgb

    pil_image = _load_rgb(image_path)
    w, h = pil_image.size
    scale = max(tile_size / w, tile_size / h, 1.0)
    if scale > 1.0:
        from PIL import Image
        pil_image = pil_image.resize((max(tile_size, round(w * scale)), max(tile_size, round(h * scale))), Image.LANCZOS)
        w, h = pil_image.size

    cmap = plt.get_cmap("magma")
    norm = Normalize(vmin=float(scores.min()), vmax=float(scores.max()))

    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(pil_image)
    for (x, y), score in zip(origins, scores):
        ax.add_patch(mpatches.Rectangle(
            (x, y), tile_size, tile_size, facecolor=cmap(norm(score)), alpha=0.5, edgecolor="none",
        ))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="OvR classifier decision_function score", shrink=0.7)
    ax.set_title("Per-tile OvR classifier score (higher = more finding-like)", fontsize=9)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    fig.tight_layout()
    return fig


def run_query_arm(
    *, arm_label: str, query_vecs: np.ndarray, patch_index, thumbnails_dir: Path,
    run_dir: Path, params: dict, logger,
):
    """1つの腕(unfiltered / ovr_filtered)について検索・可視化を実行する
    (experiments/0034 の process_query_set と同じ構造)。"""
    from lib.visualize import plot_hit_patch_gallery, plot_slide_hits_on_thumbnail
    import matplotlib.pyplot as plt

    arm_dir = run_dir / f"query__{arm_label}"
    arm_dir.mkdir(exist_ok=True)
    gallery_dir = arm_dir / "patch_gallery"
    gallery_dir.mkdir(exist_ok=True)
    plots_dir = arm_dir / "thumbnail_plots"
    plots_dir.mkdir(exist_ok=True)

    top_slides = patch_index.search_top_slides_multi(
        query_vecs, k_candidates=params["k_candidates"], nprobe=params["nprobe"],
        top_n_slides=params["top_n_slides"],
    )
    top_slides.to_csv(arm_dir / "top_slides.csv", index=False)
    logger.info(f"[{arm_label}] top slides:\n{top_slides}")

    hit_pool = patch_index.search_similar_patches_multi(
        query_vecs, k=params["rerank_pool"], nprobe=params["nprobe"],
        rerank_pool=params["rerank_pool"], max_tiles_reranked=params["max_tiles_reranked"],
    )
    n_gal = 0
    for slide_id in top_slides["slide_id"].head(params["top_n_slides_to_plot"]):
        hits = hit_pool[hit_pool["slide_id"] == slide_id]
        if hits.empty:
            continue
        try:
            fig = plot_slide_hits_on_thumbnail(slide_id, hits, thumbnails_dir, patch_index.slide_meta)
            fig.savefig(plots_dir / f"{slide_id}.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"[{arm_label}] thumbnail plot failed for slide {slide_id}: {e!r}")
        try:
            fig = plot_hit_patch_gallery(hits, params["raw_slide_dir"], patch_index.slide_meta)
            fig.savefig(gallery_dir / f"{slide_id}.png", dpi=150)
            plt.close(fig)
            n_gal += 1
        except Exception as e:
            logger.warning(f"[{arm_label}] gallery failed for slide {slide_id}: {e!r}")
    logger.info(f"[{arm_label}] galleries written: {n_gal}")
    return {
        "n_tiles": int(query_vecs.shape[0]),
        "n_top_slides": int(len(top_slides)),
        "top_slide_ids": top_slides["slide_id"].astype(str).tolist(),
        "n_galleries": n_gal,
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
        project_root, config, manifest_str, logger,
    )
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
        pil_image, origins, crops, n_blank = _tile_grid_crops(pil_image, tile_size, tile_size)
        vecs = embed_crops(crops, tile_size)
        scores = clf.decision_function(vecs)

        fig = plot_classifier_tile_heatmap(image_path, origins, crops, scores, tile_size)
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
