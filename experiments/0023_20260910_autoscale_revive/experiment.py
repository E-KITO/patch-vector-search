"""experiments/0023: autoscale 復活 — centroid 細粒度化 + ガードバンド + atlas GT 再測定。

experiments/0022 の合成再スケール評価を受けて、棚上げされていた倍率補正
(lib.query_embedding.embed_image_tiles_auto_scale)を復活させられるかを検証する。
設計の詳細は config.yml のヘッダ参照。

1 回の GPU ジョブで:
  (a)        細かい scale centroid を再ビルド → run_dir/scale_centroids_fine.npz
  (a-val)    合成再スケールクエリ(検索なし)で 旧 vs 新 centroid の推定誤差を比較
  (b)+(c)    atlas GT 7 カテゴリを 4 アームで再測定

出力:
  outputs/0023_.../revive/scale_centroids_fine.npz
  outputs/0023_.../revive/estimator_check.csv
  outputs/0023_.../revive/atlas_gt.csv
  outputs/0023_.../revive/summary.md
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# NTP atlas カテゴリ -> 確定 GT FINDING_TYPE(scripts/validate_against_ground_truth.py と同一)。
CATEGORIES = {
    "Liver, Hepatocyte - Hypertrophy - Nonneoplastic Lesion Atlas": "Hypertrophy",
    "Liver - Necrosis - Nonneoplastic Lesion Atlas": "Single cell necrosis",
    "Liver, Hepatocyte – Increased Mitosis - Nonneoplastic Lesion Atlas": "Increased mitosis",
    "Liver, Hepatocyte - Glycogen Accumulation and Depletion - Nonneoplastic Lesion Atlas": "Deposit, glycogen",
    "Liver - Extramedullary Hematopoiesis - Nonneoplastic Lesion Atlas": "Hematopoiesis, extramedullary",
    "Liver, Kupffer Cell - Hyperplasia - Nonneoplastic Lesion Atlas": "Proliferation, Kupffer cell",
    "Liver, Hepatocyte - Cytoplasmic Inclusions - Nonneoplastic Lesion Atlas": "Inclusion body, intracytoplasmic",
}


def _get_project_root() -> Path:
    r = os.environ.get("PROJECT_ROOT")
    if not r:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(r)


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="config.yml")
    p.add_argument("--overwrite", action="store_true",
                   help="scale_centroids_fine.npz / 各 CSV があっても作り直す。")
    return p.parse_args()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0023")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        return Path(staged)
    return project_root / config_rel


# ============================================================
# ガード付き autoscale
# ============================================================

def embed_autoscale(img, centroids: dict, *, max_scale: float, guard_band: float | None):
    """embed_image_tiles_auto_scale と同じ手順(投票 → リサイズ → タイル埋め込み)だが、
    推定 scale が [1/guard_band, guard_band] のときは補正しない。

    Returns: (tiles (n,1024), applied_scale, raw_scale)
    """
    from PIL import Image

    from lib.mpp_estimation import estimate_relative_scale_fm_tiled
    from lib.query_embedding import embed_image_tiles

    vote_tiles = embed_image_tiles(img, tile_size=224)
    raw_scale, _ = estimate_relative_scale_fm_tiled(
        vote_tiles, centroids, tile_size=224, max_scale=max_scale
    )
    applied = raw_scale
    if guard_band is not None and abs(math.log(raw_scale)) < math.log(guard_band):
        applied = 1.0
    w, h = img.size
    scaled = img if applied == 1.0 else img.resize(
        (max(1, round(w * applied)), max(1, round(h * applied))), Image.LANCZOS
    )
    tiles = embed_image_tiles(scaled, tile_size=224)
    return tiles, applied, raw_scale


# ============================================================
# (a) centroid 再ビルド
# ============================================================

def build_fine_centroids(project_root: Path, index_dir: Path, cfg: dict,
                         run_dir: Path, overwrite: bool, logger: logging.Logger) -> dict:
    from lib.mpp_estimation import (build_scale_reference_centroids,
                                    load_scale_centroids, save_scale_centroids)

    out = run_dir / "scale_centroids_fine.npz"
    if out.exists() and not overwrite:
        logger.info("(a) fine centroids already exist -> %s", out)
        return load_scale_centroids(out)

    raw_dir = _staged_or(project_root, cfg["raw_slide_dir"], "PVS_RAW_SLIDE_DIR")
    scales = tuple(float(s) for s in cfg["centroid_scales"])
    logger.info("(a) building centroids for scales=%s (%d slides x %d samples)",
                scales, cfg["centroid_n_slides"], cfg["centroid_samples_per_slide"])
    centroids = build_scale_reference_centroids(
        raw_wsi_dir=raw_dir,
        slide_meta_path=index_dir / "slide_meta.parquet",
        scales=scales,
        n_slides=int(cfg["centroid_n_slides"]),
        samples_per_slide=int(cfg["centroid_samples_per_slide"]),
        seed=int(cfg["seed"]),
    )
    save_scale_centroids(centroids, out)
    logger.info("(a) wrote %s (%d scales)", out, len(centroids))
    return centroids


# ============================================================
# (a-validate) 合成再スケールクエリでの推定誤差比較(検索なし)
# ============================================================

def _sample_est_regions(index, cfg: dict, rng: np.random.Generator, logger: logging.Logger):
    grid = int(cfg["est_grid_tiles"])
    min_ratio = float(min(cfg["est_ratios"]))
    slide_meta = index.slide_meta
    manifest = index.manifest
    man_sid = manifest["slide_id"].astype(str)
    all_slides = slide_meta.index.astype(str).to_numpy()
    chosen = rng.choice(all_slides, size=min(int(cfg["est_n_slides"]), len(all_slides)), replace=False)

    regions = []
    for sid in sorted(chosen):
        meta = slide_meta.loc[sid]
        P = int(round(float(meta["patch_size_level0"])))
        W, H = int(meta["level0_width"]), int(meta["level0_height"])
        block = grid * P
        win_max = math.ceil(block / min_ratio)
        margin = (win_max - block) / 2.0
        lo, hix, hiy = margin, W - block - margin, H - block - margin
        if hix <= lo or hiy <= lo:
            continue
        cand = manifest[(man_sid == sid).to_numpy()
                        & (manifest["coord_x"] >= lo).to_numpy() & (manifest["coord_x"] <= hix).to_numpy()
                        & (manifest["coord_y"] >= lo).to_numpy() & (manifest["coord_y"] <= hiy).to_numpy()]
        if len(cand) == 0:
            continue
        take = cand.sample(n=min(int(cfg["est_regions_per_slide"]), len(cand)),
                           random_state=int(rng.integers(1 << 31)))
        for _, row in take.iterrows():
            regions.append((sid, int(row["coord_x"]), int(row["coord_y"]), P))
    logger.info("(a-val) %d synthetic regions across %d slides", len(regions), len(set(r[0] for r in regions)))
    return regions


def _read_window(slide, ax, ay, P, grid, ratio):
    from PIL import Image
    block = grid * P
    win = max(1, round(block / ratio))
    x0 = int(round(ax + block / 2.0 - win / 2.0))
    y0 = int(round(ay + block / 2.0 - win / 2.0))
    region = slide.read_region((x0, y0), 0, (win, win)).convert("RGB")
    return region.resize((grid * 224, grid * 224), Image.LANCZOS)


def validate_estimator(project_root: Path, index, orig_c: dict, fine_c: dict, cfg: dict,
                       run_dir: Path, overwrite: bool, logger: logging.Logger) -> pd.DataFrame:
    out = run_dir / "estimator_check.csv"
    if out.exists() and not overwrite:
        logger.info("(a-val) estimator_check.csv exists — skipping")
        return pd.read_csv(out)

    import openslide

    from lib.mpp_estimation import estimate_relative_scale_fm_tiled
    from lib.query_embedding import embed_image_tiles

    rng = np.random.default_rng(int(cfg["seed"]) + 1)
    regions = _sample_est_regions(index, cfg, rng, logger)
    raw_dir = _staged_or(project_root, cfg["raw_slide_dir"], "PVS_RAW_SLIDE_DIR")
    grid = int(cfg["est_grid_tiles"])
    max_scale = float(cfg["auto_scale_max"])

    rows = []
    for (sid, ax, ay, P) in regions:
        try:
            slide = openslide.OpenSlide(str(raw_dir / f"{sid}.svs"))
        except Exception as e:
            logger.warning("(a-val) open failed %s: %s", sid, e)
            continue
        try:
            for r in cfg["est_ratios"]:
                img = _read_window(slide, ax, ay, P, grid, r)
                vote = embed_image_tiles(img, tile_size=224)
                s_orig, _ = estimate_relative_scale_fm_tiled(vote, orig_c, tile_size=224, max_scale=max_scale)
                s_fine, _ = estimate_relative_scale_fm_tiled(vote, fine_c, tile_size=224, max_scale=max_scale)
                true_corr = 1.0 / r
                rows.append({
                    "slide_id": sid, "anchor_x": ax, "anchor_y": ay, "ratio": r,
                    "true_correction": true_corr,
                    "scale_orig": s_orig, "scale_fine": s_fine,
                    "abslogerr_orig": abs(math.log(max(s_orig, 1e-6)) - math.log(true_corr)),
                    "abslogerr_fine": abs(math.log(max(s_fine, 1e-6)) - math.log(true_corr)),
                })
        finally:
            slide.close()
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("(a-val) wrote %s (%d rows)", out, len(df))
    return df


# ============================================================
# (b)+(c) atlas GT 再測定
# ============================================================

def _rank_stats(ranked: pd.DataFrame, gt_slides: set) -> dict:
    ranked = ranked.reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    hits = ranked[ranked["slide_id"].astype(str).isin(gt_slides)]
    return {
        "found": int(len(hits)),
        "best": int(hits["rank"].min()) if len(hits) else None,
        "mean": round(float(hits["rank"].mean()), 1) if len(hits) else None,
    }


def atlas_gt(project_root: Path, index, orig_c: dict, fine_c: dict, cfg: dict,
             run_dir: Path, overwrite: bool, logger: logging.Logger) -> pd.DataFrame:
    out = run_dir / "atlas_gt.csv"
    if out.exists() and not overwrite:
        logger.info("(c) atlas_gt.csv exists — skipping")
        return pd.read_csv(out)

    from lib.atlas_figures import query_images
    from lib.query_embedding import embed_image_tiles

    atlas_dir = project_root / cfg["atlas_dir"]
    atlas_csv = project_root / "data/query/nnl_liver_atlas_figures.csv"
    gt = pd.read_csv(project_root / cfg["gt_csv"])
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    corpus_ids = set(index.slide_meta.index.astype(str))
    n_corpus = len(corpus_ids)
    kc, npr = int(cfg["k_candidates"]), int(cfg["nprobe"])
    gband = float(cfg["guard_band"])
    ms = float(cfg["auto_scale_max"])

    def _search(tiles, gt_slides):
        ranked = index.search_top_slides_multi(tiles, k_candidates=kc, nprobe=npr, top_n_slides=n_corpus)
        return _rank_stats(ranked, gt_slides)

    rows = []
    for cat_dir_name, finding_type in CATEGORIES.items():
        cat_dir = atlas_dir / cat_dir_name
        images = [str(p) for p in query_images(cat_dir, atlas_csv=atlas_csv)]
        if not images:
            logger.warning("(c) no images for %s — skipping", cat_dir_name)
            continue
        gt_slides = set(gt.loc[gt["FINDING_TYPE"] == finding_type, "slide_id"]) & corpus_ids
        label = cat_dir_name.split(" - ")[0][:28]
        row = {"category": label, "finding_type": finding_type,
               "n_images": len(images), "n_gt": len(gt_slides)}

        # baseline
        base_tiles = np.concatenate([embed_image_tiles(f, tile_size=224) for f in images], axis=0)
        s = _search(base_tiles, gt_slides)
        row.update({"baseline_found": s["found"], "baseline_best": s["best"], "baseline_mean": s["mean"]})

        # 3 つの autoscale アーム
        arms = {
            "autoscale_orig": (orig_c, None, 2.0),
            "autoscale_fine_guard": (fine_c, gband, ms),
            "autoscale_fine_noguard": (fine_c, None, ms),
        }
        for name, (cents, gb, mx) in arms.items():
            per_img_tiles, applied, raw = [], [], []
            for f in images:
                from PIL import Image
                t, ap, rw = embed_autoscale(Image.open(f).convert("RGB"), cents, max_scale=mx, guard_band=gb)
                per_img_tiles.append(t)
                applied.append(round(ap, 3))
                raw.append(round(rw, 3))
            s = _search(np.concatenate(per_img_tiles, axis=0), gt_slides)
            row.update({
                f"{name}_found": s["found"], f"{name}_best": s["best"], f"{name}_mean": s["mean"],
                f"{name}_applied": str(applied), f"{name}_raw": str(raw),
            })
        logger.info("(c) %-28s base_best=%s  orig=%s  fine_guard=%s  fine_noguard=%s",
                    label, row["baseline_best"], row["autoscale_orig_best"],
                    row["autoscale_fine_guard_best"], row["autoscale_fine_noguard_best"])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("(c) wrote %s (%d categories)", out, len(df))
    return df


# ============================================================
# サマリ
# ============================================================

def write_summary(est_df: pd.DataFrame, gt_df: pd.DataFrame, cfg: dict, run_dir: Path) -> None:
    L = ["# experiments/0023: autoscale 復活 — centroid 細粒度化 + ガードバンド + atlas GT 再測定", ""]

    L.append("## (a-val) 合成再スケールクエリでの scale 推定誤差 |log(scale_est / (1/r))| の median")
    L.append("")
    if len(est_df):
        L.append("| r | 1/r (真) | 旧 centroid | 新 centroid |")
        L.append("|---|---|---|---|")
        for r in cfg["est_ratios"]:
            sub = est_df[est_df["ratio"] == r]
            if not len(sub):
                continue
            L.append(f"| {r} | {1.0/r:.2f} | {sub['abslogerr_orig'].median():.2f} | "
                     f"{sub['abslogerr_fine'].median():.2f} |")
        L.append("")
        L.append(f"全体 median: 旧 {est_df['abslogerr_orig'].median():.3f} / "
                 f"新 {est_df['abslogerr_fine'].median():.3f}  "
                 f"(新 < 旧 なら細粒度化が推定を改善)")
    L.append("")

    L.append("## (c) atlas GT best_rank(7 カテゴリ、0018 索引、n_hits_ratio ランキング)")
    L.append("")
    cols = ["baseline", "autoscale_orig", "autoscale_fine_guard", "autoscale_fine_noguard"]
    L.append("| category | n_gt | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(cols) + 2))
    for _, row in gt_df.iterrows():
        cells = [str(row.get(f"{c}_best")) for c in cols]
        L.append(f"| {row['category']} | {row['n_gt']} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## (c) atlas GT found(GT スライドが top-1000 に何枚入ったか)")
    L.append("")
    L.append("| category | n_gt | " + " | ".join(cols) + " |")
    L.append("|" + "---|" * (len(cols) + 2))
    for _, row in gt_df.iterrows():
        cells = [str(row.get(f"{c}_found")) for c in cols]
        L.append(f"| {row['category']} | {row['n_gt']} | " + " | ".join(cells) + " |")
    L.append("")

    # 勝敗集計(best_rank、±5 は非決定性ノイズなので引き分け扱い)
    def wins(col):
        w = l = t = 0
        for _, row in gt_df.iterrows():
            b, x = row.get("baseline_best"), row.get(f"{col}_best")
            if b is None or x is None or (isinstance(b, float) and math.isnan(b)) or (isinstance(x, float) and math.isnan(x)):
                continue
            if x <= b - 6:
                w += 1
            elif x >= b + 6:
                l += 1
            else:
                t += 1
        return w, t, l

    L.append("## baseline に対する勝敗(best_rank、±5 位はノイズとして引き分け)")
    L.append("")
    L.append("| arm | 改善 | 引き分け | 悪化 |")
    L.append("|---|---|---|---|")
    for c in cols[1:]:
        w, t, ll = wins(c)
        L.append(f"| {c} | {w} | {t} | {ll} |")
    L.append("")
    L.append("## 読み方")
    L.append("")
    L.append("- **(a-val) で 新 < 旧** なら centroid 細粒度化は推定精度を上げている(前提条件)。")
    L.append("- **(c) で autoscale_fine_guard が baseline に対し 改善 >= 悪化** なら、"
             "棚上げ時の「7/7 で悪化」は推定器の誤射が主因で、補正原理は正しい "
             "→ embed_image_tiles_auto_scale に guard_band + 新 centroid を入れて復活。")
    L.append("- **fine_guard は勝てるが fine_noguard は負ける** なら、ガードバンドが効いている "
             "(r≈1 の誤補正を止めるのが要)。")
    L.append("- **fine_guard も負ける** なら、atlas GT の低迷は倍率でなくドメインギャップが主因で、"
             "倍率補正は atlas クエリには効かない(合成クエリの >2x 回復は in-distribution だから)。")
    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")


# ============================================================
# main
# ============================================================

def main() -> None:
    args = parse_args()
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent
    cfg = load_config(exp_dir)
    exp_name = os.environ["EXP_NAME"]

    from lib.mpp_estimation import load_scale_centroids
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.search import PatchIndex

    run_dir = get_run_dir(project_root, __file__, "revive", output_root=os.environ.get("OUTPUT_ROOT"))
    write_run_metadata(run_dir, exp_name=exp_name, guard_band=cfg["guard_band"])
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    index_dir = _staged_or(project_root, cfg["index_dir"], "PVS_INDEX_DIR")
    logger.info("index_dir: %s", index_dir)
    index = PatchIndex.load(
        index_path=index_dir / "index.faiss",
        manifest_path=index_dir / "manifest.parquet",
        slide_meta_path=index_dir / "slide_meta.parquet",
        features_dir=project_root / cfg["features_dir"],
    )
    orig_c = load_scale_centroids(project_root / cfg["orig_centroids"])
    logger.info("loaded index (%d slides) + orig centroids (%d scales)", len(index.slide_meta), len(orig_c))

    fine_c = build_fine_centroids(project_root, index_dir, cfg, run_dir, args.overwrite, logger)
    est_df = validate_estimator(project_root, index, orig_c, fine_c, cfg, run_dir, args.overwrite, logger)
    gt_df = atlas_gt(project_root, index, orig_c, fine_c, cfg, run_dir, args.overwrite, logger)

    write_summary(est_df, gt_df, cfg, run_dir)
    (run_dir / "curve.json").write_text(json.dumps({
        "estimator_median_abslogerr": {
            "orig": float(est_df["abslogerr_orig"].median()) if len(est_df) else None,
            "fine": float(est_df["abslogerr_fine"].median()) if len(est_df) else None,
        },
        "atlas_gt": gt_df.to_dict("records"),
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
