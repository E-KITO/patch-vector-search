"""experiments/0022: 合成再スケール検索評価。

クエリ画像の見かけ倍率がコーパス(20x)からずれたとき、検索がどれだけ劣化するかを
コーパス由来の合成クエリ(正解スライド既知)で測る。test-time スケール探索
(option C)に本当にターゲットがあるかの go / no-go 判定。

設計の詳細は config.yml のヘッダを参照。出力:
  outputs/0022_.../eval/results.csv   1 行 = スライド×領域×倍率比×アーム
  outputs/0022_.../eval/summary.md    アーム×r の median 順位 / found@k、autoscale 推定精度
  outputs/0022_.../eval/curve.json    サマリを機械可読で
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


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="config.yml")
    p.add_argument("--overwrite", action="store_true",
                   help="results.csv があっても最初からやり直す。")
    return p.parse_args()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0022")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    """NFS の元パスより、run_slurm.sh の PRE_NATIVE_COMMAND がノードローカル SSD に
    用意したステージコピー(env_var)を優先する。"""
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        return Path(staged)
    return project_root / config_rel


# ============================================================
# 合成クエリのサンプリング
# ============================================================

def sample_regions(index, cfg: dict, rng: np.random.Generator, logger: logging.Logger):
    """(slide_id, anchor_x, anchor_y, patch_size_level0) のリストを返す。

    アンカーは各スライドの実 corpus パッチ座標(manifest)から選び、
      - grid ブロック [anchor, anchor + grid*P] が最小 ratio でも WSI 内に収まる
        (r<1 では読み出し窓 grid*P/r がブロックより大きいため中心から張り出す)
    ように制約する。
    """
    grid = int(cfg["grid_tiles"])
    min_ratio = float(min(cfg["ratios"]))
    n_slides = int(cfg["n_slides"])
    per_slide = int(cfg["regions_per_slide"])

    slide_meta = index.slide_meta  # indexed by slide_id
    manifest = index.manifest      # indexed by global_idx, has slide_id/coord_x/coord_y
    man_sid = manifest["slide_id"].astype(str)  # 18M 行、一度だけ str 化

    all_slides = slide_meta.index.astype(str).to_numpy()
    chosen_slides = rng.choice(all_slides, size=min(n_slides, len(all_slides)), replace=False)

    regions = []
    for sid in sorted(chosen_slides):
        meta = slide_meta.loc[sid]
        P = int(round(float(meta["patch_size_level0"])))
        W = meta["level0_width"]
        H = meta["level0_height"]
        block = grid * P
        win_max = math.ceil(block / min_ratio)          # 最大読み出し窓(r 最小)
        margin = (win_max - block) / 2.0                 # ブロック外への張り出し
        lo_x, hi_x = margin, W - block - margin
        lo_y, hi_y = margin, H - block - margin
        if hi_x <= lo_x or hi_y <= lo_y:
            logger.info("skip %s: slide too small for grid=%d P=%d (need %d px window)",
                        sid, grid, P, win_max)
            continue

        cand = manifest[(man_sid == sid).to_numpy()
                        & (manifest["coord_x"] >= lo_x).to_numpy()
                        & (manifest["coord_x"] <= hi_x).to_numpy()
                        & (manifest["coord_y"] >= lo_y).to_numpy()
                        & (manifest["coord_y"] <= hi_y).to_numpy()]
        if len(cand) == 0:
            logger.info("skip %s: no anchor patch satisfies the window margin", sid)
            continue
        take = cand.sample(n=min(per_slide, len(cand)), random_state=int(rng.integers(1 << 31)))
        for _, row in take.iterrows():
            regions.append((sid, int(row["coord_x"]), int(row["coord_y"]), P))

    logger.info("sampled %d regions across %d slides (requested %d x %d)",
                len(regions), len(set(r[0] for r in regions)), n_slides, per_slide)
    return regions


def read_window(slide, anchor_x: int, anchor_y: int, P: int, grid: int, ratio: float):
    """倍率比 ratio で撮影されたように grid ブロックを読み、grid*224 px の
    PIL 画像にリサンプルして返す。ブロック中心を保ち、窓の一辺 = round(grid*P/ratio)。"""
    from PIL import Image

    block = grid * P
    cx = anchor_x + block / 2.0
    cy = anchor_y + block / 2.0
    win = max(1, round(block / ratio))
    x0 = int(round(cx - win / 2.0))
    y0 = int(round(cy - win / 2.0))
    region = slide.read_region((x0, y0), 0, (win, win)).convert("RGB")
    return region.resize((grid * 224, grid * 224), Image.LANCZOS)


# ============================================================
# アーム
# ============================================================

def run_arm(arm: str, img, ratio: float, index, centroids: dict, cfg: dict) -> dict:
    from PIL import Image

    from lib.query_embedding import embed_image_tiles, embed_image_tiles_auto_scale

    scale_est = float("nan")
    if arm == "fixed":
        vecs = embed_image_tiles(img, tile_size=224)
    elif arm == "oracle":
        side = max(1, round(img.size[0] / ratio))
        vecs = embed_image_tiles(img.resize((side, side), Image.LANCZOS), tile_size=224)
    elif arm == "autoscale":
        vecs, scale_est, _ = embed_image_tiles_auto_scale(
            img, centroids, tile_size=224, max_scale=cfg["auto_scale_max"]
        )
    else:
        raise ValueError(f"unknown arm: {arm!r}")

    n_corpus = len(index.slide_meta)
    ranked = index.search_top_slides_multi(
        vecs, k_candidates=int(cfg["k_candidates"]), nprobe=int(cfg["nprobe"]),
        top_n_slides=n_corpus,
    )
    return {"ranked": ranked, "scale_est": scale_est, "n_tiles": len(vecs)}


def rank_of(ranked: pd.DataFrame, slide_id: str) -> dict:
    ids = ranked["slide_id"].astype(str).tolist()
    if slide_id in ids:
        pos = ids.index(slide_id) + 1
        row = ranked.iloc[pos - 1]
        return {"rank": pos, "nhr": float(row["n_hits_ratio"]), "max_sim": float(row["max_similarity"])}
    return {"rank": None, "nhr": float("nan"), "max_sim": float("nan")}


# ============================================================
# サマリ
# ============================================================

def summarize(df: pd.DataFrame, cfg: dict) -> dict:
    ratios = list(cfg["ratios"])
    arms = list(cfg["arms"])
    n_corpus_rank_cap = 1000
    out = {"by_arm_ratio": [], "autoscale_estimator": []}
    for arm in arms:
        for r in ratios:
            sub = df[(df["arm"] == arm) & (df["ratio"] == r)]
            if len(sub) == 0:
                continue
            ranks = sub["rank"].to_numpy(dtype=float)
            found = ~np.isnan(ranks)
            ranks_filled = np.where(found, ranks, n_corpus_rank_cap)
            out["by_arm_ratio"].append({
                "arm": arm, "ratio": r, "n": int(len(sub)),
                "median_rank": float(np.median(ranks_filled)),
                "mean_rank": float(np.mean(ranks_filled)),
                "p90_rank": float(np.percentile(ranks_filled, 90)),
                "found_at_1": float(np.mean(ranks_filled <= 1)),
                "found_at_10": float(np.mean(ranks_filled <= 10)),
                "found_at_50": float(np.mean(ranks_filled <= 50)),
                "median_source_nhr": float(np.nanmedian(sub["source_nhr"].to_numpy(dtype=float))),
                "median_n_tiles": float(np.median(sub["n_tiles"].to_numpy(dtype=float))),
            })
            if arm == "autoscale":
                est = sub["scale_est"].to_numpy(dtype=float)
                true_corr = 1.0 / r
                logerr = np.log(np.clip(est, 1e-6, None)) - math.log(true_corr)
                out["autoscale_estimator"].append({
                    "ratio": r, "true_correction": true_corr,
                    "median_scale_est": float(np.nanmedian(est)),
                    "median_abs_log_err": float(np.nanmedian(np.abs(logerr))),
                    "frac_within_1.3x": float(np.nanmean(np.abs(logerr) <= math.log(1.3))),
                })
    return out


def write_summary(summary: dict, df: pd.DataFrame, cfg: dict, run_dir: Path) -> None:
    ratios = list(cfg["ratios"])
    arms = list(cfg["arms"])
    L = ["# experiments/0022: 合成再スケール検索評価", ""]
    L.append(f"合成クエリ {len(df) // max(1, len(arms) * len(ratios))} 領域 "
             f"({df['slide_id'].nunique()} スライド) × 倍率比 {ratios} × アーム {arms}。")
    L.append(f"grid={cfg['grid_tiles']}x{cfg['grid_tiles']}、索引 = {cfg['index_dir']}、"
             f"検索は search_top_slides_multi (n_hits_ratio ランキング)。")
    L.append("順位は正解スライドの順位。未 top-1000 は 1000 として集計。")
    L.append("")

    bar = {(d["arm"], d["ratio"]): d for d in summary["by_arm_ratio"]}
    L.append("## median 順位 (アーム × 倍率比 r)")
    L.append("")
    L.append("| arm | " + " | ".join(f"r={r}" for r in ratios) + " |")
    L.append("|" + "---|" * (len(ratios) + 1))
    for arm in arms:
        cells = [f"{bar[(arm, r)]['median_rank']:.0f}" if (arm, r) in bar else "—" for r in ratios]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## found@10 (アーム × 倍率比 r)")
    L.append("")
    L.append("| arm | " + " | ".join(f"r={r}" for r in ratios) + " |")
    L.append("|" + "---|" * (len(ratios) + 1))
    for arm in arms:
        cells = [f"{bar[(arm, r)]['found_at_10']:.2f}" if (arm, r) in bar else "—" for r in ratios]
        L.append(f"| {arm} | " + " | ".join(cells) + " |")
    L.append("")

    if summary["autoscale_estimator"]:
        L.append("## autoscale 推定器の精度 (scale_est vs 真の補正係数 1/r)")
        L.append("")
        L.append("| r | 1/r (真) | median scale_est | median |log err| | 1.3x 以内 |")
        L.append("|---|---|---|---|---|")
        for d in summary["autoscale_estimator"]:
            L.append(f"| {d['ratio']} | {d['true_correction']:.2f} | {d['median_scale_est']:.2f} | "
                     f"{d['median_abs_log_err']:.2f} | {d['frac_within_1.3x']:.2f} |")
        L.append("")

    L.append("## 読み方")
    L.append("")
    L.append("- **fixed @ r=1 の median 順位が ~1** ならサニティ OK (grid 整列タイルが near-exact マッチ)。")
    L.append("- **fixed が r=2 でも順位 <=5** → UNI はスケール頑健 → C は不要、倍率問題はクローズ。")
    L.append("- **fixed が r>=2 で大きく劣化し oracle が浅いまま** → スケール不整合は実在かつ補正可能 "
             "→ C にターゲットあり。autoscale 精度が点推定で足りるか探索が要るかを示す。")
    L.append("- **fixed も oracle も劣化** → リサンプル/FOV 欠落であってスケールでは直らない → C 不発。")
    L.append("- centroid スケールは {0.5,1,2,4,8}。r>1 (拡大クエリ) の補正係数 1/r<1 は "
             "centroid 範囲外なので autoscale は原理的に当てられない — C のスケールグリッドは <1 を含める必要がある。")
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

    output_root = os.environ.get("OUTPUT_ROOT")
    run_dir = get_run_dir(project_root, __file__, "eval", output_root=output_root)
    write_run_metadata(run_dir, exp_name=exp_name,
                       n_slides=cfg["n_slides"], ratios=str(cfg["ratios"]))
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
    centroids = load_scale_centroids(project_root / cfg["scale_centroids"])
    logger.info("loaded index (%d slides) + %d scale centroids",
                len(index.slide_meta), len(centroids))

    rng = np.random.default_rng(int(cfg["seed"]))
    regions = sample_regions(index, cfg, rng, logger)

    results_csv = run_dir / "results.csv"
    expected_per_region = len(cfg["ratios"]) * len(cfg["arms"])
    done = set()
    rows = []
    if results_csv.exists() and not args.overwrite:
        prev = pd.read_csv(results_csv)
        # 完全に揃った (slide,region) のみ done 扱い。途中で落ちた領域は
        # その行を捨てて丸ごと再実行する。
        counts = prev.groupby(["slide_id", "region_idx"]).size()
        full = {(str(s), int(r)) for (s, r), n in counts.items() if n >= expected_per_region}
        prev = prev[[(str(s), int(r)) in full
                     for s, r in zip(prev["slide_id"], prev["region_idx"])]]
        rows = prev.to_dict("records")
        done = full
        logger.info("resuming: %d complete regions kept, %d rows", len(done), len(rows))

    import openslide

    raw_dir = Path(cfg["raw_slide_dir"])
    raw_dir = project_root / raw_dir if not raw_dir.is_absolute() else raw_dir
    grid = int(cfg["grid_tiles"])
    ratios = list(cfg["ratios"])
    arms = list(cfg["arms"])

    # region_idx は「同一スライド内の何番目の領域か」で、resume キーの安定のため
    # スライドごとに 0..k で振り直す。
    per_slide_counter: dict[str, int] = {}
    for (sid, ax, ay, P) in regions:
        ridx = per_slide_counter.get(sid, 0)
        per_slide_counter[sid] = ridx + 1
        if (sid, ridx) in done:
            continue
        slide_path = raw_dir / f"{sid}.svs"
        try:
            slide = openslide.OpenSlide(str(slide_path))
        except Exception as e:
            logger.warning("open failed %s: %s", slide_path, e)
            continue
        try:
            for r in ratios:
                img = read_window(slide, ax, ay, P, grid, r)
                arm_ranks = {}
                for arm in arms:
                    res = run_arm(arm, img, r, index, centroids, cfg)
                    ro = rank_of(res["ranked"], sid)
                    arm_ranks[arm] = ro["rank"]
                    rows.append({
                        "slide_id": sid, "region_idx": ridx,
                        "anchor_x": ax, "anchor_y": ay, "patch_size_level0": P,
                        "ratio": r, "arm": arm,
                        "rank": ro["rank"], "source_nhr": ro["nhr"], "source_max_sim": ro["max_sim"],
                        "n_tiles": res["n_tiles"], "scale_est": res["scale_est"],
                    })
                logger.info("%s reg%d ratio=%.2f  %s", sid, ridx, r,
                            "  ".join(f"{a}:{arm_ranks[a]}" for a in arms))
        finally:
            slide.close()
        pd.DataFrame(rows).to_csv(results_csv, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(results_csv, index=False)
    summary = summarize(df, cfg)
    (run_dir / "curve.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    write_summary(summary, df, cfg, run_dir)
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
