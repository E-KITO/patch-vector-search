"""Contact sheets for choosing the corpus background-exclusion threshold (stage 1.5).

scripts/measure_corpus_blankness.py measured mean_intensity / std_intensity /
sat_frac for every corpus patch -> outputs/measure_corpus_blankness/corpus_blankness.parquet
(job 10487, 0 failures). The provisional cut lib.patch_blankness.is_background
uses is `sat_frac < 0.10 & mean_intensity > 215` -- ~2.11% of the corpus,
~388k patches. Before freezing that into experiments/0017's manifest filter,
look at what it actually drops and what sits just above it.

Two families of sheets, written to outputs/measure_corpus_blankness/audit/:

  sat_ladder/bin_*.png -- patches binned by sat_frac, restricted to the bright
    ones (mean_intensity > --ladder-mean-min, where sat_frac is the deciding
    variable). The 0.10 boundary is where "drop" turns into "keep": bins below
    it should be all slide background; bins just above it should be real but
    sparse tissue (sinusoidal dilation, oedema, early necrosis with cell
    dropout). If real tissue shows up below 0.10, the cut is too aggressive.

  mean_guard/bin_*.png -- patches with sat_frac < 0.10 binned by mean_intensity,
    to sanity-check the mean>215 guard: are the 195..215 patches just background
    in shadow (the guard is dropping real background it shouldn't) or ink /
    fold / knife-mark (a dark-patch problem the guard correctly sidesteps)?

sampled.csv lists every sampled patch (slide_id, local_idx, coord_x, coord_y,
mean_intensity, std_intensity, sat_frac, sheet, bin) so a decision can cite
specific patches. Re-running with the same --seed samples the same patches.

Usage (adhoc script wraps this; needs the raw WSI on NFS):
    python scripts/audit_blank_threshold.py
    python scripts/audit_blank_threshold.py --per-bin 60 --seed 7
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.raw_patch import crop_patch  # noqa: E402

CORPUS_INDEX_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")
BLANKNESS_PARQUET = Path("outputs/measure_corpus_blankness/corpus_blankness.parquet")
OUT_DIR = Path("outputs/measure_corpus_blankness/audit")


def _bin_samples(df: pd.DataFrame, col: str, edges: list[float], per_bin: int, seed: int) -> list[tuple[str, pd.DataFrame]]:
    """-> [(bin_label, sampled_rows_sorted_by_col), ...] for [edges[i], edges[i+1])."""
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = df[(df[col] >= lo) & (df[col] < hi)]
        n = min(per_bin, len(pool))
        take = pool.sample(n=n, random_state=int(rng.integers(1 << 31))) if n else pool
        out.append((f"{lo:g}-{hi:g}__pool{len(pool)}", take.sort_values(col)))
    return out


def _crop_one(job):
    slide_id, cx, cy, psl = job
    try:
        return np.asarray(crop_patch(slide_id, int(cx), int(cy), RAW_SLIDE_DIR, int(psl)))
    except Exception as e:  # keep the sheet, mark the cell
        return RuntimeError(f"{slide_id}:{e!r}")


def _render(rows: pd.DataFrame, images: list, out_png: Path, title: str, n_cols: int) -> None:
    n = len(rows)
    if n == 0:
        print(f"  (empty) {out_png.name}")
        return
    n_rows = -(-n // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.7, n_rows * 1.95), squeeze=False)
    axes = axes.flatten()
    for ax, (_, r), im in zip(axes, rows.iterrows(), images):
        if isinstance(im, Exception):
            ax.text(0.5, 0.5, str(im), ha="center", va="center", fontsize=5, wrap=True)
        else:
            ax.imshow(im)
        ax.set_title(f"s={r.sat_frac:.3f}  m={r.mean_intensity:.0f}\n{r.slide_id}:{int(r.local_idx)}", fontsize=6)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}  ({n} patches)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-bin", type=int, default=45, help="patches sampled per bin")
    ap.add_argument("--n-cols", type=int, default=9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4))
    ap.add_argument("--sat-edges", type=str, default="0,0.02,0.05,0.10,0.15,0.25,0.40")
    ap.add_argument("--ladder-mean-min", type=float, default=200.0,
                    help="sat_frac ladder only samples patches brighter than this "
                    "(where sat_frac, not brightness, is what decides)")
    ap.add_argument("--mean-edges", type=str, default="0,150,195,215,240,256")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sat_edges = [float(x) for x in args.sat_edges.split(",")]
    mean_edges = [float(x) for x in args.mean_edges.split(",")]

    blank = pd.read_parquet(BLANKNESS_PARQUET)
    blank["slide_id"] = blank["slide_id"].astype(str)
    print(f"blankness: {len(blank):,} patches", flush=True)

    # sheet A: sat_frac ladder (bright patches only)
    ladder = _bin_samples(
        blank[blank["mean_intensity"] > args.ladder_mean_min], "sat_frac", sat_edges, args.per_bin, args.seed
    )
    # sheet B: mean-intensity guard, within sat_frac < 0.10
    guard = _bin_samples(blank[blank["sat_frac"] < 0.10], "mean_intensity", mean_edges, args.per_bin, args.seed + 1)

    tagged = []
    for sheet, blocks in (("sat_ladder", ladder), ("mean_guard", guard)):
        for label, sub in blocks:
            s = sub.copy()
            s["sheet"], s["bin"] = sheet, label
            tagged.append(s)
    sampled = pd.concat(tagged, ignore_index=True) if tagged else pd.DataFrame()
    if sampled.empty:
        print("nothing sampled -- check the parquet / edges")
        return

    manifest = pd.read_parquet(
        CORPUS_INDEX_DIR / "manifest.parquet", columns=["slide_id", "local_idx", "coord_x", "coord_y"]
    )
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    sampled = sampled.merge(manifest, on=["slide_id", "local_idx"], how="left", validate="many_to_one")
    if sampled["coord_x"].isna().any():
        print(f"WARNING: {int(sampled['coord_x'].isna().sum())} sampled rows had no manifest match")

    slide_meta = pd.read_parquet(CORPUS_INDEX_DIR / "slide_meta.parquet").set_index("slide_id")
    slide_meta.index = slide_meta.index.astype(str)

    sampled.to_csv(OUT_DIR / "sampled.csv", index=False)
    print(f"wrote {OUT_DIR / 'sampled.csv'}  ({len(sampled)} patches)", flush=True)

    jobs = [
        (r.slide_id, r.coord_x, r.coord_y, int(round(float(slide_meta.loc[r.slide_id, "patch_size_level0"]))))
        for r in sampled.itertuples()
    ]
    print(f"cropping {len(jobs)} patches with {args.workers} workers ...", flush=True)
    with mp.Pool(args.workers) as pool:
        images = list(pool.imap(_crop_one, jobs, chunksize=4))

    sampled["_img_idx"] = range(len(sampled))
    for sheet, blocks_src in (("sat_ladder", ladder), ("mean_guard", guard)):
        (OUT_DIR / sheet).mkdir(exist_ok=True)
        col = "sat_frac" if sheet == "sat_ladder" else "mean_intensity"
        for label, _ in blocks_src:
            block = sampled[(sampled["sheet"] == sheet) & (sampled["bin"] == label)].sort_values(col)
            imgs = [images[i] for i in block["_img_idx"]]
            lo_hi = label.split("__")[0]
            _render(block, imgs, OUT_DIR / sheet / f"bin_{lo_hi}.png",
                    f"{sheet}  {col} {lo_hi}   (provisional cut: sat_frac<0.10 & mean>215)", args.n_cols)

    n_fail = sum(isinstance(im, Exception) for im in images)
    print(f"\ndone. {n_fail} crop(s) failed of {len(images)}.")
    print(f"sheets -> {OUT_DIR}/sat_ladder/  and  {OUT_DIR}/mean_guard/")


if __name__ == "__main__":
    main()
