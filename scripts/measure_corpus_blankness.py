"""Measure blankness (brightness + saturated-pixel fraction) for every patch in
the default corpus, so a filtered manifest build can drop background rows
without re-embedding.

Why: lib.query_embedding._is_blank_tile (bright + low pixel variance) catches
only ~0.9% of the corpus as blank, but ~1.9% is background by the
saturation-based measure (job 9983, scripts/blank_patch_similarity_diagnostic).
That residual ~350k background patches sit at a moderate similarity to every
finding and, being numerous, float to the top of searches for findings with
few genuine matches (granular eosinophilic: job 9962 came back 93% background).
Removing them is CPU-only -- UNI embeds each patch in isolation, so dropping
rows from the manifest/index needs no re-encoding (README "背景パッチ").

This script only measures. It writes the raw per-patch numbers to
outputs/measure_corpus_blankness/corpus_blankness.parquet
(slide_id, local_idx, mean_intensity, std_intensity, sat_frac, is_blank_legacy).
Deciding the exclusion threshold from that distribution, and building the
filtered manifest / FAISS index, are the next steps (experiments 0017 / 0018).

Cost is dominated by openslide: ~18.4M level-0 region reads across 1000 raw
WSIs on NFS. Slide-parallel (multiprocessing over slides, one SVS handle per
slide), resumable -- a per-slide parquet is written under per_slide/ and
existing ones are skipped, so a timed-out job just needs re-submitting.

--stage-dir copies each .svs to node-local scratch (one sequential NFS read)
before opening it, rather than letting openslide do thousands of scattered
level-0 tile reads over NFS -- worth it while filesrv02's HDD is degraded.
Only the SVS reads are staged; outputs (the per-slide parquets, the merged
parquet, _progress.json) stay on NFS so a timed-out job can still resume and
so `cat _progress.json` works live.

Progress: a tqdm bar plus a checkpoint line every 25 slides in the job log,
and a live outputs/measure_corpus_blankness/_progress.json heartbeat
(slides_done / patches_measured / slides_per_s / eta_s / failed) refreshed
after every slide -- `cat` it any time to see where the job is.

Usage:
    python scripts/measure_corpus_blankness.py                 # full corpus
    python scripts/measure_corpus_blankness.py --limit 5       # smoke test
    python scripts/measure_corpus_blankness.py --concat-only   # rebuild merged parquet + summary
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.patch_blankness import blankness_metrics, is_background
from lib.raw_patch import read_patch

CORPUS_INDEX_DIR = Path("outputs/0002_20260808_build_faiss_index/default")
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")
OUT_DIR = Path("outputs/measure_corpus_blankness")
PER_SLIDE_DIR = OUT_DIR / "per_slide"
MERGED_PATH = OUT_DIR / "corpus_blankness.parquet"
PROGRESS_PATH = OUT_DIR / "_progress.json"

_COLS = ["local_idx", "mean_intensity", "std_intensity", "sat_frac", "is_blank_legacy"]


def _write_progress(payload: dict) -> None:
    """Atomic (temp + replace) so a concurrent `cat` never sees a half-written file."""
    tmp = PROGRESS_PATH.parent / (PROGRESS_PATH.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, PROGRESS_PATH)


def _measure_slide(payload):
    """Worker: crop every patch of one slide, return (slide_id, n, error).

    When stage_dir is set, the .svs is copied there (one sequential NFS read)
    before openslide opens it, instead of letting openslide do thousands of
    scattered level-0 tile reads straight over NFS.
    """
    slide_id, local_idx, coords, psl, stage_dir = payload
    import openslide

    src = RAW_SLIDE_DIR / f"{slide_id}.svs"
    staged = None
    if stage_dir is not None:
        staged = Path(stage_dir) / f"{slide_id}.svs"
        try:
            shutil.copy2(src, staged)
        except Exception as e:  # scratch full / unwritable -- fall back to NFS
            print(f"  {slide_id}: stage copy failed ({e!r}); reading from NFS", flush=True)
            staged = None
    svs_path = staged if staged is not None else src

    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        if staged is not None:
            staged.unlink(missing_ok=True)
        return slide_id, 0, f"open: {e!r}"

    recs = []
    try:
        for li, cx, cy in zip(local_idx.tolist(), coords[:, 0].tolist(), coords[:, 1].tolist()):
            img = read_patch(slide, cx, cy, psl)
            m = blankness_metrics(img)
            recs.append((
                int(li),
                m["mean_intensity"],
                m["std_intensity"],
                m["sat_frac"],
                # lib.query_embedding._is_blank_tile's rule, inlined so a worker
                # process doesn't import the encoder module (torch) for it.
                bool(m["mean_intensity"] > 240.0 and m["std_intensity"] < 8.0),
            ))
    except Exception as e:
        return slide_id, 0, f"read: {e!r}"
    finally:
        slide.close()
        if staged is not None:
            staged.unlink(missing_ok=True)

    df = pd.DataFrame(recs, columns=_COLS)
    df.insert(0, "slide_id", str(slide_id))
    df.to_parquet(PER_SLIDE_DIR / f"{slide_id}.parquet", index=False)
    return slide_id, len(df), None


def _merge_and_summarise() -> None:
    parts = sorted(PER_SLIDE_DIR.glob("*.parquet"))
    if not parts:
        print("no per-slide parquets to merge")
        return
    merged = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
    merged.to_parquet(MERGED_PATH, index=False)
    print(f"\nwrote {MERGED_PATH}: {len(merged):,} patches from {len(parts)} slides", flush=True)

    bg = is_background(merged["mean_intensity"].to_numpy(), merged["sat_frac"].to_numpy())
    legacy = merged["is_blank_legacy"].to_numpy()
    print("\n--- provisional summary (threshold NOT yet decided -- see experiments/0017) ---")
    print(f"provisional background (sat_frac<0.10 & mean>215): {int(bg.sum()):,} ({100 * bg.mean():.2f}%)")
    print(f"legacy _is_blank_tile  (mean>240 & std<8):         {int(legacy.sum()):,} ({100 * legacy.mean():.2f}%)")
    print(f"flagged by provisional but not legacy:             {int((bg & ~legacy).sum()):,}")
    qs = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]
    for col in ("mean_intensity", "std_intensity", "sat_frac"):
        q = merged[col].quantile(qs)
        print(f"{col:15s} " + "  ".join(f"p{int(p * 100)}={v:.3f}" for p, v in q.items()))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--workers", type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 4),
    )
    ap.add_argument("--limit", type=int, default=None, help="only the first N slides (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-measure slides that already have a per-slide parquet")
    ap.add_argument("--concat-only", action="store_true", help="skip measuring; just merge existing per-slide parquets")
    ap.add_argument(
        "--stage-dir", type=str, default=None,
        help="node-local scratch dir; each worker copies its .svs here (one sequential "
        "read) before openslide opens it, instead of ~18k scattered NFS tile reads. "
        "The adhoc script points this at /scratch and cleans it up on exit.",
    )
    args = ap.parse_args()

    if args.stage_dir:
        Path(args.stage_dir).mkdir(parents=True, exist_ok=True)
        print(f"staging .svs via {args.stage_dir} before each open", flush=True)

    PER_SLIDE_DIR.mkdir(parents=True, exist_ok=True)

    if args.concat_only:
        _merge_and_summarise()
        return

    _write_progress({"status": "loading manifest", "updated_at": datetime.now().isoformat(timespec="seconds")})
    manifest = pd.read_parquet(
        CORPUS_INDEX_DIR / "manifest.parquet", columns=["slide_id", "local_idx", "coord_x", "coord_y"]
    )
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    slide_meta = pd.read_parquet(CORPUS_INDEX_DIR / "slide_meta.parquet").set_index("slide_id")
    slide_meta.index = slide_meta.index.astype(str)
    print(f"corpus: {len(manifest):,} patches across {manifest.slide_id.nunique()} slides", flush=True)

    groups = {sid: g for sid, g in manifest.groupby("slide_id", sort=True)}
    slides = sorted(groups)
    if args.limit:
        slides = slides[: args.limit]

    payloads, n_skipped = [], 0
    for sid in slides:
        if (PER_SLIDE_DIR / f"{sid}.parquet").exists() and not args.overwrite:
            n_skipped += 1
            continue
        g = groups[sid]
        psl = int(round(float(slide_meta.loc[sid, "patch_size_level0"])))
        payloads.append((sid, g["local_idx"].to_numpy(), g[["coord_x", "coord_y"]].to_numpy(), psl, args.stage_dir))
    print(f"to measure: {len(payloads)} slides ({n_skipped} already done)", flush=True)

    if payloads:
        t0 = time.time()
        failures: list[tuple[str, str]] = []
        patches_done = 0
        total = len(payloads)

        def _heartbeat(done: int, status: str) -> None:
            el = time.time() - t0
            rate = done / el if el else 0.0
            _write_progress({
                "status": status,
                "slides_done": done,
                "slides_total": total,
                "slides_already_done": n_skipped,
                "patches_measured": patches_done,
                "elapsed_s": round(el, 1),
                "slides_per_s": round(rate, 3),
                "eta_s": round((total - done) / rate, 1) if rate else None,
                "failed": [s for s, _ in failures],
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })

        _heartbeat(0, "running")
        with mp.Pool(args.workers) as pool:
            bar = tqdm(
                pool.imap_unordered(_measure_slide, payloads),
                total=total, unit="slide", mininterval=15.0, smoothing=0.05,
            )
            for i, (sid, n, err) in enumerate(bar, 1):
                if err:
                    failures.append((sid, err))
                    tqdm.write(f"[{i}/{total}] {sid}: FAILED {err}")
                else:
                    patches_done += n
                bar.set_postfix_str(f"{patches_done:,} patches, {len(failures)} failed")
                _heartbeat(i, "running")
                if i % 25 == 0 or i == total:
                    el = time.time() - t0
                    eta = (total - i) / (i / el) if i else 0.0
                    tqdm.write(
                        f"[{i}/{total}] {patches_done:,} patches | {i / el:.2f} slide/s "
                        f"| ETA {eta / 60:.1f} min | {len(failures)} failed"
                    )
            bar.close()

        _heartbeat(total, "complete" if not failures else "complete_with_failures")
        if failures:
            print(f"\n{len(failures)} slide(s) FAILED:")
            for sid, err in failures:
                print(f"  {sid}: {err}")

    _merge_and_summarise()


if __name__ == "__main__":
    main()
