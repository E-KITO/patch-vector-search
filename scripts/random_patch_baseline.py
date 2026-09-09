"""Build a random-patch control set for a curated patch set, so that "the
search selected this morphology" can be told apart from "any liver patch in
this corpus looks like this".

Motivation: experiments/0015 outputs a set of patches that are all highly
similar to each other (Hypertrophy job 9932: 150 patches, similarity
0.897..0.921, visually coherent). Coherence alone proves nothing — the set is
assembled by thresholding similarity, so it is coherent by construction. The
open question for a finding like Hypertrophy, whose morphology (large
hepatocytes with abundant cytoplasm) is not far from ordinary liver
parenchyma, is whether the retrieved patches differ from what you would get by
picking patches at random. experiments/0014 already burned on the neighbouring
version of this mistake: patches that looked like the glycogen atlas plate,
where that look was simply common in normal liver.

Method: sample N patches uniformly at random from the corpus manifest (the
same patch population the search draws from), crop them at real resolution the
same way lib.patch_set does, then interleave them with the curated set and
render sheets that show only a running number. The provenance of each number
goes to blind_key.csv, so a reviewer — human or model — can look at the sheets
first and check the answers afterwards, rather than reading the labels and
then seeing what they expect to see.

Reading the result: if the two are indistinguishable, the curated set is not
selecting the finding and its contributing slides are not evidence of
unlabelled positives. If they separate cleanly, the non-GT contributing slides
are worth triaging as candidate unlabelled positives.

Output (under --out):
  random_patches/            the sampled control patches, real resolution
  blind_sheets/page_NN.png   both sets interleaved, numbered, provenance hidden
  blind_key.csv              number -> source/slide_id/coords/similarity
  summary.json               counts and parameters
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default corpus the random patches are drawn from — the current production
# index (experiments/0018, background-filtered; promoted to default 2026-09-09).
# Override with --corpus-dir to match whichever index the curated --patch-set
# was built against — e.g. outputs/0002_20260808_build_faiss_index/default for
# a pre-deblank experiments/0015 patch set — so the random control samples the
# same population the curated set was drawn from.
CORPUS_INDEX_DIR = Path("outputs/0018_20260909_build_faiss_index_deblank/default")
RAW_SLIDE_DIR = Path("data/moo_collected_tggate_wsi/raw_wsi")

DEFAULT_SEED = 42
# Oversample before blank rejection. The corpus manifest is TRIDENT's tissue
# patches, but lib/patch_set.py still had to drop 29 near-blank crops out of
# 300 for Hypertrophy, so uniform sampling will hit some too.
BLANK_OVERFETCH = 1.6
SHEET_COLS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--patch-set", required=True, type=Path,
        help="Directory holding a curated set's manifest.parquet and patches/ "
        "(e.g. outputs/0015_.../hypertrophy__validate/hypertrophy__validate).",
    )
    p.add_argument("--out", required=True, type=Path, help="Output directory.")
    p.add_argument(
        "--corpus-dir", type=Path, default=CORPUS_INDEX_DIR,
        help="Index run_dir whose manifest.parquet / slide_meta.parquet define "
        "the population the random control is sampled from. Default: %(default)s "
        "(experiments/0018). Set to the index the --patch-set was built against "
        "(e.g. outputs/0002_..._build_faiss_index/default for a pre-deblank "
        "experiments/0014 or 0015 set).",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Number of random control patches. Default: match the curated set's size.",
    )
    p.add_argument(
        "--exclude-slides", type=str, default="",
        help="Comma-separated slide_ids to exclude from sampling — pass the run's "
        "seed slides, which were excluded from its results too.",
    )
    p.add_argument(
        "--only-slides", type=str, default="",
        help="Comma-separated slide_ids to sample *from*, instead of the whole "
        "corpus. Use this to inspect what a run's seed slides actually contain: "
        "the printed blank-rejection count then measures how much of the seed is "
        "tissue-sparse, which is the suspected cause when a run's own candidates "
        "come back mostly blank (see 'Degeneration, granular, eosinophilic', "
        "job 9962: 48 of 69 crops blank). Not a control set — do not read the "
        "resulting sheets as one.",
    )
    p.add_argument("--per-page", type=int, default=50, help="Patches per blind sheet.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def sample_random_patches(
    manifest: pd.DataFrame,
    slide_meta: pd.DataFrame,
    n: int,
    exclude_slides: set[str],
    rng: np.random.Generator,
    out_dir: Path,
    only_slides: set[str] | None = None,
) -> pd.DataFrame:
    """Uniformly sample patches from the corpus, crop them, drop blank crops,
    and keep the first `n` that survive. `only_slides` narrows the population
    to those slides (see --only-slides)."""
    from lib.query_embedding import _is_blank_tile
    from lib.raw_patch import crop_patch

    pool = manifest[~manifest["slide_id"].isin(exclude_slides)]
    if only_slides:
        pool = pool[pool["slide_id"].isin(only_slides)]
        if pool.empty:
            raise SystemExit(f"no corpus patches for --only-slides {sorted(only_slides)}")
    n_draw = min(len(pool), int(round(BLANK_OVERFETCH * n)))
    drawn = pool.iloc[np.sort(rng.choice(len(pool), size=n_draw, replace=False))]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_blank = 0
    for row in drawn.itertuples():
        if len(rows) >= n:
            break
        psl = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, RAW_SLIDE_DIR, psl)
        if _is_blank_tile(patch):
            n_blank += 1
            continue
        fname = f"{len(rows) + 1:03d}_{row.slide_id}_x{row.coord_x}_y{row.coord_y}.png"
        patch.save(out_dir / fname)
        rows.append({
            "slide_id": row.slide_id,
            "coord_x": int(row.coord_x),
            "coord_y": int(row.coord_y),
            "patch_path": str(out_dir / fname),
        })

    blank_pct = 100.0 * n_blank / n_draw if n_draw else 0.0
    print(
        f"random control: drew {n_draw}, dropped {n_blank} blank ({blank_pct:.0f}%), "
        f"kept {len(rows)}"
    )
    if len(rows) < n:
        print(f"WARNING: only {len(rows)} of {n} requested — raise BLANK_OVERFETCH")
    return pd.DataFrame(rows)


def render_blind_sheets(entries: pd.DataFrame, sheets_dir: Path, per_page: int) -> int:
    """One numbered patch per cell, nothing identifying the source."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    sheets_dir.mkdir(parents=True, exist_ok=True)
    n_pages = -(-len(entries) // per_page)
    for page in range(n_pages):
        chunk = entries.iloc[page * per_page:(page + 1) * per_page]
        n_rows = -(-len(chunk) // SHEET_COLS)
        fig, axes = plt.subplots(
            n_rows, SHEET_COLS, figsize=(SHEET_COLS * 2.2, n_rows * 2.4), squeeze=False
        )
        axes = axes.flatten()
        for ax, row in zip(axes, chunk.itertuples()):
            ax.imshow(Image.open(row.patch_path))
            ax.set_title(str(row.number), fontsize=9)
            ax.axis("off")
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(sheets_dir / f"page_{page + 1:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    return n_pages


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    exclude_slides = {s.strip() for s in args.exclude_slides.split(",") if s.strip()}

    curated = pd.read_parquet(args.patch_set / "manifest.parquet")
    curated["slide_id"] = curated["slide_id"].astype(str)
    n = args.n if args.n is not None else len(curated)

    manifest = pd.read_parquet(args.corpus_dir / "manifest.parquet")
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    # lib/manifest.py writes slide_meta with slide_id as a column (index=False);
    # PatchIndex.load is what turns it into the index, so do the same here.
    slide_meta = pd.read_parquet(args.corpus_dir / "slide_meta.parquet").set_index("slide_id")
    slide_meta.index = slide_meta.index.astype(str)

    print(f"curated set: {len(curated)} patches from {curated['slide_id'].nunique()} slides")
    print(f"corpus ({args.corpus_dir}): {len(manifest)} patches, excluding {len(exclude_slides)} slide(s)")

    only_slides = {s.strip() for s in args.only_slides.split(",") if s.strip()}
    if only_slides:
        print(f"sampling only from {len(only_slides)} slide(s): {sorted(only_slides)}")

    args.out.mkdir(parents=True, exist_ok=True)
    random_df = sample_random_patches(
        manifest, slide_meta, n, exclude_slides, rng, args.out / "random_patches",
        only_slides=only_slides,
    )

    curated_entries = pd.DataFrame({
        "source": "curated",
        "slide_id": curated["slide_id"],
        "coord_x": curated["coord_x"],
        "coord_y": curated["coord_y"],
        "similarity": curated["similarity"],
        "patch_path": [str(args.patch_set / p) for p in curated["patch_file"]],
    })
    random_entries = random_df.assign(source="random", similarity=np.nan)

    entries = pd.concat([curated_entries, random_entries], ignore_index=True)
    entries = entries.iloc[rng.permutation(len(entries))].reset_index(drop=True)
    entries["number"] = np.arange(1, len(entries) + 1)

    n_pages = render_blind_sheets(entries, args.out / "blind_sheets", args.per_page)

    key_cols = ["number", "source", "slide_id", "coord_x", "coord_y", "similarity", "patch_path"]
    entries[key_cols].to_csv(args.out / "blind_key.csv", index=False)

    summary = {
        "patch_set": str(args.patch_set),
        "corpus_dir": str(args.corpus_dir),
        "n_curated": int(len(curated_entries)),
        "n_random": int(len(random_entries)),
        "n_pages": n_pages,
        "per_page": args.per_page,
        "excluded_slides": sorted(exclude_slides),
        "only_slides": sorted(only_slides),
        "seed": args.seed,
        "curated_slides": int(curated["slide_id"].nunique()),
        "random_slides": int(random_df["slide_id"].nunique()) if len(random_df) else 0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nReview {args.out / 'blind_sheets'} before opening blind_key.csv.")


if __name__ == "__main__":
    main()
