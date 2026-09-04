import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

import numpy as np
import pandas as pd


# FINDING_TYPE (data/processed_csv/single_finding_liver.csv) -> NNL atlas
# subdirectory under config["atlas_root"]. Only findings whose atlas category
# maps cleanly AND that the 2026-09-04 self-retrieval diagnostic showed the
# corpus represents well (README "自己検索診断") belong here — do not add a
# risky mapping (see the "Necrosis" -> "Single cell necrosis" mismatch that
# validate_against_ground_truth.py's CATEGORIES got wrong).
FINDING_TO_ATLAS_DIR = {
    "Deposit, glycogen": "Liver, Hepatocyte - Glycogen Accumulation and Depletion - Nonneoplastic Lesion Atlas",
    "Increased mitosis": "Liver, Hepatocyte – Increased Mitosis - Nonneoplastic Lesion Atlas",
}


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(exp_dir: Path) -> dict:
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a representative-patch set for one toxicity finding: "
        "query the index with that finding's NNL atlas figures, then curate the "
        "hits (similarity floor, per-slide spatial NMS, per-slide cap, "
        "slide-diverse truncation) into an actual patch dataset."
    )
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--finding", required=True,
        help="FINDING_TYPE to build a set for (spaces may be written as '_' so the "
        "run_slurm.sh seq GRID doesn't word-split them). One of: "
        f"{sorted(FINDING_TO_ATLAS_DIR)}",
    )
    args = parser.parse_args()
    # run_slurm.sh passes "Deposit,_glycogen" etc. — neither known finding name
    # contains a real underscore, so this is unambiguous.
    args.finding = args.finding.replace("_", " ")
    return args


def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def spatial_nms(df: pd.DataFrame, patch_size_level0: int, radius_patches: float) -> pd.DataFrame:
    """Greedy per-slide non-max suppression. df is one slide's candidate patches,
    sorted by similarity descending. Keep a patch unless an already-kept patch on
    the same slide is within radius_patches * patch_size_level0 (Euclidean, in
    level-0 pixels) — those are the same bit of tissue seen by adjacent tiles.
    """
    radius = radius_patches * patch_size_level0
    kept_xy: list[tuple[int, int]] = []
    keep_mask = []
    for x, y in zip(df["coord_x"].to_numpy(), df["coord_y"].to_numpy()):
        ok = all((x - kx) ** 2 + (y - ky) ** 2 >= radius**2 for kx, ky in kept_xy)
        keep_mask.append(ok)
        if ok:
            kept_xy.append((x, y))
    return df[np.array(keep_mask)]


def slide_diverse_truncate(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Round-robin across slides (slides ordered by their best patch's similarity)
    so one prolific slide can't dominate the final set."""
    df = df.sort_values("similarity", ascending=False)
    per_slide = {sid: g.to_dict("records") for sid, g in df.groupby("slide_id", sort=False)}
    slide_order = list(per_slide)  # already best-first from the sort above
    picked = []
    while len(picked) < target_n and any(per_slide.values()):
        for sid in slide_order:
            if per_slide[sid]:
                picked.append(per_slide[sid].pop(0))
                if len(picked) >= target_n:
                    break
    return pd.DataFrame(picked)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.raw_patch import crop_patch
    from lib.search import PatchIndex
    from lib.visualize import plot_hit_patch_gallery

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)

    if args.finding not in FINDING_TO_ATLAS_DIR:
        raise SystemExit(
            f"--finding {args.finding!r} has no atlas mapping. "
            f"Known: {sorted(FINDING_TO_ATLAS_DIR)}"
        )

    finding_slug = slugify(args.finding)
    variant_key = finding_slug
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    seed = int(config.get("seed", 42))
    index_exp_dir = project_root / config["index_exp_dir"]
    features_dir = project_root / config["features_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    atlas_root = project_root / config["atlas_root"]
    gt_csv = project_root / config["gt_csv"]
    tile_size = int(config.get("tile_size", 224))
    nprobe = int(config.get("nprobe", 64))
    k_candidate_patches = int(config.get("k_candidate_patches", 5000))
    rerank_pool = int(config.get("rerank_pool", 1000))
    max_tiles_reranked = config.get("max_tiles_reranked", None)
    sim_floor = float(config.get("sim_floor", 0.40))
    nms_radius_patches = float(config.get("nms_radius_patches", 1.5))
    max_per_slide = int(config.get("max_per_slide", 15))
    target_n_patches = int(config.get("target_n_patches", 150))

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, finding=args.finding)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"finding:           {args.finding}")
    logger.info(f"run_dir:           {run_dir}")
    logger.info(
        f"sim_floor={sim_floor} nms_radius_patches={nms_radius_patches} "
        f"max_per_slide={max_per_slide} target_n_patches={target_n_patches}"
    )

    out_dir = run_dir / finding_slug
    patches_dir = out_dir / "patches"
    sheets_dir = out_dir / "contact_sheets"
    for d in (out_dir, patches_dir, sheets_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ── 1. reference figures -> query vectors ────────────────────────────────
    atlas_dir = atlas_root / FINDING_TO_ATLAS_DIR[args.finding]
    ref_figures = sorted(atlas_dir.glob("*.jpg")) + sorted(atlas_dir.glob("*.png"))
    if not ref_figures:
        raise SystemExit(f"no reference figures in {atlas_dir}")
    logger.info(f"reference figures ({len(ref_figures)}):")
    for f in ref_figures:
        logger.info(f"  {f.name}")
    (out_dir / "reference_figures.txt").write_text(
        "\n".join(f.name for f in ref_figures) + "\n"
    )

    query_vecs = np.concatenate(
        [embed_image_tiles(str(f), tile_size=tile_size) for f in ref_figures], axis=0
    )
    logger.info(f"query tiles across {len(ref_figures)} figures: {query_vecs.shape[0]}")

    # ── 2. search ───────────────────────────────────────────────────────────
    pi = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )
    candidates = pi.search_similar_patches_multi(
        query_vecs, k=k_candidate_patches, nprobe=nprobe,
        rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked,
    )
    candidates["slide_id"] = candidates["slide_id"].astype(str)
    logger.info(
        f"raw candidates: {len(candidates)} patches from {candidates['slide_id'].nunique()} "
        f"slides, similarity {candidates['similarity'].min():.3f}..{candidates['similarity'].max():.3f}"
    )

    # ── 3. curate ───────────────────────────────────────────────────────────
    kept = candidates[candidates["similarity"] >= sim_floor].copy()
    logger.info(f"after sim_floor >= {sim_floor}: {len(kept)} patches, {kept['slide_id'].nunique()} slides")

    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)

    nms_frames = []
    for sid, g in kept.groupby("slide_id", sort=False):
        g = g.sort_values("similarity", ascending=False)
        psl = int(slide_meta.loc[sid, "patch_size_level0"])
        g = spatial_nms(g, psl, nms_radius_patches).head(max_per_slide)
        nms_frames.append(g)
    kept = pd.concat(nms_frames, ignore_index=True) if nms_frames else kept.iloc[:0]
    logger.info(f"after per-slide NMS + cap {max_per_slide}: {len(kept)} patches, {kept['slide_id'].nunique()} slides")

    final = slide_diverse_truncate(kept, target_n_patches).reset_index(drop=True)
    final["rank"] = np.arange(1, len(final) + 1)
    logger.info(f"final set: {len(final)} patches from {final['slide_id'].nunique()} slides")

    # ── 4. GT-positive fraction (coarse patch-level precision proxy) ─────────
    gt = pd.read_csv(gt_csv)
    gt["slide_id"] = gt["image_id"].str.replace(".svs", "", regex=False)
    gt_pos_slides = set(gt.loc[gt["FINDING_TYPE"] == args.finding, "slide_id"])
    final["gt_positive_slide"] = final["slide_id"].isin(gt_pos_slides)
    contributing = sorted(final["slide_id"].unique())
    gt_pos_contributing = [s for s in contributing if s in gt_pos_slides]
    logger.info(
        f"GT check: {final['gt_positive_slide'].mean():.2f} of final patches from a slide "
        f"labelled {args.finding!r}; {len(gt_pos_contributing)}/{len(contributing)} contributing "
        f"slides are GT-positive (corpus has {len(gt_pos_slides & set(slide_meta.index))} such slides)"
    )

    # ── 5. save patches + manifest ──────────────────────────────────────────
    rows = []
    for row in final.itertuples():
        psl = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, raw_slide_dir, psl)
        fname = f"{row.rank:03d}_{row.slide_id}_x{row.coord_x}_y{row.coord_y}.png"
        patch.save(patches_dir / fname)
        rows.append({
            "rank": row.rank, "finding": args.finding, "slide_id": row.slide_id,
            "coord_x": int(row.coord_x), "coord_y": int(row.coord_y),
            "similarity": float(row.similarity), "gt_positive_slide": bool(row.gt_positive_slide),
            "patch_file": f"patches/{fname}",
        })
    manifest = pd.DataFrame(rows)
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    # ── 6. contact sheets, one per contributing slide ───────────────────────
    for sid, g in final.groupby("slide_id", sort=False):
        fig = plot_hit_patch_gallery(
            g.sort_values("similarity", ascending=False), raw_slide_dir, slide_meta
        )
        tag = "gt" if sid in gt_pos_slides else "nogt"
        fig.savefig(sheets_dir / f"{sid}__{tag}.png", dpi=120, bbox_inches="tight")

    results = {
        "finding": args.finding,
        "reference_figures": [f.name for f in ref_figures],
        "n_query_tiles": int(query_vecs.shape[0]),
        "n_raw_candidates": int(len(candidates)),
        "n_after_sim_floor": int((candidates["similarity"] >= sim_floor).sum()),
        "n_final_patches": int(len(final)),
        "n_contributing_slides": len(contributing),
        "n_contributing_slides_gt_positive": len(gt_pos_contributing),
        "n_corpus_slides_with_finding": int(len(gt_pos_slides & set(slide_meta.index))),
        "frac_final_patches_from_gt_positive_slide": round(float(final["gt_positive_slide"].mean()), 3),
        "final_similarity_min": round(float(final["similarity"].min()), 4) if len(final) else None,
        "final_similarity_max": round(float(final["similarity"].max()), 4) if len(final) else None,
        "params": {
            "seed": seed, "nprobe": nprobe, "k_candidate_patches": k_candidate_patches,
            "rerank_pool": rerank_pool, "max_tiles_reranked": max_tiles_reranked,
            "sim_floor": sim_floor, "nms_radius_patches": nms_radius_patches,
            "max_per_slide": max_per_slide, "target_n_patches": target_n_patches,
        },
        "manifest_path": str(out_dir / "manifest.parquet"),
        "patches_dir": str(patches_dir),
        "contact_sheets_dir": str(sheets_dir),
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

    complete_run(run_dir)
    logger.info(f"Done. {results}")


if __name__ == "__main__":
    main()
