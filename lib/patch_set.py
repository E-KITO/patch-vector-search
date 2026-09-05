"""Turn raw index hits into a curated representative-patch set for one finding.

The productization layer on top of lib.search.PatchIndex: where
lib.visualize shows one query's hits for eyeballing, this assembles an actual
patch dataset — apply a similarity floor, suppress near-duplicate hits within a
slide, cap per slide, spread across donor slides, then crop the real-resolution
pixels and write a manifest + per-slide contact sheets.

Used by experiments/0015 (seed-slide queries). experiments/0014 (atlas-figure
queries) predates this module and keeps its own inline copy — it is a frozen
record and is not retrofitted.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def spatial_nms(df: pd.DataFrame, patch_size_level0: int, radius_patches: float) -> pd.DataFrame:
    """Greedy per-slide non-max suppression. `df` is one slide's candidate
    patches sorted by similarity descending. Keep a patch unless an
    already-kept patch on the same slide is within
    radius_patches * patch_size_level0 (Euclidean, level-0 pixels) — those are
    the same bit of tissue seen by adjacent overlapping tile searches.
    """
    radius_sq = float(radius_patches * patch_size_level0) ** 2
    kept_xy: list[tuple[int, int]] = []
    keep_mask = []
    # int() so the squared level-0 coordinates (can be ~1e10) use Python's
    # arbitrary-precision ints, not numpy int32 which overflows.
    for x, y in zip(df["coord_x"].to_numpy(), df["coord_y"].to_numpy()):
        x, y = int(x), int(y)
        ok = all((x - kx) ** 2 + (y - ky) ** 2 >= radius_sq for kx, ky in kept_xy)
        keep_mask.append(ok)
        if ok:
            kept_xy.append((x, y))
    return df[np.array(keep_mask, dtype=bool)]


def slide_diverse_truncate(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """Round-robin across slides (slides ordered by their best patch's
    similarity) so one prolific slide can't dominate the final set."""
    df = df.sort_values("similarity", ascending=False)
    per_slide = {sid: g.to_dict("records") for sid, g in df.groupby("slide_id", sort=False)}
    slide_order = list(per_slide)
    picked = []
    while len(picked) < target_n and any(per_slide.values()):
        for sid in slide_order:
            if per_slide[sid]:
                picked.append(per_slide[sid].pop(0))
                if len(picked) >= target_n:
                    break
    return pd.DataFrame(picked)


def curate_candidates(
    candidates: pd.DataFrame,
    slide_meta: pd.DataFrame,
    *,
    sim_floor: float,
    nms_radius_patches: float,
    max_per_slide: int,
    target_n_patches: int,
) -> tuple[pd.DataFrame, dict]:
    """candidates: DataFrame[slide_id, coord_x, coord_y, similarity] (string
    slide_id). slide_meta: indexed by string slide_id, has patch_size_level0.

    Returns (final set with an added integer `rank` column, per-stage counts).

    The counts matter for diagnosis: a run that falls short of its target can
    do so because the similarity floor cut it, because per-slide NMS collapsed
    near-duplicate hits, because few slides contributed at all, or because
    blank crops were dropped downstream — and those mean very different things.
    experiments/0014's inline version logged each stage; that was lost when the
    logic moved here, leaving only "candidates in, patches out" (see README
    "最終パッチ数が少ないことの意味").
    """
    n_in = len(candidates)
    kept = candidates[candidates["similarity"] >= sim_floor].copy()
    n_after_sim_floor = len(kept)

    frames = []
    for sid, g in kept.groupby("slide_id", sort=False):
        g = g.sort_values("similarity", ascending=False)
        psl = int(slide_meta.loc[sid, "patch_size_level0"])
        frames.append(spatial_nms(g, psl, nms_radius_patches).head(max_per_slide))
    kept = pd.concat(frames, ignore_index=True) if frames else kept.iloc[:0]
    n_after_nms_and_cap = len(kept)
    n_slides_after_nms = int(kept["slide_id"].nunique()) if len(kept) else 0
    n_slides_at_cap = (
        int((kept.groupby("slide_id").size() >= max_per_slide).sum()) if len(kept) else 0
    )

    final = slide_diverse_truncate(kept, target_n_patches).reset_index(drop=True)
    final["rank"] = np.arange(1, len(final) + 1)

    stats = {
        "n_candidates_in": n_in,
        "n_after_sim_floor": n_after_sim_floor,
        "n_after_nms_and_cap": n_after_nms_and_cap,
        "n_slides_after_nms": n_slides_after_nms,
        # Slides that hit max_per_slide. Many slides at the cap means the set is
        # limited by the cap (deepening the search would add more); none at the
        # cap means it is limited by how few good patches exist per slide.
        "n_slides_at_max_per_slide": n_slides_at_cap,
        "n_after_round_robin": len(final),
    }
    return final, stats


def build_patch_set(
    pi,
    query_vecs: np.ndarray,
    *,
    out_dir: Path,
    raw_slide_dir: Path,
    exclude_slides: set[str],
    gt_positive_slides: set[str],
    nprobe: int = 64,
    k_candidate_patches: int = 8000,
    rerank_pool: int = 8000,
    max_tiles_reranked: int | None = None,
    sim_floor: float = 0.40,
    nms_radius_patches: float = 1.5,
    max_per_slide: int = 15,
    target_n_patches: int = 150,
    blank_overfetch: float = 2.0,
) -> dict:
    """Search the index with `query_vecs`, curate, crop real patches, and write
    out_dir/{patches/, contact_sheets/, manifest.parquet, manifest.csv}.

    `blank_overfetch`: curate blank_overfetch * target_n_patches candidates, then
    drop near-blank crops (lib.query_embedding._is_blank_tile) while cropping and
    keep the first target_n_patches that survive. A whole-slide seed query can
    include tissue-sparse patches that then match slide background all over the
    corpus at deceptively high similarity (seen with "Increased mitosis" in
    experiments/0015) — this is the guard against that leaking into the output.

    Returns a stats dict (also suitable to drop into results.json).
    """
    from lib.query_embedding import _is_blank_tile
    from lib.raw_patch import crop_patch
    from lib.visualize import plot_hit_patch_gallery

    patches_dir = out_dir / "patches"
    sheets_dir = out_dir / "contact_sheets"
    for d in (out_dir, patches_dir, sheets_dir):
        d.mkdir(parents=True, exist_ok=True)

    slide_meta = pi.slide_meta.copy()
    slide_meta.index = slide_meta.index.astype(str)

    candidates = pi.search_similar_patches_multi(
        query_vecs, k=k_candidate_patches, nprobe=nprobe,
        rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked,
    )
    candidates["slide_id"] = candidates["slide_id"].astype(str)
    n_raw = len(candidates)
    candidates = candidates[~candidates["slide_id"].isin(exclude_slides)]
    logger.info(
        f"candidates: {n_raw} raw -> {len(candidates)} after excluding {len(exclude_slides)} "
        f"seed slides; similarity "
        f"{candidates['similarity'].min():.3f}..{candidates['similarity'].max():.3f}"
    )

    curated, curate_stats = curate_candidates(
        candidates, slide_meta,
        sim_floor=sim_floor, nms_radius_patches=nms_radius_patches,
        max_per_slide=max_per_slide,
        target_n_patches=int(round(blank_overfetch * target_n_patches)),
    )
    logger.info(
        f"curation: {curate_stats['n_candidates_in']} -> "
        f"{curate_stats['n_after_sim_floor']} after sim_floor {sim_floor} -> "
        f"{curate_stats['n_after_nms_and_cap']} after NMS + cap {max_per_slide} "
        f"({curate_stats['n_slides_after_nms']} slides, "
        f"{curate_stats['n_slides_at_max_per_slide']} at the cap) -> "
        f"{curate_stats['n_after_round_robin']} after round-robin"
    )

    # Crop in rank order, dropping near-blank crops, until target_n_patches survive.
    rows = []
    n_blank_dropped = 0
    for row in curated.itertuples():
        if len(rows) >= target_n_patches:
            break
        psl = int(slide_meta.loc[row.slide_id, "patch_size_level0"])
        patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, raw_slide_dir, psl)
        if _is_blank_tile(patch):
            n_blank_dropped += 1
            continue
        rank = len(rows) + 1
        fname = f"{rank:03d}_{row.slide_id}_x{row.coord_x}_y{row.coord_y}.png"
        patch.save(patches_dir / fname)
        rows.append({
            "rank": rank, "slide_id": row.slide_id,
            "coord_x": int(row.coord_x), "coord_y": int(row.coord_y),
            "similarity": float(row.similarity),
            "gt_positive_slide": bool(row.slide_id in gt_positive_slides),
            "patch_file": f"patches/{fname}",
        })
    manifest = pd.DataFrame(
        rows,
        columns=["rank", "slide_id", "coord_x", "coord_y", "similarity",
                 "gt_positive_slide", "patch_file"],
    )
    final = manifest  # for the contact-sheet grouping and stats below

    contributing = sorted(final["slide_id"].unique()) if len(final) else []
    gt_contributing = [s for s in contributing if s in gt_positive_slides]
    frac_gt = round(float(final["gt_positive_slide"].mean()), 3) if len(final) else None
    logger.info(
        f"final: {len(final)} patches from {len(contributing)} slides "
        f"({n_blank_dropped} blank crops dropped); "
        f"{len(gt_contributing)} contributing slides are held-out GT-positive; "
        f"frac from held-out GT slide: {frac_gt}"
    )
    manifest.to_parquet(out_dir / "manifest.parquet", index=False)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    import matplotlib.pyplot as plt

    for sid, g in final.groupby("slide_id", sort=False):
        fig = plot_hit_patch_gallery(
            g.sort_values("similarity", ascending=False), raw_slide_dir, slide_meta
        )
        tag = "gt" if sid in gt_positive_slides else "nogt"
        fig.savefig(sheets_dir / f"{sid}__{tag}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)  # 150 contributing slides = 150 figures otherwise held open

    return {
        "n_raw_candidates": n_raw,
        "n_candidates_after_exclude": int(len(candidates)),
        **curate_stats,
        "n_blank_crops_dropped": n_blank_dropped,
        "n_final_patches": int(len(final)),
        "n_contributing_slides": len(contributing),
        "n_contributing_slides_heldout_gt": len(gt_contributing),
        "frac_final_patches_from_heldout_gt_slide": round(float(final["gt_positive_slide"].mean()), 3)
        if len(final) else None,
        "final_similarity_min": round(float(final["similarity"].min()), 4) if len(final) else None,
        "final_similarity_max": round(float(final["similarity"].max()), 4) if len(final) else None,
        "contributing_slides": contributing,
        "heldout_gt_contributing_slides": gt_contributing,
    }
