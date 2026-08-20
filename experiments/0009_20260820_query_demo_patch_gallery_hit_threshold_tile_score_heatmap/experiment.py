import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
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
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed one or more --image reference images (tiled, auto-scaled to the "
        "corpus's apparent magnification) and search the patch index for similar patches / WSIs."
    )
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--image", type=str, required=True, nargs="+",
        help="Path(s) to the query image(s). Multiple reference images for the same finding "
        "are aggregated by taking each tile's best match, not by averaging their vectors "
        "(see lib.search.PatchIndex.search_similar_patches_multi / search_top_slides_multi).",
    )
    parser.add_argument(
        "--stain_reference",
        type=str,
        default=None,
        help="Optional path to a reference patch (e.g. one of "
        "outputs/average_patch_candidates/*.png); if given, every --image is "
        "Macenko stain-normalized against it before embedding. Not a safe "
        "default — see lib.query_embedding.embed_image's docstring.",
    )
    parser.add_argument(
        "--auto_scale",
        action="store_true",
        help="NOT RECOMMENDED. Apply FM-centroid magnification matching (per-tile vote, see "
        "lib.mpp_estimation.estimate_relative_scale_fm_tiled's docstring) before tiling. Visually "
        "this stops the blurring an earlier whole-image-downsize version caused, but a 7-category "
        "ground-truth comparison (see lib.query_embedding.embed_image_tiles_auto_scale's "
        "docstring) found it was the single best option in 0 of 7 categories — plain tiling with "
        "no correction won 3/7, and combining this with --stain_reference was worse than either "
        "alone in most categories. Looking cleaner is not the same as retrieving better. Off by "
        "default for this reason, not just caution.",
    )
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.mpp_estimation import load_scale_centroids
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles, embed_image_tiles_auto_scale
    from lib.search import PatchIndex
    from lib.visualize import plot_hit_patch_gallery, plot_query_tile_scores, plot_slide_hits_on_thumbnail

    import numpy as np

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Each distinct set of query images gets its own run_dir/completion guard.
    image_stems = [re.sub(r"[^A-Za-z0-9_-]", "_", Path(p).stem) for p in args.image]
    variant_key = "query__" + "+".join(image_stems)[:120]
    if args.stain_reference:
        ref_stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(args.stain_reference).stem)
        variant_key += f"__norm-{ref_stem}"
    if args.auto_scale:
        variant_key += "__autoscale"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    index_exp_dir = project_root / config["index_exp_dir"]
    features_dir = project_root / config["features_dir"]
    thumbnails_dir = project_root / config["thumbnails_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    scale_centroids_path = project_root / config.get("scale_centroids_path", "data/scale_centroids.npz")
    tile_size: int = config.get("tile_size", 224)
    k: int = config.get("k", 20)
    nprobe: int = config.get("nprobe", 32)
    rerank_pool: int = config.get("rerank_pool", 200)
    max_tiles_reranked: int = config.get("max_tiles_reranked", 4)
    k_candidates: int = config.get("k_candidates", 8000)
    top_n_slides: int = config.get("top_n_slides", 20)
    top_n_slides_to_plot: int = config.get("top_n_slides_to_plot", 3)

    write_run_metadata(
        run_dir,
        exp_name=exp_name,
        variant_key=variant_key,
        image=args.image,
        stain_reference=args.stain_reference,
        auto_scale=args.auto_scale,
    )

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:          {run_dir}")
    logger.info(f"image:            {args.image}")
    logger.info(f"stain_reference:  {args.stain_reference}")
    logger.info(f"auto_scale:       {args.auto_scale}")
    logger.info(f"index_exp_dir:    {index_exp_dir}")
    logger.info(f"seed:             {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    centroids = None
    if args.auto_scale:
        centroids = load_scale_centroids(scale_centroids_path)
        logger.info(f"loaded scale centroids: {sorted(centroids.keys())} from {scale_centroids_path}")

    all_tiles = []
    scale_info = []
    for image_path in args.image:
        if centroids is not None:
            tiles, scale, debug = embed_image_tiles_auto_scale(
                image_path, centroids, tile_size=tile_size, stain_reference=args.stain_reference
            )
            logger.info(
                f"  {image_path}: auto-scaled by {scale}x (median_unclipped={debug['median_scale_unclipped']}, n_kept={debug['n_kept']}/{debug['n_tiles']})"
            )
            scale_info.append({"image": image_path, "scale_factor": scale, "debug": debug})
        else:
            tiles = embed_image_tiles(image_path, tile_size=tile_size, stain_reference=args.stain_reference)
            scale_info.append({"image": image_path, "scale_factor": 1.0, "similarities": None})
        logger.info(f"  {image_path}: {tiles.shape[0]} tiles")
        all_tiles.append(tiles)
    query_vecs = np.concatenate(all_tiles, axis=0)
    logger.info(f"total tiles across all images: {query_vecs.shape[0]}")

    patch_index = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )

    # Which of a query image's tiles actually get exact-reranked (and can
    # therefore ever appear in similar_patches/hit_pool) depends only on
    # the raw image + tiling params, not on the search results below — so
    # this can run right after the index loads. Not supported under
    # --auto_scale (the heatmap would need to reflect the rescaled image,
    # not the raw one); skipped in that case rather than silently showing
    # the wrong tiles.
    heatmap_dir = run_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    if not args.auto_scale:
        for image_path in args.image:
            heatmap_fig = plot_query_tile_scores(
                image_path, patch_index, tile_size=tile_size, nprobe=nprobe,
                max_tiles_reranked=max_tiles_reranked,
            )
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(image_path).stem)
            heatmap_fig.savefig(heatmap_dir / f"{stem}.png", dpi=150)

    similar_patches = patch_index.search_similar_patches_multi(
        query_vecs, k=k, nprobe=nprobe, rerank_pool=rerank_pool, max_tiles_reranked=max_tiles_reranked
    )
    similar_patches.to_csv(run_dir / "similar_patches.csv", index=False)
    logger.info(f"top similar patches:\n{similar_patches}")

    top_slides = patch_index.search_top_slides_multi(
        query_vecs, k_candidates=k_candidates, nprobe=nprobe, top_n_slides=top_n_slides
    )
    top_slides.to_csv(run_dir / "top_slides.csv", index=False)
    logger.info(f"top slides (reverse lookup):\n{top_slides}")

    plots_dir = run_dir / "thumbnail_plots"
    plots_dir.mkdir(exist_ok=True)
    gallery_dir = run_dir / "patch_gallery"
    gallery_dir.mkdir(exist_ok=True)
    # similar_patches (k=k, above) is a single global top-k across the whole
    # corpus, so most individual top_slides (chosen by the unrelated
    # n_hits_ratio metric in search_top_slides_multi) have zero rows in it —
    # not because they lack any real match, just because the global top-k is
    # too small a net. An earlier version of this experiment worked around
    # that by falling back to a k=k_candidates/rerank_pool=k_candidates
    # search (near-unbounded — thousands of candidates) whenever a slide
    # missed the global top-k. That made the displayed hit count meaningless
    # across slides: a slide inside the tight global top-k showed only its
    # few rows there, while a slide that missed it showed however many
    # thousands of far weaker candidates the unbounded fallback happened to
    # return for it — the exact opposite of what hit count should signal.
    #
    # Fixed by applying one consistent, exact-similarity-based rule to every
    # plotted slide: k=rerank_pool instead of k=k. rerank_pool (200) is
    # already the number of approximate candidates per tile search_similar_
    # patches_multi exact-reranks before truncating to k — this just stops
    # discarding 180 of those 200 already-computed exact scores instead of
    # inventing a new cutoff. Any slide with truly zero rows here (its
    # patches never even placed among the top 200 exact-reranked candidates
    # of any of the max_tiles_reranked tiles) is skipped — that's honest
    # information (its n_hits_ratio ranking came from weak/approximate
    # matches only), not something to paper over with an unbounded pool.
    hit_pool = patch_index.search_similar_patches_multi(
        query_vecs, k=rerank_pool, nprobe=nprobe, rerank_pool=rerank_pool,
        max_tiles_reranked=max_tiles_reranked,
    )
    for slide_id in top_slides["slide_id"].head(top_n_slides_to_plot):
        hits = hit_pool[hit_pool["slide_id"] == slide_id]
        if hits.empty:
            continue
        fig = plot_slide_hits_on_thumbnail(slide_id, hits, thumbnails_dir, patch_index.slide_meta)
        fig.savefig(plots_dir / f"{slide_id}.png", dpi=150)

        gallery_fig = plot_hit_patch_gallery(hits, raw_slide_dir, patch_index.slide_meta)
        gallery_fig.savefig(gallery_dir / f"{slide_id}.png", dpi=150)

    results = {
        "image": args.image,
        "stain_reference": args.stain_reference,
        "auto_scale": args.auto_scale,
        "scale_info": scale_info,
        "n_tiles": int(query_vecs.shape[0]),
        "n_similar_patches": int(len(similar_patches)),
        "n_top_slides": int(len(top_slides)),
        "similar_patches_path": str(run_dir / "similar_patches.csv"),
        "top_slides_path": str(run_dir / "top_slides.csv"),
        "thumbnail_plots_dir": str(plots_dir),
        "patch_gallery_dir": str(gallery_dir),
        "tile_score_heatmap_dir": str(heatmap_dir),
    }

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info(f"Done. {results}")


if __name__ == "__main__":
    main()
