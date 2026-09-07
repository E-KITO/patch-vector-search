import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import yaml

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


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
        description="Embed one or more reference images (tiled, per-tile Macenko-normalized "
        "when config.stain_reference_per_tile is set) and search the patch index for similar "
        "patches / WSIs. Accepts a single query (--image), one folder aggregated into one "
        "query (--image-dir), or a parent folder whose every subfolder becomes its own "
        "aggregated query (--atlas-root)."
    )
    parser.add_argument("--config", type=str, default="config.yml")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=str, nargs="+",
        help="Path(s) to the query image(s). Multiple images are aggregated by taking each "
        "tile's best match, not by averaging vectors (see search_similar_patches_multi / "
        "search_top_slides_multi).",
    )
    src.add_argument(
        "--image-dir", type=str,
        help="A folder of query images (non-recursive). Every *.jpg/*.jpeg/*.png in it is "
        "aggregated into a single query, the same way multiple --image paths are.",
    )
    src.add_argument(
        "--atlas-root", type=str,
        help="A parent folder (e.g. data/query/Nonneoplastic-Lesion-Atlas-...). Each immediate "
        "subfolder is treated as one finding and its images aggregated into one query, so a "
        "single run sweeps the whole atlas. With --per-image, every image file (recursively) "
        "becomes its own query instead. Per-query runs use the completed-guard, so a re-run "
        "resumes where it stopped.",
    )
    parser.add_argument(
        "--per-image", action="store_true",
        help="With --atlas-root: one query (one run_dir) per image file, not per subfolder.",
    )
    parser.add_argument(
        "--stain_reference", type=str, default=None,
        help="Optional path to a reference patch; if given, every image is Macenko "
        "stain-normalized against it before embedding (whole-image, before tiling). Not a safe "
        "default — see lib.query_embedding.embed_image's docstring. Note this is distinct from "
        "config.stain_reference_per_tile, which normalizes each tile after cropping.",
    )
    parser.add_argument(
        "--auto_scale", action="store_true",
        help="NOT RECOMMENDED. FM-centroid magnification matching before tiling (per-tile vote). "
        "A 7-category ground-truth comparison found it was the single best option in 0 of 7 "
        "categories. Off by default for that reason, not just caution.",
    )
    parser.add_argument(
        "--no-galleries", action="store_true",
        help="Skip the real-resolution patch galleries (lib.visualize.plot_hit_patch_gallery), "
        "which are the only step that opens raw WSI files (data/moo_collected_tggate_wsi, ~600 GB, "
        "not staged to local SSD). Use this for a whole-atlas sweep: tile-score heatmaps, "
        "top_slides.csv, similar_patches.csv and the thumbnail overlays are still produced.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-run query sets even if their run_dir is already marked completed (clears the "
        "completion.json guard first). Use when re-running a finished sweep to add galleries.",
    )
    return parser.parse_args()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")


def _dir_images(d: Path) -> list[str]:
    return sorted(str(p) for p in d.iterdir() if p.suffix.lower() in _IMG_EXTS)


def resolve_query_sets(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    """-> [(variant_stem, [image_path, ...]), ...]. variant_stem feeds variant_key."""
    if args.atlas_root:
        root = Path(args.atlas_root)
        if not root.is_dir():
            raise SystemExit(f"--atlas-root {root} is not a directory")
        if args.per_image:
            imgs = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in _IMG_EXTS)
            if not imgs:
                raise SystemExit(f"--atlas-root {root} has no images")
            # 'atlas_img__' prefix so run_slurm.sh's WSI-staging glob can target
            # exactly these per-image runs.
            return [(_slug("atlas_img__" + Path(p).stem)[:120], [p]) for p in imgs]
        sets = []
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            imgs = _dir_images(sub)
            if imgs:
                sets.append((_slug(sub.name)[:120], imgs))
        if not sets:
            raise SystemExit(f"--atlas-root {root} has no subfolder with images")
        return sets
    if args.image_dir:
        d = Path(args.image_dir)
        if not d.is_dir():
            raise SystemExit(f"--image-dir {d} is not a directory")
        imgs = _dir_images(d)
        if not imgs:
            raise SystemExit(f"--image-dir {d} has no {sorted(_IMG_EXTS)} files")
        return [(_slug(d.name)[:120], imgs)]
    stems = [_slug(Path(p).stem) for p in args.image]
    return [("+".join(stems)[:120], list(args.image))]


def _staged_or(project_root: Path, config_rel: str, env_var: str, logger=None) -> Path:
    """A data path, preferring a local-SSD staged copy (env_var, set by
    run_slurm.sh's PRE_NATIVE_COMMAND) over the NFS original under project_root.

    The search path (FAISS index + per-slide .h5 feature files) is read at high
    frequency during exact re-ranking, so it must not sit on NFS during compute
    — see the storage policy in run_slurm.sh.
    """
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        if logger:
            logger.info(f"  {env_var}: using staged copy {staged}")
        return Path(staged)
    return project_root / config_rel


def _completed(*dirs: Path) -> bool:
    """True if any of these dirs holds a completion.json with status 'completed'.
    Checked before get_run_dir (whose own guard sys.exit(0)s on a completed
    canonical dir, which would kill the whole atlas loop) — so a re-run resumes
    where it stopped. Both the canonical outputs/ dir and the scratch
    OUTPUT_ROOT copy are checked, since the latter is only synced back at job end.
    """
    for d in dirs:
        f = d / "completion.json"
        if not f.exists():
            continue
        try:
            if json.loads(f.read_text()).get("status") == "completed":
                return True
        except Exception:
            pass
    return False


def process_query_set(
    *,
    variant_stem: str,
    images: list[str],
    project_root: Path,
    exp_name: str,
    output_root: str | None,
    patch_index,
    tile_transform,
    centroids,
    args: argparse.Namespace,
    params: dict,
    plot_fns: dict,
) -> None:
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles, embed_image_tiles_auto_scale

    import numpy as np

    variant_key = "query__" + variant_stem
    if args.stain_reference:
        variant_key += f"__norm-{_slug(Path(args.stain_reference).stem)}"
    if params["stain_reference_per_tile"]:
        variant_key += "__pertilenorm"
    if args.auto_scale:
        variant_key += "__autoscale"

    canonical_dir = project_root / "outputs" / exp_name / variant_key
    check_dirs = [canonical_dir]
    if output_root:
        check_dirs.append(Path(output_root) / exp_name / variant_key)
    if args.overwrite:
        for d in check_dirs:
            (d / "completion.json").unlink(missing_ok=True)
    elif _completed(*check_dirs):
        print(f"skip (already completed): {variant_key}")
        return

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, f"{exp_name}:{variant_key}")

    write_run_metadata(
        run_dir, exp_name=exp_name, variant_key=variant_key, image=images,
        stain_reference=args.stain_reference,
        stain_reference_per_tile=params["stain_reference_per_tile"],
        auto_scale=args.auto_scale, no_galleries=args.no_galleries,
    )
    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"images ({len(images)}): {images}")

    tile_size = params["tile_size"]
    nprobe = params["nprobe"]
    rerank_pool = params["rerank_pool"]
    max_tiles_reranked = params["max_tiles_reranked"]

    all_tiles = []
    scale_info = []
    for image_path in images:
        if centroids is not None:
            tiles, scale, debug = embed_image_tiles_auto_scale(
                image_path, centroids, tile_size=tile_size, stain_reference=args.stain_reference
            )
            logger.info(
                f"  {image_path}: auto-scaled by {scale}x "
                f"(median_unclipped={debug['median_scale_unclipped']}, n_kept={debug['n_kept']}/{debug['n_tiles']})"
            )
            scale_info.append({"image": image_path, "scale_factor": scale, "debug": debug})
        else:
            tiles = embed_image_tiles(
                image_path, tile_size=tile_size, stain_reference=args.stain_reference,
                tile_transform=tile_transform,
            )
            scale_info.append({"image": image_path, "scale_factor": 1.0, "similarities": None})
        logger.info(f"  {image_path}: {tiles.shape[0]} tiles")
        all_tiles.append(tiles)
    query_vecs = np.concatenate(all_tiles, axis=0)
    logger.info(f"total tiles across all images: {query_vecs.shape[0]}")

    # Which tiles get exact-reranked depends only on the raw image + tiling
    # params, so the heatmap can be drawn right after the index loads. Not
    # supported under --auto_scale (the heatmap would need the rescaled image).
    heatmap_dir = run_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    if not args.auto_scale:
        for image_path in images:
            heatmap_fig = plot_fns["tile_scores"](
                image_path, patch_index, tile_size=tile_size, nprobe=nprobe,
                max_tiles_reranked=max_tiles_reranked, tile_transform=tile_transform,
            )
            heatmap_fig.savefig(heatmap_dir / f"{_slug(Path(image_path).stem)}.png", dpi=150)
            plot_fns["close"](heatmap_fig)

    similar_patches = patch_index.search_similar_patches_multi(
        query_vecs, k=params["k"], nprobe=nprobe, rerank_pool=rerank_pool,
        max_tiles_reranked=max_tiles_reranked,
    )
    similar_patches.to_csv(run_dir / "similar_patches.csv", index=False)
    logger.info(f"top similar patches:\n{similar_patches}")

    top_slides = patch_index.search_top_slides_multi(
        query_vecs, k_candidates=params["k_candidates"], nprobe=nprobe,
        top_n_slides=params["top_n_slides"],
    )
    top_slides.to_csv(run_dir / "top_slides.csv", index=False)
    logger.info(f"top slides (reverse lookup):\n{top_slides}")

    plots_dir = run_dir / "thumbnail_plots"
    plots_dir.mkdir(exist_ok=True)
    gallery_dir = run_dir / "patch_gallery"
    gallery_dir.mkdir(exist_ok=True)
    # See the long note in git history: k=rerank_pool (not k=k) so every plotted
    # slide is scored by the same exact-similarity rule; slides with zero rows
    # here are skipped rather than back-filled from an unbounded pool.
    hit_pool = patch_index.search_similar_patches_multi(
        query_vecs, k=rerank_pool, nprobe=nprobe, rerank_pool=rerank_pool,
        max_tiles_reranked=max_tiles_reranked,
    )
    for slide_id in top_slides["slide_id"].head(params["top_n_slides_to_plot"]):
        hits = hit_pool[hit_pool["slide_id"] == slide_id]
        if hits.empty:
            continue
        fig = plot_fns["thumbnail"](slide_id, hits, params["thumbnails_dir"], patch_index.slide_meta)
        fig.savefig(plots_dir / f"{slide_id}.png", dpi=150)
        plot_fns["close"](fig)

        if not args.no_galleries:
            gallery_fig = plot_fns["gallery"](hits, params["raw_slide_dir"], patch_index.slide_meta)
            gallery_fig.savefig(gallery_dir / f"{slide_id}.png", dpi=150)
            plot_fns["close"](gallery_fig)

    results = {
        "image": images,
        "stain_reference": args.stain_reference,
        "stain_reference_per_tile": params["stain_reference_per_tile"],
        "auto_scale": args.auto_scale,
        "no_galleries": args.no_galleries,
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
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"Done. {variant_key}")


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.mpp_estimation import load_scale_centroids
    from lib.search import PatchIndex
    from lib.visualize import (
        plot_hit_patch_gallery,
        plot_query_tile_scores,
        plot_slide_hits_on_thumbnail,
    )

    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401 - kept for parity / downstream imports

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(Path(__file__).parent)

    # Data paths: prefer a local-SSD staged copy over the NFS original for the
    # search path and thumbnails (staged by run_slurm.sh's PRE_NATIVE_COMMAND).
    # raw_slide_dir (~600 GB) can't be staged wholesale; --no-galleries avoids it
    # entirely, but for a galleries run PVS_RAW_SLIDE_DIR can point at a dir
    # holding just the few dozen top-slide .svs the galleries actually open
    # (galleries are the only raw-WSI reader; each finding plots top_n_slides_to_plot).
    index_exp_dir = _staged_or(project_root, config["index_exp_dir"], "PVS_INDEX_DIR")
    features_dir = _staged_or(project_root, config["features_dir"], "PVS_FEATURES_DIR")
    thumbnails_dir = _staged_or(project_root, config["thumbnails_dir"], "PVS_THUMBNAILS_DIR")
    raw_slide_dir = _staged_or(project_root, config["raw_slide_dir"], "PVS_RAW_SLIDE_DIR")
    scale_centroids_path = project_root / config.get("scale_centroids_path", "data/scale_centroids.npz")

    stain_reference_per_tile = config.get("stain_reference_per_tile")
    if stain_reference_per_tile:
        stain_reference_per_tile = project_root / stain_reference_per_tile

    print(f"index_exp_dir: {index_exp_dir}")
    print(f"features_dir:  {features_dir}")
    print(f"thumbnails_dir: {thumbnails_dir}")

    query_sets = resolve_query_sets(args)
    print(f"query sets: {len(query_sets)} -> {[s for s, _ in query_sets]}")

    tile_transform = None
    if stain_reference_per_tile:
        from lib.torchstain_normalize import normalize_to_reference

        def tile_transform(tile):  # noqa: E731 - named for readability in tracebacks
            try:
                return normalize_to_reference(tile, stain_reference_per_tile)
            except Exception:
                return tile  # degenerate (near-blank) tile — corpus passes these through raw too

        if args.auto_scale:
            raise SystemExit(
                "stain_reference_per_tile + --auto_scale is not supported "
                "(embed_image_tiles_auto_scale has no per-tile hook)."
            )

    centroids = None
    if args.auto_scale:
        centroids = load_scale_centroids(scale_centroids_path)
        print(f"loaded scale centroids: {sorted(centroids.keys())}")

    patch_index = PatchIndex.load(
        index_path=index_exp_dir / "index.faiss",
        manifest_path=index_exp_dir / "manifest.parquet",
        slide_meta_path=index_exp_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )

    params = {
        "stain_reference_per_tile": str(stain_reference_per_tile) if stain_reference_per_tile else None,
        "thumbnails_dir": thumbnails_dir,
        "raw_slide_dir": raw_slide_dir,
        "tile_size": config.get("tile_size", 224),
        "k": config.get("k", 20),
        "nprobe": config.get("nprobe", 32),
        "rerank_pool": config.get("rerank_pool", 200),
        "max_tiles_reranked": config.get("max_tiles_reranked", 4),
        "k_candidates": config.get("k_candidates", 8000),
        "top_n_slides": config.get("top_n_slides", 20),
        "top_n_slides_to_plot": config.get("top_n_slides_to_plot", 3),
    }
    plot_fns = {
        "tile_scores": plot_query_tile_scores,
        "thumbnail": plot_slide_hits_on_thumbnail,
        "gallery": plot_hit_patch_gallery,
        "close": plt.close,
    }

    n_done = 0
    for variant_stem, images in query_sets:
        process_query_set(
            variant_stem=variant_stem, images=images, project_root=project_root,
            exp_name=exp_name, output_root=output_root, patch_index=patch_index,
            tile_transform=tile_transform, centroids=centroids, args=args,
            params=params, plot_fns=plot_fns,
        )
        n_done += 1
        print(f"[{n_done}/{len(query_sets)}] {variant_stem} done")

    print(f"All {len(query_sets)} query set(s) complete.")


if __name__ == "__main__":
    main()
