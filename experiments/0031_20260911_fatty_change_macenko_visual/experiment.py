"""experiments/0031: Fatty Change の染色正規化(Macenko) vs プレーン 目視診断。

Fatty Change はコーパス内GTスライドが0枚で定量評価ができないため、
experiments/0013(タイルスコアヒートマップ + 実解像度パッチギャラリー)と同じ
枠組みを、baseline(0018・プレーン)と macenko(0012・per-tile Macenko正規化、
scripts/validate_against_ground_truth.py の v1_macenko パイプラインと同一)の
両方で Fatty Change の全8図版に対して実行する。索引の再構築は無し。

出力(outputs/0031_.../query__fatty_change__<baseline|macenko>/):
  tile_score_heatmap/*.png
  patch_gallery/<slide_id>.png
  thumbnail_plots/<slide_id>.png
"""

import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

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
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")


def _completed(*dirs: Path) -> bool:
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
    slug: str,
    images: list[str],
    index_label: str,
    project_root: Path,
    exp_name: str,
    output_root: str | None,
    patch_index,
    tile_transform,
    thumbnails_dir: Path,
    params: dict,
    plot_fns: dict,
    overwrite: bool,
) -> str:
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles

    import numpy as np

    variant_key = f"query__{slug}__{index_label}"
    canonical_dir = project_root / "outputs" / exp_name / variant_key
    check_dirs = [canonical_dir]
    if output_root:
        check_dirs.append(Path(output_root) / exp_name / variant_key)
    if overwrite:
        for d in check_dirs:
            (d / "completion.json").unlink(missing_ok=True)
    elif _completed(*check_dirs):
        print(f"skip (already completed): {variant_key}")
        return "skipped"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, f"{exp_name}:{variant_key}")
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, index_label=index_label)
    logger.info(f"index={index_label} images={len(images)}")

    tile_size = params["tile_size"]
    nprobe = params["nprobe"]
    rerank_pool = params["rerank_pool"]
    max_tiles_reranked = params["max_tiles_reranked"]

    all_tiles = []
    for image_path in images:
        tiles = embed_image_tiles(image_path, tile_size=tile_size, tile_transform=tile_transform)
        logger.info(f"  {image_path}: {tiles.shape[0]} tiles")
        all_tiles.append(tiles)
    query_vecs = np.concatenate(all_tiles, axis=0)
    logger.info(f"total tiles: {query_vecs.shape[0]}")

    heatmap_dir = run_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    for image_path in images:
        fig = plot_fns["tile_scores"](
            image_path, patch_index, tile_size=tile_size, nprobe=nprobe,
            max_tiles_reranked=max_tiles_reranked, tile_transform=tile_transform,
        )
        fig.savefig(heatmap_dir / f"{_slug(Path(image_path).stem)}.png", dpi=150)
        plot_fns["close"](fig)

    top_slides = patch_index.search_top_slides_multi(
        query_vecs, k_candidates=params["k_candidates"], nprobe=nprobe,
        top_n_slides=params["top_n_slides"],
    )
    top_slides.to_csv(run_dir / "top_slides.csv", index=False)
    logger.info(f"top slides:\n{top_slides}")

    plots_dir = run_dir / "thumbnail_plots"
    plots_dir.mkdir(exist_ok=True)
    gallery_dir = run_dir / "patch_gallery"
    gallery_dir.mkdir(exist_ok=True)
    hit_pool = patch_index.search_similar_patches_multi(
        query_vecs, k=rerank_pool, nprobe=nprobe, rerank_pool=rerank_pool,
        max_tiles_reranked=max_tiles_reranked,
    )
    n_gal = 0
    for slide_id in top_slides["slide_id"].head(params["top_n_slides_to_plot"]):
        hits = hit_pool[hit_pool["slide_id"] == slide_id]
        if hits.empty:
            continue
        try:
            fig = plot_fns["thumbnail"](slide_id, hits, thumbnails_dir, patch_index.slide_meta)
            fig.savefig(plots_dir / f"{slide_id}.png", dpi=150)
            plot_fns["close"](fig)
        except Exception as e:
            logger.warning(f"thumbnail plot failed for slide {slide_id}: {e!r}")
        try:
            fig = plot_fns["gallery"](hits, params["raw_slide_dir"], patch_index.slide_meta)
            fig.savefig(gallery_dir / f"{slide_id}.png", dpi=150)
            plot_fns["close"](fig)
            n_gal += 1
        except Exception as e:
            logger.warning(f"gallery failed for slide {slide_id}: {e!r}")
    logger.info(f"galleries written: {n_gal}")

    (run_dir / "results.json").write_text(json.dumps({
        "index_label": index_label, "n_tiles": int(query_vecs.shape[0]),
        "n_top_slides": int(len(top_slides)),
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"done: {variant_key}")
    return "ok"


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.atlas_figures import query_images
    from lib.search import PatchIndex
    from lib.torchstain_normalize import normalize_to_reference
    from lib.visualize import plot_hit_patch_gallery, plot_query_tile_scores, plot_slide_hits_on_thumbnail

    import matplotlib.pyplot as plt

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    atlas_root = project_root / config["atlas_root"]
    target_dir = atlas_root / config["target_finding"]
    images = [str(p) for p in query_images(target_dir)]
    print(f"target finding images: {len(images)} -> {[Path(p).name for p in images]}")
    if not images:
        raise SystemExit(f"no images found under {target_dir}")

    pi_baseline = PatchIndex.load(
        index_path=project_root / config["baseline_index_dir"] / "index.faiss",
        manifest_path=project_root / config["baseline_index_dir"] / "manifest.parquet",
        slide_meta_path=project_root / config["baseline_index_dir"] / "slide_meta.parquet",
        features_dir=project_root / config["baseline_features_dir"],
    )
    pi_macenko = PatchIndex.load(
        index_path=project_root / config["macenko_index_dir"] / "index.faiss",
        manifest_path=project_root / config["macenko_index_dir"] / "manifest.parquet",
        slide_meta_path=project_root / config["macenko_index_dir"] / "slide_meta.parquet",
        features_dir=project_root / config["macenko_features_dir"],
    )

    macenko_ref = project_root / config["macenko_stain_reference"]

    def macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_ref)
        except Exception:
            return tile

    params = {
        "raw_slide_dir": project_root / config["raw_slide_dir"],
        "tile_size": int(config.get("tile_size", 224)),
        "k": int(config.get("k", 20)),
        "nprobe": int(config.get("nprobe", 32)),
        "rerank_pool": int(config.get("rerank_pool", 200)),
        "max_tiles_reranked": config.get("max_tiles_reranked", 12),
        "k_candidates": int(config.get("k_candidates", 8000)),
        "top_n_slides": int(config.get("top_n_slides", 20)),
        "top_n_slides_to_plot": int(config.get("top_n_slides_to_plot", 3)),
    }
    plot_fns = {
        "tile_scores": plot_query_tile_scores,
        "thumbnail": plot_slide_hits_on_thumbnail,
        "gallery": plot_hit_patch_gallery,
        "close": plt.close,
    }

    arms = [
        ("baseline", pi_baseline, None, project_root / config["baseline_thumbnails_dir"]),
        ("macenko", pi_macenko, macenko_tile_transform, project_root / config["macenko_thumbnails_dir"]),
    ]

    n_ok = n_skip = n_fail = 0
    for index_label, pi, tile_transform, thumbnails_dir in arms:
        try:
            r = process_query_set(
                slug="fatty_change", images=images, index_label=index_label,
                project_root=project_root, exp_name=exp_name, output_root=output_root,
                patch_index=pi, tile_transform=tile_transform, thumbnails_dir=thumbnails_dir,
                params=params, plot_fns=plot_fns, overwrite=args.overwrite,
            )
            n_ok += r == "ok"
            n_skip += r == "skipped"
        except Exception:
            n_fail += 1
            traceback.print_exc()
            print(f"!! query set FAILED: {index_label}")
        print(f"[{n_ok + n_skip + n_fail}/{len(arms)}] {index_label}")

    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed, of {len(arms)}.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
