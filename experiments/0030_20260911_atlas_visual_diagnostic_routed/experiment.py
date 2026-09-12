"""experiments/0030: whiten ルーティング先3所見の目視診断。

experiments/0013(タイルスコアヒートマップ + 実解像度パッチギャラリー)の枠組みを、
experiments/0025-0029 のルーティング作業(quantitative な best_rank/found のみ)に
初めて適用する。lib.finding_routing.WHITEN_FINDINGS に実際に倒れた3所見の atlas
フォルダだけを対象に、baseline(0018)と routed(whiten、0025)の両方で検索し、

  - タイルスコアヒートマップ(lib.visualize.plot_query_tile_scores):
    近似FAISSスコアの分布 — どのタイルが exact re-rank の土俵に上がるか
  - 実解像度パッチギャラリー(lib.visualize.plot_hit_patch_gallery):
    上位スライドの実際のパッチが所見らしく見えるか
  - スライドサムネイル上のヒット位置(lib.visualize.plot_slide_hits_on_thumbnail)

を並べて比較できるようにする。索引の再構築・新規クエリ画像は無し。

出力(outputs/0030_.../query__<finding_slug>__<baseline|routed>/):
  tile_score_heatmap/*.png
  patch_gallery/<slide_id>.png
  thumbnail_plots/<slide_id>.png
  similar_patches.csv, top_slides.csv, results.json
"""

import json
import logging
import os
import re
import sys
import traceback
from pathlib import Path

import yaml

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


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


def resolve_target_findings(project_root: Path, config: dict) -> list[tuple[str, str, list[str]]]:
    """lib.finding_routing.WHITEN_FINDINGS に倒れる atlas フォルダだけを返す
    -> [(atlas_folder_name, slug, [image_path, ...]), ...]。"""
    sys.path.insert(0, str(project_root))
    from lib.atlas_figures import query_images
    from lib.finding_routing import WHITEN_FINDINGS
    from scripts.validate_against_ground_truth import CATEGORIES

    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]

    out = []
    for folder_name, finding_type in CATEGORIES.items():
        if finding_type not in WHITEN_FINDINGS:
            continue
        cat_dir = atlas_root / folder_name
        imgs = [str(p) for p in query_images(cat_dir, atlas_csv=atlas_csv)]
        if imgs:
            out.append((folder_name, _slug(folder_name)[:120], imgs))
    return out


def process_query_set(
    *,
    finding_name: str,
    slug: str,
    images: list[str],
    index_label: str,
    project_root: Path,
    exp_name: str,
    output_root: str | None,
    patch_index,
    params: dict,
    plot_fns: dict,
    overwrite: bool,
) -> str:
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

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
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key,
                       finding=finding_name, index_label=index_label)
    logger.info(f"finding='{finding_name}' index={index_label} images={len(images)}")

    tile_size = params["tile_size"]
    nprobe = params["nprobe"]
    rerank_pool = params["rerank_pool"]
    max_tiles_reranked = params["max_tiles_reranked"]

    from lib.query_embedding import embed_image_tiles

    all_tiles = []
    for image_path in images:
        tiles = embed_image_tiles(image_path, tile_size=tile_size)
        logger.info(f"  {image_path}: {tiles.shape[0]} tiles")
        all_tiles.append(tiles)
    query_vecs = np.concatenate(all_tiles, axis=0)
    logger.info(f"total tiles: {query_vecs.shape[0]}")

    # タイルスコアヒートマップ(近似FAISSスコア = exact re-rank の土俵に上がる
    # かどうかの選抜基準)。index_label ごとに空間が違うので両方で描く。
    heatmap_dir = run_dir / "tile_score_heatmap"
    heatmap_dir.mkdir(exist_ok=True)
    for image_path in images:
        fig = plot_fns["tile_scores"](
            image_path, patch_index, tile_size=tile_size, nprobe=nprobe,
            max_tiles_reranked=max_tiles_reranked,
        )
        fig.savefig(heatmap_dir / f"{_slug(Path(image_path).stem)}.png", dpi=150)
        plot_fns["close"](fig)

    similar_patches = patch_index.search_similar_patches_multi(
        query_vecs, k=params["k"], nprobe=nprobe, rerank_pool=rerank_pool,
        max_tiles_reranked=max_tiles_reranked,
    )
    similar_patches.to_csv(run_dir / "similar_patches.csv", index=False)

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
            fig = plot_fns["thumbnail"](slide_id, hits, params["thumbnails_dir"], patch_index.slide_meta)
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
        "finding": finding_name, "index_label": index_label,
        "n_tiles": int(query_vecs.shape[0]),
        "n_top_slides": int(len(top_slides)),
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"done: {variant_key}")
    return "ok"


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.search import PatchIndex
    from lib.visualize import plot_hit_patch_gallery, plot_query_tile_scores, plot_slide_hits_on_thumbnail

    import matplotlib.pyplot as plt

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    features_dir = project_root / config["features_dir"]
    baseline_index_dir = project_root / config["baseline_index_dir"]
    whiten_index_dir = project_root / config["whiten_index_dir"]
    whiten_transform_path = project_root / config["whiten_transform_path"]
    thumbnails_dir = project_root / config["thumbnails_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]

    findings = resolve_target_findings(project_root, config)
    print(f"whiten-routed atlas findings: {len(findings)} -> {[s for _, s, _ in findings]}")
    if not findings:
        raise SystemExit("no WHITEN_FINDINGS-routed atlas folder resolved — check lib.finding_routing")

    pi_baseline = PatchIndex.load(
        index_path=baseline_index_dir / "index.faiss",
        manifest_path=baseline_index_dir / "manifest.parquet",
        slide_meta_path=baseline_index_dir / "slide_meta.parquet",
        features_dir=features_dir,
    )
    pi_whiten = PatchIndex.load(
        index_path=whiten_index_dir / "index.faiss",
        manifest_path=whiten_index_dir / "manifest.parquet",
        slide_meta_path=whiten_index_dir / "slide_meta.parquet",
        features_dir=features_dir,
        transform_path=whiten_transform_path,
    )

    params = {
        "thumbnails_dir": thumbnails_dir,
        "raw_slide_dir": raw_slide_dir,
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

    n_ok = n_skip = n_fail = 0
    total = len(findings) * 2
    for finding_name, slug, images in findings:
        for index_label, pi in (("baseline", pi_baseline), ("routed", pi_whiten)):
            try:
                r = process_query_set(
                    finding_name=finding_name, slug=slug, images=images, index_label=index_label,
                    project_root=project_root, exp_name=exp_name, output_root=output_root,
                    patch_index=pi, params=params, plot_fns=plot_fns, overwrite=args.overwrite,
                )
                n_ok += r == "ok"
                n_skip += r == "skipped"
            except Exception:
                n_fail += 1
                traceback.print_exc()
                print(f"!! query set FAILED: {slug} / {index_label}")
            print(f"[{n_ok + n_skip + n_fail}/{total}] {slug} / {index_label}")

    print(f"Done: {n_ok} ok, {n_skip} skipped, {n_fail} failed, of {total}.")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
