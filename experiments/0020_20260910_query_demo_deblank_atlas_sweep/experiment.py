"""experiments/0020: NNL アトラス全所見スイープ（背景除去 0018 vs 0002）。

初代のアトラス検索 Artifact — 所見ごとに「クエリ画像のタイル行」＋「コーパスから
引いた上位パッチ行（slide_id / sim ラベル付き）」を並べたもの — を、今回までに
改善した検索経路で作り直すためのデータ生成。

experiments/0013 の per-finding atlas sweep（--atlas-root、所見フォルダ＝1クエリ）を
踏襲するが:
  - 索引は 2 本（0018 = 背景除去済み既定、0002 = 背景除去前）を同一クエリで検索
  - クエリは uni_v1 plain タイリング（染色正規化なし = 現行既定経路。探索パラメータも
    notebooks/01_query_demo.ipynb の推奨値に合わせる: rerank_pool=200 / max_tiles_reranked=12）
  - matplotlib 図版ではなく個別 JPEG を書き出す（Artifact への base64 埋め込み用）

スイープ後の report.html 生成は scripts/build_atlas_report.py（run_slurm.sh が続けて呼ぶ）。

出力（所見ごと、outputs/0020_.../finding__<slug>/）:
  query_tiles/NN_MMM.jpg      クエリ図版のタイル（NN=図版index, MMM=タイルindex）
  hits_deblank/RR.jpg         0018 索引の第 RR 位パッチ（実解像度クロップ）
  hits_predeblank/RR.jpg      0002 索引の第 RR 位パッチ
  finding.json                所見名・図版数・タイル数・両索引のヒット表
"""

import argparse
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
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--atlas-root", type=str, default=None,
        help="所見フォルダの親（省略時は config.atlas_root）。各直下サブフォルダが "
        "1 所見 = 1 クエリ（フォルダ内の全図版を統合）。",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="completion.json が completed の所見も再実行する。",
    )
    return parser.parse_args()


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")


def _staged_or(project_root: Path, config_rel: str, env_var: str) -> Path:
    """NFS の元パスより、run_slurm.sh の PRE_NATIVE_COMMAND が用意した
    ノードローカル SSD のステージコピー（env_var）を優先する。"""
    staged = os.environ.get(env_var)
    if staged and Path(staged).exists():
        return Path(staged)
    return project_root / config_rel


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


def resolve_findings(atlas_root: Path, atlas_csv: Path) -> list[tuple[str, str, list[str]]]:
    """-> [(finding_name, slug, [image_path, ...]), ...]（画像のあるフォルダのみ）。

    `lib.atlas_figures.query_images` が「Normal liver ... for comparison」対照図版
    （`is_normal_control=yes`）を除外する。肝 NNL では Atrophy と Hepatocyte -
    Hypertrophy の計 4 枚のみが該当。
    """
    from lib.atlas_figures import query_images

    if not atlas_root.is_dir():
        raise SystemExit(f"atlas root {atlas_root} is not a directory")
    out = []
    for sub in sorted(p for p in atlas_root.iterdir() if p.is_dir()):
        imgs = [str(p) for p in query_images(sub, atlas_csv=atlas_csv)]
        if imgs:
            out.append((sub.name, _slug(sub.name)[:120], imgs))
    if not out:
        raise SystemExit(f"atlas root {atlas_root} has no subfolder with images")
    return out


def _grid_crops(image_path: str, tile_size: int):
    """embed_image_tiles と同じグリッド分割・彩度ベース空白タイル除外を行い、
    生き残ったタイル画像（PIL）を返す。検索ベクトルは embed_image_tiles 側で
    別途計算するので、これはサムネイル表示専用（除外基準は lib.patch_blankness
    で揃えてあるので枚数は embed_image_tiles と一致する）。"""
    from PIL import Image

    from lib.patch_blankness import blankness_metrics, is_background

    def _is_blank(crop):
        return bool(is_background(blankness_metrics(crop)["sat_frac"]))

    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    scale = max(tile_size / w, tile_size / h, 1.0)
    if scale > 1.0:
        image = image.resize(
            (max(tile_size, round(w * scale)), max(tile_size, round(h * scale))), Image.LANCZOS
        )
        w, h = image.size
    xs = sorted(set(list(range(0, w - tile_size + 1, tile_size)) + [w - tile_size]))
    ys = sorted(set(list(range(0, h - tile_size + 1, tile_size)) + [h - tile_size]))
    crops = [image.crop((x, y, x + tile_size, y + tile_size)) for y in ys for x in xs]
    non_blank = [c for c in crops if not _is_blank(c)]
    return non_blank or crops


def process_finding(
    *,
    finding_name: str,
    slug: str,
    images: list[str],
    project_root: Path,
    exp_name: str,
    output_root: str | None,
    indexes: dict,
    raw_slide_dir: Path,
    params: dict,
    overwrite: bool,
) -> str:
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.raw_patch import crop_patch

    import numpy as np
    from PIL import Image

    variant_key = "finding__" + slug
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
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, finding=finding_name)
    logger.info(f"finding='{finding_name}'  images={len(images)}")

    tile_size = params["tile_size"]
    thumb_px = params["thumb_px"]

    # --- クエリ: 図版ごとにタイル分割して埋め込み ---
    all_vecs = []
    per_image_tiles = []
    tiles_dir = run_dir / "query_tiles"
    tiles_dir.mkdir(exist_ok=True)
    n_shown = 0
    for img_idx, image_path in enumerate(images):
        vecs = embed_image_tiles(image_path, tile_size=tile_size)
        all_vecs.append(vecs)
        per_image_tiles.append(int(vecs.shape[0]))
        logger.info(f"  [{img_idx}] {Path(image_path).name}: {vecs.shape[0]} tiles")
        for tile_idx, crop in enumerate(_grid_crops(image_path, tile_size)):
            if n_shown >= params["n_query_tiles_shown"]:
                break
            crop.resize((thumb_px, thumb_px), Image.LANCZOS).convert("RGB").save(
                tiles_dir / f"{img_idx:02d}_{tile_idx:03d}.jpg", quality=88
            )
            n_shown += 1
    query_vecs = np.concatenate(all_vecs, axis=0)
    n_tiles = int(query_vecs.shape[0])
    logger.info(f"  total tiles: {n_tiles}")

    # --- 各索引で検索 → 実解像度パッチをクロップ ---
    results: dict[str, list[dict]] = {}
    for label, index in indexes.items():
        hits = index.search_similar_patches_multi(
            query_vecs, k=params["k"], nprobe=params["nprobe"],
            rerank_pool=params["rerank_pool"], max_tiles_reranked=params["max_tiles_reranked"],
        )
        hits_dir = run_dir / f"hits_{label}"
        hits_dir.mkdir(exist_ok=True)
        rows = []
        for rank, row in enumerate(hits.itertuples(), start=1):
            rec = {
                "rank": rank,
                "slide_id": str(row.slide_id),
                "coord_x": int(row.coord_x),
                "coord_y": int(row.coord_y),
                "similarity": round(float(row.similarity), 4),
            }
            try:
                psl0 = int(index.slide_meta.loc[row.slide_id, "patch_size_level0"])
                patch = crop_patch(row.slide_id, row.coord_x, row.coord_y, raw_slide_dir, psl0)
                patch.resize((thumb_px, thumb_px), Image.LANCZOS).convert("RGB").save(
                    hits_dir / f"{rank:02d}.jpg", quality=88
                )
                rec["patch_file"] = f"hits_{label}/{rank:02d}.jpg"
            except Exception as e:
                logger.warning(f"  crop failed ({label} rank {rank}, slide {row.slide_id}): {e!r}")
                rec["patch_file"] = None
            rows.append(rec)
        results[label] = rows
        sims = [r["similarity"] for r in rows]
        logger.info(f"  {label}: {len(rows)} hits, sim {min(sims):.3f}..{max(sims):.3f}")

    (run_dir / "finding.json").write_text(json.dumps({
        "finding": finding_name,
        "slug": slug,
        "images": [Path(p).name for p in images],
        "n_images": len(images),
        "n_tiles": n_tiles,
        "per_image_tiles": per_image_tiles,
        "n_query_tiles_saved": n_shown,
        "params": {k: params[k] for k in ("k", "nprobe", "rerank_pool", "max_tiles_reranked", "tile_size")},
        "results": results,
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info(f"done: {variant_key}")
    return "ok"


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.search import PatchIndex

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    features_dir = _staged_or(project_root, config["features_dir"], "PVS_FEATURES_DIR")
    index_deblank_dir = _staged_or(project_root, config["index_deblank_dir"], "PVS_INDEX_DEBLANK_DIR")
    index_predeblank_dir = _staged_or(project_root, config["index_predeblank_dir"], "PVS_INDEX_PREDEBLANK_DIR")
    raw_slide_dir = project_root / config["raw_slide_dir"]
    atlas_root = Path(args.atlas_root) if args.atlas_root else project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]

    print(f"features_dir:          {features_dir}")
    print(f"index_deblank_dir:     {index_deblank_dir}")
    print(f"index_predeblank_dir:  {index_predeblank_dir}")
    print(f"atlas_root:            {atlas_root}")

    findings = resolve_findings(atlas_root, atlas_csv)
    print(f"findings: {len(findings)} -> {[s for _, s, _ in findings]}")

    # 全所見が完了済みなら索引ロードを省く（run_slurm.sh が続けて呼ぶ
    # build_atlas_report.py だけを回す「report のみ再実行」の高速パス）。
    all_done = not args.overwrite and all(
        _completed(project_root / "outputs" / exp_name / ("finding__" + slug))
        for _, slug, _ in findings
    )
    if all_done:
        print("all findings already complete — skipping index load (report-only rerun)")
        indexes: dict = {}
    else:
        # "deblank" を先に（既定索引）。dict は挿入順を保つので report もこの順で並ぶ。
        indexes = {
            "deblank": PatchIndex.load(
                index_path=index_deblank_dir / "index.faiss",
                manifest_path=index_deblank_dir / "manifest.parquet",
                slide_meta_path=index_deblank_dir / "slide_meta.parquet",
                features_dir=features_dir,
            ),
            "predeblank": PatchIndex.load(
                index_path=index_predeblank_dir / "index.faiss",
                manifest_path=index_predeblank_dir / "manifest.parquet",
                slide_meta_path=index_predeblank_dir / "slide_meta.parquet",
                features_dir=features_dir,
            ),
        }

    params = {
        "tile_size": int(config.get("tile_size", 224)),
        "k": int(config.get("k", 10)),
        "nprobe": int(config.get("nprobe", 32)),
        "rerank_pool": int(config.get("rerank_pool", 200)),
        "max_tiles_reranked": config.get("max_tiles_reranked", None),
        "n_query_tiles_shown": int(config.get("n_query_tiles_shown", 24)),
        "thumb_px": int(config.get("thumb_px", 224)),
    }

    n_ok = n_skip = n_fail = 0
    for finding_name, slug, images in findings:
        try:
            r = process_finding(
                finding_name=finding_name, slug=slug, images=images,
                project_root=project_root, exp_name=exp_name, output_root=output_root,
                indexes=indexes, raw_slide_dir=raw_slide_dir, params=params,
                overwrite=args.overwrite,
            )
            n_ok += r == "ok"
            n_skip += r == "skipped"
        except Exception:
            n_fail += 1
            traceback.print_exc()
            print(f"!! finding FAILED: {slug}")
        print(f"[{n_ok + n_skip + n_fail}/{len(findings)}] {slug}")

    print(f"sweep done: {n_ok} ok, {n_skip} skipped, {n_fail} failed of {len(findings)}")
    # report.html は run_slurm.sh が続けて scripts/build_atlas_report.py を呼んで生成する。


if __name__ == "__main__":
    main()
