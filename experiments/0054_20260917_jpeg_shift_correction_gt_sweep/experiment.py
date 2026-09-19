"""experiments/0054: experiments/0053で切り分けたJPEG圧縮由来のドメインギャップ
成分から補正ベクトル(jpeg_shift)を作り、GT対応7所見で実際に補正効果を
alphaスイープ検証する。

experiments/0053は「同一パッチの生ピクセル埋め込み vs JPEG往復後埋め込み」の
ペアで、JPEG圧縮だけがatlas-corpusドメインギャップ(experiments/0038)の
8〜34%(品質次第)を説明することを確認した。この差ベクトルは中身が完全に
同一でJPEG往復の有無だけが違うため、experiments/0039・0041のdomain_shift
(atlas平均-corpus平均、生物学的要因とJPEG要因が混在)より純粋な「JPEG圧縮
だけの補正ベクトル」になっているはず、という仮説を検証する。

手順:
  1. atlas図版91枚(NNLアトラス25所見全フォルダ)のJPEGファイルからPILの
     量子化テーブル(IJG標準テーブルからの逆算)で実際の圧縮品質を推定し、
     この推定品質でjpeg_shiftベクトルを較正する。
  2. experiments/0053と同様にコーパスから生パッチ200枚を切り出し、
     (a) 生ピクセルのまま埋め込み (b) JPEG往復後に埋め込み、をplain空間・
     macenko空間(query側のnormalize_to_reference経由、experiments/0041の
     macenko domain_shift構築と同じ空間)の両方で行う。品質は較正品質に加え
     [95,85,70,50]でもシフトベクトルを作り、品質間の方向一貫性を確認する。
  3. 較正品質のjpeg_shiftで、lib.finding_routingの実ルーティング空間
     (plain/whiten/macenko)でGT対応7所見のalphaスイープ(experiments/0045
     と同じ枠組み)を行う。
  4. 「domain_shift(0039/0041) - jpeg_shift」の残差ベクトルについても、
     粗いalphaグリッドで同様に検証する。

出力(outputs/0054_.../default/):
  atlas_jpeg_quality_estimates.csv
  jpeg_shift_consistency.json
  gt_comparison_{family}_jpegshift.csv / _residual.csv
  best_alpha_by_finding_jpegshift.json / _residual.json
  summary.json
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

# IJG(ITU-T T.81 Annex K)標準輝度量子化テーブル(zigzag順、品質50基準)。
# PILがJPEGから読む im.quantization[0] と同じ並び順。
_BASE_LUMA_ZIGZAG = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]


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
    return ap.parse_args()


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _estimate_jpeg_quality(qtable: list[int]) -> float | None:
    """IJGのスケーリング式(quality<50: S=5000/quality, else S=200-2*quality、
    Tq[i]=round((base[i]*S+50)/100))を要素ごとに逆算し、中央値からqualityを
    復元する近似推定。1・255でクリップされた要素は情報を持たないため除外する。
    標準エンコーダ(libjpeg系)が生成したテーブルでのみ妥当——非標準テーブルの
    場合は不正確になりうる。
    """
    ratios = []
    for t, b in zip(qtable, _BASE_LUMA_ZIGZAG):
        if b <= 0 or t <= 1 or t >= 255:
            continue
        ratios.append((t * 100.0 - 50.0) / b)
    if len(ratios) < 16:
        return None
    s = float(np.median(ratios))
    if s <= 0:
        return 100.0
    q = 5000.0 / s if s < 100 else (200.0 - s) / 2.0
    return float(np.clip(q, 1.0, 100.0))


FAMILY_BY_FINDING = {
    "Single cell necrosis": "plain",
    "Deposit, glycogen": "plain",
    "Hematopoiesis, extramedullary": "plain",
    "Proliferation, Kupffer cell": "whiten",
    "Hypertrophy": "macenko",
    "Increased mitosis": "macenko",
    "Inclusion body, intracytoplasmic": "macenko",
}


def _pick_best_alpha_for_row(row: pd.Series, alphas: list[float], base_label: str) -> tuple[float, dict]:
    base_found = row[f"{base_label}_found"]
    base_best = row[f"{base_label}_best"]
    scored = []
    for alpha in alphas:
        found = row.get(f"alpha{alpha}_found")
        best = row.get(f"alpha{alpha}_best")
        regressed = bool(pd.isna(found) or found < base_found)
        rank_delta = float(best - base_best) if (pd.notna(best) and pd.notna(base_best)) else 0.0
        scored.append((alpha, regressed, rank_delta))
    chosen = min(scored, key=lambda t: (t[1], t[2], t[0]))
    return chosen[0], {
        "regressed": chosen[1], "rank_delta": chosen[2],
        "base_found": int(base_found) if pd.notna(base_found) else None,
        "base_best": int(base_best) if pd.notna(base_best) else None,
        "candidates": [{"alpha": a, "regressed": r, "rank_delta": d} for a, r, d in scored],
    }


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image, embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import list_atlas_folders
    from lib.raw_patch import crop_patch
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    rng = np.random.default_rng(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    # --- 1. atlas図版のJPEG品質を量子化テーブルから推定 ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    all_images = [img for _, images in folders for img in images]

    quality_rows = []
    for img_path in all_images:
        try:
            with Image.open(img_path) as im:
                qtables = getattr(im, "quantization", None)
                qtable0 = qtables[0] if qtables else None
        except Exception as e:
            logger.warning(f"failed to read {img_path}: {e!r}")
            qtable0 = None
        est = _estimate_jpeg_quality(list(qtable0)) if qtable0 else None
        quality_rows.append({"image": img_path, "estimated_quality": est})

    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(run_dir / "atlas_jpeg_quality_estimates.csv", index=False)
    valid_estimates = quality_df["estimated_quality"].dropna()
    if len(valid_estimates) < len(all_images) * 0.5:
        raise RuntimeError(
            f"JPEG品質推定に失敗した画像が多すぎる({len(valid_estimates)}/{len(all_images)}件成功)。"
            "atlas図版が非標準の量子化テーブルで生成されている可能性がある。"
        )
    calibrated_quality = int(round(float(valid_estimates.median())))
    logger.info(f"atlas JPEG quality estimates: n={len(valid_estimates)}/{len(all_images)}, "
                f"median={calibrated_quality}, min={valid_estimates.min():.1f}, max={valid_estimates.max():.1f}")

    quality_candidates = sorted(set(list(config["jpeg_quality_candidates"]) + [calibrated_quality]))

    # --- 2. probe patchesを切り出し、plain/macenko空間で生/JPEG往復embedを作る ---
    index_dir = project_root / config["index_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    n_slides = int(config["n_slides"])
    n_patches_per_slide = int(config["n_patches_per_slide"])

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))
    slide_meta = pd.read_parquet(index_dir / "slide_meta.parquet")
    if slide_meta.index.name != "slide_id":
        slide_meta = slide_meta.assign(slide_id=slide_meta["slide_id"].astype(str)).set_index("slide_id")

    all_slides = manifest["slide_id"].unique()
    n_slides = min(n_slides, len(all_slides))
    probe_slides = rng.choice(all_slides, size=n_slides, replace=False)

    raw_plain_vecs, raw_macenko_vecs = [], []
    jpeg_plain_by_q: dict[int, list[np.ndarray]] = {q: [] for q in quality_candidates}
    jpeg_macenko_by_q: dict[int, list[np.ndarray]] = {q: [] for q in quality_candidates}

    for slide_id in probe_slides:
        slide_rows = manifest[manifest["slide_id"] == slide_id]
        n_pick = min(n_patches_per_slide, len(slide_rows))
        picked = slide_rows.sample(n=n_pick, random_state=rng.integers(0, 2**31 - 1))
        patch_size_level0 = int(slide_meta.loc[slide_id, "patch_size_level0"])

        n_ok = 0
        for _, r in picked.iterrows():
            img = crop_patch(
                slide_id, int(r["coord_x"]), int(r["coord_y"]), raw_slide_dir,
                patch_size_level0, target_size=tile_size,
            )
            try:
                raw_macenko_vec = embed_image(normalize_to_reference(img, macenko_stain_ref), encoder_name="uni_v1")
                jpeg_imgs = {q: _jpeg_roundtrip(img, q) for q in quality_candidates}
                jpeg_macenko_vecs = {
                    q: embed_image(normalize_to_reference(jpeg_imgs[q], macenko_stain_ref), encoder_name="uni_v1")
                    for q in quality_candidates
                }
            except Exception as e:
                # 一部のパッチ(組織が薄い・背景寄り等)でtorchstainのMacenkoフィットが
                # 失敗しうる(lib.finding_routingの_macenko_tile_transformと同じ既知の
                # 失敗モード)。1パッチの失敗で数時間ジョブ全体を落とさないよう、
                # そのパッチはplain/macenko両方からスキップする(対応関係を保つため)。
                logger.warning(f"macenko fit failed for {slide_id} ({r['coord_x']},{r['coord_y']}): {e!r} — skipping")
                continue

            raw_plain_vecs.append(embed_image(img, encoder_name="uni_v1"))
            raw_macenko_vecs.append(raw_macenko_vec)
            for q in quality_candidates:
                jpeg_plain_by_q[q].append(embed_image(jpeg_imgs[q], encoder_name="uni_v1"))
                jpeg_macenko_by_q[q].append(jpeg_macenko_vecs[q])
            n_ok += 1
        logger.info(f"{slide_id}: {n_ok}/{n_pick} probe patches embedded")

    raw_plain_vecs = np.stack(raw_plain_vecs)
    raw_macenko_vecs = np.stack(raw_macenko_vecs)
    raw_plain_mean = raw_plain_vecs.mean(axis=0)
    raw_macenko_mean = raw_macenko_vecs.mean(axis=0)

    shift_plain_by_q = {q: np.stack(v).mean(axis=0) - raw_plain_mean for q, v in jpeg_plain_by_q.items()}
    shift_macenko_by_q = {q: np.stack(v).mean(axis=0) - raw_macenko_mean for q, v in jpeg_macenko_by_q.items()}

    jpeg_shift_plain = shift_plain_by_q[calibrated_quality]
    jpeg_shift_macenko = shift_macenko_by_q[calibrated_quality]
    np.save(run_dir / "jpeg_shift_plain.npy", jpeg_shift_plain)
    np.save(run_dir / "jpeg_shift_macenko.npy", jpeg_shift_macenko)

    # --- 品質間の方向一貫性(コサイン類似度) ---
    def _pairwise_cos(vecs_by_key: dict) -> dict:
        keys = sorted(vecs_by_key.keys())
        out = {}
        for i, ki in enumerate(keys):
            for kj in keys[i + 1:]:
                vi, vj = vecs_by_key[ki], vecs_by_key[kj]
                cos = float(np.dot(vi, vj) / (np.linalg.norm(vi) * np.linalg.norm(vj)))
                out[f"{ki}_vs_{kj}"] = cos
        return out

    consistency = {
        "calibrated_quality": calibrated_quality,
        "shift_norm_plain_by_quality": {str(q): float(np.linalg.norm(v)) for q, v in shift_plain_by_q.items()},
        "shift_norm_macenko_by_quality": {str(q): float(np.linalg.norm(v)) for q, v in shift_macenko_by_q.items()},
        "pairwise_cos_plain": _pairwise_cos(shift_plain_by_q),
        "pairwise_cos_macenko": _pairwise_cos(shift_macenko_by_q),
        "note": (
            "品質が違ってもshiftベクトルの方向(コサイン類似度)がほぼ一定なら、"
            "『単一方向をalphaでスケールする』線形補正の前提が妥当。大きくばらつく"
            "場合は品質ごとに別々の補正が必要になる。"
        ),
    }
    (run_dir / "jpeg_shift_consistency.json").write_text(json.dumps(consistency, indent=2, ensure_ascii=False))
    logger.info(f"jpeg_shift consistency:\n{json.dumps(consistency, indent=2)}")

    # --- 3. finding_routingの3空間を用意し、jpeg_shiftでGTスイープ ---
    def _plain_tile_embed(images) -> np.ndarray:
        return np.concatenate([embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0)

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _macenko_tile_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size, tile_transform=_macenko_tile_transform) for f in images],
            axis=0,
        )

    def _cached(base_fn):
        cache: dict[tuple, np.ndarray] = {}

        def _fn(images):
            key = tuple(images)
            if key not in cache:
                cache[key] = base_fn(images)
            return cache[key]

        return _fn

    def _make_corrected(cached_fn, shift, alpha):
        def _fn(images):
            return _l2_normalize(cached_fn(images) - alpha * shift)
        return _fn

    baseline_dir = project_root / finding_routing.BASELINE_INDEX_DIR
    whiten_dir = project_root / finding_routing.WHITEN_INDEX_DIR
    macenko_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    baseline_index = PatchIndex.load(
        index_path=baseline_dir / "index.faiss", manifest_path=baseline_dir / "manifest.parquet",
        slide_meta_path=baseline_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
    )
    whiten_index = PatchIndex.load(
        index_path=whiten_dir / "index.faiss", manifest_path=whiten_dir / "manifest.parquet",
        slide_meta_path=whiten_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
        transform_path=project_root / finding_routing.WHITEN_TRANSFORM_PATH,
    )
    macenko_index = PatchIndex.load(
        index_path=macenko_dir / "index.faiss", manifest_path=macenko_dir / "manifest.parquet",
        slide_meta_path=macenko_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.MACENKO_FEATURES_DIR,
    )

    plain_cached = _cached(_plain_tile_embed)
    macenko_cached = _cached(_macenko_tile_embed)

    families_jpegshift = {
        "plain": (baseline_index, plain_cached, jpeg_shift_plain,
                  ["Single cell necrosis", "Deposit, glycogen", "Hematopoiesis, extramedullary"]),
        "whiten": (whiten_index, plain_cached, jpeg_shift_plain, ["Proliferation, Kupffer cell"]),
        "macenko": (macenko_index, macenko_cached, jpeg_shift_macenko,
                    ["Hypertrophy", "Increased mitosis", "Inclusion body, intracytoplasmic"]),
    }

    def _sweep(families: dict, alphas: list[float], suffix: str) -> dict:
        best_alpha_by_finding: dict[str, dict] = {}
        for family_name, (patch_index, cached_fn, shift, findings) in families.items():
            pipelines = {"routed_base": (patch_index, cached_fn)}
            for alpha in alphas:
                pipelines[f"alpha{alpha}"] = (patch_index, _make_corrected(cached_fn, shift, alpha))

            gt_df = run_comparison(pipelines)
            assert len(gt_df) == len(CATEGORIES), (
                f"[{family_name}/{suffix}] expected {len(CATEGORIES)} category rows, got {len(gt_df)}"
            )
            gt_df["finding_type"] = list(CATEGORIES.values())
            gt_df.to_csv(run_dir / f"gt_comparison_{family_name}_{suffix}.csv", index=False)
            logger.info(f"[{family_name}/{suffix}] GT comparison:\n{gt_df.to_string()}")

            for finding in findings:
                row = gt_df.loc[gt_df["finding_type"] == finding].iloc[0]
                best_alpha, reason = _pick_best_alpha_for_row(row, alphas, base_label="routed_base")
                best_alpha_by_finding[finding] = {"family": family_name, "best_alpha": best_alpha, **reason}
                logger.info(f"[{family_name}/{suffix}] {finding}: best_alpha={best_alpha} ({reason})")
        return best_alpha_by_finding

    alphas = list(config["alphas"])
    best_alpha_jpegshift = _sweep(families_jpegshift, alphas, "jpegshift")
    (run_dir / "best_alpha_by_finding_jpegshift.json").write_text(
        json.dumps(best_alpha_jpegshift, indent=2)
    )

    # --- 4. domain_shift - jpeg_shift の残差を粗いグリッドで検証 ---
    domain_shift_plain = np.load(project_root / config["domain_shift_plain_path"])
    domain_shift_macenko = np.load(project_root / config["domain_shift_macenko_path"])
    residual_plain = domain_shift_plain - jpeg_shift_plain
    residual_macenko = domain_shift_macenko - jpeg_shift_macenko
    np.save(run_dir / "residual_shift_plain.npy", residual_plain)
    np.save(run_dir / "residual_shift_macenko.npy", residual_macenko)
    logger.info(f"domain_shift_plain norm={np.linalg.norm(domain_shift_plain):.4f}, "
                f"jpeg_shift_plain norm={np.linalg.norm(jpeg_shift_plain):.4f}, "
                f"residual_plain norm={np.linalg.norm(residual_plain):.4f}")
    logger.info(f"domain_shift_macenko norm={np.linalg.norm(domain_shift_macenko):.4f}, "
                f"jpeg_shift_macenko norm={np.linalg.norm(jpeg_shift_macenko):.4f}, "
                f"residual_macenko norm={np.linalg.norm(residual_macenko):.4f}")

    families_residual = {
        "plain": (baseline_index, plain_cached, residual_plain,
                  ["Single cell necrosis", "Deposit, glycogen", "Hematopoiesis, extramedullary"]),
        "whiten": (whiten_index, plain_cached, residual_plain, ["Proliferation, Kupffer cell"]),
        "macenko": (macenko_index, macenko_cached, residual_macenko,
                    ["Hypertrophy", "Increased mitosis", "Inclusion body, intracytoplasmic"]),
    }
    residual_alphas = list(config["residual_alphas"])
    best_alpha_residual = _sweep(families_residual, residual_alphas, "residual")
    (run_dir / "best_alpha_by_finding_residual.json").write_text(
        json.dumps(best_alpha_residual, indent=2)
    )

    # --- summary: 所見ごとにbase / jpegshift / residual のbest_rankを並べる ---
    summary_rows = []
    for finding in FAMILY_BY_FINDING:
        js = best_alpha_jpegshift.get(finding, {})
        rs = best_alpha_residual.get(finding, {})
        summary_rows.append({
            "finding": finding, "family": FAMILY_BY_FINDING[finding],
            "base_best": js.get("base_best"),
            "jpegshift_best_alpha": js.get("best_alpha"), "jpegshift_rank_delta": js.get("rank_delta"),
            "residual_best_alpha": rs.get("best_alpha"), "residual_rank_delta": rs.get("rank_delta"),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary_df.to_string()}")

    (run_dir / "summary.json").write_text(json.dumps({
        "calibrated_quality": calibrated_quality,
        "n_probe_slides": len(probe_slides),
        "n_probe_patches": len(raw_plain_vecs),
        "jpeg_shift_norm_plain": float(np.linalg.norm(jpeg_shift_plain)),
        "jpeg_shift_norm_macenko": float(np.linalg.norm(jpeg_shift_macenko)),
        "domain_shift_norm_plain": float(np.linalg.norm(domain_shift_plain)),
        "domain_shift_norm_macenko": float(np.linalg.norm(domain_shift_macenko)),
        "findings": summary_rows,
    }, indent=2, ensure_ascii=False))

    complete_run(run_dir)


if __name__ == "__main__":
    main()
