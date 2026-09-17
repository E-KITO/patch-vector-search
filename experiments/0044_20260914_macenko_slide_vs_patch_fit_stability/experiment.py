"""experiments/0044: Macenko染色ベクトル推定を「パッチごと」vs「スライド全体
(パッチ多数プール)で1回」で比較する事前検証。

ユーザー提案(図版・スライド全体を1回でMacenko正規化してからタイル分割する
案)の事前検証。現行は224pxパッチ1枚ごとに独立して染色ベクトル(HERef)・
最大濃度(maxCRef)を推定している。これをスライド単位のfitに変えると安定
するのではないか、という仮説を、全コーパスを再構築する前に小規模に検証する。

当初JPGサムネイル(data/trident_processed/thumbnails/)をスライド単位fitの
入力に使う案を検討したが、JPEG圧縮・ダウンサンプリング時の画素混合が
「.svsから直接パッチを切り出す現行パイプライン」と異なる色処理経路になり、
本来比較したい「パッチ単位 vs スライド単位」という軸に「.svs直接 vs JPG経由」
という別の交絡が混入するため却下した(ユーザー指摘)。この実験では**同じ
.svsからlib.raw_patch.crop_patchで直接切り出したパッチのみ**を使い、
(a)パッチごとに個別にfit(現行方式)と、(b)同じスライドの複数パッチを1枚に
連結してから1回だけfit(スライド単位方式)を比較する——色処理経路を完全に
同一にした上で、fitに使うピクセル量(粒度)だけを変える。

各スライドについて:
  - スライド単位fit(20パッチ連結)を「そのスライドの基準」として、各パッチの
    個別fitとの距離(orientation入れ替えの小さい方を採用)を測る
    (= 現行方式がスライド単位方式からどれだけ乖離するか)。
  - 同じ基準でorientationを揃えた上で、パッチ個別fit同士の「そのスライドの
    パッチ平均」への距離も測る(= 現行方式そのものの自己一貫性/不安定さ)。
  - 比較の物差しとして、スライド単位fit同士(スライド間)の距離も測る
    (= 保存すべき「本物のスライド間染色差」の大きさ)。パッチ内不安定さが
    スライド間差より十分小さければ問題は無く、同程度以上なら現行方式の
    ノイズが本物の信号を覆い隠している可能性が高い。

GPU不要(torchstainのNumpy実装のみ、UNIエンコーダは呼ばない)。
wsi_preprocessも不要(既存のcrop_patch + torchstainで完結)。

出力(outputs/0044_.../default/):
  per_patch_distances.csv   パッチごとの「スライド単位fitからの距離」「パッチ平均からの距離」
  per_slide_summary.csv     スライドごとの集計(パッチ内スプレッド、スライド単位fit)
  summary.json               全体集計(パッチ内 vs スライド間のスプレッド比較)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
    return ap.parse_args()


def _fit_stain_vectors(img_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from torchstain.numpy.normalizers import NumpyMacenkoNormalizer

    n = NumpyMacenkoNormalizer()
    n.fit(img_arr)
    return np.asarray(n.HERef, dtype=np.float64), np.asarray(n.maxCRef, dtype=np.float64)


def _swap_stains(HE: np.ndarray, maxC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    HE_swapped = HE[:, ::-1] if HE.shape[-1] == 2 else HE[::-1, :]
    return HE_swapped, maxC[::-1]


def _aligned_distance(HE_a, maxC_a, HE_b, maxC_b) -> tuple[float, np.ndarray, np.ndarray]:
    """(HE_a, maxC_a) を (HE_b, maxC_b) に最も近くなるorientationに揃えた上で
    距離を返す。Returns (distance, HE_a_aligned, maxC_a_aligned)。"""
    d1 = float(np.linalg.norm(HE_a - HE_b) + np.linalg.norm(maxC_a - maxC_b))
    HE_s, maxC_s = _swap_stains(HE_a, maxC_a)
    d2 = float(np.linalg.norm(HE_s - HE_b) + np.linalg.norm(maxC_s - maxC_b))
    if d1 <= d2:
        return d1, HE_a, maxC_a
    return d2, HE_s, maxC_s


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.raw_patch import crop_patch

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)
    rng = np.random.default_rng(config.get("seed", 42))

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    index_dir = project_root / config["index_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    tile_size = int(config.get("tile_size", 224))
    n_patches_per_slide = int(config["n_patches_per_slide"])

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))
    slide_meta = pd.read_parquet(index_dir / "slide_meta.parquet")
    if slide_meta.index.name != "slide_id":
        slide_meta = slide_meta.assign(slide_id=slide_meta["slide_id"].astype(str)).set_index("slide_id")

    all_slides = manifest["slide_id"].unique()
    n_slides = min(int(config["n_slides"]), len(all_slides))
    chosen_slides = rng.choice(all_slides, size=n_slides, replace=False)

    patch_rows, slide_rows_out = [], []
    slide_level_fits: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for slide_id in chosen_slides:
        slide_manifest_rows = manifest[manifest["slide_id"] == slide_id]
        n_pick = min(n_patches_per_slide, len(slide_manifest_rows))
        picked = slide_manifest_rows.sample(n=n_pick, random_state=rng.integers(0, 2**31 - 1))
        patch_size_level0 = int(slide_meta.loc[slide_id, "patch_size_level0"])

        patch_arrays, patch_coords = [], []
        for _, r in picked.iterrows():
            img = crop_patch(
                slide_id, int(r["coord_x"]), int(r["coord_y"]), raw_slide_dir,
                patch_size_level0, target_size=tile_size,
            )
            patch_arrays.append(np.array(img))
            patch_coords.append((int(r["coord_x"]), int(r["coord_y"])))
        if len(patch_arrays) < 3:
            logger.warning(f"skipping {slide_id}: only {len(patch_arrays)} patches available")
            continue

        # (b) スライド単位fit: 同じ.svs由来パッチを縦に連結して1回だけfit
        pooled_img = np.concatenate(patch_arrays, axis=0)
        try:
            HE_slide, maxC_slide = _fit_stain_vectors(pooled_img)
        except Exception as e:
            logger.warning(f"slide-level fit failed for {slide_id}: {e!r}")
            continue
        slide_level_fits[slide_id] = (HE_slide, maxC_slide)

        # (a) パッチごとに個別fit、スライド単位fitに揃えてorientation補正
        aligned = []
        for (cx, cy), arr in zip(patch_coords, patch_arrays):
            try:
                HE_p, maxC_p = _fit_stain_vectors(arr)
            except Exception as e:
                logger.warning(f"per-patch fit failed for {slide_id} ({cx},{cy}): {e!r}")
                continue
            dist_to_slide, HE_p_al, maxC_p_al = _aligned_distance(HE_p, maxC_p, HE_slide, maxC_slide)
            aligned.append((cx, cy, HE_p_al, maxC_p_al, dist_to_slide))

        if len(aligned) < 3:
            logger.warning(f"skipping {slide_id}: only {len(aligned)} successful per-patch fits")
            continue

        patch_mean_HE = np.mean([a[2] for a in aligned], axis=0)
        patch_mean_maxC = np.mean([a[3] for a in aligned], axis=0)
        for cx, cy, HE_p_al, maxC_p_al, dist_to_slide in aligned:
            dist_to_patch_mean = float(
                np.linalg.norm(HE_p_al - patch_mean_HE) + np.linalg.norm(maxC_p_al - patch_mean_maxC)
            )
            patch_rows.append({
                "slide_id": slide_id, "coord_x": cx, "coord_y": cy,
                "dist_to_slide_level_fit": dist_to_slide,
                "dist_to_patch_mean": dist_to_patch_mean,
            })

        slide_rows_out.append({
            "slide_id": slide_id, "n_patches": len(aligned),
            "patch_spread_mean": float(np.mean([r["dist_to_patch_mean"] for r in patch_rows if r["slide_id"] == slide_id])),
            "patch_to_slide_level_mean": float(np.mean([r["dist_to_slide_level_fit"] for r in patch_rows if r["slide_id"] == slide_id])),
        })
        logger.info(f"{slide_id}: {len(aligned)} patches, "
                    f"patch_spread_mean={slide_rows_out[-1]['patch_spread_mean']:.4f}, "
                    f"patch_to_slide_level_mean={slide_rows_out[-1]['patch_to_slide_level_mean']:.4f}")

    patch_df = pd.DataFrame(patch_rows)
    patch_df.to_csv(run_dir / "per_patch_distances.csv", index=False)
    slide_df = pd.DataFrame(slide_rows_out)
    slide_df.to_csv(run_dir / "per_slide_summary.csv", index=False)

    # --- 比較の物差し: スライド単位fit同士の「スライド間」距離(保存すべき本物の信号) ---
    slide_ids = list(slide_level_fits.keys())
    between_slide_dists = []
    for i in range(len(slide_ids)):
        for j in range(i + 1, len(slide_ids)):
            HE_i, maxC_i = slide_level_fits[slide_ids[i]]
            HE_j, maxC_j = slide_level_fits[slide_ids[j]]
            d, _, _ = _aligned_distance(HE_i, maxC_i, HE_j, maxC_j)
            between_slide_dists.append(d)
    between_slide_dists = np.array(between_slide_dists)

    summary = {
        "n_slides": len(slide_ids),
        "n_patches_total": len(patch_df),
        "within_slide_patch_spread": {
            "mean": float(patch_df["dist_to_patch_mean"].mean()),
            "median": float(patch_df["dist_to_patch_mean"].median()),
            "std": float(patch_df["dist_to_patch_mean"].std()),
        },
        "patch_to_slide_level_fit": {
            "mean": float(patch_df["dist_to_slide_level_fit"].mean()),
            "median": float(patch_df["dist_to_slide_level_fit"].median()),
            "std": float(patch_df["dist_to_slide_level_fit"].std()),
        },
        "between_slide_level_fit_distance": {
            "mean": float(between_slide_dists.mean()),
            "median": float(np.median(between_slide_dists)),
            "std": float(between_slide_dists.std()),
            "n_pairs": len(between_slide_dists),
        },
        "note": (
            "within_slide_patch_spread(パッチ内不安定さ) が "
            "between_slide_level_fit_distance(スライド間の本物の染色差)に比べて"
            "大きいほど、現行のパッチごとfitのノイズが実際のスライド間信号を"
            "覆い隠している可能性が高い。"
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary:\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
