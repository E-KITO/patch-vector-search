"""experiments/0043: Macenko染色正規化の基準パッチ選定を見直す。

現行の基準パッチ(data/baseline/63958_x38976_y7616.png)は
scripts/select_average_patch.py が一度だけ選んだもので、選定基準は「コーパス
40スライド(全体の4%)からサンプルしたUNI**埋め込み**の重心に最も近いパッチ」
だった。これはMacenko正規化が本来見るべき「染色ベクトル(H&E濃度プロファイル)
の典型性」を直接測ったものではなく(組織の見た目全体を混ぜた埋め込み空間の
プロキシ、スクリプト自身のdocstringが認めている)、かつ背景パッチ除去
(experiments/0017/0018)より前・今は存在しないパスでの選定だった。

この実験は、染色ベクトルそのものを直接見る、より原理に忠実な基準で選定を
やり直す診断ツール: 背景除去済みコーパス(experiments/0018)からランダム
サンプルした実パッチについて、torchstainのNumpyMacenkoNormalizerで各パッチ
自身の染色ベクトル(HERef、H&E各1本ずつのOD空間ベクトル)・最大濃度
(maxCRef)を推定し、その中央値(=コーパス全体の「典型的な染色」)への距離で
ランキングする。現行の基準パッチ自身もこの分布の中でどの位置にあるか(典型的
か外れ値か)を評価する。

Macenkoの染色ベクトル推定は画像ごとにH/Eの列順が入れ替わりうる(角度ソートの
不安定性)ため、距離計算では列を入れ替えた場合との小さい方を採用する。

注意: これは診断のみで、実際にMacenko索引(experiments/0010/0033、
data/trident_processed_uni_v1_macenko)を再構築するものではない(再構築には
wsi_preprocessパイプラインでの全コーパス再正規化が必要で、この実験の範囲外)。
GPU不要(torchstainのNumpy実装のみ、UNIエンコーダは呼ばない)。

出力(outputs/0043_.../default/):
  sampled_patches.csv          サンプルした全パッチの中央値への距離
  candidates/*.png             中央値に最も近い上位候補パッチ(新しい基準候補)
  current_reference_audit.json 現行基準パッチの距離・分布内順位
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
from PIL import Image


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
    """torchstainのNumpyMacenkoNormalizerを画像にfitし、(HERef, maxCRef)を返す。"""
    from torchstain.numpy.normalizers import NumpyMacenkoNormalizer

    n = NumpyMacenkoNormalizer()
    n.fit(img_arr)
    return np.asarray(n.HERef, dtype=np.float64), np.asarray(n.maxCRef, dtype=np.float64)


def _swap_stains(HE: np.ndarray, maxC: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H/Eの列順を入れ替えたバージョンを返す(角度ソートの不安定性対策)。"""
    HE_swapped = HE[:, ::-1] if HE.shape[-1] == 2 else HE[::-1, :]
    return HE_swapped, maxC[::-1]


def _distance_to_median(HE, maxC, HE_med, maxC_med) -> float:
    d1 = float(np.linalg.norm(HE - HE_med) + np.linalg.norm(maxC - maxC_med))
    HE_s, maxC_s = _swap_stains(HE, maxC)
    d2 = float(np.linalg.norm(HE_s - HE_med) + np.linalg.norm(maxC_s - maxC_med))
    return min(d1, d2)


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

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))
    slide_meta = pd.read_parquet(index_dir / "slide_meta.parquet")
    if slide_meta.index.name != "slide_id":
        slide_meta = slide_meta.assign(slide_id=slide_meta["slide_id"].astype(str)).set_index("slide_id")

    all_slides = manifest["slide_id"].unique()
    n_slides = min(int(config["n_slides"]), len(all_slides))
    chosen_slides = rng.choice(all_slides, size=n_slides, replace=False)

    rows = []
    n_fit_failed = 0
    for slide_id in chosen_slides:
        slide_rows = manifest[manifest["slide_id"] == slide_id]
        n_pick = min(int(config["n_patches_per_slide"]), len(slide_rows))
        picked = slide_rows.sample(n=n_pick, random_state=rng.integers(0, 2**31 - 1))
        patch_size_level0 = int(slide_meta.loc[slide_id, "patch_size_level0"])
        for _, r in picked.iterrows():
            try:
                img = crop_patch(
                    slide_id, int(r["coord_x"]), int(r["coord_y"]), raw_slide_dir,
                    patch_size_level0, target_size=tile_size,
                )
                HE, maxC = _fit_stain_vectors(np.array(img))
            except Exception as e:
                n_fit_failed += 1
                logger.warning(f"stain fit failed for {slide_id} ({r['coord_x']},{r['coord_y']}): {e!r}")
                continue
            rows.append({
                "slide_id": slide_id, "coord_x": int(r["coord_x"]), "coord_y": int(r["coord_y"]),
                "patch_size_level0": patch_size_level0, "HE": HE, "maxC": maxC,
            })
    logger.info(f"sampled {len(rows)} patches ok, {n_fit_failed} stain-fit failures, "
                f"from {n_slides} slides")

    HE_stack = np.stack([r["HE"] for r in rows], axis=0)
    maxC_stack = np.stack([r["maxC"] for r in rows], axis=0)
    HE_med = np.median(HE_stack, axis=0)
    maxC_med = np.median(maxC_stack, axis=0)
    logger.info(f"median HE:\n{HE_med}\nmedian maxC: {maxC_med}")

    for r in rows:
        r["distance_to_median"] = _distance_to_median(r["HE"], r["maxC"], HE_med, maxC_med)

    sampled_df = pd.DataFrame([
        {"slide_id": r["slide_id"], "coord_x": r["coord_x"], "coord_y": r["coord_y"],
         "distance_to_median": r["distance_to_median"]}
        for r in rows
    ]).sort_values("distance_to_median").reset_index(drop=True)
    sampled_df.to_csv(run_dir / "sampled_patches.csv", index=False)

    # --- 現行基準パッチをこの分布の中で評価 ---
    current_ref_path = project_root / config["current_reference_path"]
    current_img = Image.open(current_ref_path).convert("RGB")
    HE_cur, maxC_cur = _fit_stain_vectors(np.array(current_img))
    dist_cur = _distance_to_median(HE_cur, maxC_cur, HE_med, maxC_med)
    percentile = float((sampled_df["distance_to_median"] <= dist_cur).mean() * 100)
    audit = {
        "current_reference_path": str(config["current_reference_path"]),
        "distance_to_median": dist_cur,
        "percentile_among_sampled": percentile,
        "n_sampled": len(rows),
        "note": "percentile_among_sampledが低いほど典型(中央値に近い)側、高いほど外れ値側。",
    }
    (run_dir / "current_reference_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    logger.info(f"current reference: distance={dist_cur:.4f}, percentile={percentile:.1f}% "
                f"(among {len(rows)} sampled patches, lower=more typical)")

    # --- 中央値に近い上位候補を保存(max_candidates_per_slideで1スライド独占を防ぐ) ---
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(exist_ok=True)
    by_row = {(r["slide_id"], r["coord_x"], r["coord_y"]): r for r in rows}
    max_per_slide = int(config["max_candidates_per_slide"])
    per_slide_count: dict[str, int] = {}
    n_saved = 0
    saved_rows = []
    for _, row in sampled_df.iterrows():
        if n_saved >= int(config["top_k_candidates"]):
            break
        sid = row["slide_id"]
        if per_slide_count.get(sid, 0) >= max_per_slide:
            continue
        r = by_row[(row["slide_id"], row["coord_x"], row["coord_y"])]
        img = crop_patch(sid, row["coord_x"], row["coord_y"], raw_slide_dir, r["patch_size_level0"], target_size=tile_size)
        out_path = candidates_dir / f"{sid}_x{row['coord_x']}_y{row['coord_y']}.png"
        img.save(out_path)
        per_slide_count[sid] = per_slide_count.get(sid, 0) + 1
        n_saved += 1
        saved_rows.append({**row.to_dict(), "image_path": str(out_path)})
    pd.DataFrame(saved_rows).to_csv(run_dir / "top_candidates.csv", index=False)
    logger.info(f"saved {n_saved} candidate reference patches to {candidates_dir}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
