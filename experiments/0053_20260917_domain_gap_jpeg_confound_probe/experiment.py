"""experiments/0053: atlas vs corpus ドメインギャップ(experiments/0038)のうち
「拡張子(JPEG圧縮経路)の違い」に起因する分を切り分ける。

experiments/0038はatlas図版(全て.jpg、PIL.Image.openでロード)とcorpusパッチ
(.svsから直接切り出した生ピクセルをwsi_preprocessでUNI埋め込みしたh5、JPEGを
一度も経由しない)の間にほぼ完全なドメイン分離(所見によらずスコア差10〜12)を
見つけた。これまでの説明は「スキャナ・染色・解像度・図版特有のノイズ」だったが、
「JPEG圧縮を経由するかどうか」というパイプラインの違い自体が原因の一部になって
いないかは切り分けられていなかった(README「wsi_preprocess連携」
experiments/0044の設計メモで同種の懸念が別の比較軸で指摘されていたのと同じ論点)。

手順:
  1. experiments/0038と同じ手順(lib.ovr_scoring.embed_atlas_images_as_positives /
     sample_negative_pool / train_logreg)でatlas vs corpusのドメイン分類器を
     学習する。ただし本実験でJPEG往復プローブに使うスライドは負例プールの
     サンプリング元から除外する(同一パッチが学習・評価の両方に混入する
     リークを避ける)。
  2. コーパスから新たにn_slides×n_patches_per_slide枚の生パッチを
     lib.raw_patch.crop_patchで切り出す。各パッチについて:
       (a) 生ピクセルのままlib.query_embedding.embed_imageでUNI埋め込み
           (現行corpusパイプラインと同じ経路)
       (b) 同じピクセルを一度JPEGにエンコード→デコードしてから同じ埋め込み
           (config.jpeg_qualitiesの各品質でスイープ)
  3. (a)は同じ座標の既存h5特徴量とcos類似度を取り、再埋め込みパイプライン
     自体がcorpusと一致することを確認する(この前提が崩れていると本実験の
     結論自体が無効になるため必須のサニティチェック)。
  4. 分類器のdecision_functionで(a)と(b)各品質をスコアし、(b)-(a)の
     シフト量を、experiments/0038で確認済みのatlas-corpus間の本物のギャップ
     (所見によらずおよそ10〜12)と比較する。

出力(outputs/0053_.../default/):
  per_patch_scores.csv   パッチごとの生/JPEG各品質のスコア・cos類似度
  summary.json           サニティチェック・シフト量の集計・ギャップに対する比率
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


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.ovr_scoring import (
        list_atlas_folders, embed_atlas_images_as_positives, sample_negative_pool,
        train_logreg, _read_patch_vectors, _l2_normalize,
    )
    from lib.raw_patch import crop_patch
    from lib.query_embedding import embed_image

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
    features_dir = project_root / config["features_dir"]
    raw_slide_dir = project_root / config["raw_slide_dir"]
    tile_size = int(config.get("tile_size", 224))
    n_slides = int(config["n_slides"])
    n_patches_per_slide = int(config["n_patches_per_slide"])
    jpeg_qualities = list(config["jpeg_qualities"])

    manifest = pd.read_parquet(index_dir / "manifest.parquet")
    manifest = manifest.assign(slide_id=manifest["slide_id"].astype(str))
    slide_meta = pd.read_parquet(index_dir / "slide_meta.parquet")
    if slide_meta.index.name != "slide_id":
        slide_meta = slide_meta.assign(slide_id=slide_meta["slide_id"].astype(str)).set_index("slide_id")

    # --- 1. JPEG往復プローブに使うスライドを先に選ぶ(分類器の負例プールから除外するため) ---
    all_slides = manifest["slide_id"].unique()
    n_slides = min(n_slides, len(all_slides))
    probe_slides = rng.choice(all_slides, size=n_slides, replace=False)
    logger.info(f"probe slides ({n_slides}): {sorted(probe_slides.tolist())}")

    # --- 2. experiments/0038と同じ手順でatlas vs corpusドメイン分類器を学習 ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    pos_vecs_list = []
    for folder_name, images in folders:
        vecs, _ = embed_atlas_images_as_positives(images, tile_size)
        pos_vecs_list.append(vecs)
    pos_vecs = np.concatenate(pos_vecs_list, axis=0)
    logger.info(f"pooled atlas tiles: {pos_vecs.shape[0]} from {len(folders)} findings")

    neg_vecs, neg_slides = sample_negative_pool(
        manifest, features_dir, n=config["n_negative_patches"],
        exclude_slides=set(probe_slides.tolist()), rng=rng,
    )
    logger.info(f"negative patches sampled: {neg_vecs.shape[0]} from {len(set(neg_slides))} slides "
                f"(probe slides excluded)")

    clf = train_logreg(pos_vecs, neg_vecs, C=config["classifier_C"], seed=config.get("seed", 42))
    atlas_scores = clf.decision_function(pos_vecs)
    neg_scores = clf.decision_function(neg_vecs)
    atlas_corpus_gap = float(atlas_scores.mean() - neg_scores.mean())
    logger.info(f"atlas domain score: mean={atlas_scores.mean():.3f}, "
                f"corpus(negative pool) domain score: mean={neg_scores.mean():.3f}, "
                f"gap={atlas_corpus_gap:.3f}")

    # --- 3. probe_slidesから生パッチを切り出し、生/JPEG往復の各埋め込みをスコア ---
    rows = []
    for slide_id in probe_slides:
        slide_rows = manifest[manifest["slide_id"] == slide_id]
        n_pick = min(n_patches_per_slide, len(slide_rows))
        picked = slide_rows.sample(n=n_pick, random_state=rng.integers(0, 2**31 - 1))
        patch_size_level0 = int(slide_meta.loc[slide_id, "patch_size_level0"])

        for _, r in picked.iterrows():
            coord_x, coord_y, local_idx = int(r["coord_x"]), int(r["coord_y"]), int(r["local_idx"])
            img = crop_patch(slide_id, coord_x, coord_y, raw_slide_dir, patch_size_level0, target_size=tile_size)

            raw_vec = embed_image(img, encoder_name="uni_v1")
            precomputed_vec = _l2_normalize(
                _read_patch_vectors(slide_id, np.array([local_idx]), features_dir)
            )[0]
            cos_self = float(np.dot(raw_vec, precomputed_vec))  # 両方L2正規化済み

            row = {
                "slide_id": slide_id, "coord_x": coord_x, "coord_y": coord_y,
                "cos_self_raw_vs_precomputed": cos_self,
                "score_raw": float(clf.decision_function(raw_vec[None, :])[0]),
            }
            for q in jpeg_qualities:
                jpeg_img = _jpeg_roundtrip(img, q)
                jpeg_vec = embed_image(jpeg_img, encoder_name="uni_v1")
                row[f"score_jpeg_q{q}"] = float(clf.decision_function(jpeg_vec[None, :])[0])
                row[f"cos_raw_vs_jpeg_q{q}"] = float(np.dot(raw_vec, jpeg_vec))
            rows.append(row)
        logger.info(f"{slide_id}: {n_pick} patches processed")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_patch_scores.csv", index=False)

    # --- 4. サニティチェックとシフト量の集計 ---
    cos_self_median = float(df["cos_self_raw_vs_precomputed"].median())
    cos_self_min = float(df["cos_self_raw_vs_precomputed"].min())

    shift_summary = {}
    for q in jpeg_qualities:
        shift = df[f"score_jpeg_q{q}"] - df["score_raw"]
        shift_summary[str(q)] = {
            "mean_shift": float(shift.mean()),
            "median_shift": float(shift.median()),
            "mean_cos_raw_vs_jpeg": float(df[f"cos_raw_vs_jpeg_q{q}"].mean()),
            "frac_of_atlas_corpus_gap": float(shift.mean() / atlas_corpus_gap) if atlas_corpus_gap else None,
            "n_crossed_to_atlas_side": int(((df["score_raw"] < 0) & (df[f"score_jpeg_q{q}"] >= 0)).sum()),
        }

    summary = {
        "n_probe_slides": len(probe_slides),
        "n_probe_patches": len(df),
        "atlas_domain_score_mean": float(atlas_scores.mean()),
        "corpus_negative_pool_score_mean": float(neg_scores.mean()),
        "atlas_corpus_gap": atlas_corpus_gap,
        "raw_reembed_score_mean": float(df["score_raw"].mean()),
        "sanity_check_reembedding_cos_self": {
            "median": cos_self_median, "min": cos_self_min,
            "note": (
                "生パッチの再埋め込みが既存corpus h5特徴量とcos_sim~1.0で一致するか"
                "(README「Tier 1 roundtrip」と同じ発想)。これが崩れていると"
                "以下のJPEGシフト量の解釈は無効。"
            ),
        },
        "jpeg_shift_by_quality": shift_summary,
        "verdict": (
            "拡張子(JPEG圧縮)がドメインギャップの主要因" if
            max((v["frac_of_atlas_corpus_gap"] or 0) for v in shift_summary.values()) > 0.3
            else "拡張子(JPEG圧縮)はドメインギャップの主要因ではない"
        ),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"summary:\n{json.dumps(summary, indent=2, ensure_ascii=False)}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
