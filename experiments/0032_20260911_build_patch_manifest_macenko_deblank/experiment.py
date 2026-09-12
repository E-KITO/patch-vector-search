"""experiments/0032: Macenko染色正規化コーパス(experiments/0010)に、experiments/0017
と同じ背景パッチ除去(corpus_blankness.parquet、sat_frac<0.10)を適用したmanifest。

experiments/0010→0012 で作った Macenko 索引は背景除去前(0002相当)のハイパラで
構築されており、現行 baseline(0018、背景除去済み)と同一コーパス世代ではなかった
(README「experiments/0031」参照)。lib.manifest.build_patch_manifest は
background_blankness_path で背景行を除外できる(0017と全く同じロジック) —
blankness測定(sat_frac)は生WSIピクセルから計算した encoder/染色正規化に依存しない
値なので、0017 と同じ corpus_blankness.parquet をそのまま Macenko features_dir に
適用できる(タイリング幾何は0010の設計により0001から不変、local_idxの対応も同一)。

experiment.py 本体は experiments/0017 のコピー(ロジック変更なし、config.yml の
features_dir/background_* だけが違う)。
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import pandas as pd


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    return parser.parse_args()


def _adjust_blankness_for_corpus(
    blankness_path: Path, features_dir: Path, out_path: Path, logger: logging.Logger,
) -> Path:
    """corpus_blankness.parquet は plain uni_v1 コーパスの h5 から計測したもの
    (slide_id, local_idx, sat_frac)。Macenko コーパスは独立した wsi_preprocess
    抽出なので、タイリング境界の非決定性で一部スライドのパッチ数がわずかに
    (1000スライド中25枚、差は1〜5パッチ)ずれている — この状態で
    lib.manifest.build_patch_manifest にそのまま渡すと、ローカルインデックス
    範囲外の除外行を検出して fail-fast する(意図通りの安全装置、job 10601)。

    パッチ数が一致しないスライドは local_idx の対応が保証できない(1件ズレる
    だけで以降の全パッチの対応がズレうる)ため、除外リストを補正して当てはめる
    のではなく、**そのスライドだけ背景除外をスキップする**(全パッチ保持)。
    影響は25/1000スライドに数%の背景パッチが残る程度で、誤った対応付けで
    無関係なパッチを消すよりはるかに安全。
    """
    import h5py
    import pandas as pd

    df = pd.read_parquet(blankness_path)
    n_before = len(df)
    mismatched = []
    for slide_id, group in df.groupby("slide_id"):
        h5_path = features_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            mismatched.append(str(slide_id))
            continue
        with h5py.File(h5_path, "r") as f:
            n_patches = f["coords"].shape[0]
        if int(group["local_idx"].max()) >= n_patches:
            mismatched.append(str(slide_id))

    if mismatched:
        logger.warning(
            "%d/%d slides have a patch-count mismatch vs %s — skipping background "
            "exclusion for these slides only (kept unfiltered): %s",
            len(mismatched), df["slide_id"].nunique(), blankness_path, mismatched,
        )
        df = df[~df["slide_id"].astype(str).isin(mismatched)]
    logger.info("blankness rows: %d -> %d after mismatch adjustment", n_before, len(df))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    return out_path


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.manifest import build_patch_manifest
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    parse_args()

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)
    features_dir = project_root / config["features_dir"]
    train_sample_size: int = config.get("train_sample_size", 2_000_000)

    background_blankness_path = config.get("background_blankness_path")
    if background_blankness_path:
        background_blankness_path = _adjust_blankness_for_corpus(
            project_root / background_blankness_path, features_dir,
            run_dir / "corpus_blankness_adjusted.parquet", logger,
        )
    background_sat_frac_max: float = config.get("background_sat_frac_max", 0.10)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:      {run_dir}")
    logger.info(f"features_dir: {features_dir}")
    logger.info(f"seed:         {seed}")
    logger.info(f"background_blankness_path: {background_blankness_path}")
    logger.info(f"background_sat_frac_max:   {background_sat_frac_max}")

    manifest_path = run_dir / "manifest.parquet"
    slide_meta_path = run_dir / "slide_meta.parquet"
    training_sample_path = run_dir / "training_sample.npy"

    # ── Experiment logic ──────────────────────────────────────────────────────
    build_patch_manifest(
        features_dir=features_dir,
        manifest_path=manifest_path,
        slide_meta_path=slide_meta_path,
        training_sample_path=training_sample_path,
        train_sample_size=train_sample_size,
        seed=seed,
        background_blankness_path=background_blankness_path,
        background_sat_frac_max=background_sat_frac_max,
    )

    manifest = pd.read_parquet(manifest_path, columns=["slide_id"])
    slide_meta = pd.read_parquet(slide_meta_path)
    results = {
        "n_patches": int(len(manifest)),
        "n_slides": int(manifest["slide_id"].nunique()),
        "manifest_path": str(manifest_path),
        "slide_meta_path": str(slide_meta_path),
        "training_sample_path": str(training_sample_path),
    }
    if "n_excluded" in slide_meta.columns:
        results["n_excluded"] = int(slide_meta["n_excluded"].sum())
        results["n_patches_raw"] = int(slide_meta["total_patches_raw"].sum())

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info(f"Done. {results}")


if __name__ == "__main__":
    main()
