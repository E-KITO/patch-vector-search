"""experiments/0055: グローバルなdomain_shift(atlas平均-corpus平均、1本の
方向)ではなく、クエリごとの局所k近傍平均差による補正(local domain shift)を
GT対応7所見でalphaスイープ検証する。

experiments/0038の分類器はatlas/corpusをほぼ完璧に線形分離できるのに、
experiments/0045・0054のグローバルな線形補正(domain_shift・jpeg_shift・
その残差)は所見ごとに効果がバラバラ(Kupffer cellには効くがHypertrophyには
逆効果)だった。「1本のグローバルな方向」が所見によって欲しい補正の向きが
異なることを示唆する結果を受け、補正方向をクエリの埋め込み位置ごとに局所的に
決める(k近傍のatlas平均 - k近傍のcorpus平均)ことで、この所見依存のばらつきを
吸収できるかを検証する。

手順:
  1. atlas図版91枚全体(25所見)のタイルを埋め込み、タイルごとに由来画像を
     記録した「atlasプール」を作る(plain空間・macenko空間の両方)。
  2. コーパスから大きめのランダムサンプルを「corpusプール」として用意する
     (plain/macenko両空間)。
  3. 評価対象所見の各atlas図版タイルについて、そのタイル自身をクエリとして
     atlasプール(自分自身の由来画像のタイルは除外)・corpusプールそれぞれで
     コサイン類似度top-k近傍を取り、その平均の差(= local_shift、タイル固有の
     補正方向)を計算する。
  4. corrected = L2_normalize(tile_vec - alpha * local_shift) として、
     finding_routingの実ルーティング空間(plain/whiten/macenko)でGT対応7所見の
     alphaスイープ(experiments/0045・0054と同じ枠組み)を行う。

出力(outputs/0055_.../default/):
  gt_comparison_{family}_local.csv
  best_alpha_by_finding_local.json
  summary.json
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


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


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


def _build_atlas_pool(folders, embed_one_fn, logger) -> tuple[np.ndarray, np.ndarray]:
    """全所見フォルダの全atlas図版をタイル分割・埋め込みし、タイルごとの由来
    画像パスを記録したプールを作る(自己一致除外のためimage_idを保持する)。"""
    vecs, ids = [], []
    for folder_name, images in folders:
        for img_path in images:
            tile_vecs = embed_one_fn(img_path)
            vecs.append(tile_vecs)
            ids.extend([img_path] * len(tile_vecs))
        logger.info(f"  atlas pool: {folder_name} done")
    return np.concatenate(vecs, axis=0), np.array(ids)


def _embed_with_ids(images: list[str], embed_one_fn) -> tuple[np.ndarray, np.ndarray]:
    vecs, ids = [], []
    for img_path in images:
        tile_vecs = embed_one_fn(img_path)
        vecs.append(tile_vecs)
        ids.extend([img_path] * len(tile_vecs))
    return np.concatenate(vecs, axis=0), np.array(ids)


def _local_shift(
    tile_vecs: np.ndarray, tile_ids: np.ndarray,
    atlas_pool_vecs: np.ndarray, atlas_pool_ids: np.ndarray,
    corpus_pool_vecs: np.ndarray, k: int,
) -> np.ndarray:
    """タイルごとに、atlasプール(自分自身の由来画像は除外)・corpusプールの
    コサイン類似度top-k近傍を取り、その平均の差を返す(タイル固有の補正方向)。
    全ベクトルはL2正規化済みなので内積=コサイン類似度。
    """
    sims_atlas = tile_vecs @ atlas_pool_vecs.T  # [n, Na]
    same_image = tile_ids[:, None] == atlas_pool_ids[None, :]  # [n, Na]
    sims_atlas = np.where(same_image, -np.inf, sims_atlas)
    top_atlas = np.argpartition(-sims_atlas, k, axis=1)[:, :k]  # [n, k]
    atlas_local_mean = atlas_pool_vecs[top_atlas].mean(axis=1)  # [n, d]

    sims_corpus = tile_vecs @ corpus_pool_vecs.T  # [n, Nc]
    top_corpus = np.argpartition(-sims_corpus, k, axis=1)[:, :k]
    corpus_local_mean = corpus_pool_vecs[top_corpus].mean(axis=1)  # [n, d]

    return atlas_local_mean - corpus_local_mean


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import list_atlas_folders, sample_negative_pool
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
    k = int(config["k_neighbors"])
    n_corpus_pool = int(config["n_corpus_pool"])
    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _embed_one_plain(img_path: str) -> np.ndarray:
        return embed_image_tiles(str(img_path), tile_size=tile_size)

    def _embed_one_macenko(img_path: str) -> np.ndarray:
        return embed_image_tiles(str(img_path), tile_size=tile_size, tile_transform=_macenko_tile_transform)

    # --- 1. atlasプール(全25所見・タイルごとの由来画像id付き)をplain/macenko両空間で構築 ---
    atlas_root = project_root / config["atlas_root"]
    atlas_csv = project_root / config["atlas_figures_csv"]
    folders = list_atlas_folders(atlas_root, atlas_csv)
    logger.info(f"atlas folders: {len(folders)}")

    atlas_pool_plain_vecs, atlas_pool_plain_ids = _build_atlas_pool(folders, _embed_one_plain, logger)
    atlas_pool_macenko_vecs, atlas_pool_macenko_ids = _build_atlas_pool(folders, _embed_one_macenko, logger)
    logger.info(f"atlas pool: plain {atlas_pool_plain_vecs.shape}, macenko {atlas_pool_macenko_vecs.shape}")

    # --- 2. 索引ロード(3空間)とcorpusプール(plain/macenko、大きめのランダムサンプル) ---
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

    plain_manifest = baseline_index.manifest.assign(slide_id=baseline_index.manifest["slide_id"].astype(str))
    macenko_manifest = macenko_index.manifest.assign(slide_id=macenko_index.manifest["slide_id"].astype(str))

    corpus_pool_plain_vecs, _ = sample_negative_pool(
        plain_manifest, project_root / finding_routing.PLAIN_FEATURES_DIR,
        n=n_corpus_pool, exclude_slides=set(), rng=rng,
    )
    corpus_pool_macenko_vecs, _ = sample_negative_pool(
        macenko_manifest, project_root / finding_routing.MACENKO_FEATURES_DIR,
        n=n_corpus_pool, exclude_slides=set(), rng=rng,
    )
    logger.info(f"corpus pool: plain {corpus_pool_plain_vecs.shape}, macenko {corpus_pool_macenko_vecs.shape}")

    # --- 3. 所見ごとにローカル補正付きembed_fnを構築(local_shiftはalpha非依存でキャッシュ) ---
    def _cached_with_local_shift(embed_one_fn, atlas_pool_vecs, atlas_pool_ids, corpus_pool_vecs):
        cache: dict[tuple, tuple] = {}

        def _fn(images):
            key = tuple(images)
            if key not in cache:
                vecs, ids = _embed_with_ids(images, embed_one_fn)
                shift = _local_shift(vecs, ids, atlas_pool_vecs, atlas_pool_ids, corpus_pool_vecs, k)
                cache[key] = (vecs, shift)
            return cache[key]

        return _fn

    def _make_base(cached_fn):
        return lambda images: cached_fn(images)[0]

    def _make_local_corrected(cached_fn, alpha):
        def _fn(images):
            vecs, shift = cached_fn(images)
            return _l2_normalize(vecs - alpha * shift)
        return _fn

    plain_cached = _cached_with_local_shift(
        _embed_one_plain, atlas_pool_plain_vecs, atlas_pool_plain_ids, corpus_pool_plain_vecs
    )
    macenko_cached = _cached_with_local_shift(
        _embed_one_macenko, atlas_pool_macenko_vecs, atlas_pool_macenko_ids, corpus_pool_macenko_vecs
    )

    families = {
        "plain": (baseline_index, plain_cached,
                  ["Single cell necrosis", "Deposit, glycogen", "Hematopoiesis, extramedullary"]),
        "whiten": (whiten_index, plain_cached, ["Proliferation, Kupffer cell"]),
        "macenko": (macenko_index, macenko_cached,
                    ["Hypertrophy", "Increased mitosis", "Inclusion body, intracytoplasmic"]),
    }

    alphas = list(config["alphas"])
    best_alpha_by_finding: dict[str, dict] = {}
    for family_name, (patch_index, cached_fn, findings) in families.items():
        pipelines = {"routed_base": (patch_index, _make_base(cached_fn))}
        for alpha in alphas:
            pipelines[f"alpha{alpha}"] = (patch_index, _make_local_corrected(cached_fn, alpha))

        gt_df = run_comparison(pipelines, gt_csv=project_root / config["gt_csv"])
        assert len(gt_df) == len(CATEGORIES), (
            f"[{family_name}] expected {len(CATEGORIES)} category rows, got {len(gt_df)}"
        )
        gt_df["finding_type"] = list(CATEGORIES.values())
        gt_df.to_csv(run_dir / f"gt_comparison_{family_name}_local.csv", index=False)
        logger.info(f"[{family_name}] GT comparison:\n{gt_df.to_string()}")

        for finding in findings:
            row = gt_df.loc[gt_df["finding_type"] == finding].iloc[0]
            best_alpha, reason = _pick_best_alpha_for_row(row, alphas, base_label="routed_base")
            best_alpha_by_finding[finding] = {"family": family_name, "best_alpha": best_alpha, **reason}
            logger.info(f"[{family_name}] {finding}: best_alpha={best_alpha} ({reason})")

    (run_dir / "best_alpha_by_finding_local.json").write_text(json.dumps(best_alpha_by_finding, indent=2))

    summary_rows = [
        {"finding": f, "family": FAMILY_BY_FINDING[f], **best_alpha_by_finding.get(f, {})}
        for f in FAMILY_BY_FINDING
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"summary:\n{summary_df.to_string()}")

    (run_dir / "summary.json").write_text(json.dumps({
        "k_neighbors": k, "n_corpus_pool": n_corpus_pool,
        "n_atlas_pool_plain": int(atlas_pool_plain_vecs.shape[0]),
        "n_atlas_pool_macenko": int(atlas_pool_macenko_vecs.shape[0]),
        "findings": summary_rows,
    }, indent=2, ensure_ascii=False))

    complete_run(run_dir)


if __name__ == "__main__":
    main()
