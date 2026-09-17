"""experiments/0045: 線形ドメイン補正を finding_routing の実ルーティング空間で
GT対応7所見について細かくalphaスイープする。

experiments/0039・0040は常にplain(baseline)索引・埋め込み空間を基準にalphaを
検証していたが、lib.finding_routingは既にHypertrophy/Increased mitosis/
Inclusion body -> macenko、Proliferation Kupffer cell -> whiten、残り3所見は
baselineへルーティングしている(README「wsi_preprocess連携」直前の各節参照)。
この実験は線形補正を「所見ごとに既に選ばれている索引・埋め込み空間の上に」
追加した場合の効果を、所見ごとに正しい空間(plain/whiten/macenko)で検証する。

domain_shiftベクトル自体は再計算せず、既存の成果物を再利用する:
  - plain空間(experiments/0039のdomain_shift.npy): baseline索引・whiten索引の
    両方に使う。whitenの異方性除去変換はPatchIndex内部でクエリベクトルに事後
    適用される(lib.embedding_transform.apply_transform)ため、ドメイン補正は
    その前段のraw UNI埋め込み空間で行えば良く、whiten用に別途domain_shiftを
    計算し直す必要はない。
  - macenko空間(experiments/0041のdomain_shift_macenko.npy): macenko索引用。
    macenkoはper-tile染色正規化 -> 別コーパス埋め込みという非線形経路のため、
    plain空間のdomain_shiftは流用できない(experiments/0041の設計と同じ判断)。

alphaは0.0(補正なし、routed_baseと一致するはずの検算用)〜0.5を0.05刻みで
スイープする(experiments/0040の0.25/0.5/0.75より細かい格子)。UNI埋め込み自体は
alphaに依存しない(補正はベクトル空間の後処理のみ)ため、所見ごとに一度だけ
埋め込みをキャッシュし、alphaごとの再埋め込みコストを避ける。

所見ごとに「GTスライドが候補プールから消失しない(found減少なし)」ことを
最優先で避け、その中でbest_rankの改善が最大のalphaを選ぶ(タイはより小さい
alpha=保守的な補正を優先)。選ばれたalphaについてのみ、目視ギャラリー診断も
追加実行する(GT数値だけで判断しない、README「experiments/0037」「0041」の
教訓——Hematopoiesisで見つかったように、GT best_rankの改善がギャラリー上は
偽陽性の消失に過ぎないケースがある)。

出力(outputs/0045_.../default/):
  gt_comparison_{plain,whiten,macenko}.csv   空間ごとのGT best_rank比較
  best_alpha_by_finding.json                  所見ごとに選ばれたalphaと選定根拠
  <finding_slug>/query__{no_correction,domain_corrected}/...  選ばれたalphaの目視診断
  summary.csv                                  選ばれたalphaのギャラリー数比較
"""
from __future__ import annotations

import json
import logging
import os
import sys
import re
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


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", s).strip("_")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


# 所見ごとの補正空間(lib.finding_routingの索引選択と対応)。
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
    """所見1件分の行から最良alphaを選ぶ。found減少(GTスライドの候補プール消失)を
    最優先で避け(experiments/0039の教訓)、その中でbest_rank改善が最大、タイは
    より小さいalpha(保守的な補正)を優先する。alpha=0.0を常に候補に含めているため
    (config.alphas)、改善するalphaが無ければ自動的に補正なしへフォールバックする。
    """
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
    from lib.query_embedding import embed_image_tiles
    from lib.torchstain_normalize import normalize_to_reference
    from lib.ovr_scoring import resolve_atlas_images, run_query_arm
    from lib.search import PatchIndex
    from lib import finding_routing
    from scripts.validate_against_ground_truth import CATEGORIES, run_comparison

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()
    config = load_config(exp_dir)

    variant_key = "default"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    tile_size = int(config.get("tile_size", 224))
    alphas = list(config["alphas"])

    domain_shift_plain = np.load(project_root / config["domain_shift_plain_path"])
    domain_shift_macenko = np.load(project_root / config["domain_shift_macenko_path"])
    logger.info(f"domain_shift_plain norm={np.linalg.norm(domain_shift_plain):.4f}, "
                f"domain_shift_macenko norm={np.linalg.norm(domain_shift_macenko):.4f}")

    macenko_stain_ref = project_root / finding_routing.MACENKO_STAIN_REFERENCE

    def _plain_tile_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size) for f in images], axis=0
        )

    def _macenko_tile_transform(tile):
        try:
            return normalize_to_reference(tile, macenko_stain_ref)
        except Exception:
            return tile

    def _macenko_tile_embed(images) -> np.ndarray:
        return np.concatenate(
            [embed_image_tiles(str(f), tile_size=tile_size, tile_transform=_macenko_tile_transform)
             for f in images],
            axis=0,
        )

    def _cached(base_fn):
        """alphaに依存しないUNI埋め込みを所見1件につき1回だけ計算して使い回す
        (この実験の格子は11alpha x 3空間だが、埋め込みコストはalphaに依存しない
        後処理なので、空間ごとに所見あたり1回のTRIDENT呼び出しで済ませる)。"""
        cache: dict[tuple, np.ndarray] = {}

        def _fn(images):
            key = tuple(images)
            if key not in cache:
                cache[key] = base_fn(images)
            return cache[key]

        return _fn

    def _make_corrected(cached_fn, domain_shift, alpha):
        def _fn(images):
            return _l2_normalize(cached_fn(images) - alpha * domain_shift)
        return _fn

    # --- 索引ロード(3空間、finding_routingと同一パラメータ) ---
    baseline_dir = project_root / finding_routing.BASELINE_INDEX_DIR
    whiten_dir = project_root / finding_routing.WHITEN_INDEX_DIR
    macenko_dir = project_root / finding_routing.MACENKO_INDEX_DIR
    baseline_index = PatchIndex.load(
        index_path=baseline_dir / "index.faiss",
        manifest_path=baseline_dir / "manifest.parquet",
        slide_meta_path=baseline_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
    )
    whiten_index = PatchIndex.load(
        index_path=whiten_dir / "index.faiss",
        manifest_path=whiten_dir / "manifest.parquet",
        slide_meta_path=whiten_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.PLAIN_FEATURES_DIR,
        transform_path=project_root / finding_routing.WHITEN_TRANSFORM_PATH,
    )
    macenko_index = PatchIndex.load(
        index_path=macenko_dir / "index.faiss",
        manifest_path=macenko_dir / "manifest.parquet",
        slide_meta_path=macenko_dir / "slide_meta.parquet",
        features_dir=project_root / finding_routing.MACENKO_FEATURES_DIR,
    )

    # plain/whitenは同じraw UNI埋め込みを共有できる(whitenの変換は検索時に
    # 内部適用されるため、クエリ埋め込み自体はplainと同一)。
    plain_cached = _cached(_plain_tile_embed)
    macenko_cached = _cached(_macenko_tile_embed)

    families = {
        "plain": (
            baseline_index, plain_cached, domain_shift_plain,
            ["Single cell necrosis", "Deposit, glycogen", "Hematopoiesis, extramedullary"],
        ),
        "whiten": (
            whiten_index, plain_cached, domain_shift_plain,
            ["Proliferation, Kupffer cell"],
        ),
        "macenko": (
            macenko_index, macenko_cached, domain_shift_macenko,
            ["Hypertrophy", "Increased mitosis", "Inclusion body, intracytoplasmic"],
        ),
    }

    # --- 1. 空間ごとにGT best_rankのalpha細かいスイープ ---
    best_alpha_by_finding: dict[str, dict] = {}
    for family_name, (patch_index, cached_fn, domain_shift, findings) in families.items():
        pipelines = {"routed_base": (patch_index, cached_fn)}
        for alpha in alphas:
            pipelines[f"alpha{alpha}"] = (patch_index, _make_corrected(cached_fn, domain_shift, alpha))

        gt_df = run_comparison(pipelines)
        assert len(gt_df) == len(CATEGORIES), (
            f"[{family_name}] expected {len(CATEGORIES)} category rows, got {len(gt_df)} "
            "(an atlas category folder produced no images?)"
        )
        gt_df["finding_type"] = list(CATEGORIES.values())
        gt_df.to_csv(run_dir / f"gt_comparison_{family_name}.csv", index=False)
        logger.info(f"[{family_name}] GT comparison:\n{gt_df.to_string()}")

        for finding in findings:
            row = gt_df.loc[gt_df["finding_type"] == finding].iloc[0]
            best_alpha, reason = _pick_best_alpha_for_row(row, alphas, base_label="routed_base")
            best_alpha_by_finding[finding] = {"family": family_name, "best_alpha": best_alpha, **reason}
            logger.info(f"[{family_name}] {finding}: best_alpha={best_alpha} ({reason})")

    (run_dir / "best_alpha_by_finding.json").write_text(
        json.dumps(best_alpha_by_finding, indent=2)
    )

    # --- 2. 選ばれたalphaについてのみ目視ギャラリー診断(GT数値だけで判断しない) ---
    params = {
        "raw_slide_dir": project_root / config["raw_slide_dir"],
        "thumbnails_dir": project_root / config["baseline_thumbnails_dir"],
        "nprobe": int(config.get("nprobe", 32)),
        "rerank_pool": int(config.get("rerank_pool", 200)),
        "max_tiles_reranked": config.get("max_tiles_reranked", None),
        "k_candidates": int(config.get("k_candidates", 8000)),
        "top_n_slides": int(config.get("top_n_slides", 20)),
        "top_n_slides_to_plot": int(config.get("top_n_slides_to_plot", 3)),
    }

    rows = []
    for finding, info in best_alpha_by_finding.items():
        family_name = info["family"]
        patch_index, cached_fn, domain_shift, _ = families[family_name]
        best_alpha = info["best_alpha"]

        images = resolve_atlas_images(project_root, config, finding)
        if not images:
            logger.warning(f"no atlas images resolved for {finding!r}, skipping visual diagnostic")
            continue

        finding_dir = run_dir / _slug(finding)
        finding_dir.mkdir(exist_ok=True)

        base_vecs = cached_fn(images)
        corrected_vecs = _l2_normalize(base_vecs - best_alpha * domain_shift)

        arm_results = {}
        for arm_label, vecs in (("no_correction", base_vecs), ("domain_corrected", corrected_vecs)):
            arm_results[arm_label] = run_query_arm(
                arm_label=arm_label, query_vecs=vecs, patch_index=patch_index,
                thumbnails_dir=params["thumbnails_dir"], run_dir=finding_dir,
                params=params, logger=logger,
            )
        logger.info(f"[{finding}] (family={family_name}, alpha={best_alpha}) "
                    f"no_correction galleries: {arm_results['no_correction']['n_galleries']}, "
                    f"domain_corrected galleries: {arm_results['domain_corrected']['n_galleries']}")

        rows.append({
            "finding": finding, "family": family_name, "best_alpha": best_alpha,
            "n_galleries_no_correction": arm_results["no_correction"]["n_galleries"],
            "n_galleries_domain_corrected": arm_results["domain_corrected"]["n_galleries"],
            "top_slides_changed": arm_results["no_correction"]["top_slide_ids"][:5]
            != arm_results["domain_corrected"]["top_slide_ids"][:5],
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info(f"gallery summary:\n{summary.to_string()}")

    complete_run(run_dir)


if __name__ == "__main__":
    main()
