"""experiments/0028: lib.finding_routing(所見ごとの索引出し分け)の検証。

新規の索引構築・クエリ埋め込みは行わない — experiments/0025 の self_retrieval
CSV と experiments/0027 の atlas 図版単位 CSV を、lib.finding_routing.WHITEN_FINDINGS
で再集計するだけ。GPU 不要、ローカル実行可能。

出力(outputs/0028_.../default/):
  self_retrieval_routed.csv   finding, baseline, whiten, routed(nhr_best_rank_med_noexp)
  atlas_routed.csv            finding, image, baseline_best, whiten_best, routed_best, routed_found
  summary.md
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


def _get_project_root() -> Path:
    r = os.environ.get("PROJECT_ROOT")
    if not r:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(r)


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="config.yml")
    return p.parse_args()


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("exp0028")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "experiment.log")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ============================================================
# Stage 1: self_retrieval を再集計
# ============================================================

def build_self_retrieval_routed(project_root: Path, cfg: dict, whiten_findings: frozenset) -> pd.DataFrame:
    base = pd.read_csv(project_root / cfg["self_retrieval_baseline"])
    whit = pd.read_csv(project_root / cfg["self_retrieval_whiten"])
    metric = "nhr_best_rank_med_noexp"
    merged = base[["finding", metric]].merge(
        whit[["finding", metric]], on="finding", suffixes=("_baseline", "_whiten")
    )
    merged["routed"] = merged.apply(
        lambda r: r[f"{metric}_whiten"] if r["finding"] in whiten_findings else r[f"{metric}_baseline"],
        axis=1,
    )
    merged["routed_to"] = merged["finding"].apply(lambda f: "whiten" if f in whiten_findings else "baseline")
    return merged.rename(columns={f"{metric}_baseline": "baseline", f"{metric}_whiten": "whiten"})


# ============================================================
# Stage 2: atlas 図版単位を再集計
# ============================================================

def build_atlas_routed(project_root: Path, cfg: dict, whiten_findings: frozenset) -> pd.DataFrame:
    df = pd.read_csv(project_root / cfg["atlas_per_image"])
    df["routed_to"] = df["finding"].apply(lambda f: "whiten" if f in whiten_findings else "baseline")
    df["routed_best"] = df.apply(
        lambda r: r["whiten_best"] if r["routed_to"] == "whiten" else r["baseline_best"], axis=1
    )
    df["routed_found"] = df.apply(
        lambda r: r["whiten_found"] if r["routed_to"] == "whiten" else r["baseline_found"], axis=1
    )
    return df


# ============================================================
# Stage 3: summary
# ============================================================

def write_summary(sr: pd.DataFrame, atlas: pd.DataFrame, run_dir: Path, logger: logging.Logger) -> None:
    L = ["# experiments/0028: lib.finding_routing の検証(既存結果の再集計)", ""]
    L.append("新規計算なし — experiments/0025(self_retrieval)/ 0027(atlas 図版単位)の"
             "既存 CSV を `lib.finding_routing.WHITEN_FINDINGS` で再集計。")
    L.append("")

    L.append("## self_retrieval(nhr_best_rank_med_noexp、低いほど良い)")
    L.append("")
    L.append("| finding | routed_to | baseline | whiten | routed |")
    L.append("|---|---|---|---|---|")
    for _, r in sr.iterrows():
        L.append(f"| {r['finding']} | {r['routed_to']} | {r['baseline']} | {r['whiten']} | {r['routed']} |")
    L.append("")
    n_improved = (sr["routed_to"] == "whiten").sum()
    L.append(f"- routed で whiten 側に倒した所見: {n_improved}/{len(sr)}"
             "(GT実測で悪化しないと確認できた所見のみ)。それ以外は baseline のまま"
             "= 悪化リスクなし。")
    L.append("")

    L.append("## atlas GT(図版単位、best_rank。低いほど良い)")
    L.append("")
    agg = atlas.groupby("finding").agg(
        routed_to=("routed_to", "first"),
        n_images=("image", "count"),
        baseline_median=("baseline_best", "median"),
        whiten_median=("whiten_best", "median"),
        routed_median=("routed_best", "median"),
        baseline_found_total=("baseline_found", "sum"),
        whiten_found_total=("whiten_found", "sum"),
        routed_found_total=("routed_found", "sum"),
    ).reset_index()
    L.append("| finding | routed_to | 図版数 | baseline中央値 | whiten中央値 | routed中央値 | "
             "baseline found計 | whiten found計 | routed found計 |")
    L.append("|" + "---|" * 9)
    for _, r in agg.iterrows():
        L.append(
            f"| {r['finding']} | {r['routed_to']} | {r['n_images']} | {r['baseline_median']} | "
            f"{r['whiten_median']} | {r['routed_median']} | {r['baseline_found_total']} | "
            f"{r['whiten_found_total']} | {r['routed_found_total']} |"
        )
    L.append("")

    # 実質的なチェック: routed_median は定義上 baseline_median と whiten_median の
    # どちらか(選んだ側)に一致するので「両方より悪化していないか」は恒等的に真に
    # なり無意味。意味があるのは「選んだ側が選ばなかった側より本当に良いか」—
    # つまり routed_to="whiten" なのに whiten_median > baseline_median(またはその逆)
    # になっている所見を検出する。
    def _worse_than_alternative(r: pd.Series) -> bool:
        if r["routed_to"] == "whiten":
            return r["whiten_median"] > r["baseline_median"]
        return r["baseline_median"] > r["whiten_median"]

    bad_choice = agg[agg.apply(_worse_than_alternative, axis=1)]
    if len(bad_choice):
        L.append("## ⚠️ ルーティング選択が median best_rank では逆効果になっている所見")
        L.append("")
        L.append(bad_choice.to_markdown(index=False))
    else:
        L.append("## サニティチェック")
        L.append("")
        L.append("全所見で、選んだ側(routed_to)の median best_rank が選ばなかった側"
                 "以下(=より良いか同等)になっている。")
    L.append("")

    (run_dir / "summary.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("wrote %s", run_dir / "summary.md")


# ============================================================
# main
# ============================================================

def main() -> None:
    parse_args()
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent
    cfg = load_config(exp_dir)
    exp_name = os.environ["EXP_NAME"]

    from lib.finding_routing import WHITEN_FINDINGS
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    run_dir = get_run_dir(project_root, __file__, "default", output_root=os.environ.get("OUTPUT_ROOT"))
    write_run_metadata(run_dir, exp_name=exp_name, whiten_findings=",".join(sorted(WHITEN_FINDINGS)))
    logger = setup_logger(run_dir)
    logger.info("run_dir: %s", run_dir)

    sr = build_self_retrieval_routed(project_root, cfg, WHITEN_FINDINGS)
    sr.to_csv(run_dir / "self_retrieval_routed.csv", index=False)

    atlas = build_atlas_routed(project_root, cfg, WHITEN_FINDINGS)
    atlas.to_csv(run_dir / "atlas_routed.csv", index=False)

    write_summary(sr, atlas, run_dir, logger)

    (run_dir / "results.json").write_text(json.dumps({
        "whiten_findings": sorted(WHITEN_FINDINGS),
    }, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("done -> %s", run_dir)


if __name__ == "__main__":
    main()
