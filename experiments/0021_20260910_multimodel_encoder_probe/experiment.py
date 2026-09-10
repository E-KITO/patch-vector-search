"""experiments/0021: 多モデル埋め込み弁別力プローブ。

wsi_preprocess が作った probe セット (data/.../probe/) を読み、各エンコーダが
「融合壊死・髄外造血 (+ 参考 single_cell_necrosis)」の phenotype を confuser から
どれだけ分離できるかを、FAISS 索引を作らずに測る。全 1,800万パッチ再埋め込み
(Task B) に踏み切る前の go / no-go 判定用。

出力: outputs/0021_.../probe/results.json + summary.md
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

FINDINGS = ["necrosis", "hematopoiesis", "single_cell_necrosis"]


def _project_root() -> Path:
    r = os.environ.get("PROJECT_ROOT")
    if not r:
        print("PROJECT_ROOT unset — run via run_slurm.sh", file=sys.stderr)
        sys.exit(1)
    return Path(r)


def load_config(exp_dir: Path) -> dict:
    with open(exp_dir / "config.yml") as f:
        return yaml.safe_load(f) or {}


def l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """tie-aware ROC-AUC (Mann-Whitney U / (n_pos n_neg))。numpy のみ。"""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=float)
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0 or not np.isfinite(s).all():
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks_sorted = np.arange(1, len(s) + 1, dtype=float)
    j = 0
    while j < len(s_sorted):
        k = j
        while k + 1 < len(s_sorted) and s_sorted[k + 1] == s_sorted[j]:
            k += 1
        if k > j:
            ranks_sorted[j : k + 1] = (j + 1 + k + 1) / 2.0
        j = k + 1
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = ranks_sorted
    u = ranks[y].sum() - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n0))


def analyse_encoder(enc: str, probe_dir: Path, df: pd.DataFrame, cfg: dict) -> dict:
    import h5py

    meta = json.loads((probe_dir / f"meta_{enc}.json").read_text())
    with h5py.File(probe_dir / f"embeddings_{enc}.h5", "r") as f:
        X = f["features"][:].astype(np.float32)
    assert X.shape[0] == len(df), f"{enc}: h5 rows {X.shape[0]} != csv rows {len(df)}"
    X = l2norm(X)
    S = X @ X.T  # (N, N) cosine, N=12600 -> ~635MB float32

    slide = df["slide_id"].to_numpy().astype(str)
    exp = df["exp_id"].to_numpy().astype(str)
    group = df["group"].to_numpy().astype(str)
    conf_idx = np.where(group == "confuser")[0]

    m = int(cfg.get("auroc_top_m", 10))
    kk = int(cfg.get("precision_k", 10))
    nboot = int(cfg.get("n_bootstrap", 1000))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    out = {"dim": int(meta["dim"]), "hf_repo": meta.get("hf_repo"), "findings": {}}

    for F in FINDINGS:
        f_idx = np.where(group == F)[0]
        if len(f_idx) == 0:
            continue
        f_slides = np.unique(slide[f_idx])
        pool = np.concatenate([f_idx, conf_idx])
        y_pool = np.isin(pool, f_idx)

        # --- LOSO score: 別スライドの同所見パッチとの平均 top-m コサイン ---
        def loso_scores(pool_idx: np.ndarray) -> np.ndarray:
            sc = np.empty(len(pool_idx), dtype=float)
            for i, p in enumerate(pool_idx):
                cand = f_idx[slide[f_idx] != slide[p]]
                if len(cand) == 0:
                    sc[i] = np.nan
                    continue
                sims = S[p, cand]
                mm = min(m, len(sims))
                sc[i] = np.sort(sims)[-mm:].mean()
            return sc

        scores = loso_scores(pool)  # 実データ 1 回だけ計算
        loso = auroc(y_pool, scores)

        # bootstrap CI: スライド単位でリサンプル (パッチは非独立)。LOSO スコアは
        # 固定し、AUROC を計算する対象パッチ集合だけを resample する cluster bootstrap。
        pool_slide = slide[pool]
        by_slide = {s: np.where(pool_slide == s)[0] for s in np.unique(pool_slide)}
        conf_slides = np.unique(slide[conf_idx])
        boot = []
        for _ in range(nboot):
            picks = np.concatenate([
                rng.choice(f_slides, size=len(f_slides), replace=True),
                rng.choice(conf_slides, size=len(conf_slides), replace=True),
            ])
            sel = np.concatenate([by_slide[s] for s in picks])
            a = auroc(y_pool[sel], scores[sel])
            if np.isfinite(a):
                boot.append(a)
        ci = (
            (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
            if boot else (float("nan"), float("nan"))
        )

        # --- precision@k: 所見パッチをクエリに top-k のうち同所見の割合 ---
        precs = []
        for q in f_idx:
            cand = pool[slide[pool] != slide[q]]
            csim = S[q, cand]
            top = cand[np.argsort(csim)[::-1][:kk]]
            precs.append(np.isin(top, f_idx).mean())
        prec = float(np.mean(precs))
        # chance ~ (|F| - 平均スライド枚数) / (|pool| - 平均スライド枚数)
        avg_slide = len(f_idx) / len(f_slides)
        chance = (len(f_idx) - avg_slide) / (len(pool) - avg_slide)

        # --- margin ---
        def mean_pairs(a: np.ndarray, b: np.ndarray, exclude_same_slide: bool) -> float:
            block = S[np.ix_(a, b)]
            if exclude_same_slide:
                mask = slide[a][:, None] != slide[b][None, :]
                vals = block[mask]
            else:
                vals = block.ravel()
            return float(vals.mean()) if vals.size else float("nan")

        within = mean_pairs(f_idx, f_idx, exclude_same_slide=True)
        between = mean_pairs(f_idx, conf_idx, exclude_same_slide=False)

        # --- batch check: 同所見 同EXP vs 別EXP ---
        blk = S[np.ix_(f_idx, f_idx)]
        diff_slide = slide[f_idx][:, None] != slide[f_idx][None, :]
        same_exp = exp[f_idx][:, None] == exp[f_idx][None, :]
        w_sameexp = blk[diff_slide & same_exp]
        w_diffexp = blk[diff_slide & ~same_exp]
        batch = {
            "within_sameexp": float(w_sameexp.mean()) if w_sameexp.size else None,
            "within_diffexp": float(w_diffexp.mean()) if w_diffexp.size else None,
            "n_sameexp_pairs": int(w_sameexp.size),
        }

        # --- slide-level AUROC (self_retrieval の類似物、小サンプルで参考値) ---
        def slide_vecs(idxs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            us = np.unique(slide[idxs])
            V = np.stack([l2norm(X[idxs[slide[idxs] == s]].mean(0, keepdims=True))[0] for s in us])
            return us, V

        fs_ids, fV = slide_vecs(f_idx)
        cs_ids, cV = slide_vecs(conf_idx)
        allV = np.vstack([fV, cV])
        y_sl = np.array([True] * len(fV) + [False] * len(cV))
        sl_scores = np.empty(len(allV))
        for i in range(len(allV)):
            mask = np.ones(len(fV), bool)
            if i < len(fV):
                mask[i] = False
            sl_scores[i] = (allV[i] @ fV[mask].T).mean() if mask.any() else np.nan
        slide_auroc = auroc(y_sl, sl_scores)

        out["findings"][F] = {
            "n_slides": int(len(f_slides)),
            "n_exp_ids": int(len(np.unique(exp[f_idx]))),
            "n_patches": int(len(f_idx)),
            "loso_auroc": loso,
            "loso_auroc_ci95": ci,
            "precision_at_k": prec,
            "chance_precision": float(chance),
            "within_finding_cos": within,
            "finding_vs_confuser_cos": between,
            "margin": float(within - between),
            "batch": batch,
            "slide_level_auroc": slide_auroc,
        }
        print(f"  [{enc}/{F}] LOSO-AUROC={loso:.3f} CI{tuple(round(x,3) for x in ci)}  "
              f"P@{kk}={prec:.3f} (chance {chance:.3f})  margin={within-between:+.3f}  "
              f"batch same/diff-exp={batch['within_sameexp']}/{batch['within_diffexp']}")

    del S
    return out


def write_summary(results: dict, run_dir: Path, cfg: dict) -> None:
    encs = [e for e in cfg["encoders"] if e in results]
    lines = ["# experiments/0021: 多モデル埋め込み弁別力プローブ", ""]
    lines.append(f"probe: 融合壊死13 / 単細胞壊死4 / 髄外造血3 スライド + confuser 106、"
                 f"各スライド最大100パッチ (背景除外)。索引なし。")
    lines.append("")
    lines.append("## LOSO AUROC (patch: その所見 vs confuser、別スライド同所見パッチとの top-m 平均コサインでスコア)")
    lines.append("")
    lines.append("| finding | " + " | ".join(encs) + " |")
    lines.append("|" + "---|" * (len(encs) + 1))
    for F in FINDINGS:
        row = [F]
        for e in encs:
            d = results[e]["findings"].get(F)
            if not d:
                row.append("—")
                continue
            lo, hi = d["loso_auroc_ci95"]
            row.append(f"{d['loso_auroc']:.3f} [{lo:.2f}–{hi:.2f}]")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## precision@k / margin / batch")
    lines.append("")
    for F in FINDINGS:
        lines.append(f"### {F}")
        lines.append("")
        lines.append("| encoder | P@k | chance | margin | within(same-exp / diff-exp) | slide-AUROC |")
        lines.append("|---|---|---|---|---|---|")
        for e in encs:
            d = results[e]["findings"].get(F)
            if not d:
                continue
            b = d["batch"]
            se = f"{b['within_sameexp']:.3f}" if b["within_sameexp"] is not None else "—"
            de = f"{b['within_diffexp']:.3f}" if b["within_diffexp"] is not None else "—"
            lines.append(
                f"| {e} | {d['precision_at_k']:.3f} | {d['chance_precision']:.3f} | "
                f"{d['margin']:+.3f} | {se} / {de} | {d['slide_level_auroc']:.3f} |"
            )
        lines.append("")
    lines.append("## 読み方")
    lines.append("")
    lines.append("- **LOSO AUROC**: uni_v1 に対し CI が重ならず明確に高いエンコーダがあれば、"
                 "その所見について Task B (フル再埋め込み) の価値あり。全モデル横並び (CI 重複) なら、"
                 "その所見はエンコーダでは動かない = コーパス/タスクの問題。")
    lines.append("- **batch**: within(same-exp) が within(diff-exp) を大きく上回る所見は、"
                 "マージンが phenotype でなく実験バッチ由来。self_retrieval 診断の "
                 "`batch_dominates_finding_rate` と同じ注意 (single_cell_necrosis は元々これ)。")
    lines.append("- **slide-AUROC**: 所見スライド 3〜13 枚での参考値。ノイズ大。")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = _project_root()
    sys.path.insert(0, str(project_root))
    exp_dir = Path(__file__).parent
    cfg = load_config(exp_dir)

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    probe_dir = project_root / cfg["probe_dir"]
    df = pd.read_csv(probe_dir / "probe_patches.csv").sort_values("row_idx").reset_index(drop=True)
    assert (df["row_idx"].to_numpy() == np.arange(len(df))).all(), "probe_patches.csv row_idx not 0..N-1"

    run_dir = get_run_dir(project_root, __file__, "probe")
    write_run_metadata(run_dir, exp_name=exp_name, encoders=",".join(cfg["encoders"]))
    print(f"probe_dir: {probe_dir}  patches: {len(df)}")

    results = {}
    for enc in cfg["encoders"]:
        if not (probe_dir / f"embeddings_{enc}.h5").exists():
            print(f"skip {enc}: embeddings_{enc}.h5 not found")
            continue
        print(f"=== {enc} ===")
        results[enc] = analyse_encoder(enc, probe_dir, df, cfg)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    write_summary(results, run_dir, cfg)
    complete_run(run_dir)
    print(f"done -> {run_dir}/results.json + summary.md")


if __name__ == "__main__":
    main()
