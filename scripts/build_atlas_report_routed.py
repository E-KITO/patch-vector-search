"""experiments/0029(NNL アトラス全所見スイープ、baseline vs 所見ルーティング後)の
outputs から report.html を生成する。

experiments/0029 の run_slurm.sh がスイープ正常終了後に続けて呼ぶ:
    python scripts/build_atlas_report_routed.py --exp-name 0029_20260911_atlas_report_routed

再生成もこれ単体で可能(スイープ出力 outputs/0029_.../finding__*/ がある前提)。
experiments/0020 / scripts/build_atlas_report.py(背景除去 A/B の別レポート)は
変更しない — 本スクリプトは新規。

入力:
  outputs/{exp-name}/finding__*/    スイープ本体(baseline / routed の検索結果)

出力は self-contained な HTML(画像は base64 JPEG data URI 埋め込み、外部依存は
Google Fonts のみ)。Claude Code の Artifact として公開できる形。
"""

import base64
import io
import json
import os
from pathlib import Path


def _b64_jpeg(path: Path, px: int, q: int) -> str | None:
    try:
        from PIL import Image

        im = Image.open(path).convert("RGB")
        if max(im.size) != px:
            im = im.resize((px, px), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=q)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tile_strip(paths, px, q) -> str:
    cells = []
    for p in paths:
        uri = _b64_jpeg(p, px, q)
        if uri:
            cells.append(f'<figure class="tile"><img loading="lazy" src="{uri}" alt=""></figure>')
    return f'<div class="strip">{"".join(cells)}</div>' if cells else '<p class="empty">(タイルなし)</p>'


def _hit_strip(rows, base: Path, px, q) -> str:
    cells = []
    for r in rows:
        pf = r.get("patch_file")
        uri = _b64_jpeg(base / pf, px, q) if pf else None
        cap = f'slide {_esc(r["slide_id"])}<br>sim={float(r["similarity"]):.3f}'
        if uri:
            cells.append(f'<figure class="tile"><img loading="lazy" src="{uri}" alt=""><span class="cap">{cap}</span></figure>')
        else:
            cells.append(f'<figure class="tile miss"><div class="ph"></div><span class="cap">{cap}</span></figure>')
    return f'<div class="strip hits">{"".join(cells)}</div>' if cells else '<p class="empty">(ヒットなし)</p>'


def _finding_section(fdir: Path, fj: dict, px, q, n_query) -> str:
    routed_to = fj.get("routed_to", "baseline")
    chip = (
        '<span class="chip whiten">routed → whiten</span>'
        if routed_to == "whiten"
        else '<span class="chip baseline">routed → baseline(不変)</span>'
    )
    qtiles = sorted((fdir / "query_tiles").glob("*.jpg"))[:n_query]
    baseline = fj["results"].get("baseline", [])
    routed = fj["results"].get("routed", [])
    b_sims = [r["similarity"] for r in baseline] or [0.0]
    r_sims = [r["similarity"] for r in routed] or [0.0]
    title = _esc(
        fj["finding"].replace(" - Nonneoplastic Lesion Atlas", "")
        .replace("Liver, ", "").replace("Liver ", "").replace("Liver", "").strip(" -")
    )
    return f"""
<section class="finding">
  <header class="fhead">
    <h3>{title}</h3>
    <span class="pill">{fj['n_images']} 図版</span>
    <span class="pill">{fj['n_tiles']} タイル検索</span>
    {chip}
  </header>
  <div class="row">
    <div class="rowlabel">クエリタイル<span class="sub">NNL 図版を 224px で分割</span></div>
    {_tile_strip(qtiles, px, q)}
  </div>
  <div class="row">
    <div class="rowlabel">baseline <span class="sub">0018・変換なし<br>sim {min(b_sims):.3f}–{max(b_sims):.3f}</span></div>
    {_hit_strip(baseline, fdir, px, q)}
  </div>
  <div class="row">
    <div class="rowlabel">routed <span class="sub">{routed_to}<br>sim {min(r_sims):.3f}–{max(r_sims):.3f}</span></div>
    {_hit_strip(routed, fdir, px, q)}
  </div>
</section>"""


CSS = """
:root {
  --ground:#f7f4f6; --surface:#fff; --surface-2:#faf7f9;
  --ink:#1d1a1f; --muted:#6c6570; --faint:#9a929c;
  --accent:#7b2d52; --accent-soft:#f0e2ea; --hairline:#e6dee4;
  --whiten:#2f7d5d; --whiten-bg:#e5f1eb; --baseline:#6c6570; --baseline-bg:#f0eef1;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#151217; --surface:#1e1a20; --surface-2:#241f27;
    --ink:#ece7ec; --muted:#a89fa9; --faint:#7a7080;
    --accent:#db8fb2; --accent-soft:#392230; --hairline:#332c37;
    --whiten:#63c398; --whiten-bg:#1e3a2e; --baseline:#a89fa9; --baseline-bg:#241f27;
  }
}
:root[data-theme="dark"] {
  --ground:#151217; --surface:#1e1a20; --surface-2:#241f27;
  --ink:#ece7ec; --muted:#a89fa9; --faint:#7a7080;
  --accent:#db8fb2; --accent-soft:#392230; --hairline:#332c37;
  --whiten:#63c398; --whiten-bg:#1e3a2e; --baseline:#a89fa9; --baseline-bg:#241f27;
}
* { box-sizing:border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
  font-family:"Noto Sans JP", system-ui, sans-serif; font-size:15px; line-height:1.75;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1120px; margin:0 auto; padding:4rem 1.5rem 6rem; }
.lede { max-width:70ch; }
h1,h2,h3 { font-family:"Fraunces","Noto Serif JP",serif; text-wrap:balance; line-height:1.2; }
h1 { font-size:clamp(2rem,5vw,3rem); font-weight:900; margin:0 0 1rem; letter-spacing:-0.01em; }
h2 { font-size:1.6rem; font-weight:600; margin:0 0 1rem; color:var(--accent); }
h3 { font-size:1.15rem; font-weight:600; margin:0; }
.eyebrow { font-size:.72rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--accent); font-weight:700; margin:0 0 .8rem; font-family:"IBM Plex Mono",monospace; }
a { color:var(--accent); }
code { font-family:"IBM Plex Mono",monospace; font-size:.88em; background:var(--surface-2);
  padding:.05em .35em; border-radius:3px; }
hr { border:0; border-top:1px solid var(--hairline); margin:3.5rem 0; }
section.block { margin:3.5rem 0; }
section.block > p { max-width:74ch; }
table.routing { width:100%; border-collapse:collapse; margin:1.5rem 0; font-size:.88rem; }
table.routing th, table.routing td { text-align:left; padding:.5rem .7rem;
  border-bottom:1px solid var(--hairline); }
table.routing th { color:var(--muted); font-size:.74rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.05em; }
.finding { background:var(--surface); border:1px solid var(--hairline); border-radius:12px;
  padding:1.4rem 1.5rem; margin:1.5rem 0; }
.fhead { display:flex; flex-wrap:wrap; align-items:baseline; gap:.55rem .7rem; margin-bottom:1rem; }
.pill { font-size:.72rem; font-weight:500; color:var(--accent); background:var(--accent-soft);
  border-radius:999px; padding:.18em .7em; font-family:"IBM Plex Mono",monospace; white-space:nowrap; }
.chip { font-size:.74rem; font-weight:700; border-radius:999px; padding:.2em .75em; }
.chip.whiten { color:var(--whiten); background:var(--whiten-bg); }
.chip.baseline { color:var(--baseline); background:var(--baseline-bg); }
.row { display:grid; grid-template-columns:132px 1fr; gap:1rem; align-items:start;
  padding:.65rem 0; border-top:1px solid var(--hairline); }
.row:first-of-type { border-top:0; }
.rowlabel { font-size:.82rem; font-weight:700; color:var(--ink);
  font-family:"IBM Plex Mono",monospace; padding-top:.3rem; }
.rowlabel .sub { display:block; font-weight:400; font-family:"Noto Sans JP",sans-serif;
  font-size:.72rem; color:var(--faint); margin-top:.2rem; line-height:1.5; }
.strip { display:flex; gap:.5rem; overflow-x:auto; padding-bottom:.5rem; scrollbar-width:thin; }
.tile { margin:0; flex:0 0 auto; width:92px; }
.strip.hits .tile { width:104px; }
.tile img, .tile .ph { width:100%; aspect-ratio:1; object-fit:cover; border-radius:6px;
  border:1px solid var(--hairline); display:block; background:var(--surface-2); }
.tile.miss .ph { display:grid; place-items:center; }
.tile.miss .ph::after { content:"×"; color:var(--faint); }
.tile .cap { display:block; font-family:"IBM Plex Mono",monospace; font-size:.62rem;
  color:var(--muted); margin-top:.25rem; text-align:center; line-height:1.3; }
.empty { color:var(--faint); font-size:.85rem; }
.foot { margin-top:4rem; padding-top:1.5rem; border-top:1px solid var(--hairline);
  font-size:.78rem; color:var(--faint); line-height:1.7; }
@media (max-width:640px) {
  .row { grid-template-columns:1fr; }
  .wrap { padding:2.5rem 1rem 4rem; }
}
"""

BODY = """
<title>アトラス検索・所見ルーティング版</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Noto+Sans+JP:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>__CSS__</style>
<div class="wrap">
  <p class="eyebrow">patch-vector-search / experiments 0029</p>
  <h1>アトラス検索・所見ルーティング版</h1>
  <p class="lede">experiments/0024〜0028 で、FAISS 索引に異方性除去変換(白色化)を
  ベイクすると、コーパス内検索(self_retrieval)は大きく改善する一方、atlas クエリでは
  <strong>所見によって効果が逆転する</strong>ことが分かった。GT 実測(experiments/0027/0028)
  に基づき、所見ラベルごとに baseline(0018)/ whiten(0025)索引を出し分ける
  <code>lib/finding_routing.py</code> を実装 — このレポートは NNL アトラス
  <strong>__N_FINDINGS__ 所見</strong>を baseline と routed(所見によって baseline のまま
  か whiten に切り替わる)の両方で検索し、上下に並べたもの。__N_WHITEN__ 所見だけ
  whiten に切り替わり、残りは baseline のまま(= 変化なし、悪化しないと確認できるまで
  whiten を使わない保守方針)。</p>
  <hr>
  <section class="block">
    <h2>1. ルーティング表</h2>
    <p>GT実測(atlas 図版単位の best_rank、experiments/0027/0028)に基づく。詳細は
    README「所見ごとのルーティング表」参照。</p>
    __ROUTING_TABLE__
  </section>
  <hr>
  <section class="block">
    <h2>2. NNL アトラス全所見ギャラリー</h2>
    <p>各所見フォルダ内の全図版を統合し、<code>uni_v1</code> plain タイリング(染色正規化なし)
    で 224px タイルに分割 → 近似スコア上位 12 タイルを exact re-rank → 上位 __K__ パッチを
    実解像度でクロップ。whiten 索引側は <code>lib.search.PatchIndex.transform</code> 経由で
    exact re-rank にも異方性除去変換を適用済み(experiments/0025 の「昇格時のTODO」解消)。
    sim は正規化ベクトルの内積(コサイン類似度、baseline と routed で空間が異なるため
    直接の大小比較はできない — 比較は各々の best_rank 実測を参照)。</p>
    __FINDINGS__
  </section>
  <p class="foot">__FOOT__</p>
</div>
"""


def build(*, project_root, exp_name: str, config: dict) -> Path:
    from lib.finding_routing import WHITEN_FINDINGS
    from scripts.validate_against_ground_truth import CATEGORIES

    project_root = Path(project_root)
    out_root = project_root / "outputs" / exp_name

    px = int(config.get("report_thumb_px", 150))
    q = int(config.get("report_jpeg_q", 80))
    n_query = int(config.get("report_query_tiles", 12))
    k = int(config.get("k", 10))

    fdirs = sorted(d for d in out_root.glob("finding__*") if (d / "finding.json").exists())
    sections = []
    n_whiten = 0
    for fdir in fdirs:
        fj = json.loads((fdir / "finding.json").read_text())
        if fj.get("routed_to") == "whiten":
            n_whiten += 1
        sections.append(_finding_section(fdir, fj, px, q, n_query))
    findings_html = "\n".join(sections) if sections else '<p class="empty">(finding 出力が見つかりません)</p>'

    routing_rows = "".join(
        f"<tr><td>{_esc(ft)}</td><td>{'whiten' if ft in WHITEN_FINDINGS else 'baseline'}</td></tr>"
        for ft in sorted(CATEGORIES.values())
    )
    routing_table = (
        '<table class="routing"><thead><tr><th>finding_type(GT実測あり)</th>'
        f"<th>ルーティング</th></tr></thead><tbody>{routing_rows}</tbody></table>"
        '<p class="empty">上記以外(atlas GT 未検証の所見)はすべて baseline。</p>'
    )

    foot = (
        f"experiments/{exp_name} — NNL アトラス {len(fdirs)} 所見 × baseline/routed。"
        f"baseline = outputs/0018_20260909_build_faiss_index_deblank、"
        f"whiten = outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1。"
        f"クエリ経路 uni_v1 plain・rerank_pool={config.get('rerank_pool')}・"
        f"max_tiles_reranked={config.get('max_tiles_reranked')}。"
        f"ルーティング表の根拠は README「experiments/0027/0028」参照。"
    )

    html = (
        BODY.replace("__CSS__", CSS)
        .replace("__N_FINDINGS__", str(len(fdirs)))
        .replace("__N_WHITEN__", str(n_whiten))
        .replace("__K__", str(k))
        .replace("__ROUTING_TABLE__", routing_table)
        .replace("__FINDINGS__", findings_html)
        .replace("__FOOT__", _esc(foot))
    )
    out = out_root / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    mb = len(html.encode("utf-8")) / 1e6
    print(f"[build_atlas_report_routed] wrote {out}  ({mb:.1f} MB)")
    if mb > 15:
        print("[build_atlas_report_routed] WARNING >15MB — report_thumb_px / report_query_tiles を下げて再生成")
    return out


def _project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-name", required=True, help="experiments/{exp-name}/ のディレクトリ名")
    args = ap.parse_args()

    import yaml

    root = _project_root()
    sys.path.insert(0, str(root))
    cfg = yaml.safe_load((root / "experiments" / args.exp_name / "config.yml").read_text()) or {}
    build(project_root=root, exp_name=args.exp_name, config=cfg)
