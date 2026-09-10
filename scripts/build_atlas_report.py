"""experiments/0020（NNL アトラス全所見スイープ）の outputs から report.html を生成する。

experiments/0020 の run_slurm.sh がスイープ正常終了後に続けて呼ぶ:
    python scripts/build_atlas_report.py --exp-name 0020_20260910_query_demo_deblank_atlas_sweep

再生成もこれ単体で可能（スイープ出力 outputs/0020_.../finding__*/ がある前提）。
サムネイルサイズ等のノブは experiments/{exp-name}/config.yml の report_* キー。

入力:
  outputs/{exp-name}/finding__*/                  スイープ本体（0018 / 0002 の検索結果）
  outputs/0019_.../{finding}__deliver/            成果物ギャラリー（glycogen / ground glass / granular）
  outputs/gt_validations/self_retrieval_diagnostic_{0018,0002_predeblank}.csv

出力は self-contained な HTML（画像は base64 JPEG data URI 埋め込み、外部依存は
Google Fonts のみ）。Claude Code の Artifact として公開できる形。
"""

import base64
import csv
import io
import json
import os
from pathlib import Path

DELIVER_EXP = "0019_20260909_build_finding_patch_set_deblank"
SELF_RETRIEVAL_0018 = "gt_validations/self_retrieval_diagnostic_0018.csv"
SELF_RETRIEVAL_0002 = "gt_validations/self_retrieval_diagnostic_0002_predeblank.csv"

DELIVER_FINDINGS = [
    {
        "slug": "deposit_glycogen",
        "title": "Deposit, glycogen",
        "jp": "グリコーゲン沈着",
        "verdict": "works",
        "verdict_label": "機能する（検証済み）",
        "blind": "盲検判定精度 84%（背景除去後・job 10498。0002 索引では 91%）",
        "note": "不規則な白い空胞と淡明でレース状の細胞質。ランダム対照と明確に分離できる — "
        "検索はこの形態を確かに選別している。137 枚 / 30 スライド（GT スライドを全除外し "
        "未ラベルスライドのみから構成、病理レビュー待ち）。",
    },
    {
        "slug": "ground_glass_appearance",
        "title": "Ground glass appearance",
        "jp": "スリガラス様変化",
        "verdict": "works",
        "verdict_label": "機能する（検証済み）",
        "blind": "curated 例で校正した実効分離能 約 80%（盲検の素の判定はラベル取り違えで逆相関、job 10498）",
        "note": "均質で細かいテクスチャの中等度ピンク（glycogen よりおとなしい空胞化）。"
        "選別自体は明確に効くが、glycogen と phenotype が視覚的に重なり、"
        "「スリガラス様」というラベルとの対応づけは病理専門家の確認が要る。113 枚 / 26 スライド。",
    },
    {
        "slug": "degeneration_granular_eosinophilic",
        "title": "Degeneration, granular, eosinophilic",
        "jp": "顆粒状好酸性変性",
        "verdict": "fails",
        "verdict_label": "不成立（コーパスカバレッジ不足）",
        "blind": "n=5 で盲検シートに乗らず対照なし",
        "note": "背景除去前は非 seed 候補 74 枚のうち 48 枚（65%）が背景クロップで、"
        "残り 21 枚が「成果物」に見えていた。背景を抜くと非 seed 候補は 5 枚まで崩壊 — "
        "この所見にはコーパスに一般化可能なシグナルがほぼ無いことが露呈した。",
    },
]

FINDING_JP = {
    "Hypertrophy": "肝細胞肥大", "Microgranuloma": "微小肉芽腫", "Necrosis": "融合壊死",
    "Change, eosinophilic": "好酸性変化", "Increased mitosis": "分裂像増加",
    "Cellular infiltration": "細胞浸潤", "Swelling": "腫大", "Deposit, glycogen": "グリコーゲン沈着",
    "Degeneration, granular, eosinophilic": "顆粒状好酸性変性",
    "Ground glass appearance": "スリガラス様変化", "Single cell necrosis": "単細胞壊死",
    "Hematopoiesis, extramedullary": "髄外造血", "Lesion,NOS": "病変 NOS",
    "Alteration, cytoplasmic": "細胞質変化", "Proliferation, Kupffer cell": "クッパー細胞増殖",
    "Vacuolization, cytoplasmic": "細胞質空胞化",
}

# NNL 所見スイープで注記を付ける所見（slug → (class, ラベル)）。それ以外は中立。
NNL_VERDICT = {
    "Liver__Hepatocyte_-_Glycogen_Accumulation_and_Depletion_-_Nonneoplastic_Lesion_Atlas":
        ("works", "対応 GT: グリコーゲン沈着 — 成果物トラックで検証済み"),
    "Liver__Hepatocyte_-_Hypertrophy_-_Nonneoplastic_Lesion_Atlas":
        ("fails", "不成立（相対的基準を要する — 盲検 59%）"),
    "Liver__Hepatocyte___Increased_Mitosis_-_Nonneoplastic_Lesion_Atlas":
        ("fails", "不成立（sub-patch 局所 — シグナルがパッチの見た目を変えない）"),
    "Liver_-_Necrosis_-_Nonneoplastic_Lesion_Atlas":
        ("weak", "融合壊死はモデル表現の弱点（自己検索でも本物の弱点）"),
    "Liver_-_Extramedullary_Hematopoiesis_-_Nonneoplastic_Lesion_Atlas":
        ("weak", "髄外造血はモデル表現の弱点"),
}


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


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tile_strip(paths, px, q) -> str:
    cells = []
    for p in paths:
        uri = _b64_jpeg(p, px, q)
        if uri:
            cells.append(f'<figure class="tile"><img loading="lazy" src="{uri}" alt=""></figure>')
    return f'<div class="strip">{"".join(cells)}</div>' if cells else '<p class="empty">（タイルなし）</p>'


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
    return f'<div class="strip hits">{"".join(cells)}</div>' if cells else '<p class="empty">（ヒットなし）</p>'


def _finding_section(fdir: Path, fj: dict, px, q, n_query) -> str:
    verdict = NNL_VERDICT.get(fj["slug"])
    chip = f'<span class="chip {verdict[0]}">{_esc(verdict[1])}</span>' if verdict else ""
    qtiles = sorted((fdir / "query_tiles").glob("*.jpg"))[:n_query]
    deblank = fj["results"].get("deblank", [])
    predeblank = fj["results"].get("predeblank", [])
    d_sims = [r["similarity"] for r in deblank] or [0.0]
    p_sims = [r["similarity"] for r in predeblank] or [0.0]
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
    <div class="rowlabel">0018 <span class="sub">背景除去済み・現行既定<br>sim {min(d_sims):.3f}–{max(d_sims):.3f}</span></div>
    {_hit_strip(deblank, fdir, px, q)}
  </div>
  <div class="row">
    <div class="rowlabel">0002 <span class="sub">背景除去前・参照<br>sim {min(p_sims):.3f}–{max(p_sims):.3f}</span></div>
    {_hit_strip(predeblank, fdir, px, q)}
  </div>
</section>"""


def _deliver_section(project_root: Path, px, q, n_show) -> str:
    blocks = []
    for d in DELIVER_FINDINGS:
        base = project_root / "outputs" / DELIVER_EXP / f"{d['slug']}__deliver" / f"{d['slug']}__deliver"
        rows = _read_csv(base / "manifest.csv")
        n_total = len(rows)
        pics = []
        for r in rows[:n_show]:
            uri = _b64_jpeg(base / r["patch_file"], px, q)
            if uri:
                pics.append(
                    f'<figure class="tile"><img loading="lazy" src="{uri}" alt="">'
                    f'<span class="cap">slide {_esc(r["slide_id"])}<br>sim={float(r["similarity"]):.3f}</span></figure>'
                )
        cls = "works" if d["verdict"] == "works" else "fails"
        shown = f"（先頭 {len(pics)} / 全 {n_total} 枚）" if n_total > len(pics) else f"（全 {n_total} 枚）"
        blocks.append(f"""
  <section class="finding">
    <header class="fhead">
      <h3>{_esc(d['title'])}</h3>
      <span class="pill">{_esc(d['jp'])}</span>
      <span class="chip {cls}">{_esc(d['verdict_label'])}</span>
    </header>
    <p class="fnote">{_esc(d['note'])}</p>
    <p class="fnote small">検証: {_esc(d['blind'])}</p>
    <div class="row">
      <div class="rowlabel">deliver 集合<span class="sub">{shown}<br>seed = その所見の TG-GATEs スライド</span></div>
      <div class="strip hits">{"".join(pics) or '<span class="empty">（パッチなし）</span>'}</div>
    </div>
  </section>""")
    return "\n".join(blocks)


def _self_retrieval_table(project_root: Path) -> str:
    a = {r["finding"]: r for r in _read_csv(project_root / "outputs" / SELF_RETRIEVAL_0018)}
    b = {r["finding"]: r for r in _read_csv(project_root / "outputs" / SELF_RETRIEVAL_0002)}
    if not a:
        return '<p class="empty">（self_retrieval_diagnostic CSV が見つかりません）</p>'

    def cell(cur, prev, lower_better=True, colorize=True):
        try:
            cv, pv = float(cur), float(prev)
        except (TypeError, ValueError):
            return f"<td>{_esc(cur if cur not in (None, '') else '—')}</td>"
        if cv == pv:
            return f"<td>{_esc(cur)}</td>"
        delta = f'<span class="d">{_esc(prev)}→</span>{_esc(cur)}'
        if not colorize:
            return f"<td>{delta}</td>"
        better = cv < pv if lower_better else cv > pv
        return f'<td class="{"up" if better else "down"}">{delta}</td>'

    rows = []
    for finding, ra in a.items():
        rb = b.get(finding, {})
        jp = FINDING_JP.get(finding, "")
        rows.append(
            f"<tr><td class='fn'>{_esc(finding)}<span class='jp'>{_esc(jp)}</span></td>"
            f"<td>{_esc(ra['n_corpus_slides'])}</td>"
            # n_hits_ratio は背景除去に頑健（±1 = ノイズ）なので色を付けない。
            f"{cell(ra['nhr_best_rank_med'], rb.get('nhr_best_rank_med'), colorize=False)}"
            f"{cell(ra['sim_best_rank_med'], rb.get('sim_best_rank_med'))}"
            f"{cell(ra['sim_hit@10'], rb.get('sim_hit@10'), lower_better=False)}</tr>"
        )
    return (
        '<table class="metrics"><thead><tr>'
        "<th>finding（自己検索診断）</th><th>コーパス<br>スライド数</th>"
        "<th>best_rank<br>n_hits_ratio</th><th>best_rank<br>max_similarity</th><th>hit@10<br>max_similarity</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        '<p class="caption"><code>0002→</code> は背景除去前の中央値（すべて所見スライドの '
        "leave-one-out best_rank / hit@10）。<code>n_hits_ratio</code> 列は背景除去に頑健で"
        "全 finding ±1（ノイズ）なので色を付けていない。<code>max_similarity</code> の 2 列は"
        "色付きセルが 0018 での変化方向（緑 = 改善 / 赤 = 悪化）で、<strong>一貫して改善</strong>"
        "する — 背景パッチは多くのクエリに中程度の類似度を持ち、「スライドを最良マッチ 1 枚で"
        "順位付け」する max_similarity では背景の多いスライドが浮上して所見スライドを"
        "押し下げていたのが、背景除去で解消するため。"
        "<code>scripts/self_retrieval_diagnostic.py</code>、job 10496 / 10494。</p>"
    )


CSS = """
:root {
  --ground:#f7f4f6; --surface:#fff; --surface-2:#faf7f9;
  --ink:#1d1a1f; --muted:#6c6570; --faint:#9a929c;
  --accent:#7b2d52; --accent-soft:#f0e2ea; --hairline:#e6dee4;
  --works:#2f7d5d; --works-bg:#e5f1eb; --fails:#b23b3b; --fails-bg:#f6e4e2;
  --weak:#8a6d1f; --weak-bg:#f3ecda;
  --up:#1f7a4d; --up-bg:#e3f1e9; --down:#b03a3a; --down-bg:#f7e6e4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ground:#151217; --surface:#1e1a20; --surface-2:#241f27;
    --ink:#ece7ec; --muted:#a89fa9; --faint:#7a7080;
    --accent:#db8fb2; --accent-soft:#392230; --hairline:#332c37;
    --works:#63c398; --works-bg:#1e3a2e; --fails:#e08a8a; --fails-bg:#3d2422;
    --weak:#d7b665; --weak-bg:#362f1c;
    --up:#63c398; --up-bg:#1e3a2e; --down:#e08a8a; --down-bg:#3d2422;
  }
}
:root[data-theme="dark"] {
  --ground:#151217; --surface:#1e1a20; --surface-2:#241f27;
  --ink:#ece7ec; --muted:#a89fa9; --faint:#7a7080;
  --accent:#db8fb2; --accent-soft:#392230; --hairline:#332c37;
  --works:#63c398; --works-bg:#1e3a2e; --fails:#e08a8a; --fails-bg:#3d2422;
  --weak:#d7b665; --weak-bg:#362f1c;
  --up:#63c398; --up-bg:#1e3a2e; --down:#e08a8a; --down-bg:#3d2422;
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
.two { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1.25rem; margin:1.5rem 0; }
.card { background:var(--surface); border:1px solid var(--hairline); border-radius:10px; padding:1.25rem 1.4rem; }
.card h3 { font-family:"Noto Sans JP",sans-serif; font-size:.95rem; color:var(--accent); margin-bottom:.4rem; }
.card p { margin:0; font-size:.9rem; color:var(--muted); }
.card .big { font-family:"Fraunces",serif; font-size:2rem; font-weight:900; color:var(--ink);
  display:block; line-height:1; margin-bottom:.3rem; font-variant-numeric:tabular-nums; }
.finding { background:var(--surface); border:1px solid var(--hairline); border-radius:12px;
  padding:1.4rem 1.5rem; margin:1.5rem 0; }
.fhead { display:flex; flex-wrap:wrap; align-items:baseline; gap:.55rem .7rem; margin-bottom:1rem; }
.fnote { max-width:76ch; color:var(--muted); font-size:.9rem; margin:.2rem 0 .6rem; }
.fnote.small { font-size:.82rem; color:var(--faint); }
.pill { font-size:.72rem; font-weight:500; color:var(--accent); background:var(--accent-soft);
  border-radius:999px; padding:.18em .7em; font-family:"IBM Plex Mono",monospace; white-space:nowrap; }
.chip { font-size:.74rem; font-weight:700; border-radius:999px; padding:.2em .75em; }
.chip.works { color:var(--works); background:var(--works-bg); }
.chip.fails { color:var(--fails); background:var(--fails-bg); }
.chip.weak { color:var(--weak); background:var(--weak-bg); }
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
table.metrics { width:100%; border-collapse:collapse; margin:1.5rem 0 .5rem; font-size:.82rem;
  font-variant-numeric:tabular-nums; }
table.metrics th, table.metrics td { text-align:right; padding:.45rem .6rem;
  border-bottom:1px solid var(--hairline); }
table.metrics th { color:var(--muted); font-weight:700; font-size:.72rem; vertical-align:bottom; }
table.metrics td.fn, table.metrics th:first-child { text-align:left; }
table.metrics td.fn { font-weight:500; }
table.metrics td.fn .jp { display:block; font-size:.72rem; color:var(--faint); font-weight:400; }
table.metrics td.up { color:var(--up); background:var(--up-bg); font-weight:700; }
table.metrics td.down { color:var(--down); background:var(--down-bg); }
table.metrics td .d { color:var(--faint); font-family:"IBM Plex Mono",monospace; font-size:.78em; }
.caption { font-size:.78rem; color:var(--faint); max-width:80ch; line-height:1.7; }
.tablewrap { overflow-x:auto; }
table.taxo { width:100%; border-collapse:collapse; margin:1.5rem 0; font-size:.88rem; }
table.taxo th, table.taxo td { text-align:left; padding:.6rem .7rem;
  border-bottom:1px solid var(--hairline); vertical-align:top; }
table.taxo th { color:var(--muted); font-size:.74rem; font-weight:700;
  text-transform:uppercase; letter-spacing:.05em; }
table.taxo .v-works { color:var(--works); font-weight:700; }
table.taxo .v-fails { color:var(--fails); font-weight:700; }
ol.next { max-width:76ch; padding-left:1.2rem; }
ol.next li { margin:.5rem 0; }
.foot { margin-top:4rem; padding-top:1.5rem; border-top:1px solid var(--hairline);
  font-size:.78rem; color:var(--faint); line-height:1.7; }
@media (max-width:640px) {
  .row { grid-template-columns:1fr; }
  .wrap { padding:2.5rem 1rem 4rem; }
}
"""

BODY = """
<title>非腫瘍性病変アトラス検索・改善版</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=Noto+Sans+JP:wght@400;500;700&family=Noto+Serif+JP:wght@600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>__CSS__</style>
<div class="wrap">
  <p class="eyebrow">patch-vector-search / experiments 0020</p>
  <h1>非腫瘍性病変アトラス検索・改善版</h1>
  <p class="lede">初代の検索デモ Artifact は、NTP 非腫瘍性病変アトラス（NNL）の所見図版を
  クエリに TG-GATEs ラット肝 1000 スライドのコーパスを検索し、「望んでいるほど似ていない
  画像も多く得られてしまった」で終わっていた。この 1 か月で <strong>背景パッチの除去</strong>、
  <strong>評価軸の刷新</strong>、<strong>成果物パイプラインの確立</strong>を行った。同じ立て付け
  ——所見ごとに「クエリタイル」と「検索で引けたパッチ」を並べる——で、現行の検索経路が
  何を引くのかを作り直したのが本レポート。<strong>__N_FINDINGS__ 所見</strong>を
  <code>0018</code>（背景除去済み・現行既定）と <code>0002</code>（背景除去前）の
  両索引で検索し、上下に並べてある。</p>
  <hr>
  <section class="block">
    <h2>1. この 1 か月で改善したこと</h2>
    <div class="two">
      <div class="card"><h3>背景パッチの除去</h3><span class="big">389,959</span>
        <p>旧空白フィルタ（平均輝度&gt;240 かつ標準偏差&lt;8）は背景の半分以下しか捕まえて
        いなかった。彩度基準（<code>sat_frac &lt; 0.10</code>）に置き換え、manifest から
        2.12% を除外して索引を再構築（<code>0017</code>→<code>0018</code>、GPU 再埋め込み不要）。</p></div>
      <div class="card"><h3>評価軸の刷新</h3><span class="big">7 → 16</span>
        <p>アトラス図版クエリの GT 比較（交絡が多く 7 所見のみ）から、コーパス内
        leave-one-out の<strong>自己検索診断</strong>（16 所見、GPU 不要）へ。バッチ交絡を
        除いても「検索に足る」のは 5〜6 の common な肝所見と判明。</p></div>
      <div class="card"><h3>成果物パイプライン</h3><span class="big">137 / 113</span>
        <p><code>lib/patch_set.py</code> で、所見の TG-GATEs スライドを seed に
        「レビュー可能な代表パッチ集」を出力。glycogen 137 枚・ground glass 113 枚が
        ランダム対照で選別を確認済み（病理レビュー待ち）。</p></div>
    </div>
    <p>検索コア（<code>PatchIndex</code>: FAISS OPQ+IVF+PQ → exact re-rank → n_hits_ratio 集計）は
    初代から変えていない。染色正規化・倍率補正・手動 ROI クロップ・IVF クラスタリング仮説は
    いずれも GT 比較で既存パイプラインを上回れず棚上げした。</p>
  </section>
  <hr>
  <section class="block">
    <h2>2. NNL アトラス全所見ギャラリー</h2>
    <p>各所見フォルダ内の全図版を統合し、<code>uni_v1</code> plain タイリング（染色正規化なし）
    で 224px タイルに分割 → 近似スコア上位 12 タイルを exact re-rank（notebook 01 の推奨値）→
    上位 __K__ パッチを実解像度でクロップ。<code>0018</code> と <code>0002</code> は同一クエリ・
    同一パラメータで、差は索引（背景パッチの有無）のみ。sim は正規化ベクトルの内積（コサイン類似度）。</p>
    <p><strong>0018 と 0002 の結果は所見によって「ほぼ不変」〜「上位が入れ替わる」まで幅がある。</strong>
    融合壊死・脂肪変性のように組織マッチが強い所見は上位 10 枚が同一。色素沈着・クッパー細胞
    増殖のように<strong>背景の彩度帯に近い</strong>所見では、0002 の上位に混ざっていたパッチが
    0018 で押し出されて順位が変わる。背景除去の主効果はここではなく<strong>スライド単位の
    n_hits_ratio ランキング</strong>（下の第 4 節）に出る — max_similarity 上位パッチは元々
    ほとんど組織だからだ。</p>
    <p>多くの所見で <code>0018</code> の上位が「所見と無関係な正常組織」で埋まるのは、
    モデルの表現力ではなく<strong>アトラス→TG-GATEs のドメインギャップ</strong>が主因
    （別スキャナ・染色・倍率、図版が非所見組織だらけ、ROI なし）。</p>
    __FINDINGS__
  </section>
  <hr>
  <section class="block">
    <h2>3. 成果物ギャラリー（機能する所見）</h2>
    <p>ここまでで「patch レベル検索 + curation が機能する」と検証できた 2 所見。いずれも
    アトラス図版ではなく<strong>その所見の TG-GATEs スライドを seed</strong> にし、GT スライドを
    全除外して未ラベルスライドのみから構成した（<code>experiments/0019</code>、背景除去済み
    コーパス）。「コヒーレントに見える」ことは構成上の必然なので、<strong>コーパスから一様
    ランダム抽出した対照と混ぜた盲検判定</strong>で選別の有無を確認している。</p>
    __DELIVER__
  </section>
  <hr>
  <section class="block">
    <h2>4. 自己検索診断: 背景除去の before / after</h2>
    <p>アトラス図版を一切使わず、コーパス内の所見スライドを 1 枚ずつ抜いて残りから
    引けるかを測る（GPU 不要）。アトラス→TG-GATEs のドメインギャップという交絡が無いぶん、
    モデル／コーパスの検索天井を素直に測れる。</p>
    <div class="tablewrap">__SELF_RETRIEVAL__</div>
  </section>
  <hr>
  <section class="block">
    <h2>5. 所見の 4 クラス分け</h2>
    <p>0014 / 0015 / ランダム対照で見えた実データの構造。当初は「whole-patch テクスチャ /
    sub-patch 局所」の 2 クラスで hypertrophy を前者に入れていたが、ランダム対照で不成立と
    分かり 3 クラス目を立てた。granular eosinophilic はさらに別の失敗をする。</p>
    <table class="taxo">
      <thead><tr><th>クラス</th><th>例</th><th>patch レベル検索 + curation</th></tr></thead>
      <tbody>
        <tr><td>whole-patch テクスチャ（パッチ内で完結）</td><td>glycogen、ground glass（検証済み）</td><td class="v-works">機能する（seed が動く側なら）</td></tr>
        <tr><td>相対的基準を要する</td><td>hypertrophy、萎縮</td><td class="v-fails">機能しない（パッチ単体に基準がない）</td></tr>
        <tr><td>sub-patch 局所</td><td>分裂像増加、単一分裂像</td><td class="v-fails">機能しない（シグナルがパッチの見た目を変えない）</td></tr>
        <tr><td>コーパスカバレッジ不足</td><td>granular eosinophilic</td><td class="v-fails">機能しない（類似組織が乏しく天井が低い）</td></tr>
      </tbody>
    </table>
    <p>4 つ目だけは所見の性質ではなくコーパス側の事情なので、コーパスを広げれば解消しうる。
    融合壊死（<code>Necrosis</code>）と髄外造血は、自己検索診断でもバッチ交絡を除いた後に
    残る<strong>本物の表現の弱点</strong>。</p>
  </section>
  <section class="block">
    <h2>6. 次の一手</h2>
    <ol class="next">
      <li>glycogen 137 枚・ground glass 113 枚を<strong>病理知識のある人にレビュー</strong> —
      「検索が何かを選別している」ことは確認済みなので、残る問いは「それがその所見か」。</li>
      <li>融合壊死・髄外造血のための <code>uni_v2</code> 等他モデル検討（全所見のためではなく
      この 2 所見のため）。</li>
      <li>アトラス→TG-GATEs のドメインギャップを詰める（<code>0014</code> の #1 ブロッカー）。</li>
      <li>hypertrophy を扱うならパッチサイズを上げて基準組織を同一視野に入れる等、
      相対的基準の問題そのものに手を付ける（現行 224px の延長線上には無い）。</li>
    </ol>
  </section>
  <p class="foot">__FOOT__</p>
</div>
"""


def build(*, project_root, exp_name: str, config: dict) -> Path:
    # スイープ出力もレポートも outputs/{exp}/ の NFS canonical に置く前提
    # (experiments/0020 の run_slurm.sh は USE_LOCAL_SSD_OUTPUT=0)。スイープを
    # スキップした report のみの再実行でも finding__* をそのまま拾える。
    project_root = Path(project_root)
    out_root = project_root / "outputs" / exp_name

    px = int(config.get("report_thumb_px", 150))
    q = int(config.get("report_jpeg_q", 80))
    n_query = int(config.get("report_query_tiles", 12))
    n_deliver = int(config.get("report_deliver_patches", 28))
    k = int(config.get("k", 10))

    fdirs = sorted(d for d in out_root.glob("finding__*") if (d / "finding.json").exists())
    sections = []
    for fdir in fdirs:
        fj = json.loads((fdir / "finding.json").read_text())
        sections.append(_finding_section(fdir, fj, px, q, n_query))
    findings_html = "\n".join(sections) if sections else '<p class="empty">（finding 出力が見つかりません）</p>'

    foot = (
        f"experiments/{exp_name} — NNL アトラス {len(fdirs)} 所見 × 索引 2 本"
        f"（0018 = outputs/0018_20260909_build_faiss_index_deblank / "
        f"0002 = outputs/0002_20260808_build_faiss_index）。"
        f"クエリ経路 uni_v1 plain・rerank_pool={config.get('rerank_pool')}・"
        f"max_tiles_reranked={config.get('max_tiles_reranked')}。"
        f"成果物ギャラリーは outputs/{DELIVER_EXP}、盲検判定は scripts/random_patch_baseline.py。"
    )

    html = (
        BODY.replace("__CSS__", CSS)
        .replace("__N_FINDINGS__", str(len(fdirs)))
        .replace("__K__", str(k))
        .replace("__FINDINGS__", findings_html)
        .replace("__DELIVER__", _deliver_section(project_root, px, q, n_deliver))
        .replace("__SELF_RETRIEVAL__", _self_retrieval_table(project_root))
        .replace("__FOOT__", _esc(foot))
    )
    out = out_root / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    mb = len(html.encode("utf-8")) / 1e6
    print(f"[build_atlas_report] wrote {out}  ({mb:.1f} MB)")
    if mb > 15:
        print("[build_atlas_report] WARNING >15MB — report_thumb_px / report_query_tiles を下げて再生成")
    return out


def _project_root() -> Path:
    env = os.environ.get("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # scripts/ の親 = リポジトリルート
    return Path(__file__).resolve().parent.parent


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp-name", required=True, help="experiments/{exp-name}/ のディレクトリ名")
    args = ap.parse_args()

    import yaml

    root = _project_root()
    cfg = yaml.safe_load((root / "experiments" / args.exp_name / "config.yml").read_text()) or {}
    build(project_root=root, exp_name=args.exp_name, config=cfg)
