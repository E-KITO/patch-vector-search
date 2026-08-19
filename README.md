# patch-vector-search

UNIパッチ埋め込みに対するクラスタベースのベクトル検索
(FAISS OPQ+IVF+PQ)+WSI逆引き。任意の参照画像(既存コーパス外でもOK)を渡すと、
類似する組織パッチと、それを多く含むWSIを検索できる。  
**最終的な目標は、INHANDやNTPの非腫瘍性病変アトラスのような、所見の代表的なパッチを検索クエリとして使い、TGGATEの大規模なWSIコーパスから同じ所見を引き出してデータベース化すること。**

## 現状(2026-08-14時点)

- **動くもの**: 任意の画像(1枚〜複数枚)を渡すと、類似パッチ検索とWSI逆引きができる。
- **定量評価**: NTP非腫瘍性病変アトラス由来のクエリ画像(下記「評価に使ったデータと
  その限界」参照)で7カテゴリのground truth比較を実施済み。7カテゴリとも正解スライドを
  候補プール内に発見できているが(found=n_gt)、best_rank(最上位正解スライドの順位)は
  9〜332位とカテゴリによって大きな差がある。
- **未解決の課題は「今後やること」参照**。特にIVFクラスタ被覆率の仮説と、
  評価クエリを手動ROIクロップに変えた場合の効果は、どちらも手を付けていない。

## 評価に使ったデータとその限界

Ground truth比較(`scripts/validate_against_ground_truth.py`)のクエリ画像は、
`data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver/`に置いた
**NTP(National Toxicology Program)の非腫瘍性病変アトラス(NNL)**から取得した、
各所見カテゴリの代表的な掲載図版(26カテゴリ・94枚、1所見あたり1〜8枚)。

以下の限界を踏まえて結果を解釈すること:

- **アトラス画像1枚は所見部位を含む図版全体であり、所見が写っているのは画像の一部分に
  過ぎない**(矢印注釈・番号ラベル・周囲の正常組織や余白を含む、1800x1200px程度の
  パノラマ〜クローズアップが倍率不揃いのまま混在している)。
- **所見部位だけを人手で切り出す作業はしていない**。`embed_image_tiles`が画像全体を
  機械的にタイル分割し、白背景タイルの除外以外は所見領域かどうかの判別なしに全タイルを
  検索へ投入している。つまりこの評価は「自動化できる範囲でどこまでやれるか」を測った
  ものであり、人手でROIを切り出した場合の性能上限を示すものではない。best_rankが悪い
  カテゴリ(例: Kupffer細胞増殖 best_rank=124、封入体 best_rank=14〜297)は、モデルや
  検索アルゴリズムの限界だけでなく、クエリ画像に占める「所見と無関係なタイル」の割合が
  高いことも一因である可能性が高い。
- Ground truthとしている所見自体(`data/processed_csv/single_finding_liver.csv`)は998スライド
  コーパスの一部にしか対応しない。NNLの26カテゴリのうち、998スライドコーパス内に
  確定ラベル付きスライドが1件でもあるのは7カテゴリのみ(Hypertrophy/Necrosis/
  Increased mitosis/Glycogen/Hematopoiesis/Kupffer細胞増殖/封入体)。残り19カテゴリ
  (Fatty Change, Focus, Inflammationなど)は検証不能。
  Degeneration, fattyとFatty Changeなど、対応付けを行うべき所見もあるが現状は未対応。
  その為、Ground truth比較は基本あてにしなくていい。あくまで「7カテゴリのうち、正解スライドを候補プール内に発見できるか」という観点での評価に留める。

## 今後やること

- [ ]  **uni v2など他モデルでの埋め込みを検討する**
- [x]  ~~IVFクラスタリングの粗さが近似検索の精度を下げている可能性の検証~~
      → 2026-08-19、uni_v1で検証済み・否定的な結論。詳細は下記「IVFクラスタリング
      仮説の検証状況」参照
- [ ]  **クエリ画像を人手でROIクロップした場合の性能上限を測る**(↑の調査結果から、
      次に着手すべき最有力候補)
- [ ]  **検索結果の可視化を改善する**(現状はサムネイルJPEG上にヒット位置をプロットするだけで、実解像度WSIピクセルは無い。WSIのヒートマップ表示や、ヒットパッチのサムネイル表示など)
- [ ]  **抽出できる所見・できない所見を精査する**\
など、いろいろやる。

## 大まかな構成

```
[① manifest構築]                [② FAISSインデックス構築]
lib/manifest.py           →    lib/faiss_index.py
998個のh5を走査し                OPQ+IVF+PQ(コサイン類似度=
manifest/slide_meta/             正規化ベクトルのinner product)
学習サンプルを作成                を学習・構築。圧縮後 数GB程度
(experiments/0001)               (experiments/0002)
                                        │
                                        ▼
[③ クエリ画像埋め込み]            [④ 検索]
lib/query_embedding.py    →    lib/search.py::PatchIndex
任意画像→タイル分割→              近似候補をFAISSで取得し、
TRIDENTのUNIエンコーダで          パッチ検索はexact re-rank、
埋め込み(コーパス構築時と          WSI逆引きはn_hits_ratioで
同じ前処理・同じTRIDENT revに固定)  スライド集計
                                        │
                                        ▼
                                [⑤ 可視化]
                                lib/visualize.py
                                サムネイルJPEG上にヒット位置を
                                プロット(生WSIピクセルが無い
                                前提の代替可視化)
```

`experiments/0001`→`0002`が一度だけ実行するインデックス構築、`experiments/0003`
(または`notebooks/01_query_demo.ipynb`)が③〜⑤を毎回実行するクエリ側。
`scripts/validate_against_ground_truth.py`はground truthとの比較専用で本番の検索
パスとは独立しており、新しい前処理案の採否判断に使う。

## 使っているツール

pyproject.tomlはscaffold元のtemplateに由来する依存が大量に残っているが、このプロジェクトが
実際に使っているのは以下のみ:

| ツール | 用途 |
|---|---|
| `trident`(git依存、TRIDENTの特定commitにpin) | UNIエンコーダ(`encoder_factory`)呼び出し。コーパス側と同じ前処理を再現するため異なるrevへは上げない |
| `faiss-cpu` | OPQ+IVF+PQインデックスの学習・構築・検索(クラスタベース近似最近傍探索の実体) |
| `h5py` | コーパスのパッチ特徴量(`features_uni_v1`/`features_uni_v2`の`.h5`)読み込み |
| `torchstain` | `uni_v2`コーパスのクエリ側染色正規化(Macenko)。自作`lib/stain_normalize.py`では再現不十分と判明したため必須 |
| `torch` / `torchvision` | UNIエンコーダの推論バックエンド(TRIDENT経由) |
| `pillow` | 画像読み込み・タイル分割・リサイズ |
| `pandas` / `pyarrow` | manifest/slide_metaの`.parquet`読み書き |
| `numpy` | ベクトル演算全般 |

それ以外(jupyterlab, optuna, lightgbm, elasticsearch, spacy, vllm, wandb 等)は
scaffold元テンプレートの汎用依存で、このプロジェクトのコードからは一切参照していない。

## 使い方

前提: `experiments/0001_20260808_build_patch_manifest` →
`experiments/0002_20260808_build_faiss_index` が実行済みで、
`outputs/0002_20260808_build_faiss_index/default/` にインデックスがある状態。

```python
import numpy as np
from lib.query_embedding import embed_image_tiles
from lib.search import PatchIndex

patch_index = PatchIndex.load(
    index_path="outputs/0002_20260808_build_faiss_index/default/index.faiss",
    manifest_path="outputs/0002_20260808_build_faiss_index/default/manifest.parquet",
    slide_meta_path="outputs/0002_20260808_build_faiss_index/default/slide_meta.parquet",
    features_dir="data/trident_processed/20x_224px_0px_overlap/features_uni_v1",
)

# 同じ所見の参照画像は複数枚渡せる(1枚でも可)
query_vecs = np.concatenate([embed_image_tiles(img) for img in ["a.jpg", "b.jpg"]], axis=0)

similar_patches = patch_index.search_similar_patches_multi(query_vecs, k=20)
top_slides = patch_index.search_top_slides_multi(query_vecs, top_n_slides=20)  # n_hits_ratioで既にソート済み
```

対話的に試すなら `notebooks/01_query_demo.ipynb`、Slurm経由で実行するなら
`experiments/0003_20260808_query_demo`(`runx`で投入)。

## 何を使うべきか / 使うべきでないか

複数のアプローチを試し、ground truth(`data/processed_csv/single_finding_liver.csv`、
確定病理ラベルのある7カテゴリ)で実際に検証した結論:

| 手法 | 推奨? | 理由 |
|---|---|---|
| `embed_image_tiles`(タイル分割) | ✅ 推奨(既定) | 病変位置が不明な大きい参照画像で単一クロップより一貫して優れる |
| `search_similar_patches_multi` / `search_top_slides_multi`(タイルごと個別検索→結果統合) | ✅ 推奨(既定) | 複数参照画像はベクトル平均よりこちらの方が頑健 |
| WSI逆引きの`n_hits_ratio`ソート | ✅ 推奨(既定) | 単純な`n_hits`ソートより7カテゴリ中5カテゴリで改善、追加コストなし |
| `embed_image_tiles_auto_scale`(倍率自動補正、`lib/mpp_estimation.py`) | ❌ 非推奨 | 画質は改善するが、GT比較では検索精度がほぼ悪化(7カテゴリ中0カテゴリで最良) |
| `stain_reference=`(Macenko染色正規化) | ❌ 非推奨 | 同上。ケースによっては大きく悪化する |
| `embed_image(..., resize_mode="centercrop")`(単一クロップ) | ⚠️ 場合による | 結果の分散が大きく、GTを完全に見失うこともある |
| `encoder_name="uni_v2"`コーパス(1536次元・256px、`experiments/0004`/`0005`/`0006`) | ⚠️ v1よりやや劣るが僅差(公平な比較後) | クエリ側は`lib.torchstain_normalize.normalize_to_reference`で`data/baseline/63958_x38976_y7616.png`に正規化してから使うこと(`lib.stain_normalize`の自作Macenkoでは不十分——自己一致性0.5〜0.84止まり、torchstainなら0.96〜0.996)。正規化後の公平なGT比較でも7カテゴリ中6カテゴリでv1が優位だが、差は大幅縮小(例: Hypertrophy 94→27)。PQ量子化を細かくする(pq_m 64→96)ことも試したが改善なし。詳細は下記「uni_v2コーパスの調査状況」参照。既定は引き続き`uni_v1`(`experiments/0001`/`0002`) |

新しい前処理・パラメータのアイデアを試すときは、必ず
`scripts/validate_against_ground_truth.py`で既存パイプラインとground truth比較してから
採用すること — 目視で綺麗に見えることは検索精度が上がることを意味しない(このプロジェクトで
2回実際に踏んだ落とし穴)。

## ディレクトリ構成(このプロジェクト固有)

- `lib/manifest.py` — 998スライドを走査し、manifest/slide_meta/学習サンプルを構築
- `lib/faiss_index.py` — OPQ+IVF+PQインデックスの学習・構築
- `lib/query_embedding.py` — 任意画像→UNI埋め込み(タイル分割・倍率補正・染色正規化オプション)
- `lib/search.py` — `PatchIndex`: 類似パッチ検索・WSI逆引き
- `lib/mpp_estimation.py` — 倍率(MPP)自動推定。**非推奨**、詳細はモジュールdocstring参照
- `lib/stain_normalize.py` — 自作Macenko染色正規化。**非推奨**、詳細は`embed_image`のdocstring参照
- `lib/torchstain_normalize.py` — `torchstain`ライブラリ経由のMacenko正規化。uni_v2コーパス
  (`data/trident_processed_macenko`)のクエリ側正規化に必須(自作`stain_normalize.py`では
  再現不十分)、詳細はモジュールdocstring参照
- `lib/visualize.py` — サムネイル上へのヒット位置プロット
- `lib/raw_patch.py` — openslide経由での実解像度パッチクロップ
- `experiments/0001_..._build_patch_manifest` → `0002_..._build_faiss_index` → `0003_..._query_demo`
  (この順で依存。**現行の推奨インデックス**、`uni_v1`・1024次元)
- `experiments/0004_..._build_patch_manifest_v2` → `0005_..._build_faiss_index_v2`
  (`uni_v2`・1536次元・pq_m=64版)→ `0006_..._build_faiss_index_v2_pqm96`
  (同じ元データ、pq_m=96版。詳細は下記「uni_v2コーパスの調査状況」参照)
- `scripts/validate_against_ground_truth.py` — 新しいパイプライン案をGTで検証するツール(要・使用)
- `scripts/select_average_patch.py` — 染色正規化用の「典型的な」基準パッチを選ぶ(セットアップ用、実行済み)

## uni_v2コーパスの調査状況(2026-08-14、一旦区切り)

`data/trident_processed_macenko`(TRIDENTのuni_v2エンコーダ・256pxネイティブパッチ・
torchstainでMacenko正規化済みの生WSIから抽出)という別コーパスを試験的に評価した。
**結論: `uni_v1`(既定)を置き換えるには至らなかった。** 経緯:

1. 最初にground truth比較したところ7カテゴリ中6カテゴリでv1に劣ったが、原因は
   モデル性能ではなく、クエリ側の染色正規化がコーパス側(`torchstain`ライブラリ使用)と
   一致していなかったこと(自作`lib/stain_normalize.py`では自己一致性0.5〜0.84止まり)。
   `lib/torchstain_normalize.py`で修正し、公平な比較にしたところ差は大幅縮小
   (例: Hypertrophy best_rank 94→27)。
2. それでもv1が引き続き優位だったため、`experiments/0005`(pq_m=64、1サブベクトル
   あたり24次元)のPQ量子化が粗すぎる可能性を疑い、exact(生ベクトル)計算と近似検索の
   順位を比較する診断を実施。**Hypertrophyカテゴリで、近似検索が300〜400位台に
   埋もれさせていたスライドが、exact計算では上位10〜20位相当の強いシグナルを
   持っていることを確認**(例: スライド28741は近似313位だがexact上位11位相当)。
3. これを受けて`experiments/0006`(pq_m=96、v1と同じ1サブベクトルあたり16次元)で
   インデックスを再構築し、GT比較をやり直した。**しかし改善は誤差レベルに留まった**
   (Hypertrophy best_rankは27のまま、Glycogenはむしろ107→127に悪化)。
   nprobeを64→256に上げる追加検証でも27→25とごくわずかな改善のみだったことも
   踏まえ、**「埋もれた強いシグナルが近似検索で見つからない」問題の主因はPQの
   量子化精度ではなく、IVFの粗いクラスタリング(nlist=4096、nprobeが探索する
   クラスタ数)側にある可能性が高い**、という所見で一旦区切っている
   (深追いすればnlist自体を大きくする、GT関連スライドが同じクラスタに
   収まっているか確認する、といった方向性はあるが未着手)。

**現状のデフォルト**: `scripts/validate_against_ground_truth.py::load_v2_index()`は
`experiments/0005`(pq_m=64、改善は無いがシンプルでインデックスも小さい)を指す。
`experiments/0006`(pq_m=96)は参考用に残置(`exp_dir=`引数で指定すれば使える)。
再びこの調査を引き継ぐ場合は、上記3までで判明している「IVFクラスタリング側の
問題」という仮説の検証(nlist引き上げ、またはGT該当スライドが実際にどのクラスタに
属しているかの直接確認)から始めるのが自然な続き——だったが、下記の通りuni_v1で
この仮説自体を検証し、否定的な結論に至ったため、優先度は下がっている。

## IVFクラスタリング仮説の検証状況(uni_v1、2026-08-19)

上のuni_v2調査で浮上した「IVFクラスタリング(`nlist`/`nprobe`)側の近似誤差が
GT正解スライドの順位を大きく落としているのでは」という仮説を、**既定の
uni_v1索引(`experiments/0001`/`0002`)に対しても切り分けて検証した。
結論: uni_v1でも支持されなかった。** 0004〜0006(uni_v2系)の実行・再構築は
不要で、既存のv1索引だけで検証できた。

経緯:

1. **ベースライン取得**(`scripts/adhoc_validate_against_ground_truth.sh`、
   `nprobe=64`固定): `outputs/gt_validation_results.csv`。7カテゴリ中、
   Kupffer細胞増殖(best_rank=202)・封入体(best_rank=480)が特に悪く、
   いずれも`found=n_gt`(正解は候補プールには入っている)。
2. **nprobeスイープ**(`scripts/adhoc_nprobe_sweep.sh` → `scripts/nprobe_sweep.py`、
   `nprobe=64/256/1024/4096`): `outputs/gt_validation_nprobe_sweep.csv`。
   `nprobe=1024`以降はほぼ飽和。`nprobe=4096`(=`nlist`、全クラスタ探索、
   クラスタ探索の取りこぼしが理論上ゼロになる設定)にしてもKupffer細胞増殖は
   202→202で不変、封入体は480→399とわずかに改善するのみ。
   **→クラスタ探索の取りこぼし(coverage)が主因という仮説は否定的。**
3. **exact-vs-approximate診断**(`scripts/adhoc_exact_vs_approx_diagnostic.sh` →
   `scripts/exact_vs_approx_diagnostic.py`): `outputs/gt_validation_exact_vs_approx.csv`。
   FAISSのPQ近似スコアを、`PatchIndex._exact_similarity`による厳密(生float32)
   スコアに置き換えて同一候補プール内で再ランキング。Kupffer細胞増殖は
   462→443(ほぼ誤差)、封入体は528→532(むしろ悪化)。
   **→PQ量子化誤差が主因という仮説も否定的。**
   - 初回実装には2つのバグがあり修正済み: (a) ランキングキーが`n_hits_ratio`
     (候補プールに入るか否かの件数ベース)になっており、類似度の値(近似/厳密)
     を全く反映しない設計だった → `max_similarity`に変更。(b)
     `MAX_TILES_RERANKED=8`の上限により、Kupffer細胞増殖・封入体でGT
     スライドが候補プールに0件だった → この2カテゴリのみ全タイル対象
     (`FULL_TILE_CATEGORIES`)に変更。

**結論**: nprobe(クラスタ探索の取りこぼし)・PQ量子化誤差のどちらも、
Kupffer細胞増殖・封入体の順位低迷の主因ではない。厳密計算(近似誤差ゼロ)
でもこれらのスライドは順位が低いままなので、**`nlist`を変えて索引を
再構築しても改善する見込みは薄い**と考えられる(`nlist`はPQ近似精度にのみ
影響し、厳密スコア自体には影響しないため)。原因はモデル・索引側ではなく、
**クエリ画像(NNLアトラス図版)の内容側**(所見と無関係なタイルの混入など)
にある可能性が高い。次に着手すべきは上の「今後やること」にある、
手動ROIクロップでの性能上限測定。

検証用スクリプトは`experiments/`ではなく`scripts/`直下に置いている
(索引を作らない一回性の診断のため、既存の`validate_against_ground_truth.py`
と同じカテゴリ):

| スクリプト | 内容 | 出力 |
|---|---|---|
| `scripts/adhoc_validate_against_ground_truth.sh` | GTベースライン(`nprobe=64`) | `outputs/gt_validation_results.csv` |
| `scripts/nprobe_sweep.py` / `scripts/adhoc_nprobe_sweep.sh` | nprobeスイープ | `outputs/gt_validation_nprobe_sweep.csv` |
| `scripts/exact_vs_approx_diagnostic.py` / `scripts/adhoc_exact_vs_approx_diagnostic.sh` | 近似 vs 厳密スコア比較 | `outputs/gt_validation_exact_vs_approx.csv` |

---

以下は本プロジェクトが使っている実験管理システム(Slurm/PBS)自体の共通ドキュメント。

# Experiment Management System

SlurmおよびPBS (qsub/Miyabi) ベースの実験管理システムです。自動でスケジューラを判別し、実験の作成・投入・通知・再開をコマンド一つで行えます。

## 💡 基本コンセプト

本システムは、主に以下の2点によって動作します。

1. **シェル環境の拡張**: 便利な実験管理コマンド（`runx`, `cdx`, `lsx`, `cancelx` など）を自身のシェル環境に読み込み、実験の移動や実行を簡単に行えるようにします。
2. **実験テンプレートの自動適用 (`templates/` のコピー)**: `make create_exp` コマンドを使用して、事前に定義された実験テンプレート（コード、設定ファイル、ジョブスクリプト）を実験ディレクトリごとに展開し、実験の独立した開発・追跡を可能にします。

---

## セットアップ

### 1. 初期セットアップ

```bash
make setup
```

### 2. Python環境の同期

```bash
make uv_sync p=<partition>
```

> [!NOTE]
> `make uv_sync` および `make jupyter` コマンドは現在 Slurm 環境のみのサポートとなっています。PBS (qsub) 環境での Python 環境同期や Jupyter 起動については、今後のアップデートをお待ちいただくか、手動で実行してください。

### 3. シェルヘルパーの設定

ジョブ管理用コマンド（`runx`, `cdx`, `lsx`, `cancelx` など）を自身のシェル環境で使えるようにするため、`~/.bashrc` から本レポジトリの `shell/template-daily.sh` を source します。テンプレート側は関数定義だけで、Git のグローバル設定変更や `jq` 未導入時のシェル終了は行いません。

リポジトリのルートで一度だけ実行してください。すでに同じ設定があれば追記しません。

```bash
repo_root="$(git rev-parse --show-toplevel)"
loader_line="source \"${repo_root}/shell/template-daily.sh\""
grep -Fqx "${loader_line}" ~/.bashrc || printf '\n%s\n' "${loader_line}" >> ~/.bashrc
source ~/.bashrc
```

### 4. Slack通知の設定（任意）

`~/.bashrc` に以下を設定します。

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

---

## ディレクトリ構成

```
.
├── experiments/          # 実験ごとの設定・コード
│   └── 0001_20260101_baseline/
│       ├── run_slurm.sh  # リソース設定・実行コマンド
│       ├── experiment.py
│       └── config.yml
├── outputs/              # Pythonスクリプトの出力先
│   └── 0001_20260101_baseline/
├── logs/                 # Slurmログ・メタデータ
│   └── 0001_20260101_baseline/
│       └── 1234/         # job_id
│           ├── slurm.out
│           ├── run_metadata.yaml
│           ├── bootstrap_failure.log  # 初期化失敗時のみ
│           └── command.sh
├── lib/                  # 共通ライブラリコード
├── templates/            # 実験作成時のテンプレートファイル
├── scripts/              # ジョブ実行・監視用の内部スクリプト
│   ├── slurm_entry.sh
│   └── notify_slack.sh
└── tools/                # 各種管理ツールの実体スクリプト
    ├── create_exp.sh
    ├── resume_exp.sh
    ├── rename_exp.sh
    ├── mark_failed.sh
    ├── cancel_job.sh
    ├── uv_sync.sh
    ├── start_jupyter.sh
    └── first_setup.sh
```

---

## 主な機能と対応スケジューラ

**投入（`runx`）は Slurm (`sbatch`) 専用です。** PBS (qsub/Miyabi) 環境では `runx` は使えません。
一方、ジョブ実行中の状態管理（`slurm_entry.sh`）とキャンセル（`cancelx`）は Slurm/PBS
両方に対応しています（PBS環境で既に投入済みのジョブを管理する用途を想定）。

- **パーティション自動判定**: `make create_exp` 実行時に、ジョブの実行時間制限（Time limit）に
  応じて適切なパーティション（Slurm: `small-{owner}` 等）を自動選択し `run_slurm.sh` に焼き込みます。
- **依存ジョブ指定**: `run_slurm.sh` 内の `#SBATCH --dependency=afterok:<job_id>` を
  有効化することで指定します。job_idは依存先実験の `outputs/{exp}/latest_job_id.txt` を参照します。
- **アレイジョブ実行**: `run_slurm.sh` 内でarray/seqモードを有効化することで、探索パラメータに
  応じた複数ジョブの一括投入をSlurmのアレイジョブ機能（`--array`）で実行します。
- **ジョブキャンセル**: `cancelx <job_id>` で自動的に `scancel` または `qdel` を呼び出してジョブを停止します（Slurm/PBS両対応）。
- **終了理由の記録**: 初期化失敗は `bootstrap_failure.log` と `run_metadata.yaml` の `fail_reason` に、実行後の終了状態は `run_metadata.yaml` に記録します。通知失敗は本体ジョブを停止させず、標準エラーに警告として残します。
- **ログ自動管理**: ジョブ終了時に標準出力・標準エラーを `logs/{exp_name}/{job_id}/` 以下に `slurm.out` / `pbs.err` として自動回収します。
- **Slack通知機能**: ジョブの開始・終了・失敗（タイムアウトやOOMを含む）を判定し、Slackへ通知します（アレイジョブにも対応）。Webhook・`jq`・通信の異常は警告のみです。

---

## 実験のライフサイクル

### 1. 実験を作成する

```bash
make create_exp name=<exp_name>
```

`experiments/` 以下にディレクトリが作成され、`run_slurm.sh` / `experiment.py` / `config.yml` がテンプレートからコピーされます。

### 2. `run_slurm.sh` を編集する

`run_slurm.sh` は完全な `#SBATCH` スクリプトです。`make create_exp` の時点で
`--partition` と time-limit警告用の `--signal` マージンは自動計算済みなので、
必要ならリソースを直接編集します。

```bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80g
#SBATCH --time=24:00:00
# ↑ を大きく変える場合は --partition と --signal も手動で合わせて見直す

# 実行コマンド
RUN_COMMAND="python experiments/0001_20260101_baseline/experiment.py --config config.yml"
```

複数組み合わせを投入したい場合は、同じファイル内の Array run / Seq run セクションを
有効化します（詳細は `USAGE.md` 参照）。

### 3. ジョブを投入する

```bash
# 最新の実験を投入
runx

# 実験IDを指定して投入
runx 1

# dirty状態で投入
runx --allow-dirty
```

依存関係を指定したい場合（実験1が終わったら実験2を投入、等）は、実験2の
`run_slurm.sh` にある `#SBATCH --dependency=afterok:<job_id>` を有効化し、
実験1の `outputs/{exp}/latest_job_id.txt` の値を書き込んでから `runx` します。

### 4. 実験を確認する

```bash
# 実験一覧を表示
lsx

# 実験ディレクトリに移動
cdx      # 最新
cdx 1    # ID指定
```

### 5. ジョブをキャンセルする

```bash
cancelx <job_id>
cancelx <job_id> <reason>
```

### 実験ごとに独自のルール・スキルを持たせたい場合

`experiments/<id>_.../CLAUDE.md` や `experiments/<id>_.../.claude/skills/` を置くと、
その実験を触っている間だけ有効な追加ルール・専用スキルとして機能します
（Claude Codeがサブディレクトリ単位で自動的にlazy-loadする標準機能で、agent設定側の変更は不要）。
詳細は [`TEMPLATE_CONCEPT.md`](./TEMPLATE_CONCEPT.md#4-実験ごとに独自のルール・スキルを持たせるオプトイン) を参照。

---

## 実験の管理

### リネーム

```bash
make rename_exp name=<exp_id_or_name> new=<new_name>
```

### 失敗マーク

```bash
make mark_fail name=<exp_id_or_name> reason=<reason>
```

ディレクトリ名に `_FAILED_<date>_<reason>` が付与されます。

### 再開

```bash
# 最新の実験を再開
make resume_exp name=<exp_id_or_name>

# suffixを指定して再開
make resume_exp name=<exp_id_or_name> suffix=retry
```

元の実験の設定・コードがコピーされ、新しいIDで実験が作成されます。`config.yml` に再開元の情報が記録されます。

---

## Slack通知

ジョブの状態変化時にSlack通知が届きます。

| タイミング | 通知 |
|---|---|
| ジョブ開始 | 🚀 STARTED |
| 正常終了 | ✅ FINISHED |
| 失敗 | ❌ FAILED |
| タイムアウト | ❌ FAILED (TIMEOUT) |
| キャンセル・割り込み | ⚡ INTERRUPTED |

通知のオン・オフは環境変数で制御できます。

```bash
export SLACK_NOTIFY_ON_START=1
export SLACK_NOTIFY_ON_FINISH=1
export SLACK_NOTIFY_ON_FAIL=1
```

---

## ジョブログ

ジョブ終了後、`logs/{exp_name}/{job_id}/` に以下が保存されます。

| ファイル | 内容 |
|---|---|
| `slurm.out` | ジョブの標準出力（Slurm時は標準エラーも含む） |
| `pbs.err` | PBS (qsub) 時の標準エラー出力 |
| `run_metadata.yaml` | ジョブのメタデータ・最終ステータス |
| `command.sh` | 実行されたコマンド |

`run_metadata.yaml` のステータス一覧：

```
RUNNING / COMPLETED / FAILED / TIMEOUT / CANCELLED / OUT_OF_MEMORY / NODE_FAIL
```

---

## Jupyter

```bash
make jupyter p=<partition> mem=<memory>

# 例
make jupyter p=small-david01 mem=32g
```

---

## 更新履歴 (Change Log)

- **2026-06-19 (qsub対応)**: PBS (qsub/Miyabi) 環境への対応を追加。スケジューラ自動検知、PBS用ヘッダー・パーティション選択・ログ回収・ジョブキャンセル・依存指定の追加。
- **2026-06-16 (Slackアレイ通知)**: Slackへのアレイジョブ通知機能、およびリモート環境からのデータ収集・送信スクリプトを追加。
- **2026-06-13 (メタデータ分離)**: 設定ファイル (`config.yml`) と実行メタデータの分離、および resume/create ツールのアップデート。
- **2026-06-13 (出力ディレクトリ)**: 出力ディレクトリ構造の整理とベース設定の追加。
