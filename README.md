# patch-vector-search

UNIパッチ埋め込みに対するクラスタベースのベクトル検索
(FAISS OPQ+IVF+PQ)+WSI逆引き。任意の参照画像(既存コーパス外でもOK)を渡すと、
類似する組織パッチと、それを多く含むWSIを検索できる。  
**最終的な目標は、INHANDやNTPの非腫瘍性病変アトラスのような、所見の代表的なパッチを検索クエリとして使い、TGGATEの大規模なWSIコーパスから同じ所見を引き出してデータベース化すること。**

## 現状(2026-08-20時点)

- **動くもの**: 任意の画像(1枚〜複数枚)を渡すと、類似パッチ検索とWSI逆引きができる。
  実解像度でのヒットパッチ表示・クエリタイルごとの近似スコアヒートマップ表示も追加済み
  (下記「検索結果の可視化改善とタイル選択バイアスの発見」参照)。
- **定量評価**: NTP非腫瘍性病変アトラス由来のクエリ画像(下記「評価に使ったデータと
  その限界」参照)で7カテゴリのground truth比較を実施済み。7カテゴリとも正解スライドを
  候補プール内に発見できているが(found=n_gt)、best_rank(最上位正解スライドの順位)は
  9〜332位とカテゴリによって大きな差がある。
- **IVFクラスタリング仮説・手動ROIクロップ・染色正規化(クエリ側・コーパス側とも)は
  いずれも検証済みで、best_rankの低迷を解消する路線ではないと判断し区切った**。
  現時点で最有力の仮説は、2026-08-20に判明した**「近似スコアという選抜基準自体が
  コーパス中の出現頻度に左右され、珍しい所見ほど不利になる」というタイル選択
  バイアス**(下記参照)。未解決の課題は「今後やること」参照。

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
- [x]  ~~クエリ画像を人手でROIクロップした場合の性能上限を測る~~
      → 2026-08-20、着手したが非専門家の目視判断の限界にすぐ突き当たり区切った。
      詳細は下記「手動ROIクロップの検証状況」参照
- [x]  ~~検索結果の可視化を改善する~~(実解像度パッチギャラリー・タイルスコア
      ヒートマップを実装、`experiments/0007`〜`0009`)
      → 2026-08-20、この過程で新しい仮説(タイル選択バイアス)が見つかった。
      詳細は下記「検索結果の可視化改善とタイル選択バイアスの発見」参照
- [ ]  **タイル選択バイアス仮説の直接検証**(壊死巣内部だけを手動クロップした
      224x224画像をクエリに使い、タイル選択問題を回避した状態で検索する。
      準備済み・計算ノード混雑のため未実行。下記参照)
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
                                サムネイルJPEG上へのヒット位置
                                プロットに加え、実解像度パッチ
                                ギャラリー・クエリタイルの近似
                                スコアヒートマップも追加(2026-08-20)
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
| `stain_reference=`(Macenko染色正規化) | ❌ 非推奨 | 同上。ケースによっては大きく悪化する。自作`lib.stain_normalize`実装(自己一致性0.5〜0.84)での検証結果だったため、2026-08-20に検証済みの`lib.torchstain_normalize`(自己一致性0.96〜0.996)へ差し替えて再検証したが、結論は変わらず(7カテゴリ中4カテゴリで明確に悪化、うち1カテゴリは大幅悪化)。詳細は下記「クエリ側染色正規化の再検証」参照 |
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
- `lib/visualize.py` — サムネイル上へのヒット位置プロット、実解像度パッチギャラリー
  (`plot_hit_patch_gallery`)、クエリタイルの近似スコアヒートマップ(`plot_query_tile_scores`、
  いずれも2026-08-20追加)
- `lib/raw_patch.py` — openslide経由での実解像度パッチクロップ。`data/moo_collected_tggate_wsi/
  raw_wsi/`(全1000スライド分の生WSI)に対して既に使える状態(2026-08-20判明、モジュール
  docstringの旧記述は古い)
- `experiments/0001_..._build_patch_manifest` → `0002_..._build_faiss_index` → `0003_..._query_demo`
  (この順で依存。**現行の推奨インデックス**、`uni_v1`・1024次元)
- `experiments/0004_..._build_patch_manifest_v2` → `0005_..._build_faiss_index_v2`
  (`uni_v2`・1536次元・pq_m=64版)→ `0006_..._build_faiss_index_v2_pqm96`
  (同じ元データ、pq_m=96版。詳細は下記「uni_v2コーパスの調査状況」参照)
- `experiments/0007_..._patch_gallery` → `0008_..._patch_gallery_hit_threshold` →
  `0009_..._tile_score_heatmap`(0003からの派生。ヒットパッチギャラリー追加→
  スライド間で表示基準が不揃いだった問題の修正→クエリタイルの近似スコア
  ヒートマップ追加、の順。詳細は下記「検索結果の可視化改善とタイル選択
  バイアスの発見」参照)
- `scripts/validate_against_ground_truth.py` — 新しいパイプライン案をGTで検証するツール(要・使用)
- `scripts/select_average_patch.py` — 染色正規化用の「典型的な」基準パッチを選ぶ(セットアップ用、実行済み)
- `scripts/manual_roi_crop_diagnostic.py` — 手動ROIクロップの性能上限測定(2026-08-20、詳細は下記参照)
- `scripts/torchstain_query_normalization_diagnostic.py` — torchstainベースのクエリ側染色
  正規化をv1インデックスで再検証(2026-08-20、詳細は下記参照)
- `scripts/crop_necrosis_query_tile.py` — 壊死巣内部だけを224x224で手動クロップし、タイル
  選択バイアスを回避したクエリを作る(2026-08-20、詳細は下記参照)

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

## 手動ROIクロップの検証状況(2026-08-20、区切り)

上記IVFクラスタリング仮説の否定を受け、「次に着手すべき」とされていた手動ROIクロップの
性能上限測定に着手した。GT対応7カテゴリ・計24枚のアトラス画像を目視で確認し、所見が
明確に局在するNecrosis・Extramedullary hematopoiesisの2カテゴリについて、手動でROIを
クロップして`baseline_v1`(全画像タイル分割)と比較した(`scripts/manual_roi_crop_diagnostic.py`)。

**結論: 非専門家による目視判断の限界にすぐ突き当たり、統計的に意味のある検証には
至らなかった。** 経緯:

1. 最初は6枚(Necrosis 3枚、Hematopoiesis 3枚)をクロップ対象としたが、根拠を1枚ずつ
   検証し直したところ、「造血細胞塊」と「正常な門脈域のリンパ球カフ」、「壊死組織」と
   「出血・うっ血」(哺乳類の赤血球は無核なので、核が無く均一に赤いという特徴だけでは
   両者を区別できない)を、非専門家が見た目だけで確実に区別できない画像が複数見つかり、
   最終的に確実な根拠(矢印注釈による直接確認、または対立仮説を積極的に排除できる
   具体的な組織像の特徴)がある**2枚のみ**(Necrosis 1枚、Hematopoiesis 1枚。矢印確認済みは
   このうち1枚のみ)に絞られた。
2. この2枚での結果: Necrosis best_rank 10→8、mean_rank 298.5→286.8(改善)。
   Hematopoiesis best_rank 25→24(微改善)、mean_rank 76.3→78.7(悪化)。方向性は
   クロップ有利で一貫しているが、各カテゴリ4枚中1枚しかクロップしていないため希釈が
   大きく(残り3枚は無クロップのまま)、統計的に「クロップが効く」と結論できるだけの
   サンプルサイズではない。
3. 実パッチ(TG-GATEs自身)をクエリに使い、由来スライドを候補から除外した上で同様の
   検証を行う案も検討したが、(a) 本来の目標(コーパス外の代表画像→TG-GATEs検索)とは
   異なるユースケースになってしまう、(b) TG-GATEsの生WSIには矢印等のアノテーションが
   一切無く(`data/processed_csv/single_finding_liver.csv`にあるのはスライド単位の
   `FINDING_TYPE`と大まかな`TOPOGRAPHY_TYPE`のみ)、非専門家の判断への依存という
   同じ壁に別の形でぶつかるだけ、という2点から見送った。

**この路線は一旦区切る。** 再開する場合は、専門家(病理の知識がある人)によるROI確認を
経てサンプル数を増やすことが前提になる。

## クエリ側染色正規化の再検証(torchstain、2026-08-20)

染色正規化(`stain_reference=`)がGT比較で非推奨とされていた根拠
(`scripts/compare_query_normalization.py`)を確認したところ、内部で自作の
`lib.stain_normalize.MacenkoNormalizer`(自己一致性0.5〜0.84、上記uni_v2調査で
判明した精度問題と同じもの)を使っていたことが判明した。つまり「クエリ側染色正規化は
効果が無い」という既存の結論は、実装精度の問題を排除しないまま出されたものだった。

`lib.torchstain_normalize`(自己一致性0.96〜0.996、既にuni_v2コーパス向けには使用実績
あり)に差し替えて、v1インデックス(コーパス側は無変更・再構築不要)に対して同じ7カテゴリで
再検証した(`scripts/torchstain_query_normalization_diagnostic.py`、基準パッチは
`compare_query_normalization.py`と同じ`63958_x38976_y7616.png`を
`data/moo_collected_tggate_wsi/raw_wsi/63958.svs`から再現して使用)。

**結果: 実装精度を排除しても、依然として非推奨という結論を覆せなかった。** 7カテゴリ中
1カテゴリ(Hypertrophy: best_rank 49→17)のみ明確に改善、4カテゴリは明確に悪化
(Kupffer細胞増殖: 202→533など大幅悪化含む)、2カテゴリは指標(best_rank/mean_rank/found)
により結果が割れた(`outputs/gt_validation_torchstain_query_normalization.csv`)。

コーパス側染色正規化(uni_v2、上記「uni_v2コーパスの調査状況」参照。ただしエンコーダ・
パッチサイズも同時に変わっており交絡あり)と合わせ、**染色正規化(クエリ側・コーパス側の
いずれの形でも)は、この検索タスクの改善に寄与しないと判断し、この路線は区切る。**
大幅な悪化(特にKupffer)は、正規化が単なるノイズ除去ではなく、色素沈着・出血など
診断的意味を持つ色情報まで一緒に消してしまっている可能性を示唆している。

## 検索結果の可視化改善とタイル選択バイアスの発見(2026-08-20)

上記2つの路線がいずれも区切りとなったことを受け、「今後やること」にあった検索結果の
可視化改善に着手した。副産物として、best_rankが悪いカテゴリの主因についての、より
具体的な仮説にたどり着いた。

### 可視化の実装(experiments/0007〜0009)

- `lib/raw_patch.py`(既存、実解像度WSIパッチクロップ用)が、実は
  `data/moo_collected_tggate_wsi/raw_wsi/`(全1000スライド分の生WSI、`lib/visualize.py`の
  旧docstringが「無い」としていたもの)に対して既に使える状態だったと判明。これを使い、
  `lib.visualize.plot_hit_patch_gallery`(ヒットパッチを実解像度で並べて表示、新規)を
  追加した(`experiments/0007`)。
- ギャラリー表示で、スライドによって表示パッチ数の基準が一貫していない問題
  (`similar_patches`のグローバル上位k=20件に入るスライドは数件しか出ず、入らない
  スライドはほぼ無制限のフォールバックプールから大量に出る — 表示パッチ数が多い/少ない
  が所見の強さを全く反映していなかった)を発見・修正。全プロット対象スライドに対して
  `k=rerank_pool`(200、既に厳密再計算済みの候補数をそのまま使うだけで新たな
  恣意的な数値は導入していない)で統一した(`experiments/0008`)。
- `lib.visualize.plot_query_tile_scores`(新規)で、クエリ画像の全タイルにFAISS近似
  top-1スコアをヒートマップ表示し、`max_tiles_reranked`で実際に厳密再ランキングされる
  タイルを赤枠で明示できるようにした(`experiments/0009`)。

### 発見: 所見タイルより「ありふれた正常組織」タイルの方が近似スコアで有利

NNLアトラスのNecrosis画像(`imgi_11`、壊死巣が左35〜40%、残りは正常組織)を上記
ヒートマップで確認したところ、**壊死巣のタイルは1枚も上位12枚(`max_tiles_reranked`)に
入っておらず、赤枠は全て正常組織側のタイルに付いていた**。実際に検索した
`top_slides.csv`にも、コーパス内のGT壊死スライド4枚(41484, 58720, 58787, 59195)が
一件も入っていなかった。Extramedullary hematopoiesis画像(`imgi_6`、矢印確認済みの
造血細胞塊あり)でも同様に、矢印が指す細胞塊のタイルより、何の変哲もない正常組織タイルの
方が近似スコアが高かった。

考えられる理由: コーパス(998〜1000スライド)には「ごく普通の正常肝組織」のパッチが
圧倒的多数あるため、正常組織を写したタイルは近似top-1スコアが高く出やすい(似た正常組織
パッチがコーパス中に大量にあるので、既定の`nprobe=32`で探索する32クラスタのどこかに
必ず極めて近い候補が見つかる)。一方、壊死巣のような比較的珍しいパターンは、本当に強い
一致があっても、母数の少なさゆえに近似スコアで見劣りし、`max_tiles_reranked=12`の枠を
「量で勝る正常組織タイル」に奪われてしまう。これは`search_similar_patches_multi`側
(上位12枚のみ厳密再計算)だけでなく、`search_top_slides_multi`側(全タイル対象・
近似スコアのみで集計)にも同様に影響しうる。

**IVFクラスタリング仮説の検証状況で「原因はクエリ画像の内容側にある」と結論していたが、
今回さらに具体化できた**: 単に「所見と無関係なタイルが混入している」だけでなく、
**「近似スコアという選抜基準自体が、コーパス中の出現頻度に左右されるため、珍しい所見
ほど不利になる」**という、より構造的な問題である可能性が高い。

### 次の一手(未着手)

この仮説を直接検証するため、`imgi_11`の壊死巣内部だけを224x224で手動クロップし
(`scripts/crop_necrosis_query_tile.py`、`data/query/necrosis_imgi11_crop_x150_y400.jpg`)、
タイル選択の問題を完全に回避した状態で検索する準備をした(`experiments/0009`の
`run_slurm.sh`を単発実行用に変更済み)。計算ノード混雑のため未実行。結果次第で:

- 期待通りGT壊死スライドが上位に来れば、タイル選択バイアス仮説をさらに裏付ける。
  次はこのバイアスへの対策(例: 近似スコアの頻度補正、選抜基準の見直しなど)を検討する。
- 変化が無ければ、別の要因(モデルの表現力そのものなど)を疑う必要がある。

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
