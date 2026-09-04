# patch-vector-search

UNIパッチ埋め込みに対するクラスタベースのベクトル検索
(FAISS OPQ+IVF+PQ)+WSI逆引き。任意の参照画像(既存コーパス外でもOK)を渡すと、
類似する組織パッチと、それを多く含むWSIを検索できる。  
**最終的な目標は、INHANDやNTPの非腫瘍性病変アトラスのような、所見の代表的なパッチを検索クエリとして使い、TGGATEの大規模なWSIコーパスから同じ所見を引き出してデータベース化すること。**

## 現状(2026-09-04時点)

- **動くもの**: 任意の画像(1枚〜複数枚)を渡すと、類似パッチ検索とWSI逆引きができる。
  実解像度でのヒットパッチ表示・クエリタイルごとの近似スコアヒートマップ表示も追加済み
  (下記「検索結果の可視化改善とタイル選択バイアスの発見」参照)。
- **モデル/コーパスの検索天井**: 2026-09-04の自己検索診断
  (`scripts/self_retrieval_diagnostic.py`、下記「自己検索診断」参照)で、
  **uni_v1 + 現行コーパスは大半のcommonな肝所見を検索に足るレベルで表現できている**
  ことを確認(16所見中10所見が chance を大きく上回りhit@50 ≥ 0.75)。
  **例外は融合壊死(`Necrosis`)と髄外造血** — この2所見だけは表現の限界。
  中核仮説(代表パッチ→類似パッチ検索)は「機能する所見」については成立している。
- **区切りをつけた路線(いずれもbest_rankの低迷を解消しない)**: IVFクラスタリング仮説、
  手動ROIクロップ、染色正規化(クエリ側・コーパス側・per-tile交絡なしまで、4回検証)、
  倍率自動補正。
- **アトラスGT比較の位置づけ**: アトラス図版をクエリにする
  `scripts/validate_against_ground_truth.py` のbest_rankが悪いカテゴリの主因は、
  モデルの表現力ではなく **アトラス→TG-GATEsのドメインギャップ + 図版が所見と無関係な
  組織だらけ**(タイル選択バイアス、下記参照)であることが自己検索診断で切り分けられた。
  今後の定量評価は自己検索診断のほうが交絡が少なく所見カバレッジも広い(7 → 16所見)。
- **未解決**: アトラス図版のドメインギャップの詰め方、融合壊死・髄外造血の表現
  (uni v2等)。「今後やること」参照。

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
  高いことも一因である可能性が高い。**2026-09-04の自己検索診断で、実際にKupffer等の
  低迷はモデルの表現力ではなくこのドメインギャップ/ROI問題が主因と切り分けられた
  (下記「自己検索診断」参照)。交絡の少ない定量評価が必要ならそちらを使う。**
- Ground truthとしている所見自体(`data/processed_csv/single_finding_liver.csv`)は1000スライド
  コーパスの一部にしか対応しない。NNLの26カテゴリのうち、1000スライドコーパス内に
  確定ラベル付きスライドが1件でもあるのは7カテゴリのみ(Hypertrophy/Necrosis/
  Increased mitosis/Glycogen/Hematopoiesis/Kupffer細胞増殖/封入体)。残り19カテゴリ
  (Fatty Change, Focus, Inflammationなど)は検証不能。
  Degeneration, fattyとFatty Changeなど、対応付けを行うべき所見もあるが現状は未対応。
  その為、Ground truth比較は基本あてにしなくていい。あくまで「7カテゴリのうち、正解スライドを候補プール内に発見できるか」という観点での評価に留める。

## 今後やること

- [x]  ~~染色正規化(クエリ側・コーパス側)がこの検索タスクに寄与するか~~
      → 2026-09-04、`uni_v1`・ジオメトリ完全一致・両側per-tile Macenko・自己一致性
      検証済みの交絡なし再検証(`experiments/0010`→`0012`→GT比較→`experiments/0013`
      の可視化)まで行い、確定的に棚上げ。詳細は下記「クエリ側染色正規化の再検証」
      「可視化の再検証」参照
- [ ]  **uni v2など他モデルでの埋め込みを検討する** — ただし2026-09-04の自己検索診断で
      「uni_v1は大半のcommonな肝所見を検索に足るレベルで表現できている、例外は融合壊死と
      髄外造血のみ」と判明したため、**全所見のためではなくこの2所見のための調査**として
      位置づける。下記「自己検索診断」参照
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
- [x]  ~~タイル選択バイアス仮説の直接検証~~(壊死巣内部だけを手動クロップした
      224x224画像をクエリに使い、タイル選択問題を回避した状態で検索する)
      → 2026-09-04、`experiments/0013`で実施。壊死は改善せず(コーパスに凝固壊死に
      近いパッチが無く`sim=0.33`止まり)。一方、per-tile Macenkoが好塩基性局所所見
      (髄外造血)のタイル識別を明確に改善することが判明。下記「可視化の再検証」参照
- [x]  ~~抽出できる所見・できない所見を精査する(第一歩: 自己検索でモデル天井を測る)~~
      → 2026-09-04、`scripts/self_retrieval_diagnostic.py`。16所見中10所見は検索天井が
      十分、融合壊死と髄外造血が例外、と判明。詳細は下記「自己検索診断」参照
- [~]  **「機能する所見」でデータベース化のワークフローを一度通す**
      → 2026-09-04、`experiments/0014`(アトラス seed)/`0015`(TG-GATEs スライド seed)に着手。
      アトラス seed はドメインギャップで不成立、TG-GATEs seed は glycogen で機能・mitosis は
      所見の性質上不成立(所見の2クラス分けが見えた)。継続中。詳細は下記「成果物の試作」参照
- [ ]  **アトラス→TG-GATEsのドメインギャップを詰める**(0014 でこれが #1 ブロッカーと確定。
      アトラス図版からの組織タイル抽出の改善、複数図版のマルチクエリ化など)\
など、いろいろやる。

## 大まかな構成

```
[① manifest構築]                [② FAISSインデックス構築]
lib/manifest.py           →    lib/faiss_index.py
1000個のh5を走査し               OPQ+IVF+PQ(コサイン類似度=
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
| WSI逆引きの`n_hits_ratio`ソート | ✅ 推奨(既定) | 単純な`n_hits`ソートより7カテゴリ中5カテゴリで改善、追加コストなし。2026-09-04の自己検索診断でも`max_similarity`ソートより全所見でほぼ優位(下記「自己検索診断」発見3) |
| `embed_image_tiles_auto_scale`(倍率自動補正、`lib/mpp_estimation.py`) | ❌ 非推奨 | 画質は改善するが、GT比較では検索精度がほぼ悪化(7カテゴリ中0カテゴリで最良) |
| `stain_reference=`(Macenko染色正規化) | ❌ 非推奨(確定) | 同上。ケースによっては大きく悪化する。自作`lib.stain_normalize`実装(自己一致性0.5〜0.84)→`lib.torchstain_normalize`(自己一致性0.96〜0.996)への差し替え、さらに2026-09-04の交絡なし再検証(uni_v1・ジオメトリ完全一致・コーパス側もクエリ側もper-tile Macenko・自己一致性検証済み、`experiments/0010`/`0012`)まで行ったが、GT比較で`baseline_v1`を上回れず(best_rank 7カテゴリ中4カテゴリで悪化、うちKupffer 202→766・Necrosis 10→77等は大幅悪化。改善は所見局在型の3カテゴリのみ)。染色正規化はこれで完全に棚上げ。詳細は下記「クエリ側染色正規化の再検証」参照 |
| `embed_image(..., resize_mode="centercrop")`(単一クロップ) | ⚠️ 場合による | 結果の分散が大きく、GTを完全に見失うこともある |
| `encoder_name="uni_v2"`コーパス(1536次元・256px、`experiments/0004`/`0005`/`0006`) | ⚠️ v1よりやや劣るが僅差(公平な比較後) | クエリ側は`lib.torchstain_normalize.normalize_to_reference`で`data/baseline/63958_x38976_y7616.png`に正規化してから使うこと(`lib.stain_normalize`の自作Macenkoでは不十分——自己一致性0.5〜0.84止まり、torchstainなら0.96〜0.996)。正規化後の公平なGT比較でも7カテゴリ中6カテゴリでv1が優位だが、差は大幅縮小(例: Hypertrophy 94→27)。PQ量子化を細かくする(pq_m 64→96)ことも試したが改善なし。詳細は下記「uni_v2コーパスの調査状況」参照。既定は引き続き`uni_v1`(`experiments/0001`/`0002`) |

新しい前処理・パラメータのアイデアを試すときは、必ず
`scripts/validate_against_ground_truth.py`で既存パイプラインとground truth比較してから
採用すること — 目視で綺麗に見えることは検索精度が上がることを意味しない(このプロジェクトで
2回実際に踏んだ落とし穴)。

## ディレクトリ構成(このプロジェクト固有)

- `lib/manifest.py` — 1000スライドを走査し、manifest/slide_meta/学習サンプルを構築
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
- `experiments/0010_..._build_patch_manifest_macenko_v1` →
  `0012_..._build_faiss_index_macenko_v1`(交絡なし染色正規化再検証用の
  `uni_v1`・Macenkoコーパス。0001/0002とジオメトリ・ハイパラ完全一致。詳細は下記
  「クエリ側染色正規化の再検証」参照。既定インデックスではない)
- `experiments/0011_..._query_demo_exhaustive_tile_rerank`(0009からの派生。
  全クエリタイルを厳密再ランキングする単発検証)
- `experiments/0013_..._query_demo_macenko_per_tile`(0009と同じ可視化を
  Macenkoコーパス+per-tile正規化クエリで実行。詳細は下記「可視化の再検証」参照)
- `experiments/0014_..._build_finding_patch_set`(所見ごとの代表パッチ集を組み立てる。
  seed = NNLアトラス図版。詳細は下記「成果物の試作」参照)
- `experiments/0015_..._build_finding_patch_set_seed_slide`(同上、seed = その所見の
  TG-GATEsスライド。curation は `lib/patch_set.py`。詳細は下記「成果物の試作」参照)
- `lib/patch_set.py` — 索引ヒット → 代表パッチ集の productization レイヤー
  (類似度しきい値・スライド内NMS・スライド上限・ラウンドロビン・空白除外・切り出し・manifest)
- `scripts/validate_against_ground_truth.py` — 新しいパイプライン案をNNLアトラス由来GTで検証するツール
- `scripts/self_retrieval_diagnostic.py` — コーパス内leave-one-out自己検索でモデル/コーパスの
  検索天井を測る(2026-09-04、アトラス図版を使わない・GPU不要。詳細は下記「自己検索診断」参照)
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

> 注記(2026-08-31): wsi_preprocessのgit履歴を確認したところ、このuni_v2コーパスを
> 生成した`src/stain_norm.py`(commit 244fe66)は`torchstain`の**torchバックエンド**を使い、
> かつMacenko正規化が失敗したパッチ(縮退パッチ)を**例外を握りつぶして生画素のまま**
> 埋め込んでいた(ログ・カウントなし、TRIDENT Step3側も成否を認識しない)。
> つまりこのuni_v2コーパスは「正規化済み + 黙って素通しされた生パッチ」の混合物で、
> 割合の記録は無い。下記の結論(v1優位、PQ/IVF調査)はエンコーダ・パッチサイズの
> 交絡に加え、この半汚染コーパス上で測られていた。結論を覆すほどの影響ではないが
> 記録として。新しい`v1_macenko`コーパス(experiments/0010)はこの問題を避けている
> (numpyバックエンド、失敗率をサンプル監査で把握)。2026-09-04時点の
> サンプル監査(32/1000スライド)では失敗率0.81%・失敗パッチは全て実質白背景
> (tissue 0.0%)だったため、フル監査とmanifest行除外は行わず素通しのまま進める
> 方針(uni_v2の「割合不明の半汚染」とは異なり素通し分が背景限定と確認済み)。
> 詳細は`experiments/0010_..._build_patch_manifest_macenko_v1/config.yml`のコメント参照。

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

**再訪(2026-08-31、チーム方針):** 上記の否定的結果はいずれも交絡付きだった —
uni_v2はエンコーダ・パッチサイズも同時変化かつ半汚染コーパス、クエリ側torchstain検証は
「正規化クエリ vs 未正規化コーパス」というパイプライン不一致。そこで「uni_v1・ジオメトリ
完全一致・コーパス側もクエリ側もMacenko・自己一致性検証済み」の交絡なし再検証を実施する
(`experiments/0010` → build_faiss_index → `scripts/validate_against_ground_truth.py`の
`v1_macenko`パイプライン)。コーパス側の再エンベディングはwsi_preprocessパイプラインで実施。
この再検証でも`baseline_v1`を上回らなければ、uni_v2同様に確定的に棚上げする。

**結論(2026-09-04、確定的に棚上げ):** 交絡なし再検証を完走した
(`experiments/0010`=manifest → `experiments/0012`=FAISS index、ともに0001/0002と
ジオメトリ・ハイパラ完全一致。コーパスは1000スライド・18,368,335パッチでbaseline_v1と
2パッチ差。wsi_preprocess側の自己一致性テストは`cos_self`全パッチ1.00000でPASS。
染色正規化失敗パッチは監査で背景限定・約0.8%と判明したため除外せず素通し)。

最初のGT比較(job 9643)ではクエリ側が図版まるごとをMacenko正規化してからタイル
分割しており、コーパス側(224pxパッチごとに正規化)と粒度が不一致だった。そこで
`_embed_v1_macenko_normalized`をper-tile(タイル分割→各224pxタイルを個別に正規化→
エンコード)に修正し、`lib.query_embedding.embed_image_tiles`に`tile_transform`フックを
追加して再検証した(job 9644、`outputs/gt_validations/gt_validation_v1_macenko_per_tile.csv`)。
per-tile化で所見が局在するカテゴリ(Hypertrophy best_rank 54→14、Increased mitosis 3→1、
Inclusion body 349→97)は大きく改善したが、**`baseline_v1`との比較では依然として
best_rank 7カテゴリ中4カテゴリで悪化**(Proliferation, Kupffer cell 202→766、
Single cell necrosis 10→77、Hematopoiesis 25→84、Deposit, glycogen 7→14)、
改善は3カテゴリ(Hypertrophy 49→14、Increased mitosis 7→1、Inclusion body 480→97)。
`mean_rank`は6/7カテゴリで悪化。

**クエリ/コーパスのパイプラインを完全に一致させ交絡を全て排除しても、染色正規化は
この検索タスクを全体としては改善しないことが確定した。** 悪化するのは色情報に依存する
カテゴリ(Kupffer=色素沈着、Necrosis/Hematopoiesis=出血・細胞塊の色)に集中しており、
2026-08-20の所見(正規化が診断的意味を持つ色情報まで消す)を交絡なしで裏付ける。
染色正規化(クエリ側・コーパス側のいずれの形でも)はこれで完全に棚上げとする。
`data/trident_processed_uni_v1_macenko`コーパスと`experiments/0010`/`0012`は参照用に残置。

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

考えられる理由: コーパス(1000スライド)には「ごく普通の正常肝組織」のパッチが
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

### 次の一手(2026-09-04に実施、下記「可視化の再検証」参照)

この仮説を直接検証するため、`imgi_11`の壊死巣内部だけを224x224で手動クロップし
(`scripts/crop_necrosis_query_tile.py`、`data/query/necrosis_imgi11_crop_x150_y400.jpg`)、
タイル選択の問題を完全に回避した状態で検索する準備をした。2026-09-04に
`experiments/0013`(下記)で、Macenkoコーパス+per-tile正規化クエリと合わせて実行した。
結果は「変化が無い」側で、壊死は別の要因(モデル・コーパスの表現力そのもの)を
疑うべきという結論に傾いた。

## 可視化の再検証: Macenkoコーパス + per-tile正規化クエリ(experiments/0013、2026-09-04)

上記「クエリ側染色正規化の再検証」でMacenkoがGT比較(best_rankのみ)で`baseline_v1`を
上回れなかったが、**GT比較自体が弱い指標**(NNLアトラス図版は所見と無関係な組織が大半、
GTがあるのは7カテゴリだけ、ラベルはスライド単位)であるため、`experiments/0009`と
同じ可視化(実解像度パッチギャラリー・クエリタイル近似スコアヒートマップ)を
**Macenkoコーパス(`experiments/0012`)+ per-tile Macenko正規化クエリ**で回して、
返ってくるパッチと近似スコアの分布を目視した(`experiments/0013`、クエリ8枚:
0009の5枚 + 壊死図版 + 壊死内部クロップ + Kupffer図版。探索パラメータは0009と完全一致で
出力を横並び比較可能)。`lib.query_embedding.embed_image_tiles`と
`lib.visualize.plot_query_tile_scores`に`tile_transform`フックを追加してある。

### 発見1: per-tile Macenkoは「局所の好塩基性所見」のタイル識別を明確に改善する

髄外造血のヒートマップが決定的。アトラス図版の造血細胞塊を指す矢印2本のタイルが、
`baseline_v1`(0009)では近似スコア中位(~0.27、周囲の正常組織と大差なし)だったのが、
Macenko per-tile(0013)では**図版内で明確に最高スコア(~0.45、他は0.28〜0.35)**になった。
GT比較の数値(髄外造血 best_rank 25→84「悪化」)はこのタイル識別の改善を完全に隠していた。

### 発見2: ただしタイル識別の改善はスライド検索に伝播しない

髄外造血0013では、診断タイルが選ばれても`top_slides`(n_hits_ratioソート)上位と
`similar_patches`(厳密再ランキング上位)のスライドが一致せず、パッチギャラリーが1枚も
生成されなかった(top-3スライドのhit_poolが空)。GTスライドも依然~84位。ボトルネックが
下流(コーパスに近いパッチが無い、またはスライド集計の乖離)に移っただけ。

### 発見3: 壊死・Kupfferは前処理では動かせない層の問題

壊死巣タイルはMacenko下でも依然LOW(0.33〜0.42、正常組織は0.55〜0.70)。GT壊死スライド
(41484, 58720, 58787, 59195)はtop_slidesに一件も無し(0009と同じ)。壊死巣内部の
手動クロップ(初実行)は正常寄りの肝細胞を`sim=0.33〜0.36`という極端に低い値で返す
= コーパスに凝固壊死に近いパッチが存在しない。Kupfferもヒートマップのスコアが全域で
低く(max 0.44)、類洞のKupffer密な領域も勝たない。これらの失敗は前処理(タイリング・
スケール・染色正規化)より手前 — UNI埋め込み + このコーパスが凝固壊死・Kupffer増生を
distinctiveに表現していない。

### 結論

- **Macenkoは既定では棚上げのままで妥当**(4回目の確認)。
- ただし「染色正規化はどのタイルが選ばれるかに実際に効く(好塩基性局所所見で顕著)」は
  記録すべき発見 — タイル選択バイアスの路線を再訪する場合は関係する。
- **壊死の低迷は『コーパス/モデルにその所見が distinctive に無い』**という、前処理では
  動かせない層の問題(下記「自己検索診断」でも裏付けられた)。次の productive な動きは
  「今後やること」の「抽出できる所見・できない所見を精査する」「uni v2など他モデル」。

出力: `outputs/0013_20260904_query_demo_macenko_per_tile/`(gitignore対象、NFS上に残置)。

## 自己検索診断: モデル/コーパスの検索天井(scripts/self_retrieval_diagnostic.py、2026-09-04)

NNLアトラス図版をクエリにするGT比較は、「UNI埋め込みが同一所見のパッチを近くに置けて
いるか」と「アトラス→TG-GATEsのドメインギャップ・図版が所見と無関係な組織だらけ・
倍率不明・ROI不明」を混同している。0013で難しいカテゴリ(壊死・Kupffer)が前処理の
手前で失敗すると分かったため、**アトラス図版を一切使わず、コーパス内 leave-one-out
自己検索でモデル/コーパスの天井を測った**。`single_finding_liver.csv` の確定単一所見
スライド(コーパス内126枚、≥2枚ある16所見)について、各スライドを順にクエリにし
(自身のパッチ特徴量をh5からそのまま読む — UNIエンコーダは呼ばない、GPU不要)、
索引検索してクエリスライドを除外し、同一所見の他スライドが何位に来るかを記録
(`outputs/gt_validations/self_retrieval_diagnostic.csv`、job 9664、約16分)。

### 発見1: 大半の所見は検索に足る天井を持つ

16所見中10所見が、chance(ランダム期待順位)を大きく上回るbest_rankかつ hit@50 ≥ 0.75:
Microgranuloma(best_rank中央値2/chance 62)、Change eosinophilic(3/91)、Swelling(4/143)、
Deposit glycogen(4.5/167)、Ground glass appearance(3.5/250)、Cellular infiltration(8/143)
など。**アトラスGT比較で悪かった所見(Kupffer best 202、Inclusion body 480)は、モデルが
表現できないからではない** — Kupfferは自己検索でrank 1(ただしn=2・同群、下記注意)。
主因はアトラスのドメインギャップ側にある。

### 発見2: 凝固/融合壊死は本物の弱点

`Necrosis`(13枚)は自己検索でもbest_rank中央値49(chance 77)、hit@50 0.5 —
ほぼchance並み。0013の`sim=0.33`(コーパスに近いパッチが無い)と整合し、**融合壊死は
UNI空間でクラスタを形成しない**。一方 `Single cell necrosis`(best_rank 16.5、hit@50 0.75)
はまだマシ。**壊死サブタイプで挙動が違い、`validate_against_ground_truth.py` の
`CATEGORIES` がアトラス "Necrosis" を `"Single cell necrosis"` にマッピングしていたのは
apples-to-orangesだった。** `Hematopoiesis, extramedullary`(best_rank 125、hit@50 0.33、
n=3)も実質失敗。

### 発見3: n_hits_ratioソートはmax_similarityソートより優れている

ほぼ全所見で `nhr_best_rank ≤ sim_best_rank`(Single cell necrosis 16.5 vs 125、
Deposit glycogen 4.5 vs 26.5 等)。0013発見2の「n_hits_ratioとmax_similarityが乖離する」は
本物だが、**現行の既定(n_hits_ratio)が良い側**。集計指標をmax_similarityに変える案は
この診断で否定された。

### 発見4: バッチ効果の注意

`Change, eosinophilic` は同群(同一 EXP_ID+GROUP_ID = 同一化合物・用量・時点)スライドを
除外するとbest_rank 3 → 51 に悪化。「良い」数字の一部は所見類似ではなく実験群類似。
`n=2` の所見(Kupffer、Alteration cytoplasmic)は2枚が同群で、除外すると対象0 = 評価不能。

### 結論

- **UNI(uni_v1)+ 現行コーパスは、大半の common な肝所見を検索に足るレベルで表現できて
  いる。** プロジェクトの中核仮説(代表パッチ→類似パッチ検索)はこれらの所見については成立。
- **例外は融合壊死と髄外造血** — これらだけは表現の限界で、uni v2 等の検討が正当化される
  (全所見のためではなく、この2所見のため)。
- 現行の検索パス(n_hits_ratioソート)は妥当。集計指標の変更は不要。
- 今後の定量評価では、アトラスGT比較よりこの自己検索診断のほうが交絡が少なく所見カバレッジも
  広い(7 → 16所見)。ただし n=2〜3 の所見とバッチ効果には注意。

## 成果物の試作: 所見ごとの代表パッチ集(experiments/0014 / 0015、2026-09-04)

自己検索でモデル天井が十分と分かったので、**中核目標である「所見ごとの代表パッチ集」を
一度 end-to-end で作ってみる**トラックに着手した(README「今後やること」トラック1)。
検索コアは既存の `PatchIndex` だが、その上に productization レイヤー
(`lib/patch_set.py`: 類似度しきい値 → スライド内空間NMS → スライドあたり上限 →
由来スライドでのラウンドロビン絞り込み → 空白パッチ除外 → 実解像度パッチのファイル保存 →
manifest → スライド別コンタクトシート)を載せる。0009 系の「検索が動いたかを見る診断」
とは違い、レビュー可能なパッチデータセットそのものを出力する。

### experiments/0014: NNL アトラス図版を seed に(2026-09-04、否定的)

`Deposit, glycogen`(アトラス図版4枚)と `Increased mitosis`(同1枚)で実行。
各150枚出力したが**代表パッチ集としては不成立**:

- glycogen: 最終150枚が**88スライドに拡散**、うち GT 陽性は 1/6 スライドのみ。
- mitosis: 150枚が150スライドに1枚ずつ(最大限に拡散)、GT 陽性 4/10 スライド。
- 目視: 返ってくるパッチはアトラス図版に「見た目は似ている」(淡明な肝細胞質など)が、
  その見た目は正常肝にもありふれている。**「アトラス図版に似ている」≠「その FINDING_TYPE」**。
- **決定的**: glycogen の GT スライド 6枚中 5枚が top-5000 候補に1パッチも入らない。
  自己検索(glycogen nhr_best=4.5)では GT スライド同士がよく retrieve するのに、
  **アトラス図版をクエリにすると GT スライドを引けない = アトラス→TG-GATEs のドメイン
  ギャップが #1 ブロッカー**だと綺麗に切り分けられた。

### experiments/0015: その所見の TG-GATEs スライドを seed に(2026-09-04、所見依存)

「自己検索で動く側」を seed にする: その所見の確定スライドを seed / hold-out に分割し、
seed スライドのパッチ特徴量(h5 から直読み、エンコーダ不使用)をクエリに、seed スライドを
結果から除外、`gt_positive` フラグは hold-out GT スライドにつけて recall の指標にする。

- **glycogen: パイプラインはほぼ機能**。最終89枚/20スライドがコヒーレント(最多寄与
  スライド 22697 の15枚と hold-out GT スライド 53267 が seed と同一 morphology =
  グリコーゲン豊富な淡明・腫大肝細胞)。**hold-out GT 3枚中 2枚を回収**(0014 の 1/6 から改善)。
  非GT寄与スライドは未ラベルのグリコーゲンスライドの可能性が高い(= 本来やるべき discovery)。
- **mitosis: 所見の性質上機能しない**。返ってくるパッチが**ほぼ空白・組織希薄**(sim 0.92+)。
  増加分裂像は 224px 1枚では見た目を変えず(sub-patch シグナル)、whole-slide seed に
  混じった組織希薄パッチが「空白 vs 空白」の高類似マッチを量産していた。
- 初回は `rerank_pool=1000` が浅すぎ、top-5000 候補の 4890 が seed スライド自身のパッチ
  (自己類似)で埋まっていた → `rerank_pool` を 6000 に上げ、空白パッチ除外を
  `lib/patch_set.py` に追加して再実行する。

### 所見の2クラス分け(0014 + 0015 で見えた実データの構造)

| クラス | 例 | patch レベル検索 + curation |
|---|---|---|
| whole-patch テクスチャ | glycogen, hypertrophy, fatty change, ground glass | 機能する(seed が動く側なら) |
| sub-patch 局所 | 増加分裂像, 単一分裂像, 小さな microgranuloma | 機能しない(シグナルが patch の見た目を変えない) |

### 次の一手

1. glycogen で `rerank_pool` を上げて 0015 を再実行 → 充実した集合が出れば病理レビューへ
2. whole-patch テクスチャ系の別所見(hypertrophy 等)で 0015
3. アトラス seed の道は track 2(アトラス→TG-GATEs のドメインギャップを詰める)が前提

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
