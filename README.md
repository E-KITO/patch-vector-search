# patch-vector-search

UNIパッチ埋め込みに対するクラスタベースのベクトル検索
(FAISS OPQ+IVF+PQ)+WSI逆引き。任意の参照画像(既存コーパス外でもOK)を渡すと、
類似する組織パッチと、それを多く含むWSIを検索できる。  
**最終的な目標は、INHANDやNTPの非腫瘍性病変アトラスのような、所見の代表的なパッチを検索クエリとして使い、TGGATEの大規模なWSIコーパスから同じ所見を引き出してデータベース化すること。**

## 現状(2026-09-10時点)

- **動くもの**: 任意の画像(1枚〜複数枚)を渡すと、類似パッチ検索とWSI逆引きができる。
  実解像度でのヒットパッチ表示・クエリタイルごとの近似スコアヒートマップ表示も追加済み
  (下記「検索結果の可視化改善とタイル選択バイアスの発見」参照)。
- **検索結果 Artifact(2026-09-10、`experiments/0020`)**: 初代の「NNL アトラス所見図版 →
  類似パッチ検索」デモ Artifact を、現行の改善後経路(背景除去 0018 + notebook 01 の探索
  パラメータ)で作り直した。25所見を `0018` / `0002` の両索引で検索した before/after +
  成果物ギャラリー + 自己検索診断の総合レポート:
  https://claude.ai/code/artifact/a34cf5e1-b488-4571-8027-065c20098df3 。
  発見: 背景除去は max_similarity 上位パッチをほとんど動かさない(効果はスライド単位
  ランキング側)。下記「experiments/0020」参照。
- **成果物パイプライン**: `lib/patch_set.py` + `experiments/0015` で、所見の確定スライドを
  seed に**レビュー可能な代表パッチ集**(実解像度パッチ + manifest + スライド別コンタクト
  シート)を出力できる。`Deposit, glycogen` では GT スライドを全除外した deliver モードで
  未ラベルスライドのみから137枚を構成済み(病理レビュー待ち)。下記「成果物の試作」参照。
- **どの所見で機能するか**: **glycogen(137枚)と ground glass(113枚)は機能する**
  (ランダム対照で判定精度91%)。**hypertrophy**(同59%、chance 以下)は不成立。
  **granular eosinophilic** はコーパスに一般化可能なシグナルがほぼ無い — 背景除去後の
  `experiments/0019` deliver で非 seed 候補が5枚しか残らず(0002 時代の「候補74枚中48枚が
  背景クロップ、最終21枚」は背景ノイズの生存者だった)。所見は「パッチ内で完結する
  テクスチャ」「相対的基準を要する」「sub-patch 局所」「コーパスカバレッジ不足」の
  4クラスに分かれ、機能するのは最初の1つだけ。下記「所見の3クラス分け」参照。
  **GT recall だけでは判定を誤るので、0015 の後は必ず
  `scripts/random_patch_baseline.py` でランダム対照を取ること**(下記「ランダム対照による検証」)。
- **背景パッチ除去(2026-09-09、完了・既定昇格)**: 旧 `_is_blank_tile`(平均輝度>240 かつ
  標準偏差<8)は背景の半分以下しか捕まえておらず、manifest に約39万枚の背景パッチが
  残っていた。彩度基準(`sat_frac < 0.10`、job 10491 の閾値監査で確定)で
  `lib/patch_blankness.py` に再定義し、`experiments/0017`(manifest、389,959枚 = 2.12%
  除外)→ `0018`(FAISS 索引再構築、GPU 再埋め込み不要)を実施。`_is_blank_tile` も
  クエリ側で同じ彩度基準に統一。**効果**: max_similarity ランキングの自己検索順位が
  顕著に改善(granular eos の `sim_best` 20→2 等)、n_hits_ratio ランキングと GT 比較は
  中立、負の影響なし。→ **0018 を既定インデックスに昇格**(`scripts`/notebook の既定、
  `validate_against_ground_truth.py` の `baseline_v1`)。0002 は「背景除去前」の参照として
  残す。下記「背景パッチ」参照。
- **モデル/コーパスの検索天井**: 自己検索診断
  (`scripts/self_retrieval_diagnostic.py`、2026-09-04、2026-09-08にバッチ交絡の扱いを
  拡張、下記「自己検索診断」参照)で、**uni_v1 + 現行コーパスがバッチ交絡(同一化合物・
  同一 study)抜きで「検索に足る」と示せたのは 5〜6の common な肝所見**
  (Microgranuloma / Cellular infiltration / Swelling / Deposit glycogen /
  Ground glass appearance、Increased mitosis は境界)。当初「16所見中10所見」としていたが、
  同一 study 除外とバッチ・ネガティブコントロールで Change eosinophilic・granular
  eosinophilic・Single cell necrosis の好成績はバッチ由来と判明。**融合壊死(`Necrosis`)
  と髄外造血はこの拡張後も本物の弱点**。中核仮説(代表パッチ→類似パッチ検索)は
  「機能する所見」については成立しており、成果物トラックの glycogen・ground glass も
  ここに含まれる。**この2所見について他エンコーダ(uni_v2 / virchow2 / hibou_l)が
  uni_v1 を上回るかを `experiments/0021` の弁別力プローブ(2026-09-10)で確認したが
  否定的** — どのモデルも CI が重ならないレベルの改善を出せず、髄外造血は全モデル
  chance 以下。天井はエンコーダではなくコーパスカバレッジ/タスク定義の側(下記
  「experiments/0021」)。
  **【2026-09-10 一部修正】** `experiments/0024`(部分サンプル proxy)→ `experiments/0025`
  (ZCA 白色化をフル FAISS 索引にベイクして再測定)で、L2 正規化コサインを **whiten** に
  替えると self_retrieval の `nhr_best_rank_med_noexp` が **granular 298→20・壊死 52→34・
  肥大 59→38・髄外造血 125→91** 等、難所見 7 つで改善(2 つ悪化)。
  「融合壊死・髄外造血はエンコーダの弱点」の相当部分は**指標の hubness/異方性**だった。
  ただし whiten は atlas クエリ(ドメインギャップ)を悪化させるため既定索引への昇格は保留、
  収縮白色化 or 成果物トラック専用索引を検討中。下記「experiments/0024」「0025」。
- **区切りをつけた路線(いずれもbest_rankの低迷を解消しない)**: IVFクラスタリング仮説、
  手動ROIクロップ、染色正規化(クエリ側・コーパス側・per-tile交絡なしまで、4回検証)、
  **倍率自動補正(`experiments/0022`/`0023` で決着、下記)**。
- **倍率(magnification)— 決着**:
  - `experiments/0022`(job 10516): 正解既知の合成再スケールクエリで、**`fixed`
    パイプラインは ±2x の倍率ずれに頑健**(median 順位 1〜2)。2x を超えると崩れるが、
    それは純粋なスケール由来(oracle 補正でほぼ完全回復)。
  - `experiments/0023`(job 10517): 細かい centroid + ガードバンドで
    `embed_image_tiles_auto_scale` を復活できるか atlas GT で再測定 → **否定的**。
    FM centroid のスケール推定は**病理テクスチャの粗さを低倍率と誤認**し
    (グリコーゲン沈着・肥大肝細胞など)、しかも**推定が bimodal(0.5 か 2.0、1.0 付近を
    まず出さない)なのでガードバンドが不発**。細粒度化は synthetic の推定 median は
    改善するが atlas GT では旧 centroid より悪化(1勝2敗 vs 2勝1敗)。**動く所見
    Glycogen の best_rank が 7 → 100+ に破壊される**のが致命的。
  - **結論**: 倍率補正はコアパイプラインに入れない(棚上げ継続)。クエリはコーパスの
    20x から ±2x(概ね 10〜40x)以内で与えること。特定クエリが明らかに要補正なら
    **既知の係数で手動リサイズ**して no-correction と A/B する(自動推定は信用しない)。
- **アトラスGT比較の位置づけ**: アトラス図版をクエリにする
  `scripts/validate_against_ground_truth.py` のbest_rankが悪いカテゴリの主因は、
  モデルの表現力ではなく **アトラス→TG-GATEsのドメインギャップ + 図版が所見と無関係な
  組織だらけ**(タイル選択バイアス、下記参照)である可能性が高い。ただし Kupfferは
  コーパス内スライドが1化合物しかなく自己検索で天井を測れないため、ドメインギャップ説の
  直接的な裏付けは無い(下記「自己検索診断」発見5)。今後の定量評価は自己検索診断のほうが
  交絡が少なく所見カバレッジも広い(7 → 16所見)が、バッチ交絡には引き続き注意。
  なお、Kupffer クエリフォルダには修正前(〜2026-09-09)Hypertrophy 図版3枚
  (うち正常肝2枚)が混入しており、これが Kupffer の低迷の一因だった。誤格納を直して
  再実行(job 10467)したところ **Kupffer の baseline_v1 best_rank は 202 → 76**
  (mean 220 → 80)に改善、Hypertrophy も 49 → 23。融合壊死は 10 → 14 で不変
  (下記「評価に使ったデータとその限界」の誤格納の項)。
- **未解決**: アトラス図版のドメインギャップの詰め方、融合壊死・髄外造血の表現
  (uni v2等)。「今後やること」参照。

## 評価に使ったデータとその限界

Ground truth比較(`scripts/validate_against_ground_truth.py`)のクエリ画像は、
`data/query/Nonneoplastic-Lesion-Atlas-National-Toxicology-Program_Liver/`に置いた
**NTP(National Toxicology Program)の非腫瘍性病変アトラス(NNL)**から取得した、
各所見カテゴリの代表的な掲載図版(26カテゴリ・91枚、1所見あたり1〜8枚)。

以下の限界を踏まえて結果を解釈すること:

- **【2026-09-09 修正】クエリ図版フォルダに15件の誤格納があった。** `image_id` を
  atlas ページに照合したところ、Necrosis の図版6枚が `Hyperplasia, Nodular` /
  `Hypertrophy` フォルダに、Hypertrophy の図版5枚が `Kupffer` / `Intrahepatocellular
  Erythrocytes` フォルダに、ほか Inflammation 2枚・Focus 2枚が別フォルダに入っていた。
  2026-09-09 に全件を正しいフォルダへ移動し、重複コピー3件(` (1).jpg`)を削除した
  (94→91枚、対応表は `data/query/nnl_liver_atlas_misfiled.csv`、経緯は
  `data/query/nnl_liver_atlas_README.md`)。**この修正より前に回した
  `scripts/validate_against_ground_truth.py` / `experiments/0014` のアトラスクエリのうち、
  Necrosis(修正前はフォルダに10枚中4枚しか無かった)・Hypertrophy(修正前は5枚 = 本物4枚
  + Necrosis 図版1枚混入、本来の fig 1–4,6 は他フォルダに散逸)・Kupffer(修正前は4枚 =
  本物1枚 + Hypertrophy 図版3枚、うち2枚は正常肝)は汚染されたクエリセットで
  評価されている。** Glycogen・Increased mitosis・Hematopoiesis・封入体は影響なし。
  0014 は glycogen・mitosis でしか実行していないため直接の影響は無いが、Kupffer 等で
  再実行する場合は修正後のフォルダを使うこと。

- **【2026-09-09 修正後の再実行(job 10467)】** 誤格納を直したフォルダで
  `validate_against_ground_truth.py` を回し直した結果(修正前は
  `outputs/gt_validations/gt_validation_results_2026-09-04_pre_nnl_misfiled_fix.csv`
  に退避)。下表はいずれも **0002 コーパス**(背景除去前)での値。
  なお 2026-09-09 の 0018 昇格後、`outputs/gt_validation_results.csv` は
  `baseline_v1` = 0018 で再生成済み(job 10499)。0018 での 7 カテゴリは 0002 と
  ほぼ同じ(Kupffer best 76→78、Hypertrophy 23→24、他は不変〜±3。下記「背景パッチ」
  の GT A/B と一致)。**2026-09-10 に対照図版除外(job 10511)を反映して再々生成**、
  Hypertrophy は best 30 / found 24 に(理由は下の「2026-09-10 修正」バレット)。
  旧 0002 baseline_v1 の値は
  `outputs/gt_validations/gt_validation_results_2026-09-09_deblank_ab_job10495.csv` に残る:

  | カテゴリ | baseline_v1 best (前→後) | v1_macenko best (前→後) |
  |---|---|---|
  | Kupffer | **202 → 76**(mean 220 → 80) | 766 → 821 |
  | Hypertrophy | **49 → 23** | 14 → 21 |
  | Necrosis (→ Single cell necrosis) | 10 → 14 | 77 → 86 |
  | Increased mitosis / Glycogen / Hematopoiesis / 封入体 | 不変(7 / 7 / 25 / 480) | 不変(1 / 14 / 84 / 97) |

  非汚染4カテゴリが前後で完全一致することが、修正が Necrosis/Hypertrophy/Kupffer の
  3フォルダにしか影響していないことのサニティチェックになっている。**Kupffer の
  改善が大きい**: 修正前は「本物1枚 + Hypertrophy 図版3枚(正常肝2枚)」で正常肝タイルが
  クエリを汎用肝細胞へ引っぱっていた。本物の Kupffer 図版1枚だけにしたら best_rank 202 → 76。
  README 各所の「Kupffer は最悪カテゴリ(best_rank 202)」という記述は誤格納由来を含んでおり、
  実際の値は 76(それでも良くはないが、ドメインギャップの状況証拠としての重みは下がる)。
  融合壊死は図版を10枚に戻しても改善せず(10 → 14)、「表現の限界」という結論は変わらない。

- **【2026-09-10 修正】対照図版(`is_normal_control`)をクエリから除外した(job 10511)。**
  誤格納修正とは別に、NNL の lesion ページには「Normal liver, age and sex matched, for
  comparison with Figure N」という**正常肝の対照図版**が一部混ざっている(肝では Atrophy
  2枚・Hepatocyte - Hypertrophy 2枚の計4枚のみ。他の所見にはない)。GT 7カテゴリでこれを
  持つのは Hypertrophy だけ。全 lesion ページの caption を `data/query/nnl_liver_atlas_figures.csv`
  に収集し、`lib/atlas_figures.py` の `query_images()` が `is_normal_control=yes` を落とす
  ようにした(`validate_against_ground_truth.py` / `experiments/0020` が共有)。
  除外後、**Hypertrophy は best_rank 24 → 30 と悪化し、found も 25 → 24**(GT 25 スライド
  中1件が上位500から外れた)。正常肝タイルが「たまたま hypertrophy の GT スライドを含む
  汎用肝細胞」を引いて順位を底上げしていたため。除外は正直な数字であって改善ではない。
  他6カテゴリはフォルダ内容不変だが best_rank が 0〜5 ずれた(mitosis 7→10、髄外造血
  25→28、封入体 480→475、Kupffer 76→78、融合壊死 14・Glycogen 7 は不変。下記の注記参照)。
  `outputs/gt_validation_results.csv` は job 10511 の値で再生成済み。
- **atlas GT 数値には ±5 位程度の非決定性ノイズがある**。`lib.query_embedding.embed_image_tiles`
  の UNI 埋め込みは GPU の fp16/bf16 演算が run-to-run で完全一致せず、同じクエリ図版でも
  再実行のたびに埋め込みが僅かに変わる。job 10511 では**フォルダ内容が全く変わっていない
  6カテゴリの best_rank が 0〜5 位ずれた**。したがって atlas GT の best_rank/mean_rank は
  「±5 位以内の差は誤差」として読むこと(self_retrieval 診断はコーパス側の固定埋め込みを
  使うのでこの影響を受けない — 一次評価を self_retrieval に置く理由のひとつ)。

- **アトラス画像1枚は所見部位を含む図版全体であり、所見が写っているのは画像の一部分に
  過ぎない**(矢印注釈・番号ラベル・周囲の正常組織や余白を含む、1800x1200px程度の
  パノラマ〜クローズアップが倍率不揃いのまま混在している)。
- **所見部位だけを人手で切り出す作業はしていない**。`embed_image_tiles`が画像全体を
  機械的にタイル分割し、白背景タイルの除外以外は所見領域かどうかの判別なしに全タイルを
  検索へ投入している。つまりこの評価は「自動化できる範囲でどこまでやれるか」を測った
  ものであり、人手でROIを切り出した場合の性能上限を示すものではない。best_rankが悪い
  カテゴリ(例: 封入体 best_rank=480、Kupffer細胞増殖 best_rank=76 ※誤格納修正前は 202)は、
  モデルや検索アルゴリズムの限界だけでなく、クエリ画像に占める「所見と無関係なタイル」の
  割合が高いことも一因である可能性が高い。**自己検索診断(2026-09-04、2026-09-08拡張)は
  common な肝所見の多くがコーパス内で自己検索できることを示したが、Kupfferはコーパス内
  1化合物のため天井を測れず、この低迷がドメインギャップ由来という切り分けはできていない
  (下記「自己検索診断」参照)。交絡の少ない定量評価が必要ならそちらを使うが、
  バッチ交絡には注意。**
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
- [x]  ~~**uni v2など他モデルでの埋め込みを検討する**(融合壊死・髄外造血のため)~~
      → 2026-09-10、`experiments/0021` の弁別力プローブ(wsi_preprocess が
      necrosis 13 / single cell 4 / hematopoiesis 3 スライド + confuser 106 を
      `uni_v2` / `virchow2` / `hibou_l` で埋め込み、私が LOSO AUROC を計算)で
      **否定的**。どのモデルも uni_v1 を CI が重ならないレベルで上回れず、髄外造血は
      全モデル AUROC ≤ 0.47(chance 以下)。フル再埋め込み(Task B)は見送り。
      下記「experiments/0021」参照
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
      → 2026-09-04 実施、2026-09-08 にバッチ交絡対応で拡張。
      `scripts/self_retrieval_diagnostic.py`。バッチ交絡抜きで検索天井が十分なのは
      5〜6所見、融合壊死と髄外造血が本物の例外、と判明。詳細は下記「自己検索診断」参照
- [~]  **「機能する所見」でデータベース化のワークフローを一度通す**
      → 2026-09-04、`experiments/0014`(アトラス seed)/`0015`(TG-GATEs スライド seed)に着手。
      アトラス seed はドメインギャップで不成立、TG-GATEs seed は glycogen で機能・mitosis は
      所見の性質上不成立。2026-09-05、0015 に validate/deliver モードを分離し、
      **glycogen で deliver(未ラベルスライドのみから137枚)まで到達、ランダム対照でも
      選別が効いていることを確認**(判定精度91%)。一方 **hypertrophy は不成立**と確定し、
      所見の分類は3クラスになった。glycogen 137枚の病理レビュー待ち。詳細は下記
      「成果物の試作」「ランダム対照による検証」「所見の3クラス分け」参照
- [x]  ~~コーパスに残る背景パッチ(約39万枚)を落とす~~
      → 2026-09-09、`scripts/measure_corpus_blankness.py`(全パッチ彩度測定)→ 閾値監査
      (`sat_frac < 0.10`)→ `experiments/0017`(manifest、2.12% 除外)→ `0018`(索引再構築)。
      GPU 再埋め込み不要。max_similarity ランキングの自己検索順位は顕著に改善、
      n_hits_ratio ランキングと GT 比較は中立。granular は候補プール崩壊で成果物化せず。
      詳細は下記「背景パッチ」参照
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

前提: `experiments/0001` → `0002` → `0017_..._build_patch_manifest_deblank` →
`0018_..._build_faiss_index_deblank` が実行済みで、
`outputs/0018_20260909_build_faiss_index_deblank/default/` にインデックスがある状態。
**この 0018(背景除去済み)が 2026-09-09 以降の既定インデックス**(下記「背景パッチ」)。
背景除去前の索引が要る場合は `outputs/0002_20260808_build_faiss_index/default/`。

```python
import numpy as np
from lib.query_embedding import embed_image_tiles
from lib.search import PatchIndex

INDEX_DIR = "outputs/0018_20260909_build_faiss_index_deblank/default"
patch_index = PatchIndex.load(
    index_path=f"{INDEX_DIR}/index.faiss",
    manifest_path=f"{INDEX_DIR}/manifest.parquet",
    slide_meta_path=f"{INDEX_DIR}/slide_meta.parquet",
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
| `stain_reference=`(Macenko染色正規化) | ❌ 非推奨(確定) | 同上。ケースによっては大きく悪化する。自作`lib.stain_normalize`実装(自己一致性0.5〜0.84)→`lib.torchstain_normalize`(自己一致性0.96〜0.996)への差し替え、さらに2026-09-04の交絡なし再検証(uni_v1・ジオメトリ完全一致・コーパス側もクエリ側もper-tile Macenko・自己一致性検証済み、`experiments/0010`/`0012`)まで行ったが、GT比較で`baseline_v1`を上回れず(best_rank 7カテゴリ中4カテゴリで悪化、うちKupffer 202→766・Necrosis 10→77等は大幅悪化。改善は所見局在型の3カテゴリのみ)。染色正規化はこれで完全に棚上げ。詳細は下記「クエリ側染色正規化の再検証」参照(※Kupffer/Necrosis/Hypertrophyの数値はNNLアトラス誤格納の修正前。修正後もmacenko優位は覆らない) |
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
  (この順で依存。`uni_v1`・1024次元。**2026-09-09 以降の既定インデックスは背景除去済みの
  0018**、下記参照 — 0002 は背景除去前の索引として残す)
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
- `experiments/0016_..._image_query_path_diagnostic`(クエリ埋め込み経路が 0014 の
  結論の交絡かを測る3 tier 診断。詳細は下記「experiments/0016」参照)
- `experiments/0017_..._build_patch_manifest_deblank` → `0018_..._build_faiss_index_deblank`
  (0001/0002 のコーパスから `sat_frac < 0.10` の背景 389,959 枚を除いた版。エンコーダ・
  ジオメトリ・OPQ+IVF+PQ ハイパラは 0001/0002 と完全一致。**2026-09-09 に既定インデックス
  へ昇格**(`lib`/scripts/notebook の既定、README 各所)。詳細は下記「背景パッチ」参照)
- `experiments/0019_..._build_finding_patch_set_deblank`(0015 の deliver を 0018 コーパスで
  回し直す A/B。`index_exp_dir` のみ差分。詳細は下記「背景パッチ」参照)
- `lib/patch_blankness.py` — パッチの blankness 測定(`blankness_metrics`)と背景除外基準
  (`is_background`、`sat_frac < 0.10`)。コーパス側(`lib/manifest.py`)・クエリ側
  (`lib/query_embedding._is_blank_tile`)の両方が参照
- `lib/patch_set.py` — 索引ヒット → 代表パッチ集の productization レイヤー
  (類似度しきい値・スライド内NMS・スライド上限・ラウンドロビン・空白除外・切り出し・manifest)
- `scripts/validate_against_ground_truth.py` — 新しいパイプライン案をNNLアトラス由来GTで検証するツール
- `scripts/self_retrieval_diagnostic.py` — コーパス内leave-one-out自己検索でモデル/コーパスの
  検索天井を測る(2026-09-04、2026-09-08にバッチ交絡対応で拡張。アトラス図版を使わない・
  GPU不要。詳細は下記「自己検索診断」参照)
- `scripts/select_average_patch.py` — 染色正規化用の「典型的な」基準パッチを選ぶ(セットアップ用、実行済み)
- `scripts/manual_roi_crop_diagnostic.py` — 手動ROIクロップの性能上限測定(2026-08-20、詳細は下記参照)
- `scripts/torchstain_query_normalization_diagnostic.py` — torchstainベースのクエリ側染色
  正規化をv1インデックスで再検証(2026-08-20、詳細は下記参照)
- `scripts/crop_necrosis_query_tile.py` — 壊死巣内部だけを224x224で手動クロップし、タイル
  選択バイアスを回避したクエリを作る(2026-08-20、詳細は下記参照)
- `scripts/measure_corpus_blankness.py` — 全 18.4M パッチを raw WSI から切り出して
  `mean_intensity` / `std_intensity` / `sat_frac` を測る(background 除去の閾値決定用。
  索引・エンコーダ不使用、スライド並列。2026-09-09、詳細は下記「背景パッチ」参照)
- `scripts/audit_blank_threshold.py` — 上の parquet から `sat_frac` / `mean_intensity` で
  bin 分けしてパッチをサンプルし、コンタクトシートに落とす(閾値の目視監査。2026-09-09)

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

> **注記(2026-09-09)**: 以下の Kupffer 数値(best_rank=202 等)は、NNL アトラスの
> Kupffer クエリフォルダに Hypertrophy 図版3枚が誤格納されていた状態で測ったもの。
> 誤格納を直して再測定すると Kupffer の baseline_v1 best_rank は **76**(上記
> 「評価に使ったデータとその限界」参照)。ただし本節の結論(nprobe/PQ量子化どちらも
> 主因ではない、厳密計算でも順位が上がらない)はカテゴリ非依存の論理で、封入体
> (非汚染、best_rank=480)でも同じ結論なので、**仮説否定の結論は変わらない**。

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

> **注記(2026-09-09)**: 本節および上の「何を使うべきか」表にある Kupffer/Necrosis/
> Hypertrophy の best_rank(例: Kupffer 202→766、Hypertrophy 49→14/49→17)は、NNL
> アトラスのクエリフォルダに誤格納があった状態で測ったもの。誤格納修正後の baseline_v1 は
> Kupffer 76・Hypertrophy 23・Necrosis 14、v1_macenko は Kupffer 821・Hypertrophy 21・
> Necrosis 86(上記「評価に使ったデータとその限界」参照)。相対関係(macenko は色情報依存
> カテゴリで baseline_v1 より大きく悪化)は修正後も維持され、**「染色正規化は棚上げ」の
> 結論は変わらない**(4回の独立確認 + 悪化が色情報依存カテゴリに集中というパターン)。

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

> **注記(2026-09-09)**: 上記の発見1〜3・結論は、この8枚(`data/query/` 直下のリネーム
> コピー、いずれも `matched` 図版)に基づいており NNL アトラスの誤格納の影響を受けない。
> ただし 0013 は後日 `--atlas-root` で ①所見フォルダ集約(25クエリ、job 10429)と
> ②画像1枚ずつ(91クエリ、job 10435/10437/10438)のバルク sweep も回しており、
> **①の 9 dir**(`query__Liver_-_Necrosis_...`(4→10図版)、Kupffer(4→1)、Hypertrophy(5→9)、
> Hyperplasia-Nodular(7→2)、Intrahepatocellular-Erythrocytes(4→2)、Focus(3→5)、
> Stellate(4→2)、Inflammation(4→6)、Hepatodiaphragmatic-Nodule(8→3))が誤格納フォルダの
> 中身で計算されていた。誤格納修正後に **job 10468** で ① を回し直し
> (`--no-galleries --overwrite`、25クエリ `0 failed`)、②の孤児 dir 3件(削除した
> ` (1).jpg` 由来)を削除した。②本体は dir 名が画像ファイル名なので影響なし。

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

> **注記(2026-09-09)**: ここで見た「Kupffer図版」は本物の figure 1(a19999)。
> ただし `validate_against_ground_truth.py` の Kupffer フォルダには当時 Hypertrophy 図版が
> 3枚誤格納されており、GT比較の Kupffer best_rank=202 はその汚染を含んでいた
> (修正後は 76、上記「評価に使ったデータとその限界」)。本節のヒートマップ観察
> (図版単体では Kupffer 密領域がスコアで勝たない)自体は変わらない。

### 結論

- **Macenkoは既定では棚上げのままで妥当**(4回目の確認)。
- ただし「染色正規化はどのタイルが選ばれるかに実際に効く(好塩基性局所所見で顕著)」は
  記録すべき発見 — タイル選択バイアスの路線を再訪する場合は関係する。
- **壊死の低迷は『コーパス/モデルにその所見が distinctive に無い』**という、前処理では
  動かせない層の問題(下記「自己検索診断」でも裏付けられた)。次の productive な動きは
  「今後やること」の「抽出できる所見・できない所見を精査する」「uni v2など他モデル」。

出力: `outputs/0013_20260904_query_demo_macenko_per_tile/`(gitignore対象、NFS上に残置)。

## 自己検索診断: モデル/コーパスの検索天井(scripts/self_retrieval_diagnostic.py、2026-09-04、2026-09-08拡張)

NNLアトラス図版をクエリにするGT比較は、「UNI埋め込みが同一所見のパッチを近くに置けて
いるか」と「アトラス→TG-GATEsのドメインギャップ・図版が所見と無関係な組織だらけ・
倍率不明・ROI不明」を混同している。0013で難しいカテゴリ(壊死・Kupffer)が前処理の
手前で失敗すると分かったため、**アトラス図版を一切使わず、コーパス内 leave-one-out
自己検索でモデル/コーパスの天井を測った**。`single_finding_liver.csv` の確定単一所見
スライド(コーパス内126枚、≥2枚ある16所見)について、各スライドを順にクエリにし
(自身のパッチ特徴量をh5からそのまま読む — UNIエンコーダは呼ばない、GPU不要)、
索引検索してクエリスライドを除外し、同一所見の他スライドが何位に来るかを記録する。

**2026-09-08、化合物/実験バッチ交絡の扱いを一級の懸念に格上げして拡張(commit f735fc7、
job 10440、約19分)。** TG-GATEsは1 study(`EXP_ID`)= 1化合物のタイムコース1本で、
同一 study の2枚は所見に関係なく化合物・染色/スキャンバッチ・動物系統・固定条件を
共有するため似て見える。従来の同群除外(`EXP_ID`+`GROUP_ID` = 用量+時点完全一致のみ)
では弱い。拡張版が追加した列(`outputs/gt_validations/self_retrieval_diagnostic.csv`):
  - `n_compounds` / `n_exp_ids` — その所見のコーパス内スライドが跨がる distinct 化合物/
    study 数。**≤2 なら「所見 vs 化合物」を原理的に分離不能**。
  - `*_best_rank_med_noexp` / `nhr_hit@50_noexp` — 同一 study を丸ごと除外した best_rank。
  - `batch_dominates_finding_rate` — 各クエリで「同一 study・別所見」のスライドを
    ランキングし、その best_rank が「同一所見・別 study」ターゲット以上に上位だった
    クエリの割合。**高ければ検索は所見ではなくバッチを追っている。**

### 発見1: バッチで説明できない天井を持つのは 5〜6所見(従来「10所見」は過大評価)

同一 study 除外(`_noexp`)後も chance(random_best_rank)を明確に上回り、
`batch_dominates_finding_rate` が低いのは: **Microgranuloma**(noexp best 12 / chance 62、
hit@50 0.7)、**Cellular infiltration**(10 / 143、1.0)、**Swelling**(8 / 143、1.0)、
**Deposit, glycogen**(21 / 167、1.0)、**Ground glass appearance**(17.5 / 250、1.0)。
Increased mitosis(60 / 100、0.4)は境界線。**成果物トラックの2所見(glycogen・
ground glass)はここに残った** — deliver した137枚/113枚の妥当性を損なう結果ではない。

### 発見2: 「良い数字」がバッチ由来だった所見

- **Change, eosinophilic**: best_rank 3 → 51(同群除外)→ **136.5(同一 study 除外、
  chance 91 より悪い)**、hit@50 1.0 → 0.2、`batch_dominates` 0.5。所見類似ではなく
  実験群類似だった。
- **Degeneration, granular, eosinophilic**: コーパス内 **2化合物・2 study のみ**。
  自己検索の "best_rank 1" は、同一 study・別所見スライドが**常に(4/4)**上位に来る =
  100% バッチ。README「所見の3クラス分け」で既に「候補の93%が背景・天井0.894・
  成果物にならない見込み」としていた所見で、自己検索の数字も当てにならなかった。
- **Single cell necrosis**: 全ターゲットでの best_rank 16.5、同一 study 除外後も 45.5
  とマシに見えるが、`batch_dominates` 1.0(3/3)= 同一 study・別所見スライドが常に
  上位に来るので、その "マシ" はバッチ駆動である可能性が高い。なお
  `validate_against_ground_truth.py` の `CATEGORIES` がアトラス "Necrosis" を
  `"Single cell necrosis"` にマッピングしていたのが apples-to-oranges だった点は変わらない。

### 発見3: 融合壊死・髄外造血は本物の弱点(結論維持・補強)

`Necrosis`(13枚、11化合物)は同一 study 除外後も best_rank 49.5(chance 77)、
`batch_dominates` 0.0。多様な化合物でも駄目 = **融合壊死はUNI空間でクラスタを形成しない**
(0013の`sim=0.33`と整合)。`Hematopoiesis, extramedullary`(3枚、best 125、除外しても
不変)も同様。`Hypertrophy`(25枚、14化合物)は同一 study 除外後 best 61.5 で **chance
(40)より悪い** — 多様な化合物なのでバッチ由来ではなく、「相対基準を要する」所見の
性質(README「所見の3クラス分け」)。

### 発見4: n_hits_ratioソートはmax_similarityソートより優れている

ほぼ全所見で `nhr_best_rank ≤ sim_best_rank`(Single cell necrosis 16.5 vs 125、
Deposit glycogen 4.5 vs 26.5 等)。0013発見2の「n_hits_ratioとmax_similarityが乖離する」は
本物だが、**現行の既定(n_hits_ratio)が良い側**。集計指標をmax_similarityに変える案は
この診断で否定された。

**追記(2026-09-09、背景除去後)**: この乖離の主因は背景パッチだった。`experiments/0018`
(`sat_frac < 0.10` の背景を除いたコーパス)で自己検索を回し直すと(job 10496)、
`nhr_best_rank` は不変のまま `sim_best_rank` が一斉に縮む(granular eos 20→2、
Single cell necrosis 125→25.5、Alteration cytoplasmic 67.5→1.5)。背景の多いスライドが
「最良マッチ1枚の類似度」で浮上して GT スライドを押し下げていたのが解消された。
n_hits_ratio が良い側という結論は変わらないが、**その差は背景由来で、除去すれば
max_similarity もかなり追いつく**。詳細は上記「背景パッチ」。

### 発見5: コーパス内1化合物の所見は自己検索では評価不能

`Proliferation, Kupffer cell` / `Alteration, cytoplasmic` / `Lesion,NOS` はコーパス内
**1化合物・1 study**(`_noexp` 列が空)。**Kupfferの「自己検索でrank 1 → モデルは
表現できている」という従来の主張は撤回** — n=2 が同一 study の同一化合物なので、
自己検索は何も言えていない。アトラスGT比較でのKupffer低迷をドメインギャップに
帰属する他の状況証拠はあるが、この診断はもう根拠にならない。

### 結論

- **UNI(uni_v1)+ 現行コーパスがバッチ交絡抜きで「検索に足る」と示せたのは 5〜6所見**
  (Microgranuloma / Cellular infiltration / Swelling / Deposit glycogen / Ground glass、
  Increased mitosis は境界)。中核仮説はこれらについては成立。**特に成果物トラックの
  glycogen・ground glass は無事。**
- **融合壊死と髄外造血は表現の限界**(結論維持・補強)。uni v2 等の検討はこの2所見のため。
- **Change eosinophilic / granular eosinophilic / Single cell necrosis の「良い数字」は
  バッチ由来**。granular は他の証拠と合わせて成果物にならない見込み。
- **Kupfferの天井は自己検索では測れない**(コーパス内1化合物)。
- 現行の検索パス(n_hits_ratioソート)は妥当。集計指標の変更は不要。
- 今後の定量評価では、アトラスGT比較よりこの診断のほうが交絡は少ないが、**万能ではない**:
  n≤3 の所見、コーパス内化合物数 ≤2 の所見、`batch_dominates_finding_rate` が高い所見は
  信用しないこと。

## 成果物の試作: 所見ごとの代表パッチ集(experiments/0014 / 0015、2026-09-04〜09-05)

自己検索でモデル天井が十分と分かったので、**中核目標である「所見ごとの代表パッチ集」を
一度 end-to-end で作ってみる**トラックに着手した(README「今後やること」トラック1)。
検索コアは既存の `PatchIndex` だが、その上に productization レイヤー
(`lib/patch_set.py`: 類似度しきい値 → スライド内空間NMS → スライドあたり上限 →
由来スライドでのラウンドロビン絞り込み → 空白パッチ除外 → 実解像度パッチのファイル保存 →
manifest → スライド別コンタクトシート)を載せる。0009 系の「検索が動いたかを見る診断」
とは違い、レビュー可能なパッチデータセットそのものを出力する。

### experiments/0014: NNL アトラス図版を seed に(2026-09-04、否定的)

`Deposit, glycogen`(アトラス図版4枚)と `Increased mitosis`(同1枚)で実行。
各150枚出力したが**代表パッチ集としては不成立**(この2所見のクエリフォルダは
2026-09-09 に判明した誤格納15件の影響を受けていないので、下記の結論は有効):

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

初回(job 9671、`rerank_pool=1000`)は探索が浅すぎ、top-5000 候補の 4890 が seed
スライド自身のパッチ(自己類似 sim 0.93+)で埋まっていた。`rerank_pool` を 6000 に上げ、
空白パッチ除外を `lib/patch_set.py` に追加して再実行(job 9675)した結果:

- **glycogen: パイプラインとして機能する**。非seed候補が 110 → **689** に増え、
  **フル150枚 / 43スライドの集合**が組めた(初回は89枚止まり)。目視でコヒーレント
  (rank1 の非GTスライド 22697、非GT 12103、hold-out GT 49372・53267 が seed と同一
  morphology = グリコーゲン関連の淡明・不規則空胞化した肝細胞)。**hold-out GT 3枚中 2枚を
  回収**(0014 のアトラス seed は 1/6)。非GT寄与スライド ~40枚は未ラベルのグリコーゲン
  スライドの可能性が高く、病理レビューでトリアージできる形 = **whole-patch テクスチャ系
  所見については、この成果物ワークフローは成立する**。
- **mitosis: 深いプール + 空白フィルタでも不成立**。空白クロップを 80枚除外できた
  (フィルタは意図どおり動作)が、残り150枚は **150スライドから1枚ずつ**・hold-out GT
  回収 **1/5**・非特異的な肝組織。増加分裂像は 224px 1枚では見た目を変えない
  (sub-patch シグナル)ため、探索や curation では埋められない。**mitosis 系は patch
  レベル検索の対象外**で確定。

### job 9701: validate / deliver モード分離と hypertrophy への展開(2026-09-05 追記)

`lib/patch_set.py` の spatial_nms オーバーフロー修正(commit `4d04aff`)後、0015 に
**モードの区別**を入れた(commit `8c7948a`)。`--task '<finding>@<mode>'`:

- **validate**: GT スライドを seed / hold-out に分割し seed のみ除外。hold-out GT に
  `gt_positive` フラグが付くので recall として読める。seed→retrieval が未検証の所見用。
- **deliver**: GT スライド全体を seed にして**全て除外**。出力は未ラベルスライドから
  発見したパッチだけ = 本来の成果物。検証済みの所見用。

あわせて `max_seed_slides: 6` を導入(GT 25枚の Hypertrophy でクエリタイル数が膨らむのを
抑える)、`rerank_pool` 6000→5000、`n_query_patches_per_seed_slide` 200→120、
`sim_floor` 0.40→0.65 に調整し、job 9701 で glycogen@deliver と Hypertrophy@validate を
実行した。

**glycogen deliver: 137枚 / 30スライド**(sim 0.926〜0.945)。GT 6枚を全除外したので、
これは**未ラベルスライドのみから構成された代表パッチ集** = 病理レビューに回せる成果物。
ただし target 150 に届いていない。除外後の候補が 174 しかなく(下記の候補プール構造)、
NMS とスライドあたり上限を通すと 137 で尽きた。**deliver モードは validate より候補が
枯れやすい**(seed = 除外対象が GT 全体になるため)ことは、他の所見に展開する際の
制約になる。

**hypertrophy validate: 150枚 / 49スライド**(sim 0.897〜0.921)。ただし下記の集計バグの
ため、この run の `results.json` にある「hold-out GT 1/13」は**過小評価**である。

### 候補プールは固定窓であることに注意

`lib/search.py::search_similar_patches_multi` は全クエリタイルの結果を統合したあと
`merged.head(k)` で **top-`k_candidate_patches` に切り詰める**。seed スライドの除外
(`lib/patch_set.py`)はその**あと**に効く。seed スライド自身のパッチは自己類似 sim 0.93+ で
この窓の上位を占めるため、**seed スライドを増やすことは有効候補を直接削ることと同義**である。

| run | seed枚数 | 除外枚数 | 生候補 | 除外後 | seed占有率 |
|---|---|---|---|---|---|
| glycogen validate (9675) | 3 | 3 | 6000 | 689 | 89% |
| hypertrophy validate (9701) | 6 | 6 | 5000 | 367 | 93% |
| glycogen deliver (9701) | 6 | 6 | 5000 | 174 | 97% |

`max_seed_slides` はクエリタイル数のコスト制御として入れたものだが、この構造上
**候補プールの品質にも効いている**。seed を増やす方向(上限を緩める・撤廃する)は、
実行時間の増加と候補プールの縮小という二重の損になるため、`k_candidate_patches` を
同時に上げない限り取ってはいけない。

### 集計バグ: max_seed_slides が GT スライドを宙に浮かせていた(修正済み)

job 9701 の hypertrophy validate では、`experiment.py` の seed 分割が hold-out を
`seed_pool` の残余から導出していたため、**`max_seed_slides` で seed から溢れた 6枚
(13116, 19740, 27479, 28782, 35376, 70528)がどの集合にも属さない**状態だった —
クエリにも使われず、結果から除外もされず、`gt_positive` フラグも付かない。結果として
これらから取れたパッチは「非GT」として集計された。

実際には検索は GT スライド **4枚(27479, 28782, 35346, 35376)** を回収できていたが、
`results.json` には hold-out GT 1/13 としか記録されていない。commit `041b66e` で
hold-out を `finding_slides - seed_slides`(seed に使わなかった GT スライド全部)から
導出するよう修正し、**job 9932 で再実行して検証した**。

`seed_slides` の決まり方と `rng` の消費順序は変えていないため、**検索結果は job 9701 と
完全に同一**である(150枚の manifest を `rank, slide_id, coord_x, coord_y, similarity,
patch_file` で突き合わせて一致を確認済み。変わったのは `gt_positive_slide` が True の
パッチ数 1 → 25 だけ)。指標は以下のとおり修正された:

| | job 9701(バグあり) | job 9932(修正後) |
|---|---|---|
| hold-out スライド数 | 13 | 19 |
| hold-out GT 寄与スライド | 1 | **4** |
| hold-out GT 由来パッチの割合 | 0.007 | **0.167** |

glycogen validate(GT 6枚で上限が効かない)と deliver モードは影響を受けない。
job 9701 の出力は `outputs/.../hypertrophy__validate__9701_holdout_bug/` に退避してある
(`lib/output_utils.py` の重複実行ガードを解除するため、再実行前にリネームが必要)。

### hypertrophy の GT 指標だけでは判定できなかった

chance と比べれば有意ではある。seed 6枚を除いたコーパス994枚のうち hold-out GT は19枚
なので、49スライドを引いて GT が期待値 0.94枚のところ **4枚**(約4.3倍)、パッチ割合でも
chance 1.9% に対し **16.7%**(約8.7倍)。

しかし glycogen validate と比べると弱い。glycogen は hold-out GT **3枚中2枚(67%)** を
回収し chance 比で約15倍だったのに対し、hypertrophy は **19枚中4枚(21%)** にとどまる。

さらに hypertrophy(肝細胞肥大)は TG-GATEs で極めてありふれた所見であり、非GT寄与
スライド45枚に未ラベルの陽性が多く含まれている可能性がある。`single_finding_liver.csv`
は「確定単一所見」のスライドしか拾わないため、**GT ラベルだけでは上限も下限も測れない**。
この行き詰まりを解いたのが下記のランダム対照である。

## ランダム対照による検証: 検索は所見を選別できているか(scripts/random_patch_baseline.py、2026-09-05)

`lib/patch_set.py` が出すパッチ集は類似度しきい値で組み立てられるので、**コヒーレントで
あることは構成上の必然**であり、それ自体は何の証拠にもならない。0014 で踏んだ
「アトラス図版に似ている ≠ その FINDING_TYPE」(その見た目が正常肝にありふれていた)と
同じ罠が、seed をスライドに変えた 0015 でも成立しうる。そこで **コーパスから一様
ランダムに抽出した対照パッチ集**を作り、キュレート済みの集合と見分けられるかを測った。

手法(`scripts/random_patch_baseline.py`、索引もエンコーダも不使用):

1. コーパス manifest から一様サンプル(その run の seed スライドは除外して母集団を揃える)。
   `_is_blank_tile` で空白クロップを落とす
2. キュレート済み集合と**混ぜてシャッフルし、通し番号だけを振ったシート**を出力。
   由来は `blind_key.csv` に分離 — ラベルを見てから画像を見ると「そう見える」方向に
   バイアスがかかるため、判定を先に確定させてから答え合わせする

判定は Claude(非専門家)が各対象の先頭100枚に対して行った。

| | glycogen (9943) | ground glass (9975) | hypertrophy (9937) | granular eos (9975) |
|---|---|---|---|---|
| キュレート集合 | deliver 137枚 / 30スライド | deliver 113枚 / 26スライド | validate 150枚 / 49スライド | deliver **21枚** / 15スライド |
| 判定した枚数 | 100 | 100 | 100 | 42(全数) |
| **判定精度** | **91.0%** | **91.0%** | **59.0%** | 88.1%(※) |
| ベースライン(多数派を選ぶ) | 50.0% | 51.0% | **61.0%** | 50.0% |
| curated の recall | 98% (49/50) | **100% (49/49)** | 52% (32/61) | 76% (16/21) |
| curated の precision | 86% (49/57) | 84% (49/58) | 73% (32/44) | 100% (16/16) |
| 判定 | **機能する** | **機能する** | **不成立** | **不成立**(※) |

- **glycogen: 明確に分離できる。** 不規則な白い空胞・淡明でレース状の細胞質という
  パターンが一目で分かり、精度91%(chance 50%)。**検索は確かにこの形態を選別している。**
- **ground glass: 明確に分離できる。** 淡く均質で細かい白い抜けが散在する像として一目で
  分かり、精度91%・**curated の取りこぼしゼロ**(recall 100%)。glycogen に続く2例目の
  「機能する所見」。target 150 に届かず113枚で頭打ちになったが、コーパス内 GT が4枚しか
  ない稀な所見としては妥当(下記「job 9962」参照)。
- **hypertrophy: 分離できない。** 精度59%は、何も考えず全部 curated と答える戦略(61%)を
  下回る。唯一効いた手がかりは「明らかに肝実質でないもの」(白背景・被膜・血管腔・
  壊死様)で、確信をもって random と判定した9枚は 9/9 正解だった。しかしこの9枚を除いた
  91枚では精度55%(ベースライン67%)で、**普通の肝実質パッチの中では完全に区別できない**。
- **(※) granular eosinophilic: 精度88%は選別の証拠ではない。** キュレート21枚のうち
  **16枚(76%)が実質的な白背景**で、判定はその16枚を「背景かどうか」で当てただけである
  (16/16、誤検出ゼロ。ランダム対照側には白背景が1枚も無かった)。**所見の形態を見分けた
  のではない**ので、glycogen / ground glass の91%とは意味が違う。しかもこの16枚の類似度は
  0.908〜0.927 と、組織パッチと同じ帯域に収まっていた。原因は下記「背景パッチ」参照。

### 結論: 目視判定は機能する。hypertrophy が不成立

glycogen がポジティブコントロールとして効いたことが決定的である。同じ判定者・同じ手順・
同じブラインド設計で glycogen は91%、hypertrophy は59%。**「非専門家の目では判定でき
ない」のではなく、hypertrophy では検索が実際に選別できていない**と切り分けられた。

したがって job 9932 の chance 比 8.7倍という数字に視覚的な裏付けはなく、**非GT寄与
スライド45枚を「未ラベル陽性の候補」として病理レビューに回す根拠は現時点で無い**。
GT スライドが25枚と多い所見では、chance 比が高く出ても選別の証拠にならないことを
示した例でもある。

留保: 判定者は病理の専門家ではない。この検証は否定側に強く(見分けられなかった=
選別されていない、は glycogen の対照があって初めて言える)、肯定側は「識別可能な
形態差がある」までしか言わない。glycogen の137枚が本当にグリコーゲン沈着かどうかは、
依然として病理レビューが必要。

### 背景除去済みコーパスでの再検証(job 10498、experiments/0019、2026-09-09)

背景除去(下記「背景パッチ」)後、対照の母集団が **100% 組織**(0002 では 2.12% が背景)に
なるので、選別の証拠がまだ立つかを `experiments/0019` の deliver 集合(0015 とほぼ同一)で
測り直した(`scripts/adhoc_random_patch_baseline.sh` の 0019 ターゲット、`--corpus-dir` を
0018 に向ける)。

- **glycogen: 84/100(84%)。** 0002 の 91% から微減。誤答16枚のうち14枚が「random を
  curated と誤判定」= 偽陽性。飽食ラット肝はグリコーゲンに富むパッチが多く、背景を
  抜いた対照プールは「淡明レース状」の当たりが増えて**ベースラインが上がった**(引き分け
  にくくなった)ため。**選別自体は依然として明確に効いている。**
- **ground glass: 判定を取り違えた。** 生の盲検判定は 20/100 = 偶然(50%)から強く外れる
  **逆相関**。判定者(Claude)が "ground glass appearance" を「濃い硝子様好酸性細胞質」と
  想定したが、実際の curated 集合は「均質で細かいテクスチャの中等度ピンク」(glycogen より
  おとなしい空胞化)で、それを "正常寄り" として random 判定していた。100 試行で 20% は
  H0(50%)から p≪0.001 で外れる → **curated 集合は一貫した識別可能な phenotype を持つ**
  (ラベルの取り違えであって、選別が効いていないのではない)。curated 例で校正し直すと
  実効分離能は約80%。ただし **glycogen と ground glass の phenotype は淡明・空胞化肝細胞
  という点で視覚的に重なり、非専門家(および本モデル)は両者を切り分けられていない**。
- granular eosinophilic は 0019 で非 seed 候補が5枚しか無く盲検シートに乗らないため対照なし
  (下記「背景パッチ」の deliver A/B 参照)。

**要点**: 背景除去後も glycogen の選別は明確(84%)。ground glass も選別自体は効いている
が、"ground glass" という所見ラベルと視覚 phenotype の対応づけは**病理専門家の確認が要る**。

## job 9962: whole-patch テクスチャ系への展開(2026-09-05)

hypertrophy が不成立と分かったので、残る候補 `Ground glass appearance`(コーパス内4枚)と
`Degeneration, granular, eosinophilic`(同5枚)を deliver で回した。**fatty change は
実行できない** — `Degeneration, fatty` は GT csv に16スライドあるが、**1000枚コーパスとの
交差がゼロ**である(脂肪変性系で唯一コーパスにある `Vacuolization, cytoplasmic` は
自己検索 best_rank 826.5 / chance 500 で chance 以下、seed にできない)。

| | ground glass | granular eos | (参考) glycogen |
|---|---|---|---|
| seed / 除外スライド | 4 | 5 | 6 |
| 除外後の候補 | 295 | **74** | 174 |
| 空白クロップ除外 | 0 (0%) | **48 (70%)** | 4 (3%) |
| 最終パッチ | **113** / 150 | **21** / 150 | 137 / 150 |
| スライドあたり分布 | `15 15 15 15 10 8 8 3 3 2 …` | `5 2 2 1 1 1 1 1 …` | `15 15 15 14 9 8 8 7 6 5 …` |
| 上限15に達したスライド | 4 / 26 | **0** / 15 | 3 / 30 |

**最終パッチ数が少ないことの意味は、どの段階で減ったかによる。**

- **ground glass の113枚は「多様性が尽きた」形。** 上位4スライドだけが `max_per_slide=15`
  に達し、残り22スライドは1〜10枚。探索を深くすれば候補は増えるが、増えるのは類似度の
  低い側で、sim 0.932〜0.951 という高い帯域は取り尽くしている可能性が高い。GT 4枚の稀な
  所見としては妥当な頭打ちで、**失敗ではない**。
- **granular の21枚は失敗。** 上限に達したスライドはゼロ、最多でも5枚。候補74枚の70%が
  空白クロップとして落ち、生き残った21枚のうち16枚も実質白背景だった(上記ランダム対照)。

なお `lib/patch_set.py` はこの時点で中間段階を記録していなかったため、上の分布は
manifest から逆算したものだった。commit `ad2318e` で `n_after_sim_floor` /
`n_after_nms_and_cap` / `n_slides_at_max_per_slide` 等を stats とログに追加した
(0014 のインライン版にはあった診断が、`lib/patch_set.py` へ切り出す際に失われていた)。

## 背景パッチ: 約39万枚をコーパスから除去(2026-09-05 診断 → 2026-09-09 実施)

granular の候補の93%が背景で埋まった原因として、2つの説明が競合した。

- **(a)** UNI v1 がこの所見の形態を背景クラスタの近くに置いている(**埋め込みの問題**。
  uni_v2 検討の論拠になる)
- **(b)** コーパスに真に似た組織が乏しく、どの所見に対しても中程度の類似度を持つ背景が
  相対的に浮上しただけ(**データ側の事実**。再埋め込みでは解決しない)

コーパスから2000パッチをランダム抽出してクロップし、背景/組織に分けて各所見の seed
クエリ(0015 と同じ作り方)に対するコサイン類似度を測った(job 9983)。

| 所見 | 背景 median | 背景 p95 | 組織 median | **組織 max** |
|---|---|---|---|---|
| glycogen ✅ | 0.510 | 0.784 | 0.706 | **0.936** |
| ground glass ✅ | 0.538 | 0.709 | 0.686 | **0.928** |
| granular eos ❌ | 0.551 | 0.782 | 0.705 | **0.894** |
| hypertrophy ❌ | 0.554 | 0.840 | 0.733 | **0.894** |

**(a) は否定された。** 背景の類似度は全所見で横並び(median 0.510〜0.554)で、granular に
対して特別高くない。むしろ hypertrophy に対して最も高い。**granular のために再埋め込みを
行う論拠は得られなかった。**

**決定的だったのは組織側の天井**である。失敗した2所見がともに 0.894 で最下位、成功した
2所見が 0.928〜0.936 ときれいに分かれた。背景が近いのではなく、**その所見に本当に似た
組織がコーパスに乏しく天井が低いため、大量にある背景の裾と重なった**。

### この測定の限界: 極値を捉えていない

ランダム2000枚から得た背景39枚の granular に対する最大値は 0.803 だったが、job 9962 の
実際の検索結果では背景が **0.908〜0.927** を出していた。コーパス18,368,337パッチのうち
背景は約1.9% = **35万枚**あり、検索はその中の最も似た個体を選ぶ。39枚の最大値と35万枚の
最大値では話が違う。**背景が「特別近い」のではなく「数が多いから裾が長い」。**

### 副産物: 現行の空白フィルタは半分以下しか検出できていない

```
background: 39/2000 (1.9%) by sat_frac; 18 (0.9%) by the current _is_blank_tile
```

`_is_blank_tile`(平均輝度 > 240 **かつ** 標準偏差 < 8)は「完全な白」しか落とせず、
彩度ベースの基準で見つかる背景の半分以下しか捕まえていない。granular の21枚中16枚が
素通りしたのと整合する。コーパス全体では**約35万枚の背景パッチが manifest に残っている**
計算になる(この 2000 枚サンプルからの外挿。全パッチ実測では 389,959 枚 = 2.12%、下記)。

これを `lib/manifest.py` で落とせば、検索が背景を引くことは原理的になくなり、全所見に
効く。**GPU での再埋め込みは不要**で、h5 の特徴量はそのまま使い manifest から該当行を
除いて FAISS 索引を再構築するだけ(`experiments/0001`→`0002` の再実行相当、CPU のみ)。
uni_v2 での再埋め込み(1000スライド分の GPU 推論)とはコストが2桁違う。

### 全パッチの彩度測定と閾値確定(scripts/measure_corpus_blankness.py、2026-09-09)

上記の 2000 枚サンプルは極値を捉えられないので、**全 18,368,337 パッチを raw WSI から
切り出して** `mean_intensity` / `std_intensity` / `sat_frac`(HSV 彩度 > 0.10 の画素割合)を
測った(job 10487、1000 スライド・0 失敗、`outputs/measure_corpus_blankness/corpus_blankness.parquet`)。
`sat_frac` の分布から閾値を決めるため、bin ごとにパッチをサンプルしてコンタクトシートに
落とす監査(`scripts/audit_blank_threshold.py`、job 10491)を行った結果:

- `sat_frac < 0.10` の bin は全域がスライド背景・切片エッジの細片・空視野中の赤血球・
  カバーガラス由来アーティファクトで、診断対象の組織は `sat_frac ≈ 0.15` から現れる。
- 当初検討していた `& mean_intensity > 215` ガードは、`sat_frac < 0.10` に暗い組織は
  存在せず、明白なゴミ(ピンぼけのグレー、カバーガラスのひび、走査端の半黒パッチ)を
  約1,500枚残すだけだったので**不採用**。基準は `sat_frac < 0.10` 単独に確定。

`lib/patch_blankness.py::is_background` がこの基準。`lib/query_embedding._is_blank_tile`
もクエリ側で同じ基準に統一した(旧「輝度+分散」判定は置換)。アトラス図版クエリの
タイル分割では白マージンは旧ルールでも既に捕捉できていたため、クエリ側の変更は
GT 比較・0015/0019 の結果に影響しない(job 10495 の `baseline_v1` が誤格納修正後の
job 10467 と完全一致)。

### experiments/0017 → 0018 → 0019: 背景除去の実行と結果(2026-09-09)

| exp | 内容 | 結果 |
|---|---|---|
| **0017** | `build_patch_manifest` に `background_blankness_path` を追加し、`sat_frac < 0.10` の行を manifest・slide_meta・学習サンプルから除外(染色正規化の除外機構と union) | 389,959 枚除外(2.12%)、17,978,378 / 18,368,337 パッチ。job 10492 |
| **0018** | 0017 の manifest から FAISS 索引を再構築(`experiments/0002` のミラー、OPQ+IVF+PQ パラメータ不変) | 17,978,378 vectors。job 10493 |
| **0019** | `experiments/0015` の deliver を 0018 コーパスで回し直す A/B(`index_exp_dir` のみ差分) | job 10497、下記 |

**GT 比較(job 10495、`scripts/validate_against_ground_truth.py` に `v1_deblank` パイプライン追加)**:
7 カテゴリすべてで中立。`best_rank` は ±1〜5 の両方向のぶれ、`found` は Hypertrophy で
GT 1枚減(25→24)以外は不変。この比較は `search_top_slides_multi` = n_hits_ratio 
ランキングなので、後述のとおり背景除去の効果が出にくい指標。

**自己検索(job 10496 / 10494、`scripts/self_retrieval_diagnostic.py` に `--index-dir`)**:
2 つのランキング指標で挙動が割れた。

- `nhr_best_rank_med`(n_hits_ratio): 16 findings すべて ±1 で不変。
- `sim_best_rank_med`(max_similarity): **一貫して改善、時に大幅**。
  granular eosinophilic 20.0→**2.0**、Single cell necrosis 125.0→**25.5**、
  Alteration cytoplasmic 67.5→**1.5**、glycogen 26.5→10.5、Swelling 12.0→2.0。
  `sim_hit@10` も広く改善(granular eos 0.4→1.0、Alteration 0.0→1.0)。

背景パッチは多くのクエリに中程度〜高い類似度を持つため、「スライドを最良マッチ1枚の
類似度で順位付け」する `sim` ランキングでは背景の多いスライドが浮上して GT スライドを
押し下げていた。背景を抜くと GT スライドの `sim` 順位が上がる。n_hits_ratio は
候補集合内のパッチ数を正規化するので元々これに頑健。**0013 発見2 の「nhr と sim の
乖離」を、背景除去は sim 側を直すことで縮める**(下記「自己検索診断」発見4 に追記)。

**deliver A/B(job 10497、0015 vs 0019、同一パラメータ)**:

| finding | 0015(0002 corpus) | 0019(0018 corpus) |
|---|---|---|
| ground glass | 候補295 → blank除外0 → 最終113(寄与26スライド) | 候補295 → 最終113(寄与26スライド、バイト一致) |
| glycogen | 候補174 → blank除外4 → 最終137(寄与30スライド) | 候補170 → 最終137(寄与30スライド) |
| **granular eosinophilic** | 候補**74** → blank除外**48** → 最終**21**(寄与15スライド) | 候補**5** → 最終**5**(寄与5スライド) |

ground glass / glycogen は実質同一(glycogen の候補 174→170 は blank だった4枚が
コーパスから消えた分)。**不変チェック通過** — 背景汚染が問題でなかった所見は乱れない。

granular eosinophilic は「配信集がクリーンになる」ではなく **候補プールの崩壊**が起きた。
0002 では非 seed 候補74枚のうち48枚(65%)が背景クロップで `_is_blank_tile` に落とされ、
残り21枚が「成果物」になっていた。0018 ではその48枚がコーパスから消えているので、
5000 候補の窓が seed スライドの自己マッチでほぼ埋まり、真の非 seed パッチは5枚だけ。
→ **この所見にはコーパスに一般化可能なシグナルがほぼ無い**ことが露呈した(自己検索でも
n_compounds=2・`batch_dominates_finding_rate`=1.0)。背景除去は誤解を招く「21枚の
成果物」を正直な「5枚、シグナル不足」に変えた。self_retrieval の `sim_best` 改善は
GT スライドの順位向上として本物だが、この所見の**配信物**は元々薄い。

ground glass / glycogen(0019)のランダム対照は取得済み(job 10498、上記「ランダム対照に
よる検証」の 2026-09-09 追記): 背景除去後も glycogen の選別は明確(判定84%)、ground glass
も phenotype は識別可能。granular は n=5 で盲検シートに乗らないため対照なし。

### 既定インデックスへの昇格(2026-09-09)

上の3つの A/B(GT 中立、self_retrieval は sim 改善・nhr 不変、deliver は不変〜露呈)で
**負の影響が無いことが確認できた**ため、0018 を既定インデックスに昇格した:
`notebooks/01_query_demo.ipynb`、`scripts/self_retrieval_diagnostic.py` /
`scripts/random_patch_baseline.py` の既定、`scripts/validate_against_ground_truth.py` の
`baseline_v1`、README 各所。0002 は「背景除去前」の参照として残し、
`validate_against_ground_truth.py` では `--pipelines baseline_v1,v1_predeblank` で A/B を
再現できる。完了済みの実験(0003, 0007–0016)の config はそのまま(実行記録として保存)。
背景除去済みコーパスでの deliver は `experiments/0019` が担う。

## 所見の3クラス分け(0014 / 0015 / ランダム対照で見えた実データの構造)

当初は whole-patch テクスチャ / sub-patch 局所の2クラスで、hypertrophy は前者として
「機能するはず」に分類していた。ランダム対照(上記)で hypertrophy が不成立と分かり、
**第3のクラスを立てる必要が出た**。granular eosinophilic はさらに別の失敗の仕方をする。

| クラス | 例 | patch レベル検索 + curation |
|---|---|---|
| whole-patch テクスチャ(パッチ内で完結) | **glycogen**(検証済み), **ground glass**(検証済み) | **機能する**(seed が動く側なら) |
| 相対的基準を要する | **hypertrophy**, 萎縮 | **機能しない**(パッチ単体に基準がない) |
| sub-patch 局所 | 増加分裂像, 単一分裂像, 小さな microgranuloma | **機能しない**(シグナルが patch の見た目を変えない) |
| コーパスカバレッジ不足 | **granular eosinophilic** | **機能しない**(類似組織が乏しく天井が低い) |

4つ目は所見自体の性質ではなく**コーパス側の事情**なので、コーパスを広げれば解消しうる
点が他の3つと違う。ただし現行コーパスでは成果物にならない見込みで、これは
`experiments/0019`(背景除去済みコーパスでの deliver)で確認された: 背景を抜くと
granular の非 seed 候補は5枚まで崩壊する(0002 時代の21枚は候補の65%が背景クロップの
プールの生存者だった)。天井 0.894 が低いだけでなく、その帯域の非 seed 組織が
そもそもコーパスにほとんど無い。上記「背景パッチ」の deliver A/B 参照。

**「相対的基準を要する」クラスが機能しない理由**: 肝細胞肥大は「正常より大きい」という
相対的な判断であり、同一スライド内の正常部という基準があって初めて成立する。224px に
切り出した単一パッチではその基準が失われるため、パッチ単体をいくら見ても(人でも
モデルでも)判定できない。glycogen の空胞化のようにパッチ内で完結して見える所見とは
性質が異なる。sub-patch 局所(シグナルが小さすぎる)とも別の失敗モードであり、
**探索の深さや curation の調整では埋められない**。扱うならパッチサイズを上げて基準組織を
同一視野に入れるか、スライド内の正常部との対比を明示的にモデル化する必要がある。

上の表の `?` は未検証を示す。fatty change と ground glass は whole-patch テクスチャに
見えるが、hypertrophy で「見た目の分類だけでは当てにならない」ことが分かったので、
0015 + ランダム対照を通すまでは確定させない。

## experiments/0016: クエリ埋め込み経路は 0014 の結論の交絡か(2026-09-07、job 10434)

0014 は「アトラス図版をクエリにすると GT スライドを引けない → アトラス→TG-GATEs の
ドメインギャップが #1 ブロッカー」と結論したが、0014 の atlas アームと
self_retrieval_diagnostic / 0015 の TG-GATEs アームは2軸で違っていた: (1) 内容/ドメイン、
(2) クエリ経路(画像 → `lib.query_embedding.embed_image(_tiles)` のタイル分割・LANCZOS
リサイズ・空白フィルタ vs h5 特徴量の直読み)。0016 は (2) を単独で測る — コーパスに
既にあるパッチを使えばドメインギャップは構成上ゼロになる。glycogen と ground glass
(検証済みの2所見)で3 tier:

- **Tier 1 (roundtrip)**: GT スライドのパッチを raw WSI から `crop_patch` → `embed_image`
  で再埋め込みし、保存済み h5 ベクトルと比較。**cos_self 中央値 1.0 / min 0.99998**
  (これらのスライドは patch_size_level0=224 で `crop_patch` がリサイズ不要のため実質
  同一ピクセル)。再埋め込みベクトルでの検索(自パッチ rank / 自スライド rank / 同一所見
  他 GT スライドの recall)は **feature アームと完全一致**。
- **Tier 2 (patchset)**: 同一 seed パッチ集合で 0015 validate を h5 seed / 画像経路 seed の
  2通り実行。glycogen・ground glass とも **両アーム完全一致**(同じ寄与スライド・同じ
  150パッチ・同じ hold-out GT recall。`feature_only` / `image_only` ともゼロ)。
- **Tier 3 (region)**: GT スライドから 14×14 パッチ(~196タイル)の level-0 リージョンを
  切り出し `embed_image_tiles` で検索 = 「アトラス図版と同じ ROI なし多タイル集約」を
  ドメイン内・倍率一致で再現。**全リージョンが自スライドを rank 1 で retrieve**、
  他 GT スライドも target hit@10 0.67〜0.75 / hit@50 0.88〜1.0。

**結論**: クエリ埋め込み経路(`embed_image` / `embed_image_tiles`・タイル分割・リサイズ・
空白フィルタ・多タイル集約)は 0014 の結論の意味ある交絡ではない。ドメイン内ピクセルを
正しい倍率で与えれば、画像経路は h5 直読みと同等に検索できる。したがって 0014 の
「アトラス図版で GT スライドを引けない」失敗は**アトラス図版そのもの**(別スキャナ・
染色・倍率、図版が非所見組織だらけ、ROI なし)に帰属でき、ツーリングの副作用ではない。
「ドメインギャップが #1 ブロッカー」は補強された。

ストレージ注: 0016 / 0013 とも exact re-rank の h5 ランダムアクセスが高頻度なので、
`run_slurm.sh` の `PRE_NATIVE_COMMAND` で index + h5 をノードローカル NVMe にステージ
してから読む(`PVS_INDEX_DIR` / `PVS_FEATURES_DIR` 等)。`USE_LOCAL_SSD_INPUT=1` は
`data/`(844G)全体を rsync するので使わない(job 10421 はこれで TIMEOUT した)。

## experiments/0020: 初代アトラス検索 Artifact の作り直し(2026-09-10、job 10501)

プロジェクト最初期に作った検索デモ Artifact — NNL アトラスの所見図版をクエリに、
所見ごとに「クエリタイル行」と「コーパスから引いた上位パッチ行(slide_id / sim)」を
並べ、「望んでいるほど似ていない画像も多く得られてしまった」で終わっていたもの —
を、この1か月の改善(背景除去 0018、評価軸の刷新、成果物パイプライン)を反映して
同じ立て付けで作り直した。

- **スイープ(`experiments/0020` / `experiment.py`、job 10501)**: NNL アトラス25所見を
  フォルダ単位で集約(所見フォルダ = 1クエリ、フォルダ内の全図版を統合)。
  `uni_v1` plain タイリング(染色正規化なし = 現行既定経路)、探索は
  `notebooks/01_query_demo.ipynb` の推奨値(`rerank_pool=200` / `max_tiles_reranked=12`)。
  各所見を **`0018`(背景除去済み・現行既定)と `0002`(背景除去前)の両索引**で検索し、
  上位10パッチを raw WSI から実解像度クロップ。所見ごとに `query_tiles/` `hits_deblank/`
  `hits_predeblank/` `finding.json` を出力。25/25 成功・クロップ失敗0件、約10分。
- **レポート(`scripts/build_atlas_report.py`、run_slurm.sh が続けて呼ぶ)**: スイープ出力 +
  `outputs/0019`(glycogen/ground glass/granular の deliver 成果物)+ 自己検索診断 CSV
  (`self_retrieval_diagnostic_{0018,0002_predeblank}.csv`)から self-contained な
  `outputs/0020_.../report.html` を生成(画像は base64 JPEG 埋め込み、外部依存は
  Google Fonts のみ)。Artifact: https://claude.ai/code/artifact/a34cf5e1-b488-4571-8027-065c20098df3
- **発見: 背景除去は max_similarity 上位パッチをほとんど動かさない**。所見別の 0018 vs 0002
  上位10枚は、組織マッチが強い所見(融合壊死・脂肪変性)では**完全一致**、背景の彩度帯に
  近い所見(色素沈着・クッパー細胞増殖)では**上位が入れ替わる**という幅。背景除去の主効果は
  ここではなく**スライド単位の `n_hits_ratio` ランキング**に出る(「背景パッチ」節の
  self_retrieval `sim_best_rank` 改善と整合。max_similarity 上位パッチは元々ほぼ組織)。
- sim レンジは 0.47〜0.71 で初代 Artifact(0.5〜0.69)と同水準。「あまり似ていない」ままの
  所見が多いのは、多くの所見で 0018 の上位が「所見と無関係な正常組織」で埋まるため —
  モデルの表現力ではなく **アトラス→TG-GATEs のドメインギャップ**が主因(0014 の #1 ブロッカー、
  0016 で補強)。
- **対照図版の除外(job 10513、`--overwrite` 再実行)**: NNL の lesion ページには一部
  「Normal liver ... for comparison with Figure N」の**対照図版**が混ざる(肝では Atrophy
  2枚・Hepatocyte - Hypertrophy 2枚の計4枚のみ)。`lib.atlas_figures.query_images()` が
  `data/query/nnl_liver_atlas_figures.csv` の `is_normal_control=yes` を見て除外する
  ようにし、0020 のクエリセットも Atrophy 4→2枚・Hypertrophy 9→7枚に絞って作り直した。
  除外対象外の23所見は sim レンジ・上位パッチとも前回(job 10504)と一致(スイープ経路は
  実質再現的。`validate_against_ground_truth.py` 側は下記の通り数枚の順位揺れが出る)。
- **運用メモ**: `USE_LOCAL_SSD_OUTPUT=0`(NFS 直書き)。=1 にすると「スイープをスキップして
  レポートだけ作り直す」再実行で `build_atlas_report.py` が空のスクラッチを見て所見出力を
  取りこぼす(job 10503 で発生、10504 で修正)。`experiment.py` は全所見完了済みなら索引
  ロードも省くので、レポートのみの再実行は1〜2分。25所見完了済みなら `PRE_NATIVE` の
  72G ステージングもスキップする。`run_slurm.sh` の `--overwrite` は反映後に外すこと
  (job 10513 後に外した。figure フォルダやクエリパラメータを変えたときだけ一時的に足す)。

## experiments/0021: 多モデル埋め込み弁別力プローブ — 融合壊死・髄外造血(2026-09-10、job 10510)

自己検索診断(下記)で uni_v1 の明確な弱点だった融合壊死(`Necrosis`)と髄外造血
(`Hematopoiesis, extramedullary`)について、他エンコーダが phenotype をより分離できるかを
**全 1,800万パッチの再埋め込みに踏み切る前に**安価に測った。埋め込みは wsi_preprocess
(GPU、`HANDOFF_multimodel_encoder_probe.md`)、弁別力の計算は patch-vector-search 側
(`experiments/0021`、索引なし、numpy のみ)。

- **probe セット**: `single_finding_liver.csv` の単一所見コーパススライド —
  necrosis 13(11 EXP)/ single cell necrosis 4(参考)/ hematopoiesis 3 — と
  confuser 106(3ターゲット以外の単一所見スライド全部)を、各スライド最大100パッチ
  (背景除外)で `uni_v1` / `hibou_l` / `uni_v2` / `virchow2` の4モデルに埋め込み
  (12,600 パッチ × 4)。TRIDENT `encoder_factory`、224px 入力、L2 正規化なし。
- **主指標 LOSO AUROC**(パッチを「その所見 vs confuser」に分類。スコア = 別スライドの
  同所見パッチとの top-10 平均コサイン。スライド単位 cluster bootstrap で 95% CI):

  | finding | uni_v1 | hibou_l | uni_v2 | virchow2 |
  |---|---|---|---|---|
  | necrosis | 0.655 [0.57–0.72] | 0.541 [0.45–0.64] | 0.703 [0.61–0.78] | 0.648 [0.58–0.72] |
  | hematopoiesis | 0.437 [0.25–0.60] | 0.471 [0.35–0.65] | 0.322 [0.24–0.41] | 0.316 [0.27–0.36] |
  | single cell necrosis | 0.744 [0.49–0.90] | 0.461 [0.28–0.65] | 0.687 [0.24–0.96] | 0.673 [0.46–0.82] |

- **結論: 否定的。フル再埋め込み(Task B)は見送り。**
  - **どのモデルも uni_v1 を CI が重ならないレベルで上回れない**。necrosis で uni_v2 の
    点推定がわずかに高い(0.703 vs 0.655)が CI は大きく重なり、しかも uni_v2 の
    within(same-EXP) 0.678 vs within(diff-EXP) 0.402 という開きは、その差が phenotype
    でなく**実験バッチ由来**であることを示す(uni_v1 は 0.561 / 0.492 で開きが小さい)。
  - **髄外造血は全モデル AUROC ≤ 0.47(chance 0.5 以下)**。3スライドと極小で決定的では
    ないが、方向は明確 — どのエンコーダも髄外造血パッチを引けない。極小・sub-patch な
    焦点性所見で、分裂像と同じ「patch レベル検索の対象外」クラスの可能性が高い。
  - hibou_l(非ヒト事前学習を含むとされる)は全所見で最下位に近く、「非ヒト事前学習が
    ラット肝に効く」仮説はこのプローブでは支持されなかった。
- **留保**: n が小さい(13 / 4 / 3 スライド)。テストしたのは4モデルのみ(phikon_v2 /
  hoptimus1 / conch / gigapath 未試験)だが、独立系統の大規模 FM である uni_v2 と
  virchow2 が**両方とも** uni_v1 を超えられなかったことが強いシグナル。プローブは
  patch レベルの弁別(検索の必要条件)を測ったもので十分条件ではないが、必要条件を
  満たせない以上フル索引でも改善しないと判断。→ この2所見の弱点はエンコーダではなく
  **コーパスカバレッジ(融合壊死スライドの化合物多様性)とタスク定義**の側にある。

## experiments/0022: 合成再スケール検索評価 — 倍率ずれの影響を正解既知で定量(2026-09-10、job 10516)

「クエリ画像の見かけ倍率がコーパス(20x)からずれると検索がどれだけ落ちるか」を、
**コーパス自身から作った正解スライド既知の合成クエリ**で測った。test-time スケール探索
(option C)に着手する前に、そもそも補正すべき劣化が存在するか・それがスケール由来か
を確かめるための前段。atlas GT(ノイズ大・de-emphasize 済み)ではこの問いに答えられない。

- **合成**: 0018 索引のコーパススライドを random 40 枚 → 各スライドの実パッチ座標に
  アンカーした 4×4 ブロックを raw WSI から「倍率比 r で撮影されたように」読み出し
  (level-0 で `round(4·P/r)` px の窓 → 4·224px にリサンプル、P = スライドごとの
  `patch_size_level0`)。120 領域 × r ∈ {0.35, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83}。
  r>1 = コーパスより拡大、r<1 = 縮小。
- **3 アーム**(同じ合成クエリ画像に対して): `fixed`(そのまま `embed_image_tiles`
  = 現行パイプライン)/ `oracle`(r 既知として 1/r リサイズしてからタイル化 = 完全補正の
  上限)/ `autoscale`(`embed_image_tiles_auto_scale` の FM centroid 投票)。
  指標は `search_top_slides_multi`(n_hits_ratio ランキング)での正解スライド順位。
- **median 順位 / found@10**:

  | | r=0.35 | r=0.5 | r=0.71 | r=1.0 | r=1.41 | r=2.0 | r=2.83 |
  |---|---|---|---|---|---|---|---|
  | fixed  median | 14.5 | 2 | 1 | 1 | 1 | 2 | 21.5 |
  | oracle median | 4 | 1 | 1 | 1 | 1 | 1 | 1 |
  | autoscale median | 4 | 1 | 1 | 1 | 1 | 1 | 1 |
  | fixed found@10 | 0.42 | 0.90 | 0.97 | 0.99 | 0.95 | 0.84 | 0.33 |
  | oracle found@10 | 0.62 | 0.97 | 0.97 | 0.99 | 0.97 | 0.93 | 0.91 |

- **発見1: UNI は ±2x の倍率ずれに対して頑健**。`fixed` は r ∈ [0.5, 2.0](4 倍幅)で
  median 順位 1〜2・found@10 ≥ 0.84。**実運用上、2 倍以内の倍率不一致は非問題**。
  r=1 サニティ(median 1・found@1 0.78)も OK。
- **発見2: 2x を超えると崩れ、その崩れはスケール由来で補正可能**。
  - r=2.83(拡大しすぎ): `fixed` median 21.5・found@10 0.33 → `oracle` median **1**・
    found@10 **0.91**。ほぼ完全に回復 = 純粋なスケール不一致。
  - r=2.0: found@10 0.84 → 0.93、mean 順位 14.1 → 5.9(裾が縮む)。
  - r=0.35(縮小しすぎ): `fixed` median 14.5 → `oracle` **4**、found@10 0.42 → 0.62。
    **部分回復のみ**。広視野クエリを固定 px にリサンプルした時点で失われる情報
    (corpus タイル相当あたりの解像度低下)は完全補正でも戻らない。
- **発見3: 点推定補正(`autoscale`)が oracle とほぼ同じ**。棚上げ時の「GT 7/7 で悪化」は
  **真の倍率が不明でコンテンツ混在の atlas 図版**で測ったもの。単一倍率・正解既知の
  合成クエリでは、投票推定は完全知識と同等に >2x の劣化を回復する。→ **フルな
  スケール探索(C)は不要。必要なのは `embed_image_tiles_auto_scale` の復活。**
  - ただし centroid グリッドが粗い({0.5,1,2,4,8})。r=0.71/1.41(真の補正 1.41/0.71)は
    最近傍 centroid にスナップして |log err| 0.35 になる。retrieval 結果には響いていない
    (その帯は `fixed` が既に良いため)が、**centroid に 0.71 / 1.41 を足すべき**。
  - r>1(拡大クエリ、補正係数 1/r<1)は centroid 下限 0.5 で頭打ち。r=2 は 1/r=0.5 で
    たまたま合うが、それ以上の拡大は当てられない。
- **この測定の限界**: 合成クエリは単一倍率かつ検索対象と同一スライド由来(in-distribution・
  完全一致パッチが必ず存在する易しいケース)なので、順位の絶対値は best-case。
  アーム間の相対比較が目的なのでそこは問題ない。→ 実際の atlas 図版でどうかは
  `experiments/0023` で検証(下記、否定的)。

## experiments/0023: autoscale 復活の検証 — 否定的、倍率補正は決着(2026-09-10、job 10517)

0022 の「点推定補正で >2x を回復できる」を受けて、`embed_image_tiles_auto_scale` を
(a) centroid 細粒度化 + (b) ガードバンド付きで復活できるか、(c) atlas GT 7 カテゴリで
再測定した(0018 索引、対照図版除外済み、4 アーム: baseline / autoscale_orig /
autoscale_fine_guard / autoscale_fine_noguard)。

- **(a) centroid 再ビルド**: `build_scale_reference_centroids` を
  scales `{0.35,0.5,0.71,1,1.41,2,2.83}` で(20 スライド × 4 サンプル)。
  `outputs/0023_.../revive/scale_centroids_fine.npz`(**昇格せず**)。
- **(a-val) 推定精度**(合成再スケールクエリ・検索なし、旧 vs 新 centroid の
  `|log(scale_est / (1/r))|` median): 全体 median は旧 0.049 → 新 0.010 と改善するが、
  **内訳はまだら** — 新は r=0.71/2.83 を直す一方 r=0.5 を新たに悪化させ(0.00→0.35)、
  r=1.41 は両方 0.34 のまま。細粒度化で個々の投票のノイズが増える。
- **(c) atlas GT best_rank(baseline → 各アーム)**:

  | category | n_gt | baseline | orig | fine_guard | fine_noguard |
  |---|---|---|---|---|---|
  | Hypertrophy | 25 | 30 | **23** | 30 | 30 |
  | Single cell necrosis | 4 | 14 | 13 | 13 | 13 |
  | Increased mitosis | 10 | 10 | 9 | 9 | 9 |
  | **Deposit, glycogen** | 6 | **7** | **113** | **101** | **101** |
  | Hematopoiesis, extramedullary | 3 | 28 | 25 | **51** | **51** |
  | Proliferation, Kupffer cell | 2 | 78 | **33** | **33** | **33** |
  | Inclusion body | 1 | 475 | 消失 | 消失 | 消失 |

  勝敗(±5 位はノイズで引き分け): orig **2勝1敗**、fine_guard/fine_noguard **1勝2敗**
  (+ Inclusion body は全アームで GT スライドが top-1000 から消失、found 1→0)。

- **結論: 否定的。autoscale は棚上げ継続。** 決め手:
  - **推定器は病理テクスチャの粗さを低倍率と誤認する**。Glycogen 図版の推定スケールは
    `[1.0, 0.5, 0.5, 0.5]`(= 「2x 拡大されている」)で 2x 縮小補正がかかるが、baseline
    best_rank 7 が示す通り実際はほぼ正しいスケール。グリコーゲンで膨れた明るい
    hepatocyte が「粗い = 高倍率」と読まれる(fatty change の既知失敗と同型、
    `lib/mpp_estimation.py` docstring 参照)。結果 best_rank 7 → 100+ に破壊。
  - **推定が bimodal でガードバンドが不発**。7 カテゴリの全図版の推定は 0.35 / 0.5 /
    2.0 / 2.83 ばかりで 1.0 付近(ガード帯 [0.625, 1.6])をまず出さない。だから
    `fine_guard` と `fine_noguard` の結果が完全一致(ガードが一度も発火しない)。
  - **細粒度化は逆効果**。synthetic の推定 median は良くなるが、atlas 図版では
    0.5→0.35 / 2.0→2.83 とより極端な補正になり、旧 centroid より悪化。
  - 「勝ち」の 2 つも脆い: Kupffer は GT 2 枚・1 化合物でバッチと分離不能、
    Hypertrophy 30→23 は atlas GT の ±5 ノイズ + 単発の域。
- **これで倍率スレッドは閉じる**。0022 = `fixed` は ±2x に頑健、0023 = それを超える
  領域の自動推定は out-of-domain クエリでは信用できない。要補正なら既知係数で手動。

## experiments/0024: 類似度指標の再ランク比較 — 異方性除去 / hubness 補正(2026-09-10、jobs 10525/10534)

引き継ぎ資料の「類似度の算出方法はほかにあるのか」。現行の「L2 正規化コサイン」には
2 つの既知の弱点があり(下記)、それを self_retrieval と同じ土俵(コーパス内 LOO、
atlas なし、正解 = 同一 FINDING_TYPE の別スライド)で、**FAISS を使わずスライドあたり
100 パッチの厳密行列**で 9 アーム A/B した(GPU 不要、9 分)。

- **弱点1: 異方性**。UNI 埋め込みは球面に一様分布せず、平均ベクトル μ のノルムが大きい
  → 任意 2 パッチのコサインが軒並み 0.4-0.6(atlas の「0.5..0.69」の下駄)。ただし
  **測ってみると軽度** —— 共分散の第1主成分の寄与率は 9.1%、上位8個で 40%(単一の
  支配的 nuisance 軸は無い)。
- **弱点2: hubness**。一部のコーパスパッチ(分布中心に近い「ありふれた正常肝細胞」)が
  多数のクエリの kNN に出現し、所見パッチの枠を奪う(0007-0009 で文書化済み)。
- **アーム**: `center`(μ 除去)/ `abtt{1,2,4}`(上位 d 主成分を除去、Mu & Viswanath 2018)/
  `whiten`(ZCA 白色化 `W=(Σ+εI)^{-1/2}`)/ `csls`(Conneau et al. 2018、
  `csls(q,c)=2cos − r_T(c) − r_S(q)`、r_T = c のコーパス内 kNN 平均コサインで hub にペナルティ)/
  `csls_abtt2` / `csls_whiten`。

- **所見横断 median best_rank(低いほど良い)**:

  | arm | sim (max_similarity) | **nhr (n_hits_ratio proxy)** |
  |---|---|---|
  | cosine(baseline) | 62.5 | 86.2 |
  | abtt4 | 56.0 | 43.2 |
  | **whiten** | 52.5 | **42.0** |
  | csls | **39.5** | 66.8 |
  | csls_abtt2 | **37.2** | 61.0 |
  | csls_whiten | 51.5 | 45.5 |

- **発見: 2 つの対策が別々の失敗モードを叩き、併用では stack しない**。
  - **CSLS は `sim`(max_similarity)ランキングを 62.5 → 37-40**(約 40% 減)。hubness 補正が
    予測通り効く。
  - **whiten / abtt4 は `nhr`(n_hits_ratio、本番のランキングキー)を 86 → 42-43**(約 50% 減)。
  - `csls_whiten` は両者の良いとこ取りにならない(sim 51.5・nhr 45.5)。白色化で
    hub 構造が既に崩れているため CSLS の伸びしろが消える。
- **最大の伸びは repo が「壊れている」と分類していた所見**(nhr、best_rank median):

  | finding | cosine | 最良アーム |
  |---|---|---|
  | Hematopoiesis, extramedullary | **1000(未検出)** | abtt4: **35** |
  | Degeneration, granular, eosinophilic | 160 | whiten: **11** |
  | Single cell necrosis | 124 | csls_whiten: **52** |
  | Necrosis(凝固壊死) | 48 | abtt4/center: 36 |
  | Hypertrophy | 41 | whiten: 26 |

  → **「融合壊死・髄外造血はエンコーダの本物の弱点」という自己検索診断の結論は一部修正が
  必要** —— その相当部分は**エンコーダではなく類似度指標の hubness/異方性**だった。
  コーパスカバレッジで詰んでいる所見(Kupffer, Lesion NOS, Vacuolization/Alteration
  cytoplasmic)は全アーム 1000 のまま(想定通り、指標では動かない)。
- **本番ランキングキーは n_hits_ratio なので `whiten` を採る**。`abtt4`(行列逆計算・ε
  不要、上位 4 成分を引くだけ)を安価な代替として A/B。回帰リスクは低い(whiten は
  Swelling 12→13 以外ほぼ中立〜改善)。
- **留保**: コーパス部分サンプル(100/slide)・nhr は proxy(top-hit_k)・所見あたり
  ~10 クエリスライドの median(Change eosinophilic は 175〜1000 で振れる)。本番
  パイプラインでの確認が必須。
- **次**: `whiten` / `abtt4` を索引再構築時のコーパス変換として入れてフル測定 → `experiments/0025`。

## experiments/0025: 異方性除去変換を索引にベイクしてフル測定(2026-09-10、job 10546)

`experiments/0024` の whiten / abtt4 を、コーパス部分サンプルの厳密行列ではなく
**本番の OPQ+IVF+PQ 索引にベイク**(`lib.embedding_transform.make_faiss_pretransform` →
`faiss.LinearTransform` + `NormalizationTransform` を index チェーンに prepend。
`faiss.write_index`/`read_index` で変換ごと保存・復元され、**add/search 双方に自動適用**
= `self_retrieval_diagnostic` も `validate_against_ground_truth` もコード変更ゼロ)して
フル 18M パッチで再構築(`outputs/0017` の training_sample から fit、0018 と同一ハイパラ)。
`scripts/validate_against_ground_truth.py` に `--index-dir` を追加。

- **self_retrieval_diagnostic(フル FAISS、`nhr_best_rank_med_noexp` = バッチ交絡除外の主指標)**:

  | finding | baseline(0018) | whiten | abtt4 |
  |---|---|---|---|
  | Degeneration, granular, eosinophilic | 298 | **20** | 54 |
  | Hypertrophy | 59.5 | **38.5** | 50 |
  | Necrosis | 52 | **34.5** | **31** |
  | Increased mitosis | 59.5 | **39** | 48.5 |
  | Deposit, glycogen | 21 | **11.5** | 13 |
  | Ground glass appearance | 17 | **8.5** | 14 |
  | Hematopoiesis, extramedullary | 125 | **91** | 98 |
  | Change, eosinophilic | 146 | **239** ✗ | 267 ✗ |
  | Cellular infiltration | 12 | **23** ✗ | 22 ✗ |
  | Single cell necrosis | 44 | 45.5 | 63.5 ✗ |

  **whiten: 7 所見改善 / 2 所見悪化 / 残り中立** —— 主指標では正味プラス。特に
  granular(298→20)・壊死・肥大・glycogen・ground glass・髄外造血で大きい。
  **abtt4 は 3 改善 / 3 悪化とほぼ相殺、whiten に劣る**(0024 の部分サンプルでは両者
  近かったが、フル FAISS では差がついた)。
- **atlas GT(7 カテゴリ、de-emphasize 済み・±5 ノイズ)は whiten で悪化**:

  | category | baseline | whiten | abtt4 |
  |---|---|---|---|
  | Hypertrophy | 30 | 21 | 49 |
  | Single cell necrosis | 14 | **45** ✗ | 52 ✗ |
  | Hematopoiesis, extramedullary | 28 | **109** ✗✗ | 179 ✗✗ |
  | Deposit, glycogen | 7 | **15** ✗ | 12 |
  | Kupffer | 78 | 72 | 48 |

  (ただし `found` はむしろ増える: whiten は全カテゴリで GT 全数を top-1000 内に retrieve、
  baseline は Hypertrophy 24/25・mitosis 9/10。)
- **機序**: 白色化は共分散を等方化する = 小さい固有値の方向を `1/√λ` 倍に増幅する。
  コーパス内クエリでは埋もれた病理シグナルが浮上する(self_retrieval で改善)が、
  **atlas 図版のように corpus-μ から遠い out-of-distribution クエリでは、その増幅が
  ドメインギャップのノイズを増幅する**(atlas GT で悪化)。self_retrieval の
  Change eos / Cellular infiltration の悪化も、白色化が効きすぎて元のジオメトリの
  有用な部分まで崩している兆候。
- **結論: 有望だが既定索引への昇格は保留**。whiten は主指標(コーパス内検索)で
  難しい所見を大きく改善するが、(a) 動いていた common 所見 2 つを悪化、(b) atlas
  クエリを悪化。次の選択肢:
  - **収縮白色化**(identity と full-whiten を α で内挿: `(αΛ + (1−α)·meanλ·I + εI)^{-1/2}`、
    α ∈ {0.3,0.5,0.7})で効きすぎを抑える → `experiments/0026`(0025 の infra 再利用、W だけ差し替え)
  - **PCA 白色化 + 裾切り**(上位 k 次元だけ白色化)
  - **whiten_v1 を成果物トラック専用の索引に**(0015/0019 の per-finding パッチ集は
    コーパス内処理で atlas クエリ無し → whiten の恩恵だけ受けられる。特に granular を再挑戦)。
    対話/atlas デモは 0018 のまま。
- **昇格時の TODO**: `lib.search._exact_similarity`(パッチ再ランク、生 h5 を読む)にも
  同じ変換を通すこと。0025 の測定は slide ランキング(`search_top_slides_multi`、h5 再ランク
  なし)のみなので現状は未対応。
- **既知の小欠陥**: `lib.faiss_index` は `logging.getLogger("lib.faiss_index")` に出力するが
  実験側は `exp****` ロガーにしかハンドラを付けないので、索引構築の進捗ログ
  (「added X/Y slides」等)が experiment.log に出ない(0018 でも同様)。

## 次の一手(成果物トラック)

1. **glycogen deliver の137枚(job 9701)を病理知識のある人にレビューしてもらう** —
   採用率・多様性・所見レンジのカバー。GT スライドを全除外して未ラベルスライドだけから
   組んだ集合なので、「本当にグリコーゲン沈着か」がそのまま成果物の妥当性になる。
   ランダム対照で「検索が何かを選別している」ことは確認済み(判定精度91%)なので、
   残る問いは「選別しているそれがグリコーゲン沈着か」に絞られた
2. **ground glass deliver の113枚も病理レビューに回す** — 2例目の「機能する所見」
   (判定精度91%・recall 100%)。glycogen と同じく未ラベルスライドのみから組んだ集合
3. ~~**背景パッチをコーパスから落とす**~~ **(2026-09-09 完了、`experiments/0017`→`0018`)** —
   `sat_frac < 0.10` の背景 389,959 枚(2.12%)を manifest から除外し索引を再構築(GPU 不要)。
   `_is_blank_tile` もクエリ側で同基準に統一。max_similarity ランキングの自己検索順位は
   顕著に改善したが(上記「背景パッチ」)、GT 比較・n_hits_ratio ランキングは中立、
   granular は「候補プールの崩壊」で成果物化せず(コーパスに一般化可能なシグナルが
   ほぼ無いことが露呈)。ground glass / glycogen の配信集は不変。
   → 残タスク: 0019 のランダム対照、0018 コーパスを既定に昇格するかの判断
4. **deliver モードの候補枯れ対策** — glycogen 137枚、ground glass 113枚と、いずれも
   target 150 に未達。ただし ground glass は「多様性が尽きた」形で失敗ではないため、
   `k_candidate_patches` を上げる前に、その所見のコーパス内スライド数から**現実的な
   目標枚数を見積もる**ほうが筋が良い(上記「job 9962」参照)
5. hypertrophy を扱うなら、パッチサイズを上げて基準組織を同一視野に入れる等、
   **相対的基準の問題そのもの**に手を付ける必要がある(上記「所見の3クラス分け」)。
   現行の 224px パッチ検索の延長線上には無い
6. **fatty change は現行コーパスでは実行不可能** — `Degeneration, fatty` の16スライドは
   1枚もコーパスに無い。やるならコーパス側にスライドを足す必要がある
5. アトラス seed の道(0014)は track 2(アトラス→TG-GATEs のドメインギャップを詰める)が前提
6. mitosis 系(sub-patch)を扱うなら別アプローチ(細胞検出 → パッチ内カウント等)が必要で、
   現行の埋め込み検索の枠外

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
