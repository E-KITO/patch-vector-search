# wsi_preprocess への依頼: 多モデル埋め込みプローブ + (条件付き) フル再埋め込み

作成: 2026-09-10 / patch-vector-search 側エージェント
宛先: wsi_preprocess capsule のエージェント

---

## 0. 前提と分担

- **背景**: patch-vector-search の自己検索診断 (`scripts/self_retrieval_diagnostic.py`、
  `outputs/gt_validations/self_retrieval_diagnostic_0018.csv`) で、uni_v1 + 現行コーパスが
  バッチ交絡を除いても「検索に足る」と示せたのは 5〜6 の common な肝所見。**明確な例外が
  融合壊死 (`Necrosis`) と髄外造血 (`Hematopoiesis`)** で、この 2 所見だけがエンコーダの
  表現の弱点に見える。他モデルがこの 2 所見でだけ uni_v1 を上回るかを、**全 1,800万パッチの
  再埋め込みに踏み切る前に**安価に確かめたい。
- **分担**:
  - **wsi_preprocess (あなた)** = パッチをエンコーダに通す GPU 推論。TRIDENT の
    `encoder_factory` を使う。
  - **patch-vector-search (私)** = 出てきた特徴量で manifest / FAISS 索引 / 評価
    (自己検索診断・GT 比較・deliver) を回す。
- **このリポジトリ (`patch-vector-search`) を bind した場合の規律**:
  **data 専用**。`data/**` への h5 書き込みのみ。**git 操作・コード編集・実験ディレクトリ
  作成はしないこと**(私が別 capsule で同時に触るため。`.agentrun.lock` は各 capsule の自
  repo にしか効かない)。

---

## Task A: 多モデル弁別力プローブ (まずこれ)

索引は作らない。少数のパッチを各エンコーダで埋め込むだけ。私が弁別力メトリクスを計算し、
uni_v1 を明確に上回るモデルがあればフル (Task B) に進む。無ければ打ち切り。

### A-1. 候補エンコーダ

| trident `encoder_factory` 名 | モデル | dim (目安) | 備考 |
|---|---|---|---|
| `uni_v1` | UNI (現行既定) | 1024 | **既存の `data/trident_processed/20x_224px_0px_overlap/features_uni_v1/*.h5` を流用可。再埋め込み不要** |
| `uni_v2` | UNI2 (ViT-g, MahmoodLab) | 1536 | HF gated (ライセンス承認要)。native 256px |
| `virchow2` | Virchow2 (ViT-H, Paige, ~310万 WSI) | 1280 or 2560 | HF gated。本命 (別ラボ・最大規模) |
| `hibou_l` | Hibou-L (Histai, DINOv2) | 1024 | **事前学習に非ヒト/獣医組織を含むとされる** → ラット肝に効く可能性。オープン (gating なし) |

- gating が通らないモデルはスキップして構わない (どれが通ったか報告してください)。
- `phikon_v2` / `conch_v1` は予備。余力があれば追加。
- **入力パッチサイズ**: コーパスパッチは 224px (`20x_224px_0px_overlap`)。各エンコーダには
  224px パッチをそのモデルの期待入力へ (TRIDENT の transform 任せで) リサイズして通す。
  実際に何 px を食わせたかを meta に記録してください。

### A-2. プローブ用パッチセットの作り方

`patch-vector-search` の以下から決定的に生成する (seed=42):

1. `data/processed_csv/single_finding_liver.csv` を読む。`slide_id = image_id`
   から `.svs` を除去。
2. コーパス内スライドに限定: `data/trident_processed/20x_224px_0px_overlap/features_uni_v1/{slide_id}.h5`
   が存在するものだけ。
3. **陽性スライド**: `FINDING_TYPE` が
   - `"Necrosis"` → group `necrosis`
   - `"Single cell necrosis"` → group `single_cell_necrosis`
   - `"Hematopoiesis"` → group `hematopoiesis`
   のいずれか、かつ**そのスライドが単一 FINDING_TYPE** (CSV 内でそのスライドに紐づく
   `FINDING_TYPE` の集合が 1 要素) のものだけ。目安: necrosis ~13、single cell ~4、
   hematopoiesis ~3 スライド。
4. **confuser スライド**: 上記 3 グループの FINDING_TYPE を **持たない**コーパススライドから
   一様ランダムに 60 枚 (seed=42)。group `confuser`。
5. **各スライドから 100 パッチを一様ランダムサンプル**:
   - 座標は `data/trident_processed/20x_224px_0px_overlap/features_uni_v1/{slide_id}.h5`
     の行順 (= `patches/{slide_id}_patches.h5` と同順、= manifest 行順) に対応。
     行インデックスをランダムに 100 個引く。
   - **背景パッチを除外**: `lib/patch_blankness.py::is_background(blankness_metrics(img)["sat_frac"])`
     が True のものを落とす (パッチ画素は `data/trident_processed/20x_224px_0px_overlap/patches/{slide_id}_patches.h5`、
     または `data/moo_collected_tggate_wsi/raw_wsi/{slide_id}.svs` から `lib/raw_patch.py::crop_patch`
     で 224px 切り出し)。100 に満たなければあるだけ。
6. `probe/probe_patches.csv` を書く。列:
   `row_idx,slide_id,coord_x,coord_y,group,finding_type,exp_id`
   (`exp_id` = CSV の `EXP_ID`。私がバッチ交絡チェックに使う)。
   `row_idx` は 0..N-1 の通し番号 = 全 h5 の行順の正。

### A-3. 埋め込みと出力

各エンコーダについて (uni_v1 含む、uni_v1 は既存 h5 から該当行を引くだけでよい):

- `probe/embeddings_{encoder}.h5`
  - dataset `features`: shape `(N, dim)` float32。**行順は `probe_patches.csv` の `row_idx` と一致**。
  - **L2 正規化はしない** (生のエンコーダ出力。`features_uni_v1` と同じ規約 —
    patch-vector-search 側が検索時に normalize する)。
- `probe/meta_{encoder}.json`
  - `encoder_name` (trident id)、`dim`、`hf_repo` / `revision` or commit、
    `input_px` (実際に食わせたサイズ)、`hf_gated` (bool)、`n_rows`、
    `notes` (native パッチサイズ、正規化の有無 等)。

### A-4. 出力先

`data/trident_processed/20x_224px_0px_overlap/probe/` に置いてください
(`patch-vector-search` を bind していれば同じパス。していなければ NFS 上の任意の場所を
私に教えてくれれば `features_dir` をそこに向けます)。

### A-5. この後私がやること

`experiments/0021` として、各 `embeddings_{encoder}.h5` を読み、所見グループごとに
「同一所見・別スライド間のコサイン類似度」対「所見 vs confuser のコサイン類似度」の
マージン、および所見パッチをクエリにした precision@10 (同一所見の割合、同一スライド除外) を
計算。エンコーダ横断で比較し、necrosis / hematopoiesis で uni_v1 を明確に上回るものを判定。

---

## Task B: フル再埋め込み (Task A で勝者が出た場合のみ、私の合図後)

勝者エンコーダ 1 本で全 1000 スライドのパッチを再埋め込みし、`features_uni_v1` と
**バイト単位で対応の取れるレイアウト**にする。

- 出力: `data/trident_processed/20x_224px_0px_overlap/features_{encoder}/{slide_id}.h5`
  - dataset `features`: shape `(n_patches_slide, dim)` float32、**未正規化**。
  - **行順 = `patches/{slide_id}_patches.h5` の格納順 = 既存 `features_uni_v1/{slide_id}.h5`
    の行順 = manifest.parquet / slide_meta.parquet の行順**。これがずれると既存の座標・
    manifest が使えなくなる。TRIDENT の Step3 を同じ `coords_dir=20x_224px_0px_overlap`
    で回せば自動的に揃うはず。
- `_config_feats_{encoder}.json` を `_config_feats_uni_v1.json` と同じ様式で併置。
- 1000 スライド全部。欠けると私の索引ビルドが fail-fast で止まる。
- この後私が: `experiments/0021` で manifest (`lib/manifest.py`) → FAISS 索引
  (`lib/faiss_index.py`、0002/0018 と同じ OPQ+IVF+PQ パラメータ) → 評価バッテリー
  (`self_retrieval_diagnostic.py --index-dir` / `validate_against_ground_truth.py` /
  necrosis の `experiments/0015` deliver + `random_patch_baseline.py`)。

---

## 確認したいこと (先に教えてください)

1. **HF gating**: `uni_v2` / `virchow2` のライセンス承認は取得済み or 取得可能か。
   無理なものはスキップで OK (どれが通ったか報告を)。
2. **パッチ画素の入手経路**: あなたの側に patch cache
   (`.../20x_224px_0px_overlap/patches/`) or raw WSI の自前コピーがあるか、
   それとも bind した patch-vector-search の `data/` を読むか。
3. **compute 見積**: Task A は数千パッチ × 3〜4 モデル (~1 GPU 時間以下)。
   Task B は 1,800万パッチ × 1 モデル (uni_v1 のコーパス構築と同規模)。後者に入る前に
   私から合図します。
4. **進捗の受け渡し**: 完了したら `data/.../probe/` に置いて、この repo の
   `HANDOFF_multimodel_encoder_probe.md` に追記するか、私 (patch-vector-search capsule) に
   直接メッセージをください。
