#!/bin/bash
#SBATCH --job-name=0016_20260907_image_query_path_diagnostic
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0016_20260907_image_query_path_diagnostic/%j_0016_20260907_image_query_path_diagnostic.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0016_20260907_image_query_path_diagnostic/%j_0016_20260907_image_query_path_diagnostic.out
#SBATCH --signal=B:USR1@180
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=12:00:00
# GPU: Tier 1/2/3 とも raw WSI からパッチを切り出して UNI(embed_image /
# embed_image_tiles)で再埋め込みするため必要。重いのは Tier 2 の
# build_patch_set(rerank_pool=5000・全クエリタイル厳密再ランキング ×
# 2 arm × 2 finding、0015 mitosis で ~24 分)と Tier 1c の
# search_similar_patches ループ。実計算は 2〜3 時間見込み。
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと。
#          partition ごとの時間上限: small-creator-i=1h / medium-creator-i=2h /
#          large-creator-i=4h / x-large-creator-i=無制限(--time 指定は必須)。
#
# job 10421 は USE_LOCAL_SSD_INPUT=1 のせいで data/(844G, 生 WSI 634G 込み)を
# ジョブ開始前に丸ごと rsync し、それだけで 5〜6h の壁を使い切って TIMEOUT した。
# 下記のとおり targeted staging に変更済み。

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0016_20260907_image_query_path_diagnostic/%A_%a_0016_20260907_image_query_path_diagnostic.out
# #SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0016_20260907_image_query_path_diagnostic/%A_%a_0016_20260907_image_query_path_diagnostic.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0016_20260907_image_query_path_diagnostic"

# =====================================================
# Storage — filesrv01 の HDD 障害調査中につき、計算中の高頻度 I/O は
# NFS ではなく計算ノードのローカル NVMe(/scratch)で行う。
#
# USE_LOCAL_SSD_INPUT=1 は data/(844G, 生 WSI 634G 込み)全体を rsync するため
# 使わない(job 10421 の TIMEOUT 原因)。代わりに PRE_NATIVE_COMMAND で必要な
# ものだけ(baseline v1 index 1.3G + baseline v1 h5 72G + gt_csv)を
# ${SCRATCH_DIR}/staged/ にステージし、experiment.py に PVS_INDEX_DIR /
# PVS_FEATURES_DIR で読ませる。exact re-rank の h5 ランダムアクセス
# (Tier 1c / Tier 2)がここに乗る。
# 生 WSI は 634G でステージ不可 — Tier 1/2/3 の crop_patch / read_region は
# 全体で数千回・~10〜50 スライドに限られる(0014/0015 と同程度)ので NFS 直読み。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _S="${SCRATCH_DIR}/staged"
  mkdir -p "${_S}"
  echo "[stage] FAISS index (baseline v1) -> ${_S}/index"
  rsync -a "${PROJECT_ROOT}/outputs/0002_20260808_build_faiss_index/default/" "${_S}/index/"
  echo "[stage] baseline v1 per-slide h5 features (~72G) -> ${_S}/features"
  rsync -a "${PROJECT_ROOT}/data/trident_processed/20x_224px_0px_overlap/features_uni_v1/" "${_S}/features/"
  export PVS_INDEX_DIR="${_S}/index"
  export PVS_FEATURES_DIR="${_S}/features"
  echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
'

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
# )
# GRID_VALUES=(
#     "bert roberta"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
