#!/bin/bash
#SBATCH --job-name=0023_20260910_autoscale_revive
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0023_20260910_autoscale_revive/%j_0023_20260910_autoscale_revive.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0023_20260910_autoscale_revive/%j_0023_20260910_autoscale_revive.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# 棚上げされていた倍率補正 (embed_image_tiles_auto_scale) を復活させられるか検証する。
#   (a)     細かい scale centroid を再ビルド (build_scale_reference_centroids)
#   (a-val) 合成再スケールクエリで 旧 vs 新 centroid の推定誤差を比較 (検索なし)
#   (b)+(c) atlas GT 7 カテゴリを baseline / autoscale_orig / autoscale_fine_guard /
#           autoscale_fine_noguard の 4 アームで再測定
#
# GPU: centroid 合成と全クエリのタイル埋め込み (uni_v1) に必要。
# ⚠️ リソースを変えたら --partition / --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0023_20260910_autoscale_revive"

# =====================================================
# Storage
# =====================================================
# search_top_slides_multi は FAISS 索引 + manifest + slide_meta しか触らないので
# 72G features は不要。索引一式 (1.3G) だけ NVMe に。centroid 合成の生 WSI 読みと
# atlas 図版 JPEG は openslide/PIL がヘッダ + 領域しか読まないので NFS 直読み。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _SRC="${PROJECT_ROOT}/outputs/0018_20260909_build_faiss_index_deblank/default"
  _S="${SCRATCH_DIR}/staged/index"
  mkdir -p "${_S}"
  echo "[stage] 0018 index -> ${_S}"
  rsync -a "${_SRC}/" "${_S}/"
  export PVS_INDEX_DIR="${_S}"
  echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
'

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

RUN_MODE="single"
# 3 ステップとも成果物 (npz / CSV) の存在で冪等。TIMEOUT しても再投入で続きから。
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
