#!/bin/bash
#SBATCH --job-name=0022_20260910_synthetic_rescale_eval
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0022_20260910_synthetic_rescale_eval/%j_0022_20260910_synthetic_rescale_eval.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0022_20260910_synthetic_rescale_eval/%j_0022_20260910_synthetic_rescale_eval.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# クエリ倍率がコーパス 20x からずれたときの検索劣化を、コーパス由来の合成クエリ
# (正解スライド既知) で測る。fixed / oracle / autoscale の 3 アームで
# search_top_slides_multi の正解スライド順位を比較する。
#
# GPU: 合成クエリタイルの uni_v1 埋め込みに必要。
# ⚠️ リソース (--gres/--cpus-per-task/--mem/--time) を変えたら --partition と
#    --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0022_20260910_synthetic_rescale_eval"

# =====================================================
# Storage
# =====================================================
# search_top_slides_multi は FAISS 索引 + manifest + slide_meta しか触らない
# (h5 の exact re-rank はしない) ので、72G features はステージ不要。索引一式
# (index.faiss 1.3G + manifest 118M) だけノードローカル NVMe に置く。
# 生 WSI は openslide がヘッダ + 領域窓しか読まないので NFS 直読み。
USE_LOCAL_SSD_INPUT=0
# 出力は results.csv / summary.md / curve.json のみ。NFS 直書き
# (experiment.py が results.csv を領域ごとに書き直すので resume も NFS 側で効く)。
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
# experiment.py は results.csv を領域ごとに書き、resume 時は (slide,region) 済みを
# スキップするので TIMEOUT しても再投入で続きから走る。
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
