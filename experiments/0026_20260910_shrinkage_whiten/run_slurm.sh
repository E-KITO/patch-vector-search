#!/bin/bash
#SBATCH --job-name=0026_20260910_shrinkage_whiten
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0026_20260910_shrinkage_whiten/%j_0026_20260910_shrinkage_whiten.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0026_20260910_shrinkage_whiten/%j_0026_20260910_shrinkage_whiten.out
#SBATCH --signal=B:USR1@180
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=24:00:00
# 収縮白色化 α ∈ {0.25,0.5,0.75} を FAISS 索引にベイクしてフル 18M パッチで再構築し、
# self_retrieval_diagnostic + atlas GT で baseline(0018)/ full whiten(0025)と並べる。
# experiments/0025 と同じ構成: Stage 1-2 CPU(索引 3 本、各 ~31分)、Stage 3 CPU、Stage 4 GPU。
# 各ステージ冪等。partition 時間上限: small 1h / medium 2h / large 4h / x-large 無制限。
# ⚠️ リソースを変えたら --partition / --signal も手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0026_20260910_shrinkage_whiten"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _NIDX=$(find "${PROJECT_ROOT}/outputs/${EXP_NAME}" -name index.faiss 2>/dev/null | wc -l)
  if [ "${_NIDX}" -ge 3 ]; then
    echo "[stage] skip — ${_NIDX} indexes already built (resume: measurement only)"
  else
    _S="${SCRATCH_DIR}/staged/features"
    mkdir -p "${_S}"
    echo "[stage] uni_v1 per-slide h5 (~72G) -> ${_S}"
    rsync -a "${PROJECT_ROOT}/data/trident_processed/20x_224px_0px_overlap/features_uni_v1/" "${_S}/"
    export PVS_FEATURES_DIR="${_S}"
    echo "[stage] done: $(du -sh "${_S}" 2>/dev/null | cut -f1 || true)"
  fi
'

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
