#!/bin/bash
#SBATCH --job-name=0025_20260910_whiten_index_rebuild
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0025_20260910_whiten_index_rebuild/%j_0025_20260910_whiten_index_rebuild.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0025_20260910_whiten_index_rebuild/%j_0025_20260910_whiten_index_rebuild.out
#SBATCH --signal=B:USR1@180
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=24:00:00
# 異方性除去変換 (ZCA 白色化 / ABTT) を FAISS 索引にベイクしてフル 18M パッチで
# 再構築し、self_retrieval_diagnostic + atlas GT で baseline (0018) と A/B する。
#   Stage 1-2 (CPU): 変換 fit + OPQ+IVF+PQ 索引を 2 本再構築 (I/O 支配、各 ~30-60分)
#   Stage 3   (CPU): self_retrieval_diagnostic --index-dir (生 h5 ベクトル、encoder 不要)
#   Stage 4   (GPU): validate_against_ground_truth --index-dir (atlas 図版を uni_v1 埋め込み)
# 各ステージは出力の存在で冪等。TIMEOUT しても再投入で続きから。
# partition 時間上限: small 1h / medium 2h / large 4h / x-large 無制限。
# ⚠️ リソースを変えたら --partition / --signal も手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0025_20260910_whiten_index_rebuild"

# =====================================================
# Storage
# =====================================================
# 索引再構築は全 1000 h5 (72G) をフルスキャンして add_with_ids する I/O 支配パス。
# 2 本ぶん読むのでノードローカル NVMe にステージしてから読む。
# (self_retrieval のクエリベクトルと atlas 図版は少量なので NFS 直読み。)
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PRE_NATIVE_COMMAND='
  set -euo pipefail
  _NIDX=$(find "${PROJECT_ROOT}/outputs/${EXP_NAME}" -name index.faiss 2>/dev/null | wc -l)
  if [ "${_NIDX}" -ge 2 ]; then
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
