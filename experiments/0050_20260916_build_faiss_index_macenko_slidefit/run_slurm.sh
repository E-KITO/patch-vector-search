#!/bin/bash
#SBATCH --job-name=0050_20260916_build_faiss_index_macenko_slidefit
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0050_20260916_build_faiss_index_macenko_slidefit/%j_0050_20260916_build_faiss_index_macenko_slidefit.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0050_20260916_build_faiss_index_macenko_slidefit/%j_0050_20260916_build_faiss_index_macenko_slidefit.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# CPU-only: FAISSのKMeans/OPQ/PQ学習+全~1000ファイルを読んでadd_with_idsする
# 重量級パス(I/O支配)。0012/0018/0033と同一設定(nlist=4096,pq_m=64,pq_nbits=8,
# opq_niter=10)。experiments/0049(スライド単位fitコーパスの背景除去済み
# manifest)が先に完了している必要がある。GPU不要。
#
# 依存ジョブとして投入する場合(手動、0049完了後に0049のjob_idを埋める):
# #SBATCH --dependency=afterok:<0049のjob_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0050_20260916_build_faiss_index_macenko_slidefit"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
