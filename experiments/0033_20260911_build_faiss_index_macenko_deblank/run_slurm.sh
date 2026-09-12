#!/bin/bash
#SBATCH --job-name=0033_20260911_build_faiss_index_macenko_deblank
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0033_20260911_build_faiss_index_macenko_deblank/%j_0033_20260911_build_faiss_index_macenko_deblank.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0033_20260911_build_faiss_index_macenko_deblank/%j_0033_20260911_build_faiss_index_macenko_deblank.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=4:00:00
# CPU-only: FAISSのKMeans/OPQ/PQ学習+全~1000ファイルを読んでadd_with_idsする
# 重量級パス(I/O支配)。0012/0018と同一設定(nlist=4096,pq_m=64,pq_nbits=8,
# opq_niter=10)。experiments/0032(背景除去済みMacenko manifest)が先に
# 完了している必要がある。GPU不要。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0033_20260911_build_faiss_index_macenko_deblank"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
