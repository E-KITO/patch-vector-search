#!/bin/bash
#SBATCH --job-name=0024_20260910_similarity_metric_rerank
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0024_20260910_similarity_metric_rerank/%j_0024_20260910_similarity_metric_rerank.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0024_20260910_similarity_metric_rerank/%j_0024_20260910_similarity_metric_rerank.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# 現行のコサイン類似度に対し、異方性除去 (center / all-but-the-top / 白色化) と
# hubness 補正 (CSLS) が self-retrieval の順位を改善するかを A/B する。
# GPU 不要 (埋め込みは既存、FAISS も使わない)。コーパスをスライドあたり
# n_corpus_subsample パッチに絞った厳密行列 (~120k x 1024) の BLAS matmul のみ。
# ⚠️ リソースを変えたら --partition / --signal のマージンも手動で見直すこと。
#   partition 時間上限: small 1h / medium 2h / large 4h / x-large 無制限。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0024_20260910_similarity_metric_rerank"

# =====================================================
# Storage
# =====================================================
# FAISS 索引は不要。manifest.parquet (118M) を1回、h5 features から ~121k 行を
# 散発読みするだけ。experiment.py が corpus_sub.npz / query_vecs.npz にキャッシュ
# するので、TIMEOUT しても再投入時は h5 を一切読まずアームの続きから走る。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

RUN_MODE="single"
# アームごとに results.csv を書き、resume 時は済みアームをスキップ。
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
