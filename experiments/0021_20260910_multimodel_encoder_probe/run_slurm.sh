#!/bin/bash
#SBATCH --job-name=0021_20260910_multimodel_encoder_probe
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0021_20260910_multimodel_encoder_probe/%j_0021_20260910_multimodel_encoder_probe.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0021_20260910_multimodel_encoder_probe/%j_0021_20260910_multimodel_encoder_probe.out
#SBATCH --signal=B:USR1@36
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=1:00:00
# wsi_preprocess が作った probe セット (data/.../probe/embeddings_{encoder}.h5) を読み、
# 融合壊死・髄外造血で他エンコーダが uni_v1 を上回るかの弁別力メトリクスを計算する。
# GPU 不要 (埋め込みは済んでいる)。索引も作らない。~12,600 ベクトルの numpy 演算のみ。
# ⚠️ リソースを変えたら --partition / --signal も手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0021_20260910_multimodel_encoder_probe"

# probe/ は ~300MB、NFS 直読みで十分。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH}"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
