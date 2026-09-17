#!/bin/bash
#SBATCH --job-name=0048_20260914_deliver_candidate_pool_analysis
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0048_20260914_deliver_candidate_pool_analysis/%j_0048_20260914_deliver_candidate_pool_analysis.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0048_20260914_deliver_candidate_pool_analysis/%j_0048_20260914_deliver_candidate_pool_analysis.out
#SBATCH --signal=B:USR1@54
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=1:30:00
# GT対応7所見について、deliverモードの候補プール構造(生候補→seed除外後→
# sim_floor後→NMS+上限後→round-robin後)を一括集計する。実際のパッチ画像は
# 切り出さない(raw WSIクロップ・コンタクトシート生成なし)ため、
# experiments/0015・0019(同じくCPUのみ、--gres=gpu無し)より大幅に軽い。
# シード側クエリ埋め込みはGTスライド自身の既存h5特徴量を直読みするだけで
# UNIエンコーダは使わない。GPU不要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0048_20260914_deliver_candidate_pool_analysis"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
