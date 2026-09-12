#!/bin/bash
#SBATCH --job-name=0028_20260911_finding_routing_validation
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0028_20260911_finding_routing_validation/%j_0028_20260911_finding_routing_validation.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0028_20260911_finding_routing_validation/%j_0028_20260911_finding_routing_validation.out
#SBATCH --signal=B:USR1@60
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=0:10:00
# CPU-only、GPU不要: experiments/0025(self_retrieval)/ 0027(atlas 図版単位)の
# 既存 CSV を lib.finding_routing.WHITEN_FINDINGS で再集計するだけ。新規の索引
# 構築・クエリ埋め込みは無い。実際には capsule 内でローカル実行して検証済み
# (job投入なし) — このスクリプトは再現性のための記録用。
# ⚠️ 注意: リソースを変更したら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0028_20260911_finding_routing_validation"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
