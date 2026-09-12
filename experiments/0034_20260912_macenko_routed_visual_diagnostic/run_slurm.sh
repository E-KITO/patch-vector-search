#!/bin/bash
#SBATCH --job-name=0034_20260912_macenko_routed_visual_diagnostic
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0034_20260912_macenko_routed_visual_diagnostic/%j_0034_20260912_macenko_routed_visual_diagnostic.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0034_20260912_macenko_routed_visual_diagnostic/%j_0034_20260912_macenko_routed_visual_diagnostic.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=4:00:00
# lib.finding_routing.MACENKO_FINDINGS の4所見(Hypertrophy / Increased mitosis /
# Inclusion body / Fatty Change、計18図版)を baseline(0018)と routed(macenko、
# 0033=背景除去済み)の両方で目視診断(experiments/0013の枠組み: タイルスコア
# ヒートマップ + パッチギャラリー)。全タイル厳密re-rank(experiments/0031の
# job 10599の教訓、max_tiles_reranked=null)なのでNFS h5ランダムアクセスが多い。
# 索引・features_dir・macenkoのper-tile染色正規化はすべて lib.finding_routing
# 経由で解決するため局所SSDステージングは行わずNFS直読み。
#
# GPU: クエリ図版のタイル埋め込み(uni_v1 UNI エンコーダ、macenko腕はper-tile
# torchstain正規化込み)に必要。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0034_20260912_macenko_routed_visual_diagnostic"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
