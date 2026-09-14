#!/bin/bash
#SBATCH --job-name=0042_20260914_macenko_domain_correction_adhoc_queries
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0042_20260914_macenko_domain_correction_adhoc_queries/%j_0042_20260914_macenko_domain_correction_adhoc_queries.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0042_20260914_macenko_domain_correction_adhoc_queries/%j_0042_20260914_macenko_domain_correction_adhoc_queries.out
#SBATCH --signal=B:USR1@36
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=1:00:00
# experiments/0041のMacenko+線形補正併用(macenko空間で計算したdomain_shift、
# alpha=0.25)を、NNLアトラス外の単発参照タイル(data/query/query_001〜003.png)
# に適用し、baseline/macenko単独/macenko+線形補正併用の3腕で目視ギャラリー
# 診断する。domain_shiftはexperiments/0041が計算済みのものを再利用(再計算
# しない)。クエリ画像3枚のみなので実験0037/0041よりずっと軽量。
# ⚠️ experiments/0041 の outputs/0041_.../default/domain_shift_macenko.npy に
#    依存する(config.yml の domain_shift_macenko_path)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0042_20260914_macenko_domain_correction_adhoc_queries"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
