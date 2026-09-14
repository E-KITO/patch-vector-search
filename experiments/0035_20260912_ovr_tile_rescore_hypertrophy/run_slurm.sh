#!/bin/bash
#SBATCH --job-name=0035_20260912_ovr_tile_rescore_hypertrophy
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0035_20260912_ovr_tile_rescore_hypertrophy/%j_0035_20260912_ovr_tile_rescore_hypertrophy.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0035_20260912_ovr_tile_rescore_hypertrophy/%j_0035_20260912_ovr_tile_rescore_hypertrophy.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# Hypertrophy(コーパス内GT25枚)を対象に、OvR(One-vs-Rest)ロジスティック回帰で
# クエリタイルを事前スコアリングし、スコア上位keep_frac(config.yml)だけを検索に
# 回した場合に experiments/0034 で0枚だったパッチギャラリーが生成されるようになるか
# を見るプローブ(README「タイル選択バイアス」参照)。
#
# GPU: atlas図版のタイル埋め込み(uni_v1 UNIエンコーダ)に必要。分類器の学習自体は
# CPU(scikit-learn LogisticRegression、正例・負例あわせて1万パッチ強)。負例サンプル
# (コーパス全体からランダム抽出)がコーパスのほぼ全スライドのh5に触れるため、
# experiments/0034と同様NFS h5ランダムアクセスが発生する→局所SSDステージングなし。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0035_20260912_ovr_tile_rescore_hypertrophy"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
