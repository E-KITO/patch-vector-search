#!/bin/bash
#SBATCH --job-name=0055_20260917_local_knn_domain_correction_gt_sweep
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0055_20260917_local_knn_domain_correction_gt_sweep/%j_0055_20260917_local_knn_domain_correction_gt_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0055_20260917_local_knn_domain_correction_gt_sweep/%j_0055_20260917_local_knn_domain_correction_gt_sweep.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=4:00:00
# experiments/0045・0054のグローバルなdomain_shift/jpeg_shift補正が所見依存で
# 明暗が分かれた(Kupffer cellには効くがHypertrophyには逆効果)ことを受け、
# 補正方向をクエリの埋め込み位置ごとに局所k近傍(atlas平均-corpus平均)で
# 決める補正を試す(ユーザー提案)。atlasプール(91枚全タイル、自己一致除外用に
# 由来画像id付き)とcorpusプール(3万パッチ)をplain/macenko両空間で構築し、
# GT対応7所見でfinding_routing実ルーティング空間ごとにalphaスイープする。
#
# GPU: atlas図版91枚のタイル埋め込み(plain/macenko両空間)。時間の大半は
# corpusプール3万パッチのh5サンプリング(NFS)と、experiments/0045と同規模の
# alphaスイープ(13alpha)×3空間のFAISS exact rerank。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0055_20260917_local_knn_domain_correction_gt_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
