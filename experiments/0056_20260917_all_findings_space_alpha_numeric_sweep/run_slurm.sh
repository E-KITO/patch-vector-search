#!/bin/bash
#SBATCH --job-name=0056_20260917_all_findings_space_alpha_numeric_sweep
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0056_20260917_all_findings_space_alpha_numeric_sweep/%j_0056_20260917_all_findings_space_alpha_numeric_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0056_20260917_all_findings_space_alpha_numeric_sweep/%j_0056_20260917_all_findings_space_alpha_numeric_sweep.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# フェーズA: NNLアトラス全25所見×3空間(plain/whiten/macenko)×局所k近傍補正
# 13alpha=975通りを、画像生成なしの軽量な数値指標(FAISS IVF検索のみ、
# 生WSIクロップ・matplotlib描画なし)でスクリーニングする。実際に画像を目視
# するフェーズBは、この結果を見て絞り込んだ候補だけに限定して別実験で行う。
#
# GPU: atlas図版91枚のタイル埋め込み(plain/macenko両空間)。時間の大半は
# corpusプール3万パッチ×2空間のh5サンプリング(NFS)。975通りのFAISS検索
# 自体はexact rerankを伴わないため軽量。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0056_20260917_all_findings_space_alpha_numeric_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
