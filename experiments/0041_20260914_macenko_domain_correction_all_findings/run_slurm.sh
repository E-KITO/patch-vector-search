#!/bin/bash
#SBATCH --job-name=0041_20260914_macenko_domain_correction_all_findings
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0041_20260914_macenko_domain_correction_all_findings/%j_0041_20260914_macenko_domain_correction_all_findings.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0041_20260914_macenko_domain_correction_all_findings/%j_0041_20260914_macenko_domain_correction_all_findings.out
#SBATCH --signal=B:USR1@108
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=3:00:00
# macenko(per-tile染色正規化+別コーパス索引)と線形ドメイン補正
# (experiments/0039・0040)を「両方とも」適用した場合の効果を、NNLアトラス
# 全25所見(GTの有無問わず)で目視ギャラリー診断する。domain_shiftはmacenko
# 空間で独立に再計算する(plain空間のexperiments/0038/0039のものは埋め込み
# 分布が違うため流用しない)。GT対応7所見についてはGT best_rank比較も追加。
#
# GPU: atlas図版91枚・約4900タイルのmacenko per-tile正規化+埋め込み
# (domain_shift計算)、GT対応7所見の追加re-embed(GT比較腕、約1600タイル×2)
# に必要。macenko正規化はplain埋め込みよりCPU側処理が重いため
# experiments/0037/0039より時間に余裕を持たせた(large-creator-i、3h)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0041_20260914_macenko_domain_correction_all_findings"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
