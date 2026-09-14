#!/bin/bash
#SBATCH --job-name=0040_20260914_atlas_domain_shift_alpha_sweep
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0040_20260914_atlas_domain_shift_alpha_sweep/%j_0040_20260914_atlas_domain_shift_alpha_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0040_20260914_atlas_domain_shift_alpha_sweep/%j_0040_20260914_atlas_domain_shift_alpha_sweep.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0039(alpha=1.0の平均シフト補正)はGT best_rankが7所見中5所見で
# 悪化、うち2所見はGTスライドが候補プールから完全に消失という否定的結果だった。
# 補正が強すぎた可能性を検証するため、experiments/0039が保存した
# domain_shift.npy を再利用し(再計算しない、同じ負例サンプルで比較するため)、
# alpha=0.25/0.5/0.75 でGT best_rankをスイープする。最良alphaについてのみ
# コーパスGT対応7所見の目視ギャラリー診断も追加実行する。
#
# GPU: GT対応7所見(約1635タイル)をalpha数(3)+最良alpha再診断(1)の計4回
# 再埋め込みする。domain_shift自体は再計算しないため experiments/0039 より軽い。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。
# ⚠️ experiments/0039 の outputs/0039_.../default/domain_shift.npy に依存する
#    (config.yml の domain_shift_path)。0039のoutputsが残っていることが前提。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0040_20260914_atlas_domain_shift_alpha_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
