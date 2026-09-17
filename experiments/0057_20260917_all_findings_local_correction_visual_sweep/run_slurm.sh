#!/bin/bash
#SBATCH --job-name=0057_20260917_all_findings_local_correction_visual_sweep
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0057_20260917_all_findings_local_correction_visual_sweep/%j_0057_20260917_all_findings_local_correction_visual_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0057_20260917_all_findings_local_correction_visual_sweep/%j_0057_20260917_all_findings_local_correction_visual_sweep.out
#SBATCH --signal=B:USR1@216
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=6:00:00
# フェーズB: experiments/0056(フェーズA)で数値スクリーニングが局所補正の
# 同語反復的アーティファクトにより機能しないと判明したため、experiments/0046
# の方針(粗い格子で全所見を網羅し実際に目視する)に戻す。GTの無い18所見
# それぞれについて、3空間(plain/whiten/macenko)×alpha2点(0.0=補正なし、
# 0.3=中程度)=108アームで実ギャラリー(生WSIクロップ+matplotlib描画)を
# 生成する。現行の既定(plain)以外の空間の方が良く見えるかも含めて比較する。
#
# GPU: atlas図版タイル埋め込み(plain/macenko両空間、experiments/0055・0056と
# 同一)。時間の大半はexperiments/0046と同様、108アーム分のFAISS検索+
# 生WSIクロップ+ギャラリー描画。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0057_20260917_all_findings_local_correction_visual_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
