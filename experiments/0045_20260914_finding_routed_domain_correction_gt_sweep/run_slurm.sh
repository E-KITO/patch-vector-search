#!/bin/bash
#SBATCH --job-name=0045_20260914_finding_routed_domain_correction_gt_sweep
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0045_20260914_finding_routed_domain_correction_gt_sweep/%j_0045_20260914_finding_routed_domain_correction_gt_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0045_20260914_finding_routed_domain_correction_gt_sweep/%j_0045_20260914_finding_routed_domain_correction_gt_sweep.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0039・0040はplain(baseline)空間だけを基準にalphaを検証していたが、
# lib.finding_routingは所見ごとにplain/whiten/macenkoの3空間に振り分けている。
# この実験は「所見ごとに既に選ばれているルーティング先の上に」線形ドメイン補正を
# 重ねた場合の効果を、GT対応7所見について正しい空間で0.0〜0.5(0.05刻み)の細かい
# alpha格子で検証する。domain_shiftはexperiments/0039・0041の成果物を再利用し
# (再計算しない)、UNI埋め込み自体はalphaに依存しない後処理なので所見・空間ごとに
# 1回だけキャッシュする(alphaごとの再埋め込みはしない)。選ばれたalphaについて
# のみ目視ギャラリー診断も追加実行する。
#
# GPU: 3空間(plain/whiten/macenko)x GT対応7カテゴリの埋め込み(埋め込みは
# alphaに依存せずキャッシュ、0040のような per-alpha 再埋め込みは無い)+
# 選定後の目視ギャラリー診断7所見分。0040(1空間・4パイプライン再埋め込み)と
# 同程度〜やや軽い想定。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。
# ⚠️ experiments/0039 の domain_shift.npy と experiments/0041 の
#    domain_shift_macenko.npy に依存する(config.yml参照)。両方のoutputsが
#    残っていることが前提。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0045_20260914_finding_routed_domain_correction_gt_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
