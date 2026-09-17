#!/bin/bash
#SBATCH --job-name=0046_20260914_finding_routed_domain_correction_visual_sweep
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0046_20260914_finding_routed_domain_correction_visual_sweep/%j_0046_20260914_finding_routed_domain_correction_visual_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0046_20260914_finding_routed_domain_correction_visual_sweep/%j_0046_20260914_finding_routed_domain_correction_visual_sweep.out
#SBATCH --signal=B:USR1@288
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=8:00:00
# NNLアトラス25所見のうちコーパスGTが無い18所見(experiments/0045が扱う7所見の
# 残り)について、finding_routingの実ルーティング空間(plain/whiten/macenko)で
# alpha候補(config.alphas、experiments/0045と同一の0.0〜0.5を0.05刻み11点、
# ユーザー指示で0046側も精度を合わせた)のギャラリーを生成する。alpha候補自体の
# 採否判断は目視で別途行う——このジョブは生成のみ。domain_shiftは再計算せず
# experiments/0039・0041の成果物を再利用する。UNI埋め込みはalphaに依存しない
# ので所見あたり1回だけ計算する。
#
# GPU: 18所見の埋め込み(alphaに依存せず所見ごとに1回)+ 18所見 x 11アーム
# (alpha候補)の検索・ギャラリー生成=198アーム実行。experiments/0041
# (25所見 x 2アーム=50アーム、2時間)からアーム数ベースで比例外挿し、
# 余裕を見て8時間を確保(x-large-creator-i、wsi_preprocess側の報告に倣った
# パーティション名)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。
# ⚠️ experiments/0039 の domain_shift.npy と experiments/0041 の
#    domain_shift_macenko.npy に依存する(config.yml参照)。両方のoutputsが
#    残っていることが前提。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0046_20260914_finding_routed_domain_correction_visual_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
