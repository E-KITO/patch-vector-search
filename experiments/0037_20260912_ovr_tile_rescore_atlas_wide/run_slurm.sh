#!/bin/bash
#SBATCH --job-name=0037_20260912_ovr_tile_rescore_atlas_wide
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0037_20260912_ovr_tile_rescore_atlas_wide/%j_0037_20260912_ovr_tile_rescore_atlas_wide.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0037_20260912_ovr_tile_rescore_atlas_wide/%j_0037_20260912_ovr_tile_rescore_atlas_wide.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0035・0036(コーパス内GT正例スライドを使うOvR、7所見限定)を、
# 正例の出所をatlas図版自体のタイルに変えることでNNLアトラス全25所見に展開する
# (コーパスGTの無い18所見にも同じ枠組みを適用できるようにするため)。検証は
# study単位LOSOの代わりにatlas図版単位のLOIO(leave-one-image-out)。
# 所見ごとに分類器学習→ヒートマップ→検索2腕(unfiltered/ovr_filtered)を回し、
# summary.csvで横並び比較する。1所見のエラーは他所見の実行を止めない
# (experiment.pyのper-folder try/except、experiments/0036と同じ構造)。
#
# GPU: atlas図版のタイル埋め込み(uni_v1 UNIエンコーダ)に必要。負例サンプリングは
# 所見ごとにコーパスのほぼ全スライドのh5に触れるため(experiments/0035/0036と
# 同様)、25所見分でNFS h5ランダムアクセスがさらに増える→局所SSDステージングなし。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0037_20260912_ovr_tile_rescore_atlas_wide"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
