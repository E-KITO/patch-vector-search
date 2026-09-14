#!/bin/bash
#SBATCH --job-name=0036_20260912_ovr_tile_rescore_sweep
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0036_20260912_ovr_tile_rescore_sweep/%j_0036_20260912_ovr_tile_rescore_sweep.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0036_20260912_ovr_tile_rescore_sweep/%j_0036_20260912_ovr_tile_rescore_sweep.out
#SBATCH --signal=B:USR1@54
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=1:30:00
# experiments/0035(Hypertrophy単体、OvRタイル事前重み付け)の否定的な結果(LOSO AUROC
# の分散が異常に大きい、フィルタしてもギャラリー0枚のまま)が、Hypertrophy固有の限界
# なのか手法自体の限界なのかを、atlas図版が存在する7所見(config.yml
# target_findings)全部に展開して切り分ける。所見ごとに分類器学習(正例:GTスライド
# パッチ、負例:コーパス全体ランダム、経費目0035とほぼ同じ)→LOSO AUROC→
# atlas図版タイルヒートマップ→検索2腕(unfiltered/ovr_filtered)を回し、
# summary.csvで横並び比較する。1所見のGT不足・atlas図版欠如・想定外エラーは
# 他所見の実行を止めない(experiment.pyのper-finding try/except)。
#
# GPU: atlas図版のタイル埋め込み(uni_v1 UNIエンコーダ)に必要。負例サンプリングは
# 所見ごとにコーパスのほぼ全スライドのh5に触れるため(experiments/0035と同様)、
# 7所見分でNFS h5ランダムアクセスが7倍になる→局所SSDステージングなし。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0036_20260912_ovr_tile_rescore_sweep"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
