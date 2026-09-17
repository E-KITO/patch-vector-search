#!/bin/bash
#SBATCH --job-name=0039_20260914_atlas_domain_shift_correction
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0039_20260914_atlas_domain_shift_correction/%j_0039_20260914_atlas_domain_shift_correction.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0039_20260914_atlas_domain_shift_correction/%j_0039_20260914_atlas_domain_shift_correction.out
#SBATCH --signal=B:USR1@72
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=2:00:00
# experiments/0038で確認された atlas vs corpus ドメインギャップ(所見によらず
# ほぼ一定の差、10.0〜11.7)に対する最も単純な線形補正(平均シフト)を、
# scripts.validate_against_ground_truth.run_comparison によるGT best_rank比較
# (baseline_v1 vs domain_corrected)と、コーパスGT対応7所見の目視ギャラリー
# 診断(lib.ovr_scoring.run_query_arm)の両方で検証する。索引・コーパス側は
# 一切変更しない(baseline 0018のまま)、クエリ埋め込みベクトルへの後処理のみ。
#
# GPU: atlas図版91枚・約4900タイル(domain_shift計算)+ GT対応7所見の再埋め込み
# (GT比較腕・目視診断腕で複数回、計約1600タイル×2〜3)に必要。負例サンプリングは
# コーパスのほぼ全スライドのh5に触れる(experiments/0035-0038と同様)。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0039_20260914_atlas_domain_shift_correction"

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"
RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
