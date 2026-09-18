#!/bin/bash
#SBATCH --job-name=adhoc_validate_against_ground_truth
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_validate_against_ground_truth/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_validate_against_ground_truth/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# 7カテゴリ・計91枚程度の画像をembed+検索するだけの軽量ジョブ
# (experiments/0003_..._query_demo と同等の資源設定)。
# GPUはUNI推論を速くするために確保しているが、lib/query_embedding.pyは
# GPU無し(cuda利用不可)でも自動的にfloat32のCPU実行にフォールバックする。
#
# baseline_v1 は 2026-09-09 に背景除去済みインデックス(experiments/0018)へ昇格済み。
# 既定では baseline_v1 + (存在すれば)baseline_v2 / v1_macenko を回す。
#
# 挙動切替(env-var を sbatch の前に前置。--export のカンマ分割を避けるため):
#   VGT_PIPELINES=baseline_v1,v1_predeblank
#       default_pipelines() の部分集合だけ回す。背景除去 A/B を再実行するなら
#       baseline_v1(0018)と v1_predeblank(0002)だけあれば十分で、torchstain を
#       伴う v2/macenko の遅い埋め込みを省ける。
#   VGT_OUT=outputs/gt_validations/gt_validation_results_deblank_ab.csv
#       出力先。既定は outputs/gt_validation_results.csv(上書きされる)。
#       退避が必要なら別名を渡すこと。
#   VGT_GT_CSV=data/processed_csv/full_finding_liver.csv
#       単一所見フィルタ(既定の single_finding_liver.csv)を外した完全版GTで
#       回す(2026-09-18、data/tggate_csv/ の生病理データから再構築)。
#       VGT_OUT と併用して出力先を別名にすること(既定のまま重ねると上書きされる)。
#   VGT_INDEX_DIR=outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1
#       単一パイプライン(--index-dir)モード。既存の(索引再構築済みの)variant
#       ディレクトリに対して、素の plain embedding クエリで再評価する
#       (変換は FAISS 索引自体にベイクされているため、クエリ側で追加変換は不要)。
#   VGT_INDEX_DIRS="outputs/a/default/x outputs/b/default/y"
#       スペース区切りで複数 --index-dir を1ジョブ内で直列に回す。各variantの
#       出力は自動的に outputs/gt_validations/gt_validation_results_<variant名>
#       [_<gt-csvのstem>].csv に分かれる(VGT_OUT/VGT_INDEX_DIRの指定は無視)。
#
#   VGT_PIPELINES=baseline_v1,v1_predeblank VGT_OUT=outputs/gt_validations/gt_validation_results_deblank_ab.csv sbatch scripts/adhoc_validate_against_ground_truth.sh
#   VGT_GT_CSV=data/processed_csv/full_finding_liver.csv VGT_OUT=outputs/gt_validations/gt_validation_results_full_finding_liver.csv sbatch scripts/adhoc_validate_against_ground_truth.sh
#   VGT_INDEX_DIRS="outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1 outputs/0025_20260910_whiten_index_rebuild/default/abtt4_v1 outputs/0026_20260910_shrinkage_whiten/default/a025_v1 outputs/0026_20260910_shrinkage_whiten/default/a050_v1 outputs/0026_20260910_shrinkage_whiten/default/a075_v1" VGT_GT_CSV=data/processed_csv/full_finding_liver.csv sbatch scripts/adhoc_validate_against_ground_truth.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

COMMON_ARGS=""
if [ -n "${VGT_PIPELINES:-}" ]; then COMMON_ARGS="${COMMON_ARGS} --pipelines ${VGT_PIPELINES}"; fi
if [ -n "${VGT_GT_CSV:-}" ]; then COMMON_ARGS="${COMMON_ARGS} --gt-csv ${VGT_GT_CSV}"; fi

if [ -n "${VGT_INDEX_DIRS:-}" ]; then
    INDEX_DIR_LIST=(${VGT_INDEX_DIRS})
else
    INDEX_DIR_LIST=("")
fi

run_one() {
    local run_args="$1"
    echo "--- ${2:-single run} ---"
    echo "RUN_ARGS: ${run_args}"
    if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
        apptainer exec \
            --nv \
            "${SIF_PATH}" \
            bash -c "
                set -euo pipefail
                source ${PROJECT_ROOT}/.venv/bin/activate
                export CUDA_HOME=/usr/local/cuda
                cd ${PROJECT_ROOT}
                python scripts/validate_against_ground_truth.py ${run_args}
            "
    else
        echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
        source "${PROJECT_ROOT}/.venv/bin/activate"
        cd "${PROJECT_ROOT}"
        python scripts/validate_against_ground_truth.py ${run_args}
    fi
}

echo "Running scripts/validate_against_ground_truth.py on $(hostname) ..."
if [ -n "${VGT_INDEX_DIRS:-}" ]; then
    for idx_dir in "${INDEX_DIR_LIST[@]}"; do
        variant_tag=$(basename "${idx_dir}")
        gt_tag=""
        if [ -n "${VGT_GT_CSV:-}" ]; then gt_tag="_$(basename "${VGT_GT_CSV}" .csv)"; fi
        out="outputs/gt_validations/gt_validation_results_${variant_tag}${gt_tag}.csv"
        run_one "--index-dir ${idx_dir} --out ${out} ${COMMON_ARGS}" "${idx_dir} -> ${out}"
    done
else
    RUN_ARGS="${COMMON_ARGS}"
    if [ -n "${VGT_OUT:-}" ]; then RUN_ARGS="${RUN_ARGS} --out ${VGT_OUT}"; fi
    if [ -n "${VGT_INDEX_DIR:-}" ]; then RUN_ARGS="${RUN_ARGS} --index-dir ${VGT_INDEX_DIR}"; fi
    run_one "${RUN_ARGS}"
fi

echo "Done. Results written under outputs/gt_validations/ (or ${VGT_OUT:-outputs/gt_validation_results.csv} for a single non-list run)."
