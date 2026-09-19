#!/bin/bash
#SBATCH --job-name=adhoc_self_retrieval_diagnostic
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_self_retrieval_diagnostic/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# 複数 --index-dir をまとめて1ジョブで回す場合(SRD_INDEX_DIRS、下記)は
# 1本あたり0018規模の実測(35所見で約30分)を variant 数だけ直列で回すため、
# sbatch 前に --partition=large-creator-i --time=4:00:00 等へ手動で上書きすること
# (例: sbatch --partition=large-creator-i --time=4:00:00 scripts/adhoc_self_retrieval_diagnostic.sh)。
# scripts/self_retrieval_diagnostic.py: コーパス内 leave-one-out 自己検索で
# uni_v1 索引の「モデル天井」を測る。UNI エンコーダは一切呼ばない
# (クエリベクトルは h5 の格納特徴量をそのまま読む)ので GPU 不要・CPU のみ。
# ~16 findings x 最大12 LOO クエリ、1クエリあたり search_top_slides_multi を
# 1500 パッチ分。実測を見てから MAX_QUERY_SLIDES_PER_FINDING / N_QUERY_PATCHES を調整。
#
# 既定は experiments/0018(背景除去済み、現行の既定索引)。出力は
# self_retrieval_diagnostic.csv。
#
# 挙動切替(env-var を sbatch の前に前置):
#   SRD_INDEX_DIR=outputs/0002_20260808_build_faiss_index/default
#       背景除去前の索引(experiments/0002)で回す。出力名に実験IDが付く
#       (self_retrieval_diagnostic_0002.csv)ので既定の記録を上書きしない。
#
#   SRD_GT_CSV=data/processed_csv/full_finding_liver.csv
#       単一所見フィルタ(既定の single_finding_liver.csv)を外した完全版GTで
#       回す(2026-09-18、data/tggate_csv/ の生病理データから再構築)。
#       出力名にファイル名が付く(self_retrieval_diagnostic_full_finding_liver.csv)。
#
#   SRD_INDEX_DIRS="outputs/a/default outputs/b/default"
#       スペース区切りで複数 --index-dir を1ジョブ内で直列に回す(SRD_GT_CSV等の
#       他オプションは全variant共通で適用される)。指定時は SRD_INDEX_DIR(単数)
#       は無視される。
#
#   sbatch scripts/adhoc_self_retrieval_diagnostic.sh
#   SRD_INDEX_DIR=outputs/0002_20260808_build_faiss_index/default sbatch scripts/adhoc_self_retrieval_diagnostic.sh
#   SRD_GT_CSV=data/processed_csv/full_finding_liver.csv sbatch scripts/adhoc_self_retrieval_diagnostic.sh
#   SRD_INDEX_DIRS="outputs/0025_20260910_whiten_index_rebuild/default/whiten_v1 outputs/0025_20260910_whiten_index_rebuild/default/abtt4_v1" SRD_GT_CSV=data/processed_csv/full_finding_liver.csv sbatch --partition=large-creator-i --time=4:00:00 scripts/adhoc_self_retrieval_diagnostic.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

COMMON_ARGS=""
if [ -n "${SRD_GT_CSV:-}" ]; then COMMON_ARGS="${COMMON_ARGS} --gt-csv ${SRD_GT_CSV}"; fi

if [ -n "${SRD_INDEX_DIRS:-}" ]; then
    INDEX_DIR_LIST=(${SRD_INDEX_DIRS})
elif [ -n "${SRD_INDEX_DIR:-}" ]; then
    INDEX_DIR_LIST=("${SRD_INDEX_DIR}")
else
    INDEX_DIR_LIST=("")
fi

run_one() {
    local run_args="$1"
    echo "--- index-dir: ${2:-<default>} ---"
    echo "RUN_ARGS: ${run_args}"
    if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
        apptainer exec \
            "${SIF_PATH}" \
            bash -c "
                set -euo pipefail
                source ${PROJECT_ROOT}/.venv/bin/activate
                cd ${PROJECT_ROOT}
                python scripts/self_retrieval_diagnostic.py ${run_args}
            "
    else
        echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
        source "${PROJECT_ROOT}/.venv/bin/activate"
        cd "${PROJECT_ROOT}"
        python scripts/self_retrieval_diagnostic.py ${run_args}
    fi
}

echo "Running scripts/self_retrieval_diagnostic.py on $(hostname) for ${#INDEX_DIR_LIST[@]} index-dir(s) ..."
for idx_dir in "${INDEX_DIR_LIST[@]}"; do
    args="${COMMON_ARGS}"
    if [ -n "${idx_dir}" ]; then
        # --index-dir の自動タグ(親ディレクトリ名)は variant 間で衝突しうる
        # (例: outputs/0025.../default/whiten_v1 と .../abtt4_v1 は両方とも
        # 親が "default" になる)ので、複数 index-dir をまとめて回す場合は
        # variant 自身のディレクトリ名を --out に明示する。
        variant_tag=$(basename "${idx_dir}")
        gt_tag=""
        if [ -n "${SRD_GT_CSV:-}" ]; then gt_tag="_$(basename "${SRD_GT_CSV}" .csv)"; fi
        args="--index-dir ${idx_dir} --out outputs/gt_validations/self_retrieval_diagnostic_${variant_tag}${gt_tag}.csv ${args}"
    fi
    run_one "${args}" "${idx_dir}"
done

echo "Done. Results written under outputs/gt_validations/ (self_retrieval_diagnostic.csv, or _<expid>.csv for a non-default --index-dir)."
