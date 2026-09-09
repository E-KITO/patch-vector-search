#!/bin/bash
#SBATCH --job-name=adhoc_audit_blank_threshold
#SBATCH --partition=medium-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_audit_blank_threshold/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_audit_blank_threshold/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# scripts/audit_blank_threshold.py: 段階1.5。measure_corpus_blankness の parquet から
# sat_frac / mean_intensity で bin 分けしてパッチをサンプルし、コンタクトシートに落とす。
# 暫定基準 sat_frac<0.10 & mean>215 の境界付近を目視して、コーパスから落とす閾値を確定する。
#
#   sat_ladder/bin_*.png  sat_frac の bin(明るいパッチのみ)。0.10 の左右を見比べる:
#                         左は全部背景、右は希薄でも本物の組織であってほしい。
#   mean_guard/bin_*.png  sat_frac<0.10 を mean_intensity で bin 分け。mean>215 ガードの妥当性。
#   sampled.csv           サンプルした全パッチ(slide_id / local_idx / coord / 指標 / sheet / bin)。
#
# GPU 不要。~540 パッチを raw WSI から切り出すだけの軽いジョブ。
# 挙動切替(--export):
#   ABT_PER_BIN=N   bin あたりのサンプル数(既定 45)
#   ABT_SEED=N      乱数シード(既定 42)
#
#   sbatch scripts/adhoc_audit_blank_threshold.sh
#   sbatch --export=ALL,ABT_PER_BIN=60,ABT_SEED=7 scripts/adhoc_audit_blank_threshold.sh

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

RUN_ARGS="--workers ${SLURM_CPUS_PER_TASK:-16}"
if [ -n "${ABT_PER_BIN:-}" ]; then RUN_ARGS="${RUN_ARGS} --per-bin ${ABT_PER_BIN}"; fi
if [ -n "${ABT_SEED:-}" ]; then RUN_ARGS="${RUN_ARGS} --seed ${ABT_SEED}"; fi

echo "Running scripts/audit_blank_threshold.py on $(hostname) ..."
echo "RUN_ARGS: ${RUN_ARGS}"

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            python scripts/audit_blank_threshold.py ${RUN_ARGS}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/audit_blank_threshold.py ${RUN_ARGS}
fi

echo "Done. -> outputs/measure_corpus_blankness/audit/{sat_ladder,mean_guard}/bin_*.png"
