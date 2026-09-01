#!/bin/bash
#SBATCH --job-name=adhoc_stain_norm_self_consistency
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_stain_norm_self_consistency/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_stain_norm_self_consistency/%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# Independent self-consistency check for the delivered uni_v1 Macenko corpus
# (scripts/stain_norm_self_consistency_diagnostic.py). Re-crops a sample of
# corpus patches from the raw WSIs, re-runs the Macenko + uni_v1 pipeline, and
# compares to the stored feature vectors (cos_self ~1.0 is the gate before
# experiments/0012 build_faiss_index). Default sample is 20 slides x 5 patches
# = 100 uni_v1 embeddings — a few minutes on one GPU. Same GPU/CPU/mem shape as
# adhoc_exact_vs_approx_diagnostic.sh.
#
# Run only AFTER the corpus + rename are done (features_uni_v1_macenko present).
# For a pre-delivery smoke test against whatever slides exist, add --n-slides 1
# (the python script falls back to the pre-rename features_uni_v1 dir).
#
# Once the audit has produced stain_norm_failures.json, pass it through so the
# manifest-excluded degenerate patches are skipped when sampling:
#   ... adhoc_stain_norm_self_consistency_diagnostic.sh \
#         --failures-json data/trident_processed_uni_v1_macenko/stain_norm_failures.json

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

echo "Running scripts/stain_norm_self_consistency_diagnostic.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        --nv \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            export CUDA_HOME=/usr/local/cuda
            cd ${PROJECT_ROOT}
            python scripts/stain_norm_self_consistency_diagnostic.py \"\$@\"
        " _ "$@"
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    python scripts/stain_norm_self_consistency_diagnostic.py "$@"
fi

echo "Done. Results written to outputs/stain_norm_self_consistency.csv"
