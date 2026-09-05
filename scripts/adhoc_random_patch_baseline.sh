#!/bin/bash
#SBATCH --job-name=adhoc_random_patch_baseline
#SBATCH --partition=small-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_random_patch_baseline/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_random_patch_baseline/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=1:00:00
# scripts/random_patch_baseline.py: experiments/0015 が出したパッチ集に対する
# ランダム対照を作る。「検索がその所見を選別できている」のか「このコーパスの
# 肝パッチなら大体こう見える」だけなのかを切り分けるための診断。
# 索引もエンコーダも使わず、manifest からランダムサンプルして openslide で
# 実解像度クロップするだけなので GPU 不要・CPU のみ。150枚程度なら数分。
#
# 既定は Hypertrophy(job 9932)。--exclude-slides にはその run の seed スライドを
# 渡している(結果から除外されていた側と母集団を揃えるため)。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

PATCH_SET="${PATCH_SET:-outputs/0015_20260904_build_finding_patch_set_seed_slide/hypertrophy__validate/hypertrophy__validate}"
OUT_DIR="${OUT_DIR:-outputs/random_patch_baseline/hypertrophy}"
EXCLUDE_SLIDES="${EXCLUDE_SLIDES:-27537,28741,35367,35374,40637,44299}"

RUN_CMD="python scripts/random_patch_baseline.py \
    --patch-set ${PATCH_SET} \
    --out ${OUT_DIR} \
    --exclude-slides ${EXCLUDE_SLIDES}"

echo "Running scripts/random_patch_baseline.py on $(hostname) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            ${RUN_CMD}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    ${RUN_CMD}
fi

echo "Done. Review ${OUT_DIR}/blind_sheets/ before opening ${OUT_DIR}/blind_key.csv"
