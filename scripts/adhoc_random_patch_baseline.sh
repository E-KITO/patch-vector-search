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
# 実解像度クロップするだけなので GPU 不要・CPU のみ。150枚で数分。

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"

# =====================================================
# 対象パッチ集: "<name>|<patch_set_dir>|<exclude_slides>"
#
# exclude_slides にはその run の seed スライドを渡す(検索結果から除外されて
# いた側と、ランダム対照の母集団を揃えるため)。validate モードなら seed
# スライド、deliver モードなら GT スライド全体。
#
# 対象を変えるときはここを書き換えてコミットしてから投入すること
# (実行設定を git に残すため。環境変数で外から差し替えない)。
#
# 実行済み:
#   hypertrophy (job 9937) — キュレート150枚とランダム150枚を目視で区別できず。
#     詳細は README「ランダム対照による検証」参照。
#     "hypertrophy|outputs/0015_20260904_build_finding_patch_set_seed_slide/hypertrophy__validate/hypertrophy__validate|27537,28741,35367,35374,40637,44299"
# =====================================================

TARGETS=(
    # glycogen は 0015 で唯一「機能する」と判定された所見。hypertrophy が
    # 本当に不成立なのか、それとも目視判定自体が使えないのかを切り分ける
    # ポジティブコントロールとして回す。
    "glycogen|outputs/0015_20260904_build_finding_patch_set_seed_slide/deposit_glycogen__deliver/deposit_glycogen__deliver|49244,49372,52799,53036,53267,58070"
)

RUN_CMDS=""
for target in "${TARGETS[@]}"; do
    IFS='|' read -r NAME PATCH_SET EXCLUDE_SLIDES <<< "${target}"
    RUN_CMDS+="echo '--- ${NAME} ---'"$'\n'
    RUN_CMDS+="python scripts/random_patch_baseline.py"
    RUN_CMDS+=" --patch-set ${PATCH_SET}"
    RUN_CMDS+=" --out outputs/random_patch_baseline/${NAME}"
    RUN_CMDS+=" --exclude-slides ${EXCLUDE_SLIDES}"$'\n'
done

echo "Running scripts/random_patch_baseline.py on $(hostname) for ${#TARGETS[@]} target(s) ..."

if command -v apptainer &>/dev/null && [ -n "${SIF_PATH:-}" ] && [ -f "${SIF_PATH}" ]; then
    apptainer exec \
        "${SIF_PATH}" \
        bash -c "
            set -euo pipefail
            source ${PROJECT_ROOT}/.venv/bin/activate
            cd ${PROJECT_ROOT}
            ${RUN_CMDS}
        "
else
    echo "⚠️ Apptainer not found or SIF_PATH not set. Running on host system."
    source "${PROJECT_ROOT}/.venv/bin/activate"
    cd "${PROJECT_ROOT}"
    eval "${RUN_CMDS}"
fi

echo "Done. Review outputs/random_patch_baseline/*/blind_sheets/ before opening blind_key.csv"
