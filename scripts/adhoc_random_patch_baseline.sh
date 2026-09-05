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
# 4番目のフィールド(省略可)は --only-slides。指定するとコーパス全体ではなく
# そのスライドからのみサンプルする = 対照ではなく **seed スライドの中身の調査**。
# 出力される blank 率がそのまま「seed のうち組織が疎な割合」になる。
#
# 実行済み:
#   hypertrophy (job 9937) — 判定精度59%(ベースライン61%)= 区別できず、不成立。
#     "hypertrophy|outputs/0015_.../hypertrophy__validate/hypertrophy__validate|27537,28741,35367,35374,40637,44299"
#   glycogen (job 9943) — 判定精度91%(ベースライン50%)= 明確に選別できている。
#     "glycogen|outputs/0015_.../deposit_glycogen__deliver/deposit_glycogen__deliver|49244,49372,52799,53036,53267,58070"
# =====================================================

TARGETS=(
    # job 9962 の2所見。どちらも target 150 に未達(113枚 / 21枚)なので、
    # まず「選別が効いているか」を対照で確かめてから深追いするか決める。
    "ground_glass|outputs/0015_20260904_build_finding_patch_set_seed_slide/ground_glass_appearance__deliver/ground_glass_appearance__deliver|28113,28140,37688,6371"
    "granular_eos|outputs/0015_20260904_build_finding_patch_set_seed_slide/degeneration_granular_eosinophilic__deliver/degeneration_granular_eosinophilic__deliver|29935,29965,29969,29984,35058"

    # granular の seed 調査。job 9962 では候補69枚中48枚(70%)が空白クロップとして
    # 落ちており、クエリ側が組織の疎なパッチを掴んでいる疑いがある。seed 5枚から
    # 直接サンプルして blank 率と見た目を測る(--only-slides、対照ではない)。
    # exclude は空にする — seed 自身を見たいので除外してはいけない。
    "granular_eos_seed|outputs/0015_20260904_build_finding_patch_set_seed_slide/degeneration_granular_eosinophilic__deliver/degeneration_granular_eosinophilic__deliver||29935,29965,29969,29984,35058"
)

RUN_CMDS=""
for target in "${TARGETS[@]}"; do
    IFS='|' read -r NAME PATCH_SET EXCLUDE_SLIDES ONLY_SLIDES <<< "${target}"
    RUN_CMDS+="echo '--- ${NAME} ---'"$'\n'
    RUN_CMDS+="python scripts/random_patch_baseline.py"
    RUN_CMDS+=" --patch-set ${PATCH_SET}"
    RUN_CMDS+=" --out outputs/random_patch_baseline/${NAME}"
    RUN_CMDS+=" --exclude-slides '${EXCLUDE_SLIDES}'"
    RUN_CMDS+=" --only-slides '${ONLY_SLIDES:-}'"$'\n'
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
