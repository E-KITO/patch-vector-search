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
# 対象パッチ集: "<name>|<patch_set_dir>|<exclude_slides>|<only_slides>|<corpus_dir>"
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
# 5番目のフィールド(省略可、既定 outputs/0002_.../default)は --corpus-dir。
# ランダム対照を引くコーパス。curated 集合を作った索引に合わせること —
# experiments/0019 の集合なら outputs/0018_20260909_build_faiss_index_deblank/default
# (背景除去済みの母集団から対照を引かないと A/B が非対称になる)。
#
# 実行済み:
#   hypertrophy (job 9937) — 判定精度59%(ベースライン61%)= 区別できず、不成立。
#     "hypertrophy|outputs/0015_.../hypertrophy__validate/hypertrophy__validate|27537,28741,35367,35374,40637,44299"
#   glycogen (job 9943、0002 corpus) — 判定精度91%(ベースライン50%)= 明確に選別できている。
#     "glycogen|outputs/0015_.../deposit_glycogen__deliver/deposit_glycogen__deliver|49244,49372,52799,53036,53267,58070"
#   ground_glass / granular_eos (job 9962、0002 corpus) — granular は候補74枚中48枚が
#     背景クロップ、最終21枚。
# =====================================================

TARGETS=(
    # experiments/0019 (背景除去済み 0018 corpus) の deliver 集合に対する対照。
    # 0015(0002)の対照(glycogen job 9943 = 判定91%)との比較で、背景を抜いた
    # コーパスでも curated 集合が「一般的な肝パッチ」と区別できるかを見る。
    # 5番目のフィールド = --corpus-dir: 対照も 0018 の母集団から引く。
    # exclude は deliver モードなので GT スライド全体。
    #
    # granular eosinophilic は 0019 で非 seed 候補が5枚しか無く盲検シートに
    # 乗らない(job 10497)ので対照は取らない。
    "ground_glass_0019|outputs/0019_20260909_build_finding_patch_set_deblank/ground_glass_appearance__deliver/ground_glass_appearance__deliver|28113,28140,37688,6371||outputs/0018_20260909_build_faiss_index_deblank/default"
    "glycogen_0019|outputs/0019_20260909_build_finding_patch_set_deblank/deposit_glycogen__deliver/deposit_glycogen__deliver|49244,49372,52799,53036,53267,58070||outputs/0018_20260909_build_faiss_index_deblank/default"
)

RUN_CMDS=""
for target in "${TARGETS[@]}"; do
    IFS='|' read -r NAME PATCH_SET EXCLUDE_SLIDES ONLY_SLIDES CORPUS_DIR <<< "${target}"
    RUN_CMDS+="echo '--- ${NAME} ---'"$'\n'
    RUN_CMDS+="python scripts/random_patch_baseline.py"
    RUN_CMDS+=" --patch-set ${PATCH_SET}"
    RUN_CMDS+=" --out outputs/random_patch_baseline/${NAME}"
    RUN_CMDS+=" --exclude-slides '${EXCLUDE_SLIDES}'"
    RUN_CMDS+=" --only-slides '${ONLY_SLIDES:-}'"
    if [ -n "${CORPUS_DIR:-}" ]; then RUN_CMDS+=" --corpus-dir ${CORPUS_DIR}"; fi
    RUN_CMDS+=$'\n'
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
