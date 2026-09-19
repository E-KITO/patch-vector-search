#!/bin/bash
#SBATCH --job-name=0058_20260918_build_finding_patch_set_deblank_sparse_findings
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0058_20260918_build_finding_patch_set_deblank_sparse_findings/%j_0058_20260918_build_finding_patch_set_deblank_sparse_findings.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0058_20260918_build_finding_patch_set_deblank_sparse_findings/%j_0058_20260918_build_finding_patch_set_deblank_sparse_findings.out
#SBATCH --signal=B:USR1@144
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=4:00:00
# 所見ごとに: seed スライドのパッチ特徴量を h5 から読む + 索引検索 + 後処理 +
# 実解像度パッチ切り出し。UNI エンコーダは呼ばないので GPU 不要。
# rerank_pool=6000 で全クエリタイルを厳密再ランキングするので h5 読み込みが
# 支配的(タイル × 候補が触るスライド数だけ h5 open)。初回(rerank_pool=1000)は
# 数分だったが 6x 深いので余裕を見て 4h / large-creator-i にしている。
# ⚠️ リソースを変えたら --partition と --signal のマージンも手動で見直すこと。

# 他の実験のジョブに依存させたい場合:
# #SBATCH --dependency=afterok:<job_id>

export PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
export EXP_NAME="0058_20260918_build_finding_patch_set_deblank_sparse_findings"

# =====================================================
# Storage
# =====================================================
# index/manifest(数GB)+ exact re-rank 用の h5 一部 + 生 WSI 数枚しか読まない
# 軽量クエリなので data/ 全体のステージングは不要。NFS 直読み。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Seq run(0019 から resume、2026-09-18): glycogen / ground glass に続く
# 「成果物トラック」の第3・第4・第5候補として、Microgranuloma・
# Cellular infiltration・Swelling を deliver モードで初めて実行する。
#
# 選定理由(README「次の一手」議論、0058 追加分): 当初 track B 候補に挙げた
# Tension Lipidosis・Cholangiofibrosis はコーパスGT 0枚(Fatty Changeと同型)で
# 検証不能と判明、代わりにこの3所見に切り替えた。この3所見は
# self_retrieval_diagnostic.csv で batch_dominates_finding_rate=0.0・
# nhr_best_rank_med_noexp が一桁台(11.5/12/8)と、バッチ交絡ではない本物の
# シグナルが確認済み。単純な label 数(single_finding_liver.csv 全体では
# 339/130/142枚)だけ見ると豊富に見えるが、この索引が実際に埋め込んでいる
# コーパスではGTスライドはそれぞれ16/7/7枚しかなく、glycogen(6枚)と同水準の
# スパースさ——検索による拡張が意味を持つ所見という点でglycogen/ground glassと
# 同じ立場にある。finding_routing.py 上はどちらもbaseline(0018)固定。
#
# deliver: 全GTスライドをseedにして除外し、未ラベルスライドから集合を作る。
# recall指標は出ないので **実行後に必ず scripts/random_patch_baseline.py で
# ランダム対照を取ること**(job 9937 の教訓と同じ)。
#
# k_candidate_patches / rerank_pool は 0015/0019 と同じ 5000。deliver は候補が
# 枯れやすい(glycogen deliver は174候補で137枚止まり)ので、GTが16/7/7枚と
# 少ないこの3所見でも同様に target 150 未達の可能性がある。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--task"
)
GRID_VALUES=(
    "Microgranuloma@deliver Cellular_infiltration@deliver Swelling@deliver"
)

# GRID_VALUES はスペース区切りで seq に展開される。所見名のスペースは "_" で書き、
# experiment.py 側で "_"→" " に戻す。"@" 以降がモード(下記 argparse 参照)。

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
