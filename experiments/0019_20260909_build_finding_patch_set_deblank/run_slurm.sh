#!/bin/bash
#SBATCH --job-name=0019_20260909_build_finding_patch_set_deblank
#SBATCH --partition=large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/0019_20260909_build_finding_patch_set_deblank/%j_0019_20260909_build_finding_patch_set_deblank.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/0019_20260909_build_finding_patch_set_deblank/%j_0019_20260909_build_finding_patch_set_deblank.out
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
export EXP_NAME="0019_20260909_build_finding_patch_set_deblank"

# =====================================================
# Storage
# =====================================================
# index/manifest(数GB)+ exact re-rank 用の h5 一部 + 生 WSI 数枚しか読まない
# 軽量クエリなので data/ 全体のステージングは不要。NFS 直読み。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Seq run: experiments/0015 の deliver を、背景除去済みコーパス
# (experiments/0018、sat_frac<0.10 の背景パッチ 389,959件 = 2.12% を除外)で
# 回し直す A/B。variant 対 variant で 0015 の同名出力と比較する。
#
# 背景除去の狙いは「配信パッチ集を背景が占拠する」所見の是正なので、GT validation
# (job 10495、中立)ではなくこの deliver 出力が payoff テスト:
#   Degeneration,_granular,_eosinophilic@deliver — 本命。job 9962 で配信集の
#     93% が背景だった所見。背景占拠が解消するか。
#     (コーパス内5枚のうち4枚が EXP 184 gemfibrozil、残りも fenofibrate =
#      同フィブラート系なので、引けたものが「顆粒状好酸性変性一般」か
#      「フィブラート系肝細胞変化」かは解釈時に注意 — 0015 から変わらない留意点。)
#   Ground_glass_appearance@deliver / Deposit,_glycogen@deliver — 不変チェック。
#     元々背景汚染が問題になっていなかったので 0015 とほぼ同じ集合になるはず。
#
# deliver: 全 GT スライドを seed にして除外し、未ラベルスライドから集合を作る。
# recall 指標は出ないので **実行後に必ず scripts/random_patch_baseline.py で
# ランダム対照を取ること**(job 9937: hypertrophy は GT recall が良く見えて
# 実際は選別できていなかった)。
#
# k_candidate_patches / rerank_pool は 0015 と同じ 5000。deliver は候補が枯れ
# やすい(glycogen deliver は 174 候補で 137枚止まり)ので target 150 に届か
# なければ次で深くする。config.yml の index_exp_dir だけが 0015 と異なる。
# experiment.py の SUPPORTED_FINDINGS / MODES を参照。
# =====================================================

RUN_MODE="seq"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--task"
)
GRID_VALUES=(
    "Degeneration,_granular,_eosinophilic@deliver Ground_glass_appearance@deliver Deposit,_glycogen@deliver"
)

# GRID_VALUES はスペース区切りで seq に展開される。所見名のスペースは "_" で書き、
# experiment.py 側で "_"→" " に戻す。"@" 以降がモード(下記 argparse 参照)。

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
