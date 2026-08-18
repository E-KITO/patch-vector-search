#!/bin/bash
#SBATCH --job-name=extract_trident_processed
#SBATCH --partition=x-large-creator-i
#SBATCH --output=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_extract_trident_processed/%j.out
#SBATCH --error=/workspace/filesrv02/kito/patch-vector-search/logs/adhoc_extract_trident_processed/%j.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=8:00:00

set -euo pipefail

PROJECT_ROOT="/workspace/filesrv02/kito/patch-vector-search"
ARCHIVE="${PROJECT_ROOT}/data/moo_collected_tggate_wsi/trident_processed.tar.gz"
DEST="${PROJECT_ROOT}/data"

echo "Extracting ${ARCHIVE} into ${DEST}/ ..."
tar -xzf "${ARCHIVE}" -C "${DEST}"
echo "Done. Result should be under ${DEST}/trident_processed/"
