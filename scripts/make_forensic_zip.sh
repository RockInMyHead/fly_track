#!/usr/bin/env bash
# Build forensic delivery package: code ZIP + video + outputs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="${1:-/Users/artem/Downloads/VID00001_100-130.mp4}"
FORENSIC_OUT="${2:-output/forensic_VID00001_100-130}"
BUNDLE="${ROOT}/output/malecns_forensic_package"
CODE_ZIP="${BUNDLE}/malecns_vo_code.zip"

mkdir -p "${BUNDLE}/video" "${BUNDLE}/outputs"
cp -f "${VIDEO}" "${BUNDLE}/video/$(basename "${VIDEO}")"
rsync -a --delete "${ROOT}/${FORENSIC_OUT}/" "${BUNDLE}/outputs/"

# Code archive (exclude heavy/runtime dirs and prior run outputs)
cd "${ROOT}"
zip -r -q "${CODE_ZIP}" . \
  -x '.venv/*' \
  -x '**/__pycache__/*' \
  -x '.git/*' \
  -x 'output/*' \
  -x 'archive/*' \
  -x 'uploads/*' \
  -x '*.pyc'

# Top-level delivery ZIP
cd "${ROOT}/output"
rm -f malecns_forensic_package.zip
zip -r -q malecns_forensic_package.zip malecns_forensic_package

echo "Created: ${ROOT}/output/malecns_forensic_package.zip"
echo "  video:   $(basename "${VIDEO}")"
echo "  outputs: ${FORENSIC_OUT}"
echo "  code:    malecns_vo_code.zip ($(du -h "${CODE_ZIP}" | cut -f1))"
