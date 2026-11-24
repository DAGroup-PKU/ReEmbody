#!/usr/bin/env bash
# Fetch pinned third-party robot descriptions and MuJoCo Menagerie assets required
# to compile Dual-Arm Franka FR3 + Robotiq 2F-85 scenes locally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${REEMBODY_THIRD_PARTY_DIR:-${REPO_ROOT}/third_party}"
mkdir -p "${THIRD_PARTY}"

FRANKA_DESC_DIR="${THIRD_PARTY}/franka_description"
FRANKA_DESC_COMMIT="7aeeddc449edf8d62b594f9e36a81da53e7796f9"

if [ ! -d "${FRANKA_DESC_DIR}/.git" ]; then
  echo "[1/2] Cloning franka_description at commit ${FRANKA_DESC_COMMIT}..."
  git clone --no-checkout https://github.com/frankarobotics/franka_description.git "${FRANKA_DESC_DIR}"
  git -C "${FRANKA_DESC_DIR}" checkout "${FRANKA_DESC_COMMIT}"
else
  echo "[1/2] franka_description already present at ${FRANKA_DESC_DIR}"
fi

MENAGERIE_DIR="${THIRD_PARTY}/mujoco_menagerie"
if [ ! -d "${MENAGERIE_DIR}/.git" ]; then
  echo "[2/2] Sparse-cloning mujoco_menagerie (franka_fr3_v2, robotiq_2f85, franka_emika_panda)..."
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git "${MENAGERIE_DIR}"
  git -C "${MENAGERIE_DIR}" sparse-checkout set franka_fr3_v2 robotiq_2f85 franka_emika_panda
else
  echo "[2/2] mujoco_menagerie already present at ${MENAGERIE_DIR}"
fi

echo ""
echo "Third-party robot assets ready under: ${THIRD_PARTY}"
echo "Verify with: python scripts/verify_examples.py --check-mujoco"
