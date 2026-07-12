#!/usr/bin/env bash
# Production acceptance test (Linux/macOS / CI wrapper).
#
# Runs the full end-to-end flow: build image, verify deps, obtain the real
# nuScenes v1.0-mini dataset, ingest real clips, and assert correctness.
#
# Examples:
#   ./scripts/test_vlm_app_production.sh                 # docker mode
#   ACCEPTANCE_MODE=local ./scripts/test_vlm_app_production.sh
#   NUSCENES_DIR=/data/nuscenes ./scripts/test_vlm_app_production.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Prefer the project venv, else python3.
if [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
    PY="${REPO_ROOT}/.venv/bin/python"
else
    PY="$(command -v python3 || command -v python)"
fi

exec "${PY}" "${REPO_ROOT}/scripts/production_acceptance.py" "$@"
