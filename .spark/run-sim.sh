#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

export CYCLONEDDS_HOME="$repo_root/.deps/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec "$repo_root/.venv_sim/bin/python" \
  "$repo_root/gear_sonic/scripts/run_sim_loop.py" \
  --interface "${GROOT_WBC_SIM_INTERFACE:-docker0}" \
  --no-enable-onscreen \
  --enable-offscreen \
  --enable-image-publish \
  --camera-port "${GROOT_WBC_CAMERA_PORT:-5555}" \
  "$@"
