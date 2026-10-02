#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
gr00t_root="/home/arm-seattle-spark-02/workspaces/humanoid-trt-bench/upstream/Isaac-GR00T"

export CYCLONEDDS_HOME="$repo_root/.deps/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$gr00t_root${PYTHONPATH:+:$PYTHONPATH}"

exec "$repo_root/.venv_inference/bin/python" \
  "$repo_root/gear_sonic/scripts/run_vla_inference.py" \
  "$@"
