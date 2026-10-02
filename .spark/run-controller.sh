#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${GROOT_WBC_IMAGE:-gr00t-wbc-spark:trt10.13-ort1.16.3}"
tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args+=(-t)
fi

exec docker run --rm "${tty_args[@]}" \
  --gpus all \
  --ipc host \
  --network host \
  --env "GROOT_WBC_SIM_INTERFACE=${GROOT_WBC_SIM_INTERFACE:-docker0}" \
  --volume "$repo_root/gear_sonic_deploy:/workspace/gear_sonic_deploy" \
  --workdir /workspace/gear_sonic_deploy \
  --entrypoint /workspace/gear_sonic_deploy/deploy.sh \
  "$image" "$@"
