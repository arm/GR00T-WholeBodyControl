#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${GROOT_WBC_IMAGE:-gr00t-wbc-spark:trt10.13-ort1.16.3}"
network_mode="${GROOT_WBC_NETWORK_MODE:-host}"
tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args+=(-t)
fi

if [[ "$network_mode" == "bridge" ]]; then
  network_args=(
    --network bridge
    --add-host host.docker.internal:host-gateway
    --publish 127.0.0.1:5557:5557
  )
  sim_interface="${GROOT_WBC_SIM_INTERFACE:-eth0}"
elif [[ "$network_mode" == "host" ]]; then
  network_args=(--network host)
  sim_interface="${GROOT_WBC_SIM_INTERFACE:-docker0}"
else
  echo "Unsupported GROOT_WBC_NETWORK_MODE: $network_mode" >&2
  exit 2
fi

exec docker run --rm "${tty_args[@]}" \
  --name "${GROOT_WBC_CONTROLLER_CONTAINER_NAME:-gr00t-wbc-controller}" \
  --gpus all \
  --ipc host \
  "${network_args[@]}" \
  --env "GROOT_WBC_SIM_INTERFACE=$sim_interface" \
  --env "GROOT_WBC_AUTO_APPROVE=${GROOT_WBC_AUTO_APPROVE:-}" \
  --volume "$repo_root/gear_sonic_deploy:/workspace/gear_sonic_deploy" \
  --workdir /workspace/gear_sonic_deploy \
  --entrypoint /workspace/gear_sonic_deploy/deploy.sh \
  "$image" "$@"
