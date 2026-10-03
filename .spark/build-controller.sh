#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${GROOT_WBC_IMAGE:-gr00t-wbc-spark:trt10.13-ort1.16.3}"

docker build --pull=false --tag "$image" "$script_dir"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp \
  --volume "$repo_root/gear_sonic_deploy:/workspace/gear_sonic_deploy" \
  --workdir /workspace/gear_sonic_deploy \
  --entrypoint bash \
  "$image" \
  -lc 'cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON && cmake --build build --parallel "$(nproc)"'
