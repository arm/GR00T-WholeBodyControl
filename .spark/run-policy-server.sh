#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /absolute/path/to/unitree-g1-sonic-checkpoint [port]" >&2
  exit 2
fi

if [[ ! -d "$1" ]]; then
  echo "Checkpoint directory does not exist: $1" >&2
  exit 1
fi

checkpoint="$(realpath "$1")"
port="${2:-5550}"
repo_root="/home/arm-seattle-spark-02/workspaces/humanoid-trt-bench/upstream/Isaac-GR00T"
hf_cache="/home/arm-seattle-spark-02/workspaces/humanoid-trt-bench/cache/huggingface"
cosmos_snapshot="$hf_cache/hub/models--nvidia--Cosmos-Reason2-2B/snapshots/9ce19a195e423419c349abfc86fd07178b230561"
image="${GROOT_POLICY_IMAGE:-gr00t-spark:latest}"

if [[ ! -f "$checkpoint/config.json" ]]; then
  echo "Checkpoint is missing config.json: $checkpoint" >&2
  exit 1
fi
if [[ ! -f "$repo_root/gr00t/eval/run_gr00t_server.py" ]]; then
  echo "Pinned Isaac-GR00T checkout is missing: $repo_root" >&2
  exit 1
fi
if [[ ! -f "$cosmos_snapshot/preprocessor_config.json" ]]; then
  echo "Pinned Cosmos processor snapshot is missing: $cosmos_snapshot" >&2
  exit 1
fi

tty_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  tty_args+=(-t)
fi

exec docker run --rm "${tty_args[@]}" \
  --name "${GROOT_POLICY_CONTAINER_NAME:-gr00t-wbc-policy}" \
  --gpus all \
  --ipc host \
  --network host \
  --volume "$repo_root:/workspace/Isaac-GR00T:ro" \
  --volume "$checkpoint:/models/unitree-g1-sonic:ro" \
  --volume "$hf_cache:/root/.cache/huggingface:ro" \
  --volume "$(dirname "$(dirname "$cosmos_snapshot")"):/models/cosmos:ro" \
  --volume "$script_dir/python-overlay:/opt/gr00t-wbc-overlay:ro" \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH=/opt/gr00t-wbc-overlay:/workspace/Isaac-GR00T \
  --env "GROOT_COSMOS_PROCESSOR_PATH=/models/cosmos/snapshots/$(basename "$cosmos_snapshot")" \
  --workdir /workspace/Isaac-GR00T \
  --entrypoint /opt/gr00t-venv/bin/python \
  "$image" \
  gr00t/eval/run_gr00t_server.py \
  --model-path /models/unitree-g1-sonic \
  --embodiment-tag UNITREE_G1_SONIC \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "$port"
