#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${GROOT_WBC_APPROVE_EXCLUSIVE:-}" != "YES" ]]; then
  echo "Set GROOT_WBC_APPROVE_EXCLUSIVE=YES to permit stopping protected GPU services." >&2
  exit 2
fi

repo="/home/arm-seattle-spark-02/workspaces/gr00t-whole-body-control"
gr00t_root="/home/arm-seattle-spark-02/workspaces/humanoid-trt-bench/upstream/Isaac-GR00T"
model="${1:-$repo/models/sii-linzy-grab-bottle-checkpoint-10000}"
tag="${2:-integration-$(date -u +%Y%m%dT%H%M%SZ)}"
result="/home/arm-seattle-spark-02/workspaces/gr00t-wbc-results/$tag"
env_name="${GROOT_WBC_ENV_NAME:-default}"
prompt="${GROOT_WBC_PROMPT:-grab the bottle}"
task_scenario="${GROOT_WBC_TASK_SCENARIO:-single_bottle}"
task_target="${GROOT_WBC_TASK_TARGET:-bottle}"
task_seed="${GROOT_WBC_TASK_SEED:-0}"
task_duration="${GROOT_WBC_TASK_DURATION_S:-}"
if [[ -z "$task_duration" ]]; then
  if [[ "$env_name" == "pnp_bottle" ]]; then task_duration=90; else task_duration=45; fi
fi
task_metrics="$result/task-metrics.json"
task_arm_file="$result/task-armed"
protected=(pi05-fp8-production qwen3-vl-judge triton-spark)
bridge_gateway="$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')"
stopped=()
sim_pid=""
client_pid=""
policy_pid=""
controller_pid=""

if [[ ! -f "$model/config.json" ]]; then
  echo "Invalid checkpoint directory: $model" >&2
  exit 1
fi
if [[ -e "$result" ]]; then
  echo "Result tag already exists: $result" >&2
  exit 1
fi
mkdir -p "$result"

if [[ ! "$task_duration" =~ ^[0-9]+$ ]] || (( task_duration < 1 )); then
  echo "GROOT_WBC_TASK_DURATION_S must be a positive integer" >&2
  exit 1
fi
if [[ "$env_name" != "default" && "$env_name" != "pnp_bottle" ]]; then
  echo "Unsupported GROOT_WBC_ENV_NAME: $env_name" >&2
  exit 1
fi

cleanup() {
  status=$?
  set +e
  if [[ -n "$client_pid" ]]; then kill -TERM -- "-$client_pid" 2>/dev/null; fi
  if [[ -n "$sim_pid" ]]; then kill -TERM -- "-$sim_pid" 2>/dev/null; fi
  docker rm -f gr00t-wbc-controller gr00t-wbc-policy gr00t-wbc-dds-bridge >/dev/null 2>&1
  for container in "${stopped[@]}"; do
    docker start "$container" >/dev/null
  done
  sleep 5
  docker ps --format '{{.Names}}|{{.ID}}|{{.Status}}' >"$result/containers-after.txt" 2>&1
  nvidia-smi >"$result/nvidia-smi-after.txt" 2>&1
  printf '%s\n' "$status" >"$result/exit-code.txt"
  find "$result" -type f ! -name evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >"$result/evidence.sha256"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

cd "$repo"
git rev-parse HEAD >"$result/source-commit.txt"
printf '%s\n' '5fdb36c78c88b9cc3a2c584fcd8993e9955b2384' >"$result/model-revision.txt"
sha256sum "$model"/model-*.safetensors >"$result/model-sha256.txt"
printf 'env_name=%s\nprompt=%s\nscenario=%s\ntarget=%s\nseed=%s\nduration_s=%s\n' \
  "$env_name" "$prompt" "$task_scenario" "$task_target" "$task_seed" "$task_duration" \
  >"$result/evaluation-config.txt"
docker ps --format '{{.Names}}|{{.ID}}|{{.Status}}' >"$result/containers-before.txt"
nvidia-smi >"$result/nvidia-smi-before.txt"

for container in "${protected[@]}"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    echo "Protected container is not running before maintenance: $container" >&2
    exit 1
  fi
  stopped+=("$container")
done
docker stop --timeout 30 "${protected[@]}" | tee "$result/stopped-containers.txt"

# The protected bridge-attached services normally keep docker0 at carrier-up.
# Preserve an isolated multicast-capable DDS interface while they are stopped.
docker run --detach --rm \
  --name gr00t-wbc-dds-bridge \
  --network bridge \
  --entrypoint sleep \
  gr00t-wbc-spark:trt10.13-ort1.16.3 infinity \
  >"$result/dds-bridge-container.txt"
for _ in $(seq 1 20); do
  [[ "$(cat /sys/class/net/docker0/operstate)" == "up" ]] && break
  sleep 1
done
if [[ "$(cat /sys/class/net/docker0/operstate)" != "up" ]]; then
  echo "docker0 did not reach carrier-up for isolated DDS" >&2
  exit 1
fi

GROOT_POLICY_CONTAINER_NAME=gr00t-wbc-policy \
  .spark/run-policy-server.sh "$model" 5550 >"$result/policy.log" 2>&1 &
policy_pid=$!

policy_ready=false
for _ in $(seq 1 180); do
  if ! docker inspect gr00t-wbc-policy >/dev/null 2>&1; then
    kill -0 "$policy_pid" 2>/dev/null || { tail -100 "$result/policy.log" >&2 || true; exit 1; }
    sleep 2
    continue
  fi
  if ! docker inspect -f '{{.State.Running}}' gr00t-wbc-policy 2>/dev/null | grep -qx true; then
    tail -100 "$result/policy.log" >&2 || true
    exit 1
  fi
  if timeout 10 env PYTHONPATH="$gr00t_root" \
      "$repo/.venv_inference/bin/python" -c \
      'from gr00t.policy.server_client import PolicyClient; p=PolicyClient(host="127.0.0.1", port=5550); ok=p.ping(); p.close(); raise SystemExit(0 if ok else 1)' \
      >/dev/null 2>&1; then
    policy_ready=true
    break
  fi
  sleep 2
done
if [[ "$policy_ready" != true ]]; then
  echo "Policy server did not become ready" >&2
  tail -100 "$result/policy.log" >&2 || true
  exit 1
fi

GROOT_WBC_TASK_SCENARIO="$task_scenario" \
GROOT_WBC_TASK_TARGET="$task_target" \
GROOT_WBC_TASK_SEED="$task_seed" \
GROOT_WBC_TASK_DURATION_S="$task_duration" \
GROOT_WBC_TASK_METRICS_PATH="$task_metrics" \
GROOT_WBC_TASK_ARM_FILE="$task_arm_file" \
PYTHONUNBUFFERED=1 setsid .spark/run-sim.sh --env-name "$env_name" \
  >"$result/sim.log" 2>&1 &
sim_pid=$!
sim_ready=false
for _ in $(seq 1 60); do
  if ! kill -0 "$sim_pid" 2>/dev/null; then
    tail -100 "$result/sim.log" >&2 || true
    exit 1
  fi
  if ss -ltn | grep -q ':5555 '; then
    sim_ready=true
    break
  fi
  sleep 1
done
if [[ "$sim_ready" != true ]]; then
  echo "Simulator camera server did not become ready" >&2
  exit 1
fi

GROOT_WBC_AUTO_APPROVE=YES \
GROOT_WBC_NETWORK_MODE=bridge \
GROOT_WBC_CONTROLLER_CONTAINER_NAME=gr00t-wbc-controller \
  .spark/run-controller.sh \
    --cp policy/sonic_v1_1/model \
    --obs-config policy/sonic_v1_1/observation_config.yaml \
    --planner planner/target_vel/V2/planner_sonic.onnx \
    --motion-data reference/example \
    --input-type zmq_manager \
    --output-type zmq \
    --zmq-host host.docker.internal \
    sim >"$result/controller.log" 2>&1 &
controller_pid=$!

controller_ready=false
for _ in $(seq 1 300); do
  if ! docker inspect gr00t-wbc-controller >/dev/null 2>&1; then
    kill -0 "$controller_pid" 2>/dev/null || { tail -120 "$result/controller.log" >&2 || true; exit 1; }
    sleep 2
    continue
  fi
  if ! docker inspect -f '{{.State.Running}}' gr00t-wbc-controller 2>/dev/null | grep -qx true; then
    tail -120 "$result/controller.log" >&2 || true
    exit 1
  fi
  if grep -q 'Initialized ZMQ output interface' "$result/controller.log"; then
    controller_ready=true
    break
  fi
  sleep 2
done
if [[ "$controller_ready" != true ]]; then
  echo "SONIC controller did not become ready" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 setsid .spark/run-inference-client.sh \
  --host 127.0.0.1 \
  --port 5550 \
  --embodiment-tag unitree_g1_sonic \
  --prompt "$prompt" \
  --camera-host 127.0.0.1 \
  --camera-port 5555 \
  --action-zmq-host "$bridge_gateway" \
  --verbose-timing >"$result/client.log" 2>&1 &
client_pid=$!

client_ready=false
for _ in $(seq 1 60); do
  if ! kill -0 "$client_pid" 2>/dev/null; then
    tail -120 "$result/client.log" >&2 || true
    exit 1
  fi
  if grep -q 'Starting the policy loop' "$result/client.log"; then
    client_ready=true
    break
  fi
  sleep 1
done
if [[ "$client_ready" != true ]]; then
  echo "VLA client did not initialize" >&2
  exit 1
fi

"$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py k
control_started=false
for _ in $(seq 1 20); do
  if grep -q 'Started C++ control loop' "$result/client.log"; then
    control_started=true
    break
  fi
  sleep 1
done
if [[ "$control_started" != true ]]; then
  echo "SONIC planner control did not start" >&2
  exit 1
fi

action_ready=false
for _ in $(seq 1 120); do
  if ! kill -0 "$client_pid" 2>/dev/null; then
    tail -120 "$result/client.log" >&2 || true
    exit 1
  fi
  if grep -q 'New action chunk' "$result/client.log"; then
    action_ready=true
    break
  fi
  sleep 2
done
if [[ "$action_ready" != true ]]; then
  echo "VLA client did not receive an action chunk" >&2
  exit 1
fi

controller_lines_before_init=$(wc -l <"$result/controller.log")
"$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py i
pose_mode_ready=false
for _ in $(seq 1 20); do
  if ! kill -0 "$client_pid" 2>/dev/null; then
    tail -120 "$result/client.log" >&2 || true
    exit 1
  fi
  if tail -n "+$((controller_lines_before_init + 1))" "$result/controller.log" \
      | grep -q 'ZMQ STREAMING MODE: ENABLED'; then
    pose_mode_ready=true
    break
  fi
  sleep 1
done
if [[ "$pose_mode_ready" != true ]]; then
  echo "SONIC did not enter streamed pose mode" >&2
  tail -120 "$result/controller.log" >&2 || true
  exit 1
fi
# The planner-to-pose transition clears any token already queued. Resend the
# initial pose after streamed mode is active so the controller can consume it.
"$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py i
initial_motion_complete=false
for _ in $(seq 1 20); do
  if ! kill -0 "$client_pid" 2>/dev/null; then
    tail -120 "$result/client.log" >&2 || true
    exit 1
  fi
  if ! docker inspect -f '{{.State.Running}}' gr00t-wbc-controller 2>/dev/null | grep -qx true; then
    tail -120 "$result/controller.log" >&2 || true
    exit 1
  fi
  if tail -n "+$((controller_lines_before_init + 1))" "$result/controller.log" \
      | grep -q 'Temporary motion completed.'; then
    initial_motion_complete=true
    break
  fi
  sleep 1
done
if [[ "$initial_motion_complete" != true ]]; then
  echo "SONIC did not complete the initial-pose transition" >&2
  tail -120 "$result/controller.log" >&2 || true
  exit 1
fi
sleep 1
"$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py p
if [[ "$env_name" == "pnp_bottle" ]]; then
  install -m 0644 /dev/null "$task_arm_file"
fi

for _ in $(seq 1 $((task_duration + 5))); do
  kill -0 "$client_pid"
  docker inspect -f '{{.State.Running}}' gr00t-wbc-policy | grep -qx true
  docker inspect -f '{{.State.Running}}' gr00t-wbc-controller | grep -qx true
  sleep 1
  if [[ "$env_name" == "pnp_bottle" && -s "$task_metrics" ]]; then
    task_status=$("$repo/.venv_inference/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$task_metrics")
    if [[ "$task_status" == "success" || "$task_status" == "object_off_table" || "$task_status" == "simulator_unstable" || "$task_status" == "complete" ]]; then
      echo "Task reached terminal status: $task_status"
      break
    fi
  fi
done

if [[ "$env_name" == "pnp_bottle" ]]; then
  if [[ ! -s "$task_metrics" ]]; then
    echo "Task evaluator did not produce metrics" >&2
    exit 1
  fi
  "$repo/.venv_inference/bin/python" - "$task_metrics" <<'PY'
import json
import sys

with open(sys.argv[1]) as stream:
    metrics = json.load(stream)
required = {
    "scenario",
    "target",
    "seed",
    "contact_observed",
    "lift_observed",
    "success",
    "robot_falls",
}
missing = required.difference(metrics)
if missing:
    raise SystemExit(f"Task metrics missing keys: {sorted(missing)}")
print(json.dumps(metrics, sort_keys=True))
PY
fi

action_chunks=$(grep -c 'New action chunk' "$result/client.log" || true)
action_frames=$(grep -c 'ZMQ: Sent latent action' "$result/client.log" || true)
controller_frames=$(grep -c 'frame_index:' "$result/controller.log" || true)
if (( action_chunks < 3 || controller_frames < 1 )); then
  echo "Insufficient closed-loop activity: chunks=$action_chunks controller_frames=$controller_frames" >&2
  exit 1
fi
if grep -Eqi 'Traceback|segmentation fault|CUDA error|out of memory' \
    "$result/policy.log" "$result/sim.log" "$result/controller.log" "$result/client.log"; then
  echo "Fatal signature detected in evaluation logs" >&2
  exit 1
fi

nvidia-smi >"$result/nvidia-smi-evaluation.txt"
"$repo/.venv_inference/bin/python" .spark/analyze-eval.py "$result" \
  >"$result/metrics.json"
printf 'action_chunks=%s\naction_frame_markers=%s\ncontroller_frames=%s\n' \
  "$action_chunks" "$action_frames" "$controller_frames" | tee "$result/summary.txt"
