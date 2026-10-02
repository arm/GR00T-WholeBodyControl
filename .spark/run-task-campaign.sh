#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "${GROOT_WBC_APPROVE_EXCLUSIVE:-}" != "YES" ]]; then
  echo "Set GROOT_WBC_APPROVE_EXCLUSIVE=YES to permit stopping protected GPU services." >&2
  exit 2
fi

repo="/home/arm-seattle-spark-02/workspaces/gr00t-whole-body-control"
gr00t_root="/home/arm-seattle-spark-02/workspaces/humanoid-trt-bench/upstream/Isaac-GR00T"
model="${1:-$repo/models/sii-linzy-grab-bottle-checkpoint-10000}"
tag="${2:-task-campaign-$(date -u +%Y%m%dT%H%M%SZ)}"
result="/home/arm-seattle-spark-02/workspaces/gr00t-wbc-results/$tag"
trials_per_scenario="${GROOT_WBC_TRIALS_PER_SCENARIO:-30}"
task_duration="${GROOT_WBC_TASK_DURATION_S:-90}"
protected=(pi05-fp8-production qwen3-vl-judge triton-spark)
bridge_gateway="$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')"
stopped=()
sim_pid=""
client_pid=""
policy_pid=""
controller_pid=""

if [[ ! "$trials_per_scenario" =~ ^[0-9]+$ ]] || (( trials_per_scenario < 1 )); then
  echo "GROOT_WBC_TRIALS_PER_SCENARIO must be a positive integer" >&2
  exit 1
fi
if [[ ! "$task_duration" =~ ^[0-9]+$ ]] || (( task_duration < 1 )); then
  echo "GROOT_WBC_TASK_DURATION_S must be a positive integer" >&2
  exit 1
fi
if [[ ! -f "$model/config.json" ]]; then
  echo "Invalid checkpoint directory: $model" >&2
  exit 1
fi
if [[ -e "$result" ]]; then
  echo "Result tag already exists: $result" >&2
  exit 1
fi
mkdir -p "$result/trials"

stop_trial() {
  set +e
  if [[ -n "$client_pid" ]]; then kill -TERM -- "-$client_pid" 2>/dev/null; fi
  if [[ -n "$sim_pid" ]]; then kill -TERM -- "-$sim_pid" 2>/dev/null; fi
  docker rm -f gr00t-wbc-controller >/dev/null 2>&1
  wait "$client_pid" 2>/dev/null
  wait "$sim_pid" 2>/dev/null
  client_pid=""
  sim_pid=""
  controller_pid=""
  set -e
}

cleanup() {
  status=$?
  set +e
  stop_trial
  set +e
  docker rm -f gr00t-wbc-policy gr00t-wbc-dds-bridge >/dev/null 2>&1
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
printf 'trials_per_scenario=%s\ntask_duration_s=%s\n' \
  "$trials_per_scenario" "$task_duration" >"$result/campaign-config.txt"
docker ps --format '{{.Names}}|{{.ID}}|{{.Status}}' >"$result/containers-before.txt"
nvidia-smi >"$result/nvidia-smi-before.txt"

for container in "${protected[@]}"; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
    echo "Protected container is not running before maintenance: $container" >&2
    exit 1
  fi
  docker inspect -f '{{.Name}}|{{.Id}}|{{.RestartCount}}' "$container"
  stopped+=("$container")
done >"$result/protected-container-identities.txt"
docker stop --timeout 30 "${protected[@]}" | tee "$result/stopped-containers.txt"

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

run_trial() {
  local scenario="$1"
  local target="$2"
  local prompt="$3"
  local seed="$4"
  local trial_name
  local trial_result
  local task_metrics
  local task_arm_file
  local sim_ready=false
  local controller_ready=false
  local client_ready=false
  local control_started=false
  local action_ready=false
  local action_chunks
  local action_frames
  local controller_frames
  local task_status="running"

  trial_name="${scenario}-${target}-seed-$(printf '%02d' "$seed")"
  trial_result="$result/trials/$trial_name"
  task_metrics="$trial_result/task-metrics.json"
  task_arm_file="$trial_result/task-armed"
  mkdir -p "$trial_result"
  printf 'scenario=%s\ntarget=%s\nprompt=%s\nseed=%s\nduration_s=%s\n' \
    "$scenario" "$target" "$prompt" "$seed" "$task_duration" \
    >"$trial_result/evaluation-config.txt"

  GROOT_WBC_TASK_SCENARIO="$scenario" \
  GROOT_WBC_TASK_TARGET="$target" \
  GROOT_WBC_TASK_SEED="$seed" \
  GROOT_WBC_TASK_DURATION_S="$task_duration" \
  GROOT_WBC_TASK_METRICS_PATH="$task_metrics" \
  GROOT_WBC_TASK_ARM_FILE="$task_arm_file" \
  PYTHONUNBUFFERED=1 setsid .spark/run-sim.sh --env-name pnp_bottle \
    >"$trial_result/sim.log" 2>&1 &
  sim_pid=$!
  for _ in $(seq 1 60); do
    kill -0 "$sim_pid" 2>/dev/null || { tail -100 "$trial_result/sim.log" >&2 || true; return 1; }
    if ss -ltn | grep -q ':5555 '; then sim_ready=true; break; fi
    sleep 1
  done
  [[ "$sim_ready" == true ]] || { echo "Simulator camera server did not become ready" >&2; return 1; }

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
      sim >"$trial_result/controller.log" 2>&1 &
  controller_pid=$!
  for _ in $(seq 1 300); do
    if grep -q 'Initialized ZMQ output interface' "$trial_result/controller.log"; then
      controller_ready=true
      break
    fi
    kill -0 "$controller_pid" 2>/dev/null || { tail -120 "$trial_result/controller.log" >&2 || true; return 1; }
    sleep 2
  done
  [[ "$controller_ready" == true ]] || { echo "SONIC controller did not become ready" >&2; return 1; }

  PYTHONUNBUFFERED=1 setsid .spark/run-inference-client.sh \
    --host 127.0.0.1 \
    --port 5550 \
    --embodiment-tag unitree_g1_sonic \
    --prompt "$prompt" \
    --camera-host 127.0.0.1 \
    --camera-port 5555 \
    --action-zmq-host "$bridge_gateway" \
    --verbose-timing >"$trial_result/client.log" 2>&1 &
  client_pid=$!
  for _ in $(seq 1 60); do
    kill -0 "$client_pid" 2>/dev/null || { tail -120 "$trial_result/client.log" >&2 || true; return 1; }
    if grep -q 'Starting the policy loop' "$trial_result/client.log"; then client_ready=true; break; fi
    sleep 1
  done
  [[ "$client_ready" == true ]] || { echo "VLA client did not initialize" >&2; return 1; }

  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py k
  for _ in $(seq 1 20); do
    if grep -q 'Started C++ control loop' "$trial_result/client.log"; then control_started=true; break; fi
    sleep 1
  done
  [[ "$control_started" == true ]] || { echo "SONIC planner control did not start" >&2; return 1; }

  for _ in $(seq 1 120); do
    kill -0 "$client_pid" 2>/dev/null || { tail -120 "$trial_result/client.log" >&2 || true; return 1; }
    if grep -q 'New action chunk' "$trial_result/client.log"; then action_ready=true; break; fi
    sleep 2
  done
  [[ "$action_ready" == true ]] || { echo "VLA client did not receive an action chunk" >&2; return 1; }

  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py i
  sleep 2
  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py p
  install -m 0644 /dev/null "$task_arm_file"

  for _ in $(seq 1 "$task_duration"); do
    kill -0 "$client_pid"
    docker inspect -f '{{.State.Running}}' gr00t-wbc-policy | grep -qx true
    docker inspect -f '{{.State.Running}}' gr00t-wbc-controller | grep -qx true
    sleep 1
    if [[ -s "$task_metrics" ]]; then
      task_status=$("$repo/.venv_inference/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$task_metrics")
      if [[ "$task_status" == "success" || "$task_status" == "object_off_table" ]]; then
        echo "Trial $trial_name reached terminal status: $task_status"
        break
      fi
    fi
  done

  action_chunks=$(grep -c 'New action chunk' "$trial_result/client.log" || true)
  action_frames=$(grep -c 'ZMQ: Sent latent action' "$trial_result/client.log" || true)
  controller_frames=$(grep -c 'frame_index:' "$trial_result/controller.log" || true)
  if (( action_chunks < 3 || controller_frames < 1 )); then
    echo "Insufficient closed-loop activity: chunks=$action_chunks controller_frames=$controller_frames" >&2
    return 1
  fi
  if [[ ! -s "$task_metrics" ]]; then
    echo "Task evaluator did not produce metrics" >&2
    return 1
  fi
  if grep -Eqi 'Traceback|segmentation fault|CUDA error|out of memory' \
      "$result/policy.log" "$trial_result/sim.log" \
      "$trial_result/controller.log" "$trial_result/client.log"; then
    echo "Fatal signature detected in trial logs" >&2
    return 1
  fi

  "$repo/.venv_inference/bin/python" .spark/analyze-eval.py "$trial_result" \
    >"$trial_result/performance.json"
  printf 'action_chunks=%s\naction_frame_markers=%s\ncontroller_frames=%s\ntermination_reason=%s\n' \
    "$action_chunks" "$action_frames" "$controller_frames" "$task_status" \
    >"$trial_result/summary.txt"
  stop_trial
  sleep 2
}

scenarios=(single_bottle bottle_apple bottle_apple)
targets=(bottle bottle apple)
prompts=('grab the bottle' 'grab the bottle' 'grab the red apple')

for index in "${!scenarios[@]}"; do
  for seed in $(seq 0 $((trials_per_scenario - 1))); do
    echo "Starting ${scenarios[$index]}/${targets[$index]} seed=$seed"
    run_trial "${scenarios[$index]}" "${targets[$index]}" "${prompts[$index]}" "$seed"
  done
done

nvidia-smi >"$result/nvidia-smi-evaluation.txt"
"$repo/.venv_inference/bin/python" .spark/analyze-task-campaign.py "$result" \
  >"$result/campaign-metrics.json"
cat "$result/campaign-metrics.json"
