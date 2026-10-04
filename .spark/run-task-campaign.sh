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
campaign_scenarios="${GROOT_WBC_CAMPAIGN_SCENARIOS:-all}"
campaign_seeds="${GROOT_WBC_SEEDS:-}"
model_revision="${GROOT_WBC_MODEL_REVISION:-5fdb36c78c88b9cc3a2c584fcd8993e9955b2384}"
resume="${GROOT_WBC_RESUME:-NO}"
resuming=false
protected=(pi05-fp8-production qwen3-vl-judge triton-spark)
bridge_gateway="$(docker network inspect bridge -f '{{(index .IPAM.Config 0).Gateway}}')"
stopped=()
sim_pid=""
client_pid=""
policy_pid=""
controller_pid=""
video_pid=""

if [[ ! "$trials_per_scenario" =~ ^[0-9]+$ ]] || (( trials_per_scenario < 1 )); then
  echo "GROOT_WBC_TRIALS_PER_SCENARIO must be a positive integer" >&2
  exit 1
fi
if [[ ! "$task_duration" =~ ^[0-9]+$ ]] || (( task_duration < 1 )); then
  echo "GROOT_WBC_TASK_DURATION_S must be a positive integer" >&2
  exit 1
fi
if [[ "$campaign_scenarios" != "all" && "$campaign_scenarios" != "single_bottle" ]]; then
  echo "GROOT_WBC_CAMPAIGN_SCENARIOS must be all or single_bottle" >&2
  exit 1
fi
if [[ -n "$campaign_seeds" && ! "$campaign_seeds" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GROOT_WBC_SEEDS must be a comma-separated list of non-negative integers" >&2
  exit 1
fi
if [[ -n "$campaign_seeds" ]]; then
  IFS=',' read -r -a campaign_seed_values <<<"$campaign_seeds"
else
  mapfile -t campaign_seed_values < <(seq 0 $((trials_per_scenario - 1)))
fi
if [[ ! -f "$model/config.json" ]]; then
  echo "Invalid checkpoint directory: $model" >&2
  exit 1
fi
if [[ -e "$result" && "$resume" != "YES" ]]; then
  echo "Result tag already exists: $result" >&2
  exit 1
fi
if [[ -e "$result" ]]; then
  resuming=true
  [[ -f "$result/source-commit.txt" ]] || {
    echo "Cannot resume result without source-commit.txt: $result" >&2
    exit 1
  }
  original_commit=$(<"$result/source-commit.txt")
  if ! git -C "$repo" diff --quiet "$original_commit" HEAD -- \
      gear_sonic/utils/mujoco_sim/base_sim.py .spark/analyze-task-campaign.py; then
    echo "Evaluator changed since the campaign began; refusing mixed-code resume." >&2
    exit 1
  fi
  interruption="$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$result/interrupted-trials" "$result/interruptions/$interruption"
  for artifact in evidence.sha256 exit-code.txt containers-after.txt nvidia-smi-after.txt; do
    [[ ! -e "$result/$artifact" ]] || mv "$result/$artifact" "$result/interruptions/$interruption/$artifact"
  done
fi
mkdir -p "$result/trials"

stop_trial() {
  set +e
  if [[ -n "$video_pid" ]]; then kill -TERM "$video_pid" 2>/dev/null; fi
  if [[ -n "$client_pid" ]]; then kill -TERM -- "-$client_pid" 2>/dev/null; fi
  if [[ -n "$sim_pid" ]]; then kill -TERM -- "-$sim_pid" 2>/dev/null; fi
  docker rm -f gr00t-wbc-controller >/dev/null 2>&1
  wait "$client_pid" 2>/dev/null
  wait "$sim_pid" 2>/dev/null
  wait "$video_pid" 2>/dev/null
  client_pid=""
  sim_pid=""
  controller_pid=""
  video_pid=""
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
  if [[ ! -f "$result/campaign-complete" && "$status" -eq 0 ]]; then
    status=125
  fi
  printf '%s\n' "$status" >"$result/exit-code.txt"
  find "$result" -type f ! -name evidence.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum >"$result/evidence.sha256"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

cd "$repo"
if [[ "$resuming" == true ]]; then
  git rev-parse HEAD >"$result/resume-commit.txt"
else
  git rev-parse HEAD >"$result/source-commit.txt"
  printf '%s\n' "$model_revision" >"$result/model-revision.txt"
  sha256sum "$model"/model-*.safetensors >"$result/model-sha256.txt"
  printf 'trials_per_scenario=%s\ntask_duration_s=%s\ncampaign_scenarios=%s\ncampaign_seeds=%s\n' \
    "$trials_per_scenario" "$task_duration" "$campaign_scenarios" "$campaign_seeds" \
    >"$result/campaign-config.txt"
fi
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
  local task_history
  local task_arm_file
  local task_reset_file
  local sim_ready=false
  local controller_ready=false
  local client_ready=false
  local control_started=false
  local action_ready=false
  local action_chunks
  local action_frames
  local controller_frames
  local controller_lines_before_init
  local initial_motion_complete=false
  local pose_mode_ready=false
  local object_reset_ready=false
  local policy_paused_on_lift=false
  local task_status="running"

  trial_name="${scenario}-${target}-seed-$(printf '%02d' "$seed")"
  trial_result="$result/trials/$trial_name"
  task_metrics="$trial_result/task-metrics.json"
  task_history="$trial_result/task-history.jsonl"
  task_arm_file="$trial_result/task-armed"
  task_reset_file="$trial_result/reset-objects"
  mkdir -p "$trial_result"
  printf 'scenario=%s\ntarget=%s\nprompt=%s\nseed=%s\nduration_s=%s\nlift_height_m=%s\nhold_time_s=%s\nbottle_position=%s\nrobot_body_q=%s\nlock_lower_body=%s\nwaist_yaw_bounds_rad=%s\nassisted_grasp=%s\npause_on_lift=%s\npolicy_inference_seed=%s\nsonic_checkpoint=%s\nsonic_obs_config=%s\nego_video_override=%s\nobservation_trace_override=%s\naction_trace_override=%s\nhand_action_lead_frames=%s\n' \
    "$scenario" "$target" "$prompt" "$seed" "$task_duration" \
    "${GROOT_WBC_TASK_LIFT_HEIGHT_M:-0.045}" \
    "${GROOT_WBC_TASK_HOLD_TIME_S:-0.5}" \
    "${GROOT_WBC_TASK_BOTTLE_POSITION:-grid}" \
    "${GROOT_WBC_TASK_ROBOT_BODY_Q:-controller-initialized}" \
    "${GROOT_WBC_LOCK_LOWER_BODY:-0}" \
    "${GROOT_WBC_WAIST_YAW_BOUNDS_RAD:-unconstrained}" \
    "${GROOT_WBC_ASSISTED_GRASP:-0}" \
    "${GROOT_WBC_PAUSE_ON_LIFT:-0}" \
    "${GROOT_POLICY_INFERENCE_SEED:-random}" \
    "${GROOT_WBC_SONIC_CHECKPOINT:-policy/release/model}" \
    "${GROOT_WBC_SONIC_OBS_CONFIG:-policy/release/observation_config.yaml}" \
    "${GROOT_WBC_EGO_VIDEO_OVERRIDE:-none}" \
    "${GROOT_WBC_OBSERVATION_TRACE_OVERRIDE:-none}" \
    "${GROOT_WBC_ACTION_TRACE_OVERRIDE:-none}" \
    "${GROOT_WBC_HAND_ACTION_LEAD_FRAMES:-0}" \
    >"$trial_result/evaluation-config.txt"

  GROOT_WBC_TASK_SCENARIO="$scenario" \
  GROOT_WBC_TASK_TARGET="$target" \
  GROOT_WBC_TASK_SEED="$seed" \
  GROOT_WBC_TASK_DURATION_S="$task_duration" \
  GROOT_WBC_TASK_LIFT_HEIGHT_M="${GROOT_WBC_TASK_LIFT_HEIGHT_M:-0.045}" \
  GROOT_WBC_TASK_HOLD_TIME_S="${GROOT_WBC_TASK_HOLD_TIME_S:-0.5}" \
  GROOT_WBC_TASK_METRICS_PATH="$task_metrics" \
  GROOT_WBC_TASK_HISTORY_PATH="$task_history" \
  GROOT_WBC_TASK_ARM_FILE="$task_arm_file" \
  GROOT_WBC_TASK_RESET_FILE="$task_reset_file" \
  GROOT_WBC_TASK_BOTTLE_POSITION="${GROOT_WBC_TASK_BOTTLE_POSITION:-}" \
  GROOT_WBC_TASK_ROBOT_BODY_Q="${GROOT_WBC_TASK_ROBOT_BODY_Q:-}" \
  GROOT_WBC_LOCK_LOWER_BODY="${GROOT_WBC_LOCK_LOWER_BODY:-0}" \
  GROOT_WBC_WAIST_YAW_BOUNDS_RAD="${GROOT_WBC_WAIST_YAW_BOUNDS_RAD:-}" \
  GROOT_WBC_ASSISTED_GRASP="${GROOT_WBC_ASSISTED_GRASP:-0}" \
  GROOT_WBC_ASSISTED_GRASP_CLOSE_THRESHOLD="${GROOT_WBC_ASSISTED_GRASP_CLOSE_THRESHOLD:-0.5}" \
  GROOT_WBC_ASSISTED_GRASP_RELEASE_THRESHOLD="${GROOT_WBC_ASSISTED_GRASP_RELEASE_THRESHOLD:-0.15}" \
  GROOT_WBC_ASSISTED_GRASP_CONTACT_GRACE_S="${GROOT_WBC_ASSISTED_GRASP_CONTACT_GRACE_S:-0.75}" \
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
      --cp "${GROOT_WBC_SONIC_CHECKPOINT:-policy/release/model}" \
      --obs-config "${GROOT_WBC_SONIC_OBS_CONFIG:-policy/release/observation_config.yaml}" \
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

  GROOT_WBC_NORMALIZED_HAND_ACTIONS="${GROOT_WBC_NORMALIZED_HAND_ACTIONS:-1}" \
  GROOT_WBC_ZERO_HAND_STATE="${GROOT_WBC_ZERO_HAND_STATE:-1}" \
  GROOT_WBC_LOG_ACTION_SUMMARY="${GROOT_WBC_LOG_ACTION_SUMMARY:-1}" \
  GROOT_WBC_LOG_OBSERVATION_Q="${GROOT_WBC_LOG_OBSERVATION_Q:-1}" \
  GROOT_WBC_EGO_VIDEO_OVERRIDE="${GROOT_WBC_EGO_VIDEO_OVERRIDE:-}" \
  GROOT_WBC_OBSERVATION_TRACE_OVERRIDE="${GROOT_WBC_OBSERVATION_TRACE_OVERRIDE:-}" \
  GROOT_WBC_ACTION_TRACE_OVERRIDE="${GROOT_WBC_ACTION_TRACE_OVERRIDE:-}" \
  GROOT_WBC_HAND_ACTION_LEAD_FRAMES="${GROOT_WBC_HAND_ACTION_LEAD_FRAMES:-0}" \
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

  controller_lines_before_init=$(wc -l <"$trial_result/controller.log")
  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py i
  pose_mode_ready=false
  for _ in $(seq 1 20); do
    kill -0 "$client_pid" 2>/dev/null || { tail -120 "$trial_result/client.log" >&2 || true; return 1; }
    if tail -n "+$((controller_lines_before_init + 1))" "$trial_result/controller.log" \
        | grep -q 'ZMQ STREAMING MODE: ENABLED'; then
      pose_mode_ready=true
      break
    fi
    sleep 1
  done
  if [[ "$pose_mode_ready" != true ]]; then
    echo "SONIC did not enter streamed pose mode" >&2
    tail -120 "$trial_result/controller.log" >&2 || true
    return 1
  fi
  # The planner-to-pose transition clears any token already queued. Resend the
  # initial pose after streamed mode is active so the controller can consume it.
  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py i
  initial_motion_complete=false
  for _ in $(seq 1 20); do
    kill -0 "$client_pid" 2>/dev/null || { tail -120 "$trial_result/client.log" >&2 || true; return 1; }
    if ! docker inspect -f '{{.State.Running}}' gr00t-wbc-controller 2>/dev/null | grep -qx true; then
      tail -120 "$trial_result/controller.log" >&2 || true
      return 1
    fi
    if tail -n "+$((controller_lines_before_init + 1))" "$trial_result/controller.log" \
        | grep -q 'Temporary motion completed.'; then
      initial_motion_complete=true
      break
    fi
    sleep 1
  done
  if [[ "$initial_motion_complete" != true ]]; then
    echo "SONIC did not complete the initial-pose transition" >&2
    tail -120 "$trial_result/controller.log" >&2 || true
    return 1
  fi
  sleep 1
  install -m 0644 /dev/null "$task_reset_file"
  object_reset_ready=false
  for _ in $(seq 1 20); do
    if [[ -s "$task_metrics" ]] && "$repo/.venv_inference/bin/python" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("reset_applied") else 1)' \
        "$task_metrics"; then
      object_reset_ready=true
      break
    fi
    sleep 0.25
  done
  if [[ "$object_reset_ready" != true ]]; then
    echo "Simulator did not apply the post-initialization object reset" >&2
    return 1
  fi
  PYTHONPATH="$repo" "$repo/.venv_sim/bin/python" /tmp/gr00t-record-camera.py \
    "$trial_result/ego-view.mp4" --duration "$((task_duration + 5))" \
    >"$trial_result/video.log" 2>&1 &
  video_pid=$!
  resume_markers_before=$(grep -c \
    'Cleared cached action chunk for a fresh resume observation' \
    "$trial_result/client.log" || true)
  "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py p
  resume_marker_ready=false
  for _ in $(seq 1 40); do
    resume_markers_now=$(grep -c \
      'Cleared cached action chunk for a fresh resume observation' \
      "$trial_result/client.log" || true)
    if (( resume_markers_now > resume_markers_before )); then
      resume_marker_ready=true
      break
    fi
    sleep 0.25
  done
  if [[ "$resume_marker_ready" != true ]]; then
    echo "VLA client did not acknowledge policy resume" >&2
    return 1
  fi
  resume_marker_line=$(grep -n \
    'Cleared cached action chunk for a fresh resume observation' \
    "$trial_result/client.log" | tail -1 | cut -d: -f1)
  resume_action_ready=false
  for _ in $(seq 1 40); do
    kill -0 "$client_pid" 2>/dev/null || { tail -120 "$trial_result/client.log" >&2 || true; return 1; }
    if tail -n "+$((resume_marker_line + 1))" "$trial_result/client.log" \
        | grep -q 'New action chunk'; then
      resume_action_ready=true
      break
    fi
    sleep 0.25
  done
  if [[ "$resume_action_ready" != true ]]; then
    echo "VLA client did not produce a fresh action chunk after resume" >&2
    return 1
  fi
  install -m 0644 /dev/null "$task_arm_file"

  for _ in $(seq 1 $(((task_duration + 5) * 10))); do
    kill -0 "$client_pid"
    docker inspect -f '{{.State.Running}}' gr00t-wbc-policy | grep -qx true
    docker inspect -f '{{.State.Running}}' gr00t-wbc-controller | grep -qx true
    sleep 0.1
    if [[ -s "$task_metrics" ]]; then
      read -r task_status task_lifted task_grasp_active < <(
        "$repo/.venv_inference/bin/python" -c \
          'import json,sys; x=json.load(open(sys.argv[1])); print(x["status"], int(x.get("lift_observed", False)), int(x.get("assisted_grasp_active", False)))' \
          "$task_metrics"
      )
      if [[ "${GROOT_WBC_PAUSE_ON_LIFT:-0}" != "0" \
          && "$policy_paused_on_lift" == false \
          && "$task_status" == "running" \
          && "$task_lifted" == 1 \
          && "$task_grasp_active" == 1 ]]; then
        pause_markers_before=$(grep -c 'Paused policy loop' "$trial_result/client.log" || true)
        "$repo/.venv_inference/bin/python" .spark/send-keyboard-command.py p
        pause_marker_ready=false
        for _ in $(seq 1 20); do
          pause_markers_now=$(grep -c 'Paused policy loop' "$trial_result/client.log" || true)
          if (( pause_markers_now > pause_markers_before )); then
            pause_marker_ready=true
            break
          fi
          sleep 0.1
        done
        if [[ "$pause_marker_ready" != true ]]; then
          echo "VLA client did not acknowledge pause-on-lift" >&2
          return 1
        fi
        install -m 0644 /dev/null "$trial_result/policy-paused-on-lift"
        policy_paused_on_lift=true
        echo "Paused policy updates after latched lift in $trial_name"
      fi
      if [[ "$task_status" == "success" || "$task_status" == "object_off_table" || "$task_status" == "simulator_unstable" || "$task_status" == "complete" ]]; then
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
if [[ "$campaign_scenarios" == "single_bottle" ]]; then
  scenarios=(single_bottle)
  targets=(bottle)
  prompts=('grab the bottle')
fi

for index in "${!scenarios[@]}"; do
  for seed in "${campaign_seed_values[@]}"; do
    trial_name="${scenarios[$index]}-${targets[$index]}-seed-$(printf '%02d' "$seed")"
    trial_result="$result/trials/$trial_name"
    if [[ -f "$trial_result/task-metrics.json" \
        && -f "$trial_result/performance.json" \
        && -f "$trial_result/summary.txt" ]]; then
      task_status=$("$repo/.venv_inference/bin/python" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$trial_result/task-metrics.json")
      if [[ "$task_status" == "success" || "$task_status" == "object_off_table" \
          || "$task_status" == "simulator_unstable" || "$task_status" == "complete" ]]; then
        echo "Skipping completed $trial_name ($task_status)"
        continue
      fi
    fi
    if [[ -e "$trial_result" ]]; then
      archived="$result/interrupted-trials/$trial_name-$(date -u +%Y%m%dT%H%M%SZ)"
      echo "Archiving incomplete $trial_name as ${archived##*/}"
      mv "$trial_result" "$archived"
    fi
    echo "Starting ${scenarios[$index]}/${targets[$index]} seed=$seed"
    run_trial "${scenarios[$index]}" "${targets[$index]}" "${prompts[$index]}" "$seed"
  done
done

nvidia-smi >"$result/nvidia-smi-evaluation.txt"
"$repo/.venv_inference/bin/python" .spark/analyze-task-campaign.py "$result" \
  >"$result/campaign-metrics.json"
install -m 0644 /dev/null "$result/campaign-complete"
cat "$result/campaign-metrics.json"
