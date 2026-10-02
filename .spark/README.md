# GR00T + SONIC + Whole-Body Control on DGX Spark

This directory makes the upstream four-process simulation evaluation reproducible on the ARM64 DGX Spark. It pins the source and container dependencies, builds the TensorRT controller, and provides integration and task-level launchers.

## Status and pins

- GR00T-WholeBodyControl: `b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`
- Isaac-GR00T: `376ba890cff8c9de64d71d982772a9c36185fdd7`
- Controller image: `gr00t-wbc-spark:trt10.13-ort1.16.3`
- Base image: `nvcr.io/nvidia/deepstream@sha256:c25d5449fa3b2a82aeb6cdf224bdb422361ccf3e98db98077dfac3c2305048bb`
- TensorRT 10.13.2.6, CUDA 13.0, ONNX Runtime 1.16.3
- SONIC v1.1 encoder SHA256: `fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae`
- SONIC v1.1 decoder SHA256: `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509`
- Target-velocity planner SHA256: `39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`

The checked-in scripts do not contain model weights or generated build products.

## Required policy checkpoint

The closed-loop evaluation requires a GR00T N1.7 checkpoint fine-tuned for the `UNITREE_G1_SONIC` embodiment. A LIBERO or generic checkpoint is not compatible: the policy must emit the SONIC 64-value motion latent plus seven joints for each hand. Supply the checkpoint as an absolute local directory containing `config.json` and its weights.

For the initial integration evaluation, the Spark uses the community checkpoint `SII-Linzy/groot-g1-sonic-grab-bottle-deploy-checkpoint-10000`, pinned at Hugging Face revision `5fdb36c78c88b9cc3a2c584fcd8993e9955b2384`. It is a narrow bottle-grasp policy rather than an NVIDIA reference model; its published results are simulation-only. The local path is:

```text
models/sii-linzy-grab-bottle-checkpoint-10000
```

The policy launcher keeps the checkpoint metadata unchanged. A small Python startup overlay redirects only the gated Cosmos processor lookup to the already-pinned local snapshot; model loading then runs offline.

## Build

From the repository root:

```bash
.spark/build-controller.sh
```

## Run the simulation evaluation

Use four terminals on the Spark, all from this repository. These processes communicate through host networking.

Terminal 1 — start the GR00T policy server:

```bash
.spark/run-policy-server.sh \
  "$PWD/models/sii-linzy-grab-bottle-checkpoint-10000" 5550
```

Terminal 2 — start headless MuJoCo, its camera server, and Unitree DDS bridge:

```bash
.spark/run-sim.sh
```

Terminal 3 — start the TensorRT SONIC whole-body controller. Confirm the prompt only after the simulator is ready:

```bash
GROOT_WBC_NETWORK_MODE=bridge .spark/run-controller.sh \
  --cp policy/sonic_v1_1/model \
  --obs-config policy/sonic_v1_1/observation_config.yaml \
  --planner planner/target_vel/V2/planner_sonic.onnx \
  --motion-data reference/example \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host host.docker.internal \
  sim
```

Terminal 4 — connect observations to GR00T and stream its actions to SONIC:

```bash
.spark/run-inference-client.sh \
  --host 127.0.0.1 \
  --port 5550 \
  --embodiment-tag unitree_g1_sonic \
  --prompt "walk to the table and pick up the object" \
  --camera-host 127.0.0.1 \
  --camera-port 5555 \
  --action-zmq-host 172.17.0.1
```

Use the upstream keyboard publisher to send `k` to start/stop control, `p` to pause/resume policy inference, and `i` to return to the initial pose. For a scripted one-shot command:

```bash
.venv_inference/bin/python .spark/send-keyboard-command.py k
.venv_inference/bin/python .spark/send-keyboard-command.py p
```

## Verification and operational notes

The retained `integration-20261002-v7` Spark run passed end to end with 89 GR00T inference samples and 2,250 streamed SONIC latent-action frames. GR00T action-chunk latency was 179 ms median, 199.6 ms p95, and 200.12 ms p99; the camera averaged 32.51 Hz with zero dropped messages. This validates integration and runtime stability.

Evidence is retained at `/home/arm-seattle-spark-02/workspaces/gr00t-wbc-results/integration-20261002-v7`. Recompute its metrics with:

```bash
.venv_inference/bin/python .spark/analyze-eval.py \
  /home/arm-seattle-spark-02/workspaces/gr00t-wbc-results/integration-20261002-v7
```

## Bottle-task verification campaign

The task evaluator restores the repository's original `pnp_bottle_43dof.xml` scene and checks its camera framing against the checkpoint's published first- and third-person simulation videos. The original evaluation harness and placement manifest were not published, so this is a pinned reconstruction rather than a claim of exact reproduction. As in the checkpoint's published evaluation, the floating base and twelve leg joints are captured and locked when each task trial begins; the arms, waist, and hands remain policy-controlled. The evaluator adds deterministic placement seeds, a bottle-plus-red-apple variant, and a machine-readable success rule: the requested object must contact the right hand and remain at least 45 mm above its initial height for 0.5 seconds. Wrong-object lifts and robot falls are recorded separately.

Run a single task trial through the integration runner:

```bash
GROOT_WBC_APPROVE_EXCLUSIVE=YES \
GROOT_WBC_ENV_NAME=pnp_bottle \
GROOT_WBC_TASK_SCENARIO=single_bottle \
GROOT_WBC_TASK_TARGET=bottle \
GROOT_WBC_TASK_SEED=0 \
GROOT_WBC_PROMPT="grab the bottle" \
  .spark/run-integration-eval.sh \
    "$PWD/models/sii-linzy-grab-bottle-checkpoint-10000" \
    task-smoke-single-bottle-seed-00
```

Run the complete 90-trial campaign—30 deterministic placements for each published scenario:

```bash
GROOT_WBC_APPROVE_EXCLUSIVE=YES \
GROOT_WBC_TRIALS_PER_SCENARIO=30 \
GROOT_WBC_TASK_DURATION_S=90 \
  .spark/run-task-campaign.sh \
    "$PWD/models/sii-linzy-grab-bottle-checkpoint-10000" \
    task-campaign-20261002-v1
```

For a three-trial smoke campaign, set `GROOT_WBC_TRIALS_PER_SCENARIO=1`. A trial ends early only after a successful sustained lift or after the requested object falls irrecoverably below the table; otherwise it runs for the full duration. The campaign retains per-trial logs, task metrics, latency/camera metrics, container identities, model hashes, and an evidence manifest under `/home/arm-seattle-spark-02/workspaces/gr00t-wbc-results/<tag>`.

The checkpoint repository's reference videos are pinned at revision `5fdb36c78c88b9cc3a2c584fcd8993e9955b2384`. The successful reference clips used to validate camera framing have SHA256 values `b9fa72cb3522de0e3221c0ea02efd326b3712a949ca6cb268e9c6b03754243aa` for the ego view and `62e8e2d2835fe4ea90325533f842f2d5db4620f7636e46af2258aae8fe194f60` for the third-person view.

- `run-sim.sh --help`, `run-inference-client.sh --help`, controller compilation, ARM64 dynamic linkage, PolicyClient construction, and SONIC message serialization have been checked on this Spark.
- A headless simulation smoke test reached camera-server readiness on port 5655.
- The upstream Python test currently fails collection because it imports deleted input-reader helpers. The upstream C++ `run_tests` binary expects an absent `reference/bones_072925_test` fixture and then crashes; neither failure is caused by this Spark port.
- Full controller startup converts ONNX models to TensorRT engines and the GR00T server occupies the GPU. Run the closed-loop evaluation in an exclusive maintenance window; do not overlap it with production inference services.
- Spark loopback has no multicast. Simulation mode remains enabled while DDS binds to `docker0` through `GROOT_WBC_SIM_INTERFACE`.
- The exclusive runner starts a network-only bridge keeper while protected services are stopped. The simulator binds DDS to host `docker0`, while the controller uses bridge `eth0`; controller state returns through port 5557 and actions enter through the bridge gateway.
