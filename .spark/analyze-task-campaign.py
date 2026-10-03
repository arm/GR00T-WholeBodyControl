#!/usr/bin/env python3
"""Aggregate deterministic GR00T + SONIC bottle-task campaign results."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1 + z * z / total
    center = (probability + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(probability * (1 - probability) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    args = parser.parse_args()

    trials = []
    for trial_dir in sorted((args.campaign_dir / "trials").iterdir()):
        if not trial_dir.is_dir():
            continue
        task_path = trial_dir / "task-metrics.json"
        performance_path = trial_dir / "performance.json"
        if not task_path.exists() or not performance_path.exists():
            raise SystemExit(f"Incomplete trial: {trial_dir}")
        task = json.loads(task_path.read_text())
        performance = json.loads(performance_path.read_text())
        if task["status"] not in {"success", "object_off_table", "simulator_unstable", "complete"}:
            raise SystemExit(f"Nonterminal task metrics in {trial_dir}: {task['status']}")
        trials.append(
            {
                "name": trial_dir.name,
                "scenario": task["scenario"],
                "target": task["target"],
                "seed": task["seed"],
                "status": task["status"],
                "task_time_s": task["task_time_s"],
                "simulator_time_s": task["simulator_time_s"],
                "success": task["success"],
                "object_off_table": task["object_off_table"],
                "contact_observed": task["contact_observed"],
                "lift_observed": task["lift_observed"],
                "wrong_object_lifted": task["wrong_object_lifted"],
                "robot_falls": task["robot_falls"],
                "simulator_instabilities": task.get("simulator_instabilities", 0),
                "latency_ms": performance["latency_ms"],
                "camera_frequency_hz": performance["camera_frequency_hz"],
                "camera_dropped_messages": performance["camera_dropped_messages"],
            }
        )

    groups = defaultdict(list)
    for trial in trials:
        groups[(trial["scenario"], trial["target"])].append(trial)

    scenarios = {}
    for (scenario, target), members in groups.items():
        successes = sum(bool(member["success"]) for member in members)
        key = f"{scenario}/{target}"
        scenarios[key] = {
            "trials": len(members),
            "successes": successes,
            "success_rate": successes / len(members),
            "success_rate_wilson_95": wilson_interval(successes, len(members)),
            "contacts": sum(bool(member["contact_observed"]) for member in members),
            "lifts": sum(bool(member["lift_observed"]) for member in members),
            "wrong_object_lifts": sum(
                bool(member["wrong_object_lifted"]) for member in members
            ),
            "objects_off_table": sum(
                bool(member["object_off_table"]) for member in members
            ),
            "robot_falls": sum(member["robot_falls"] for member in members),
            "simulator_instabilities": sum(
                member["simulator_instabilities"] for member in members
            ),
            "latency_ms": {
                "median_mean": statistics.fmean(
                    member["latency_ms"]["median"] for member in members
                ),
                "p95_mean": statistics.fmean(
                    member["latency_ms"]["p95"] for member in members
                ),
                "max": max(member["latency_ms"]["max"] for member in members),
            },
            "camera_frequency_hz_mean": statistics.fmean(
                member["camera_frequency_hz"]["mean"] for member in members
            ),
            "camera_dropped_messages": sum(
                member["camera_dropped_messages"] for member in members
            ),
        }

    output = {
        "schema_version": 1,
        "campaign": "gr00t-n1.7-sonic-wbc-bottle-task",
        "total_trials": len(trials),
        "technical_failures": 0,
        "task_successes": sum(bool(trial["success"]) for trial in trials),
        "scenarios": scenarios,
        "trials": trials,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
