#!/usr/bin/env python3
"""Summarize a retained GR00T + SONIC integration result directory."""

import argparse
import json
import math
import re
import statistics
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    client = (args.result_dir / "client.log").read_text(errors="replace")
    simulator = (args.result_dir / "sim.log").read_text(errors="replace")
    latencies = [
        float(value) * 1000
        for value in re.findall(r"New action chunk .*?latency: ([0-9.]+)s", client)
    ]
    frequencies = [
        float(value)
        for value in re.findall(r"Image publish frequency:\s+([0-9.]+)", simulator)
    ]
    frames = [int(value) for value in re.findall(r"frame: ([0-9]+)", client)]
    drops = [int(value) for value in re.findall(r"message dropped: (\d+)", simulator)]
    if not latencies or not frequencies or not frames:
        raise SystemExit("result is missing latency, camera, or action-frame samples")

    metrics = {
        "inference_samples": len(latencies),
        "latency_ms": {
            "min": min(latencies),
            "mean": statistics.fmean(latencies),
            "median": percentile(latencies, 0.50),
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies),
        },
        "streamed_action_frames": max(frames) + 1,
        "camera_frequency_hz": {
            "samples": len(frequencies),
            "mean": statistics.fmean(frequencies),
            "min": min(frequencies),
            "max": max(frequencies),
        },
        "camera_dropped_messages": max(drops, default=0),
    }
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
