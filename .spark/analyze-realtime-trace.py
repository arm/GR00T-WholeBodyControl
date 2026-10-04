#!/usr/bin/env python3
"""Build a synchronized GR00T/SONIC/WBC real-time trace report."""

import argparse
import json
from pathlib import Path

from gear_sonic.utils.inference.realtime_trace import (
    perfetto_trace,
    read_trace_events,
    summarize_events,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--action-rate-hz", type=float, default=50.0)
    parser.add_argument("--action-horizon", type=int, default=40)
    parser.add_argument("--perfetto-output", type=Path)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    log_paths = [
        args.trial_dir / "client.log",
        args.trial_dir / "controller.log",
        args.trial_dir / "sim.log",
    ]
    events = read_trace_events(log_paths)
    sources = {event["source"] for event in events}
    required_sources = {"vla_client", "wbc", "simulator"}
    missing_sources = required_sources - sources
    if missing_sources and not args.allow_partial:
        raise SystemExit(
            "Trace is missing required sources: " + ", ".join(sorted(missing_sources))
        )

    summary = summarize_events(
        events,
        action_rate_hz=args.action_rate_hz,
        action_horizon=args.action_horizon,
    )
    if not args.allow_partial:
        missing_metrics = [
            name
            for name in (
                "inference_latency_ms",
                "horizon_margin_ms",
                "action_transport_ms",
                "publish_interval_ms",
            )
            if summary[name] is None
        ]
        if missing_metrics:
            raise SystemExit(
                "Trace is missing required metrics: " + ", ".join(missing_metrics)
            )
    summary["sources"] = sorted(sources)

    if args.events_output:
        args.events_output.write_text(
            "".join(json.dumps(event, sort_keys=True) + "\n" for event in events)
        )
    if args.perfetto_output:
        args.perfetto_output.write_text(
            json.dumps(perfetto_trace(events), separators=(",", ":")) + "\n"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
