"""Low-overhead real-time tracing for the GR00T to SONIC/WBC pipeline."""

from __future__ import annotations

from collections import defaultdict, deque
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Iterable, TextIO


TRACE_PREFIX = "GROOT_TRACE "
TRACE_SCHEMA_VERSION = 1


def _env_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }


class EventTracer:
    """Emit one atomic JSON event per line using the host monotonic clock."""

    def __init__(
        self,
        source: str,
        *,
        enabled: bool | None = None,
        stream: TextIO | None = None,
        monotonic_ns=time.monotonic_ns,
        wall_time_ns=time.time_ns,
    ):
        self.source = source
        self.enabled = (
            _env_enabled(os.environ.get("GROOT_WBC_TRACE_EVENTS"))
            if enabled is None
            else enabled
        )
        self.stream = sys.stdout if stream is None else stream
        self._monotonic_ns = monotonic_ns
        self._wall_time_ns = wall_time_ns
        self._lock = threading.Lock()

    def emit(self, event: str, **fields) -> dict | None:
        if not self.enabled:
            return None
        payload = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "source": self.source,
            "event": event,
            "mono_ns": self._monotonic_ns(),
            "wall_ns": self._wall_time_ns(),
        }
        payload.update(fields)
        line = TRACE_PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock:
            print(line, file=self.stream, flush=True)
        return payload


def read_trace_events(paths: Iterable[Path]) -> list[dict]:
    """Read embedded trace records from one or more retained process logs."""

    events = []
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(errors="replace").splitlines(), start=1
        ):
            prefix_at = line.find(TRACE_PREFIX)
            if prefix_at < 0:
                continue
            try:
                event = json.loads(line[prefix_at + len(TRACE_PREFIX) :])
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed trace event in {path}:{line_number}") from error
            if event.get("schema_version") != TRACE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported trace schema in {path}:{line_number}: "
                    f"{event.get('schema_version')}"
                )
            if not isinstance(event.get("mono_ns"), int):
                raise ValueError(f"Trace event has no monotonic timestamp: {path}:{line_number}")
            events.append(event)
    return sorted(events, key=lambda event: event["mono_ns"])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict | None:
    if not values:
        return None
    return {
        "samples": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def _active_window(events: list[dict]) -> tuple[int, int]:
    resumes = [
        event["mono_ns"]
        for event in events
        if event["source"] == "vla_client" and event["event"] == "policy_resumed"
    ]
    starts = [
        event["mono_ns"]
        for event in events
        if event["source"] == "simulator" and event["event"] == "task_armed"
    ]
    if not resumes and not starts:
        raise ValueError("Trace has no policy_resumed or task_armed event")
    start_ns = max(starts or resumes)
    pauses = [
        event["mono_ns"]
        for event in events
        if event["source"] == "vla_client"
        and event["event"] == "policy_paused"
        and event["mono_ns"] > start_ns
    ]
    terminals = [
        event["mono_ns"]
        for event in events
        if event["source"] == "simulator"
        and event["event"] in {"task_success", "task_failed"}
        and event["mono_ns"] > start_ns
    ]
    candidates = pauses + terminals
    end_ns = min(candidates) if candidates else events[-1]["mono_ns"]
    return start_ns, end_ns


def summarize_events(
    events: list[dict],
    *,
    action_rate_hz: float = 50.0,
    action_horizon: int = 40,
) -> dict:
    """Calculate GR00T/WBC real-time behavior from correlated trace events."""

    if not events:
        raise ValueError("No real-time trace events found")
    start_ns, end_ns = _active_window(events)
    active = [
        event for event in events if start_ns <= event["mono_ns"] <= end_ns
    ]

    completed = [
        event
        for event in active
        if event["source"] == "vla_client"
        and event["event"] == "inference_completed"
    ]
    accepted = [
        event
        for event in active
        if event["source"] == "vla_client" and event["event"] == "chunk_accepted"
    ]
    published = [
        event
        for event in active
        if event["source"] == "vla_client" and event["event"] == "action_published"
    ]
    received = [
        event
        for event in active
        if event["source"] == "wbc" and event["event"] == "action_received"
    ]

    inference_ms = [event["duration_ns"] / 1e6 for event in completed]
    observation_ms = [event["observation_ns"] / 1e6 for event in completed]
    accepted_times = [event["mono_ns"] for event in accepted]
    chunk_interval_ms = [
        (current - previous) / 1e6
        for previous, current in zip(accepted_times, accepted_times[1:])
    ]
    horizon_margin_ms = [
        (action_horizon - 1 - event["action_index"]) * 1000.0 / action_rate_hz
        for event in accepted
    ]

    inference_start_by_chunk = {
        event["chunk_id"]: event["mono_ns"]
        for event in active
        if event["source"] == "vla_client"
        and event["event"] == "inference_started"
    }
    action_age_ms = [
        (event["mono_ns"] - inference_start_by_chunk[event["chunk_id"]]) / 1e6
        for event in published
        if event.get("chunk_id") in inference_start_by_chunk
    ]

    publish_times = [event["mono_ns"] for event in published]
    publish_interval_ms = [
        (current - previous) / 1e6
        for previous, current in zip(publish_times, publish_times[1:])
    ]
    expected_interval_ms = 1000.0 / action_rate_hz

    receive_queues: dict[int, deque[dict]] = defaultdict(deque)
    for event in received:
        receive_queues[event["frame_index"]].append(event)
    transport_ms = []
    unmatched_published = 0
    for event in published:
        matches = receive_queues[event["frame_index"]]
        while matches and matches[0]["mono_ns"] < event["mono_ns"]:
            matches.popleft()
        if not matches:
            unmatched_published += 1
            continue
        transport_ms.append((matches.popleft()["mono_ns"] - event["mono_ns"]) / 1e6)

    frame_indices = [event["frame_index"] for event in published]
    gaps = 0
    duplicates = 0
    out_of_order = 0
    for previous, current in zip(frame_indices, frame_indices[1:]):
        if current == previous:
            duplicates += 1
        elif current < previous:
            out_of_order += 1
        elif current > previous + 1:
            gaps += current - previous - 1

    last_index_counts: dict[int, int] = defaultdict(int)
    for event in published:
        if event["action_index"] == action_horizon - 1:
            last_index_counts[event["chunk_id"]] += 1
    repeated_last_frames = sum(
        max(0, count - 1) for count in last_index_counts.values()
    )
    max_repeated_last_frames = max(
        (max(0, count - 1) for count in last_index_counts.values()),
        default=0,
    )

    marker_names = (
        "contact_started",
        "assisted_grasp_activated",
        "lift_started",
        "task_success",
        "task_failed",
        "policy_paused",
    )
    task_markers_s = {}
    event_precursors = {}
    for marker_name in marker_names:
        marker = next(
            (
                event
                for event in events
                if event["event"] == marker_name and event["mono_ns"] >= start_ns
            ),
            None,
        )
        if marker is None:
            continue
        task_markers_s[marker_name] = (marker["mono_ns"] - start_ns) / 1e9
        prior_action = next(
            (
                event
                for event in reversed(published)
                if event["mono_ns"] <= marker["mono_ns"]
            ),
            None,
        )
        if prior_action is not None:
            event_precursors[marker_name] = {
                "action_to_event_ms": (
                    marker["mono_ns"] - prior_action["mono_ns"]
                )
                / 1e6,
                "chunk_id": prior_action["chunk_id"],
                "action_index": prior_action["action_index"],
                "hand_action_index": prior_action.get("hand_action_index"),
                "frame_index": prior_action["frame_index"],
            }

    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "active_window": {
            "start_mono_ns": start_ns,
            "end_mono_ns": end_ns,
            "duration_s": (end_ns - start_ns) / 1e9,
        },
        "counts": {
            "events": len(active),
            "inferences": len(completed),
            "accepted_chunks": len(accepted),
            "actions_published": len(published),
            "actions_received": len(received),
        },
        "inference_latency_ms": distribution(inference_ms),
        "observation_capture_ms": distribution(observation_ms),
        "chunk_interval_ms": distribution(chunk_interval_ms),
        "horizon_margin_ms": distribution(horizon_margin_ms),
        "action_age_ms": distribution(action_age_ms),
        "publish_interval_ms": distribution(publish_interval_ms),
        "publish_deadlines": {
            "expected_interval_ms": expected_interval_ms,
            "over_1_5x": sum(
                interval > expected_interval_ms * 1.5
                for interval in publish_interval_ms
            ),
            "over_2x": sum(
                interval > expected_interval_ms * 2.0
                for interval in publish_interval_ms
            ),
        },
        "action_transport_ms": distribution(transport_ms),
        "frame_continuity": {
            "gaps": gaps,
            "duplicates": duplicates,
            "out_of_order": out_of_order,
            "unmatched_published": unmatched_published,
        },
        "horizon_exhaustion": {
            "repeated_last_action_frames": repeated_last_frames,
            "chunks_repeating_last_action": sum(
                count > 1 for count in last_index_counts.values()
            ),
            "max_repeated_last_action_frames": max_repeated_last_frames,
            "max_repeated_last_action_ms": (
                max_repeated_last_frames * 1000.0 / action_rate_hz
            ),
        },
        "task_markers_s": task_markers_s,
        "event_precursors": event_precursors,
    }


def perfetto_trace(events: list[dict]) -> dict:
    """Convert retained events into a Perfetto/Chrome trace-event document."""

    trace_events = []
    lanes = {"vla_client": (1, 1), "wbc": (2, 1), "simulator": (3, 1)}
    for event in events:
        source = event["source"]
        pid, tid = lanes.get(source, (9, 1))
        timestamp_us = event["mono_ns"] / 1000.0
        args = {
            key: value
            for key, value in event.items()
            if key not in {"schema_version", "source", "event", "mono_ns", "wall_ns"}
        }
        if event["event"] == "inference_completed":
            trace_events.append(
                {
                    "name": "GR00T inference",
                    "cat": "gr00t",
                    "ph": "X",
                    "ts": (event["mono_ns"] - event["duration_ns"]) / 1000.0,
                    "dur": event["duration_ns"] / 1000.0,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        elif event["event"] == "sim_state":
            trace_events.append(
                {
                    "name": "physical state",
                    "cat": "simulation",
                    "ph": "C",
                    "ts": timestamp_us,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
        else:
            trace_events.append(
                {
                    "name": event["event"],
                    "cat": source,
                    "ph": "i",
                    "s": "t",
                    "ts": timestamp_us,
                    "pid": pid,
                    "tid": tid,
                    "args": args,
                }
            )
    return {
        "displayTimeUnit": "ms",
        "traceEvents": trace_events,
    }
