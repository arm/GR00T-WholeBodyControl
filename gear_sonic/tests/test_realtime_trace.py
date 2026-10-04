import io
import json

import pytest

from gear_sonic.utils.inference.realtime_trace import (
    EventTracer,
    TRACE_PREFIX,
    perfetto_trace,
    read_trace_events,
    summarize_events,
)


def event(source, name, mono_ms, **fields):
    return {
        "schema_version": 1,
        "source": source,
        "event": name,
        "mono_ns": int(mono_ms * 1e6),
        "wall_ns": int((1_700_000_000_000 + mono_ms) * 1e6),
        **fields,
    }


def synthetic_events():
    return [
        event("vla_client", "policy_resumed", 0),
        event("simulator", "task_armed", 10),
        event("vla_client", "inference_started", 20, chunk_id=1),
        event(
            "vla_client",
            "inference_completed",
            120,
            chunk_id=1,
            duration_ns=100_000_000,
            observation_ns=10_000_000,
        ),
        event("vla_client", "chunk_accepted", 125, chunk_id=1, action_index=5),
        event(
            "vla_client",
            "action_published",
            130,
            chunk_id=1,
            action_index=5,
            frame_index=0,
        ),
        event("wbc", "action_received", 131, frame_index=0),
        event(
            "vla_client",
            "action_published",
            150,
            chunk_id=1,
            action_index=6,
            frame_index=1,
        ),
        event("wbc", "action_received", 151, frame_index=1),
        event(
            "simulator",
            "sim_state",
            160,
            object_z=0.875,
            contact=False,
            lifted=False,
        ),
        event(
            "vla_client",
            "action_published",
            190,
            chunk_id=1,
            action_index=8,
            frame_index=3,
        ),
        event("wbc", "action_received", 191, frame_index=3),
        event("simulator", "task_success", 200, status="success"),
    ]


def test_event_tracer_emits_atomic_json_line():
    output = io.StringIO()
    monotonic_values = iter([123])
    wall_values = iter([456])
    tracer = EventTracer(
        "test",
        enabled=True,
        stream=output,
        monotonic_ns=lambda: next(monotonic_values),
        wall_time_ns=lambda: next(wall_values),
    )

    payload = tracer.emit("sample", chunk_id=7)

    assert payload["mono_ns"] == 123
    line = output.getvalue()
    assert line.startswith(TRACE_PREFIX)
    assert json.loads(line[len(TRACE_PREFIX) :]) == payload


def test_read_trace_events_extracts_embedded_records(tmp_path):
    path = tmp_path / "client.log"
    first, second = synthetic_events()[:2]
    path.write_text(
        "ordinary log line\n"
        + TRACE_PREFIX
        + json.dumps(second)
        + "\n"
        + "prefix "
        + TRACE_PREFIX
        + json.dumps(first)
        + "\n"
    )

    events = read_trace_events([path])

    assert events == [first, second]


def test_summarize_events_correlates_gr00t_and_wbc():
    summary = summarize_events(synthetic_events())

    assert summary["active_window"]["duration_s"] == pytest.approx(0.19)
    assert summary["counts"]["inferences"] == 1
    assert summary["counts"]["actions_published"] == 3
    assert summary["inference_latency_ms"]["mean"] == pytest.approx(100)
    assert summary["observation_capture_ms"]["mean"] == pytest.approx(10)
    assert summary["horizon_margin_ms"]["mean"] == pytest.approx(680)
    assert summary["action_transport_ms"]["mean"] == pytest.approx(1)
    assert summary["publish_deadlines"]["over_1_5x"] == 1
    assert summary["frame_continuity"] == {
        "gaps": 1,
        "duplicates": 0,
        "out_of_order": 0,
        "unmatched_published": 0,
    }


def test_perfetto_trace_contains_inference_span_and_state_counter():
    trace = perfetto_trace(synthetic_events())

    inference = next(
        item for item in trace["traceEvents"] if item["name"] == "GR00T inference"
    )
    state = next(
        item for item in trace["traceEvents"] if item["name"] == "physical state"
    )
    assert inference["ph"] == "X"
    assert inference["dur"] == pytest.approx(100_000)
    assert state["ph"] == "C"
