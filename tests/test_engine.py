"""Engine unit tests: causal classification, folding order and cycles."""

from app.engine import (
    BLOCKED,
    GROUNDED,
    INVALID,
    Segment,
    build_timeline,
    classify_sensor,
    compare_segments,
    compare_sensor,
    evaluate_batch,
    evaluate_quality_gate,
    query_sample,
    scan_quality_gate,
)
from app.storage import BatchStore, StoredEvent


def _ev(rev, parent, start=0, end=10, value=1.0, withdrawn=False,
        batch="B", sensor="S", key=None):
    return StoredEvent(
        batch_id=batch,
        sensor_id=sensor,
        revision_id=rev,
        parent_id=parent,
        idempotency_key=key or f"k-{rev}",
        start_ns=start,
        end_ns=end,
        value=None if withdrawn else float(value),
        unit="u",
        withdrawn=withdrawn,
    )


def test_classify_root_missing_and_cycle():
    events = {
        "root": _ev("root", "ROOT"),
        "ok": _ev("ok", "root"),
        "orphan": _ev("orphan", "ghost"),
        "c1": _ev("c1", "c2"),
        "c2": _ev("c2", "c1"),
        "down": _ev("down", "c1"),
    }
    states, depths = classify_sensor(events)
    assert states["root"] == GROUNDED and depths["root"] == 0
    assert states["ok"] == GROUNDED and depths["ok"] == 1
    assert states["orphan"] == BLOCKED
    assert states["c1"] == INVALID and states["c2"] == INVALID
    assert states["down"] == INVALID


def test_self_loop_is_invalid():
    events = {"x": _ev("x", "x")}
    states, _ = classify_sensor(events)
    assert states["x"] == INVALID


def test_fold_order_depth_then_id():
    # Two incomparable children of root overlap at [20,30); deterministic
    # tie-break: lexicographically larger revision id applies last/wins.
    events = {
        "root": _ev("root", "ROOT", 0, 100, value=0),
        "a": _ev("a", "root", 20, 40, value=1),
        "b": _ev("b", "root", 10, 30, value=2),
    }
    states, depths = classify_sensor(events)
    segs = build_timeline(list(events.values()), depths)
    assert [(s.start_ns, s.end_ns, s.value, s.revision_id) for s in segs] == [
        (0, 10, 0.0, "root"),
        (10, 30, 2.0, "b"),
        (30, 40, 1.0, "a"),
        (40, 100, 0.0, "root"),
    ]


def test_descendant_beats_lexicographic_order():
    events = {
        "root": _ev("root", "ROOT", 0, 50, value=0),
        "z": _ev("z", "root", 0, 50, value=1),
        "a": _ev("a", "z", 0, 50, value=2),  # deeper -> wins regardless of id
    }
    states, depths = classify_sensor(events)
    segs = build_timeline(list(events.values()), depths)
    assert len(segs) == 1 and segs[0].value == 2.0 and segs[0].revision_id == "a"


def test_blocked_events_excluded_from_timeline_and_hash_stable():
    store = BatchStore()
    for e in [_ev("root", "ROOT", 0, 20, value=7.0), _ev("lost", "ghost", 0, 5, value=9.0)]:
        store.events[e.revision_id] = e
        store.idempotency[e.idempotency_key] = "x"
    doc1 = evaluate_batch("B", store)
    doc2 = evaluate_batch("B", store)
    assert doc1["timeline_hash"] == doc2["timeline_hash"]
    sensor = doc1["sensors"][0]
    assert len(sensor["segments"]) == 1
    assert sensor["segments"][0]["value"] == 7.0
    assert doc1["revisions_blocked"] == 1
    assert doc1["complete"] is False


def test_query_sample_uses_same_fold_and_half_open_rule():
    events = {
        "root": _ev("root", "ROOT", 0, 100, value=0),
        "a": _ev("a", "root", 20, 40, value=1),
        "b": _ev("b", "root", 10, 30, value=2),
    }
    states, depths = classify_sensor(events)
    # Same-depth overlap [20,30): larger id "b" wins, exactly like the fold.
    at25 = query_sample("S", 25, events, states, depths)
    assert at25["state"] == "calibrated"
    assert at25["value"] == 2.0 and at25["revision_id"] == "b"
    assert at25["parent_chain"] == ["b", "root", "ROOT"]
    # End boundary of b/a window: [30,40) belongs to "a".
    assert query_sample("S", 30, events, states, depths)["revision_id"] == "a"
    assert query_sample("S", 100, events, states, depths)["state"] == "uncovered"


def test_query_sample_blocked_wins_over_grounded_and_skips_invalid():
    events = {
        "root": _ev("root", "ROOT", 0, 100, value=0),
        "orphan": _ev("orphan", "ghost", 0, 50, value=9),
        "c1": _ev("c1", "c2", 80, 90, value=3),
        "c2": _ev("c2", "c1", 80, 90, value=4),
    }
    states, depths = classify_sensor(events)
    blocked_point = query_sample("S", 20, events, states, depths)
    assert blocked_point["state"] == "blocked"
    assert blocked_point["missing_ancestors"] == ["ghost"]
    # A point covered only by invalid (cyclic) revisions is resolved normally.
    assert query_sample("S", 85, events, states, depths)["value"] == 0.0
    assert query_sample("S", 85, events, states, depths)["revision_id"] == "root"


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def test_scan_merges_abutting_out_of_bounds_segments():
    segments = [
        Segment(0, 10, "calibrated", 5.0, "u", "v1"),
        Segment(10, 25, "calibrated", 6.0, "u", "v2"),
        Segment(25, 40, "calibrated", 1.0, "u", "v3"),
    ]
    violations, gaps = scan_quality_gate(segments, [], 0.0, 4.0, 20, True)
    assert violations == [
        {
            "start_ns": 0,
            "end_ns": 25,
            "duration_ns": 25,
            "observed_min": 5.0,
            "observed_max": 6.0,
        }
    ]
    assert gaps == []


def test_scan_drops_candidates_shorter_than_min_duration():
    segments = [Segment(0, 10, "calibrated", 9.0, "u", "v1")]
    violations, _ = scan_quality_gate(segments, [], 0.0, 4.0, 11, True)
    assert violations == []
    # A run lasting exactly the threshold is a violation.
    violations, _ = scan_quality_gate(segments, [], 0.0, 4.0, 10, True)
    assert [v["duration_ns"] for v in violations] == [10]


def test_scan_gaps_break_runs_and_never_merge():
    # OOB [0,10) -> withdrawn [10,20) -> OOB [20,30): two 10ns candidates.
    segments = [
        Segment(0, 10, "calibrated", 9.0, "u", "v1"),
        Segment(10, 20, "withdrawn", None, None, "w1"),
        Segment(20, 30, "calibrated", 9.0, "u", "v2"),
    ]
    violations, gaps = scan_quality_gate(segments, [], 0.0, 4.0, 15, True)
    assert violations == []
    assert [(g["start_ns"], g["end_ns"], g["state"]) for g in gaps] == [
        (10, 20, "withdrawn")
    ]
    # With a low enough threshold both candidates become separate violations.
    violations, _ = scan_quality_gate(segments, [], 0.0, 4.0, 10, True)
    assert [(v["start_ns"], v["end_ns"]) for v in violations] == [(0, 10), (20, 30)]


def test_scan_uncovered_holes_and_blocked_spans_are_gaps():
    segments = [
        Segment(0, 10, "calibrated", 1.0, "u", "v1"),
        Segment(30, 40, "calibrated", 1.0, "u", "v2"),
    ]
    violations, gaps = scan_quality_gate(segments, [(20, 30)], 0.0, 4.0, 5, True)
    assert violations == []
    assert [(g["start_ns"], g["end_ns"], g["state"]) for g in gaps] == [
        (10, 20, "uncovered"),
        (20, 30, "blocked"),
    ]


def test_scan_blocked_takes_precedence_over_calibrated():
    segments = [Segment(0, 20, "calibrated", 9.0, "u", "v1")]  # OOB throughout
    violations, gaps = scan_quality_gate(segments, [(5, 15)], 0.0, 4.0, 10, True)
    # The blocked span splits the OOB run into two 5ns candidates.
    assert violations == []
    assert [(g["start_ns"], g["end_ns"], g["state"]) for g in gaps] == [
        (5, 15, "blocked")
    ]


def test_scan_adjacent_same_state_gaps_merge_across_revisions():
    segments = [
        Segment(0, 10, "withdrawn", None, None, "w1"),
        Segment(10, 20, "withdrawn", None, None, "w2"),
    ]
    _, gaps = scan_quality_gate(segments, [], 0.0, 4.0, 5, True)
    assert [(g["start_ns"], g["end_ns"], g["state"]) for g in gaps] == [
        (0, 20, "withdrawn")
    ]


def test_scan_record_gaps_false_drops_gap_intervals():
    segments = [Segment(0, 10, "withdrawn", None, None, "w1")]
    _, gaps = scan_quality_gate(segments, [], 0.0, 4.0, 5, False)
    assert gaps == []


def test_scan_bounds_are_inclusive():
    segments = [Segment(0, 10, "calibrated", 4.0, "u", "v1")]
    violations, _ = scan_quality_gate(segments, [], 0.0, 4.0, 1, True)
    assert violations == []


def _gate_store(events):
    store = BatchStore()
    for e in events:
        store.events[e.revision_id] = e
    return store


def test_evaluate_quality_gate_unknown_sensor_returns_none():
    store = _gate_store([_ev("v1", "ROOT", 0, 10, value=1.0)])
    assert evaluate_quality_gate("B", "OTHER", 0.0, 2.0, 1, "warn", store) is None


def test_evaluate_quality_gate_pass_fail_and_hashes():
    events = [
        _ev("v1", "ROOT", 0, 100, value=1.0),
        _ev("v2", "v1", 10, 30, value=9.0),  # 20ns above the max
    ]
    store = _gate_store(events)
    doc = evaluate_quality_gate("B", "S", 0.0, 5.0, 10, "warn", store)
    assert doc["passed"] is False
    assert [
        (v["start_ns"], v["end_ns"], v["duration_ns"]) for v in doc["violations"]
    ] == [(10, 30, 20)]
    assert doc["gaps"] == []
    batch = evaluate_batch("B", store)
    assert doc["segments_hash"] == batch["sensors"][0]["segments_hash"]
    assert doc["timeline_hash"] == batch["timeline_hash"]


def test_evaluate_quality_gate_missing_policies():
    events = [
        _ev("v1", "ROOT", 0, 10, value=1.0),
        _ev("w", "v1", 10, 20, withdrawn=True),
    ]
    store = _gate_store(events)
    ignored = evaluate_quality_gate("B", "S", 0.0, 5.0, 1, "ignore", store)
    assert ignored["passed"] is True and ignored["gaps"] == []
    warned = evaluate_quality_gate("B", "S", 0.0, 5.0, 1, "warn", store)
    assert warned["passed"] is True
    assert [(g["start_ns"], g["end_ns"], g["state"]) for g in warned["gaps"]] == [
        (10, 20, "withdrawn")
    ]
    failed = evaluate_quality_gate("B", "S", 0.0, 5.0, 1, "fail", store)
    assert failed["passed"] is False and failed["violations"] == []


# ---------------------------------------------------------------------------
# Cross-batch comparison
# ---------------------------------------------------------------------------


def test_compare_segments_merges_identical_payloads_across_revisions():
    # Same effective value on both sides; revision provenance is ignored, so
    # the left revision boundary at t=10 does not split the diff interval.
    left = [
        Segment(0, 10, "calibrated", 1.0, "u", "v1"),
        Segment(10, 20, "calibrated", 1.0, "u", "v2"),
    ]
    right = [Segment(0, 20, "calibrated", 1.0, "u", "w1")]
    intervals = compare_segments(left, right)
    assert [(i["start_ns"], i["end_ns"], i["kind"]) for i in intervals] == [
        (0, 20, "equal")
    ]


def test_compare_segments_scans_union_span_and_classifies_pairs():
    left = [Segment(0, 10, "calibrated", 1.0, "u", "v1")]
    right = [
        Segment(5, 10, "withdrawn", None, None, "w1"),
        Segment(20, 30, "calibrated", 1.0, "u", "w2"),
    ]
    intervals = compare_segments(left, right)
    assert [(i["start_ns"], i["end_ns"], i["kind"]) for i in intervals] == [
        (0, 5, "left_only"),
        (5, 10, "state_mismatch"),
        (10, 20, "equal"),  # both sides uncovered
        (20, 30, "right_only"),
    ]
    assert intervals[1]["left"]["state"] == "calibrated"
    assert intervals[1]["right"]["state"] == "withdrawn"
    assert intervals[2]["left"] is None and intervals[2]["right"] is None


def test_compare_segments_empty_both_sides():
    assert compare_segments([], []) == []


def test_compare_sensor_reports_blocked_and_invalid():
    left = {
        "root": _ev("root", "ROOT", 0, 10, value=1.0),
        "orphan": _ev("orphan", "ghost", 0, 5, value=2.0),
    }
    right = {
        "c1": _ev("c1", "c2", 0, 10, value=1.0),
        "c2": _ev("c2", "c1", 0, 10, value=1.0),
    }
    doc = compare_sensor("L", "R", "S", left, right)
    assert doc["incomplete"] is True
    reasons = {(r["side"], r["reason"]) for r in doc["incomplete_reasons"]}
    assert reasons == {("left", "gaps"), ("right", "invalid_revisions")}
    gaps = next(r for r in doc["incomplete_reasons"] if r["reason"] == "gaps")
    assert gaps["affected_revisions"] == ["orphan"]
    assert gaps["missing_revision_ids"] == ["ghost"]
    # Left grounded fold covers [0,10); the right fold is empty (cycle).
    assert [(i["start_ns"], i["end_ns"], i["kind"]) for i in doc["intervals"]] == [
        (0, 10, "left_only")
    ]
