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


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


def _gate_store(*events):
    store = BatchStore()
    for e in events:
        store.events[e.revision_id] = e
    return store


def _gate(store, lo=0.0, hi=3.0, min_dur=1, policy="fail", batch="B", sensor="S"):
    return evaluate_quality_gate(batch, sensor, lo, hi, min_dur, policy, store)


def test_gate_all_in_band_passes_with_empty_lists():
    store = _gate_store(_ev("r", "ROOT", 0, 100, value=1.0))
    doc = _gate(store)
    assert doc["status"] == "pass"
    assert doc["violations"] == [] and doc["gaps"] == []
    assert (doc["scan_start_ns"], doc["scan_end_ns"]) == (0, 100)


def test_gate_below_min_and_above_max_candidate_runs():
    # A compliant middle segment keeps the two runs apart; both out-of-band
    # pieces are descendants so they win their windows over the root value.
    store = _gate_store(
        _ev("r", "ROOT", 0, 60, value=1.0),
        _ev("low", "r", 0, 20, value=-1.0, key="low"),
        _ev("high", "low", 40, 60, value=9.0, key="high"),
    )
    doc = _gate(store, 0.0, 3.0)
    assert [
        (v["start_ns"], v["end_ns"], v["side"], v["peak_revision_id"])
        for v in doc["violations"]
    ] == [(0, 20, "below_min", "low"), (40, 60, "above_max", "high")]
    assert doc["violations"][0]["duration_ns"] == 20
    assert doc["status"] == "fail"


def test_gate_merges_abutting_violations_across_revision_boundaries():
    store = _gate_store(
        _ev("r", "ROOT", 0, 50, value=1.0),
        _ev("a", "r", 10, 30, value=5.0),
        _ev("b", "a", 30, 50, value=6.0),
    )
    doc = _gate(store, 0.0, 3.0, min_dur=40)
    assert [
        (v["start_ns"], v["end_ns"], v["duration_ns"], v["side"], v["peak_value"])
        for v in doc["violations"]
    ] == [(10, 50, 40, "above_max", 6.0)]
    # The peak revision is the one carrying the most extreme value.
    assert doc["violations"][0]["peak_revision_id"] == "b"


def test_gate_mixed_sides_merge_and_report_mixed_side():
    store = _gate_store(
        _ev("a", "ROOT", 0, 40, value=-5.0, key="a"),
        _ev("b", "a", 20, 60, value=8.0, key="b"),
        _ev("c", "b", 0, 20, value=-1.0, key="c"),
    )
    doc = _gate(store, 0.0, 3.0, min_dur=50)
    assert [
        (v["start_ns"], v["end_ns"], v["side"], v["peak_value"], v["peak_revision_id"])
        for v in doc["violations"]
    ] == [(0, 60, "mixed", 8.0, "b")]


def test_gate_min_duration_is_inclusive_and_filters_short_runs():
    store = _gate_store(
        _ev("r", "ROOT", 0, 100, value=1.0),
        _ev("v4", "r", 70, 80, value=9.0, key="v4"),
    )
    assert _gate(store, 0.0, 3.0, min_dur=11)["violations"] == []
    keep = _gate(store, 0.0, 3.0, min_dur=10)["violations"]
    assert [(v["start_ns"], v["end_ns"]) for v in keep] == [(70, 80)]


def test_gate_withdrawn_gap_breaks_violation_merge_and_fails_by_policy():
    store = _gate_store(
        _ev("r", "ROOT", 0, 30, value=9.0),
        _ev("w", "r", 10, 20, withdrawn=True, key="w"),
    )
    fail = _gate(store, 0.0, 3.0, min_dur=15, policy="fail")
    # Two 10ns violation runs are both below the 15ns threshold: no violation,
    # but the withdrawn gap still fails the gate under policy="fail".
    assert fail["violations"] == []
    assert [(g["start_ns"], g["end_ns"], g["kind"]) for g in fail["gaps"]] == [
        (10, 20, "withdrawn")
    ]
    assert fail["status"] == "fail"

    ignored = _gate(store, 0.0, 3.0, min_dur=15, policy="ignore")
    assert ignored["gaps"] and ignored["status"] == "pass"

    # With threshold 10 both split runs are reported separately.
    keep = _gate(store, 0.0, 3.0, min_dur=10, policy="ignore")
    assert [(v["start_ns"], v["end_ns"]) for v in keep["violations"]] == [
        (0, 10),
        (20, 30),
    ]


def test_gate_uncovered_gap_inside_hull():
    store = _gate_store(
        _ev("a", "ROOT", 0, 10, value=1.0, key="a"),
        _ev("b", "ROOT", 20, 30, value=2.0, key="b"),
    )
    doc = _gate(store, 0.0, 3.0)
    assert [(g["start_ns"], g["end_ns"], g["duration_ns"], g["kind"]) for g in doc["gaps"]] == [
        (10, 20, 10, "uncovered")
    ]
    assert doc["status"] == "fail"
    assert _gate(store, 0.0, 3.0, policy="ignore")["status"] == "pass"


def test_gate_blocked_span_is_gap_takes_precedence_and_merges():
    store = _gate_store(
        _ev("r", "ROOT", 0, 100, value=1.0),
        _ev("o1", "g1", 10, 40, value=9.0, key="o1"),
        _ev("o2", "g2", 40, 60, value=9.0, key="o2"),
    )
    doc = _gate(store, 0.0, 3.0)
    assert [
        (g["start_ns"], g["end_ns"], g["kind"], g["missing_ancestors"])
        for g in doc["gaps"]
    ] == [(10, 60, "blocked", ["g1", "g2"])]
    # The grounded fold says 9.0 out-of-band there, but blocked wins.
    assert doc["violations"] == []


def test_gate_blocked_gap_union_overlapping_missing_sets():
    store = _gate_store(
        _ev("o1", "g1", 0, 20, value=9.0, key="o1"),
        _ev("o2", "g2", 10, 30, value=9.0, key="o2"),
    )
    doc = _gate(store, 0.0, 3.0)
    [gap] = doc["gaps"]
    assert (gap["start_ns"], gap["end_ns"]) == (0, 30)
    assert gap["missing_ancestors"] == ["g1", "g2"]
    assert doc["scan_start_ns"] == 0 and doc["scan_end_ns"] == 30


def test_gate_overlapping_blocked_intervals_with_nonmonotonic_ends():
    # o1 starts earlier but outlives the middle blocked revision; after o2
    # ends at t=20 only o1 must still cover [20, 60), and nothing after 60.
    store = _gate_store(
        _ev("r", "ROOT", 0, 100, value=1.0),
        _ev("o1", "g1", 10, 60, value=9.0, key="o1"),
        _ev("o2", "g2", 10, 20, value=9.0, key="o2"),
    )
    doc = _gate(store, 0.0, 3.0, policy="ignore")
    assert [(g["start_ns"], g["end_ns"], g["kind"]) for g in doc["gaps"]] == [
        (10, 60, "blocked")
    ]
    # No blocked coverage past 60, and the grounded 1.0 there is in band.
    assert doc["violations"] == []


def test_gate_gaps_of_different_kinds_never_merge():
    store = _gate_store(
        _ev("r", "ROOT", 0, 40, value=1.0),
        _ev("w", "r", 0, 10, withdrawn=True, key="w"),
        _ev("o", "ghost", 10, 20, value=9.0, key="o"),
    )
    doc = _gate(store, 0.0, 3.0, policy="ignore")
    assert [(g["start_ns"], g["end_ns"], g["kind"]) for g in doc["gaps"]] == [
        (0, 10, "withdrawn"),
        (10, 20, "blocked"),
    ]


def test_gate_bounds_are_inclusive_and_equal_bounds_allowed():
    store = _gate_store(_ev("r", "ROOT", 0, 30, value=5.0))
    doc = _gate(store, 5.0, 5.0)
    assert doc["status"] == "pass" and doc["violations"] == []


def test_gate_cyclic_only_sensor_is_empty_scan_that_passes():
    store = _gate_store(
        _ev("c1", "c2", 0, 10, value=1.0, key="c1"),
        _ev("c2", "c1", 0, 10, value=2.0, key="c2"),
    )
    doc = _gate(store)
    assert doc["status"] == "pass"
    assert doc["scan_start_ns"] is None and doc["scan_end_ns"] is None
    assert doc["violations"] == [] and doc["gaps"] == []


def test_gate_unknown_sensor_returns_none():
    store = _gate_store(_ev("r", "ROOT", 0, 10, value=1.0))
    assert evaluate_quality_gate("B", "NOPE", 0.0, 3.0, 1, "fail", store) is None


def test_gate_timeline_hash_matches_folded_segments_hash():
    store = _gate_store(
        _ev("r", "ROOT", 0, 50, value=1.0),
        _ev("a", "r", 10, 30, value=5.0),
    )
    gate_doc = _gate(store)
    batch_doc = evaluate_batch("B", store)
    assert gate_doc["timeline_hash"] == batch_doc["sensors"][0]["segments_hash"]
