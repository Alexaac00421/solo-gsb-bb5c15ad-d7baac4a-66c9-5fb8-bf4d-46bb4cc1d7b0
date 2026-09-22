"""Pydantic schemas for the calibration timeline service.

Time coordinates are nanoseconds since an arbitrary batch-local epoch.
Intervals are half-open ``[start_ns, end_ns)`` so that abutting fragments do
not share any sample instant.
"""

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# A revision root uses this literal as its (missing) parent.
ROOT_PARENT = "ROOT"

VersionId = Annotated[str, Field(min_length=1, max_length=200)]
BatchId = Annotated[str, Field(min_length=1, max_length=200)]
SensorId = Annotated[str, Field(min_length=1, max_length=200)]
IdemKey = Annotated[str, Field(min_length=1, max_length=200)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MeasurementEvent(StrictModel):
    """One measurement-fragment revision delivered to the ingest endpoint."""

    batch_id: BatchId
    sensor_id: SensorId
    revision_id: VersionId
    # Parent revision for this sensor; the literal "ROOT" marks a causal root.
    parent_id: VersionId
    # Idempotency key; the same event payload is accepted exactly once in
    # effect and every repeated submission is acknowledged as a duplicate.
    idempotency_key: IdemKey
    start_ns: int = Field(ge=0)
    end_ns: int = Field(ge=1)
    # Calibration value for the fragment; required unless the fragment is
    # withdrawn (withdrawn fragments erase instead of writing a value).
    value: Optional[float] = None
    unit: Optional[str] = Field(default=None, max_length=40)
    withdrawn: bool = False

    @model_validator(mode="after")
    def _check_interval_and_value(self) -> "MeasurementEvent":
        if self.revision_id == ROOT_PARENT:
            raise ValueError(f"revision_id may not be the reserved value {ROOT_PARENT!r}")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns must be strictly greater than start_ns")
        if not self.withdrawn and self.value is None:
            raise ValueError("value is required when withdrawn is false")
        return self


class TimelineSegment(StrictModel):
    start_ns: int
    end_ns: int
    state: Literal["calibrated", "withdrawn"]
    # Populated for calibrated segments; null on withdrawn (erased) spans.
    value: Optional[float] = None
    unit: Optional[str] = None
    revision_id: str


class Gap(StrictModel):
    """A missing revision that blocks one or more known children."""

    batch_id: str
    sensor_id: str
    missing_revision_id: str
    awaited_by: list[str]
    # Revision that referenced the missing parent (may repeat across gaps).
    referenced_from: str


class SensorTimeline(StrictModel):
    sensor_id: str
    grounded: bool
    segments: list[TimelineSegment]
    segments_hash: str
    covered_start_ns: Optional[int] = None
    covered_end_ns: Optional[int] = None


class BatchResult(StrictModel):
    batch_id: str
    sensors: list[SensorTimeline]
    gaps: list[Gap]
    revisions_total: int
    revisions_grounded: int
    revisions_blocked: int
    revisions_invalid: int
    # True when every known revision is grounded (no gaps, no cycles).
    complete: bool
    timeline_hash: str


class SamplePointResponse(StrictModel):
    """Effective calibration state at one queried nanosecond of one sensor."""

    batch_id: str
    sensor_id: str
    timestamp_ns: int
    # calibrated: a grounded, value-bearing segment wins the fold at the point;
    # withdrawn:   a tombstone segment wins (value/unit are null);
    # uncovered:   no grounded segment covers the instant;
    # blocked:     a revision covering the instant is waiting on missing
    #              ancestors, so the eventual winner cannot yet be determined.
    state: Literal["calibrated", "withdrawn", "uncovered", "blocked"]
    # Populated only for calibrated points; null otherwise.
    value: Optional[float] = None
    unit: Optional[str] = None
    # Winning revision (last applied in deterministic fold order); null unless
    # the state is calibrated/withdrawn.
    revision_id: Optional[str] = None
    # Ordered parent chain from revision_id back to the causal root, e.g.
    # ["v3", "v2", "v1", "ROOT"]; empty unless the point is resolved.
    parent_chain: list[str] = Field(default_factory=list)
    # Distinct not-yet-uploaded revisions that block every covering revision;
    # populated only for blocked points (sorted, ready for backfill).
    missing_ancestors: list[str] = Field(default_factory=list)


class CompareSide(StrictModel):
    """Effective folded state one side contributes to a compare interval."""

    state: Literal["calibrated", "withdrawn"]
    # Populated for calibrated spans; null on withdrawn (erased) spans.
    value: Optional[float] = None
    unit: Optional[str] = None


class CompareInterval(StrictModel):
    """One half-open diff interval over the union of both folded timelines."""

    start_ns: int
    end_ns: int
    # equal:          both sides agree (both uncovered counts as agreement);
    # left_only:      only the left batch covers the span;
    # right_only:     only the right batch covers the span;
    # value_mismatch: both calibrated but value or unit differ;
    # state_mismatch: one side calibrated, the other withdrawn.
    kind: Literal["equal", "left_only", "right_only", "value_mismatch", "state_mismatch"]
    # Effective payload of each side on this span; null where uncovered.
    left: Optional[CompareSide] = None
    right: Optional[CompareSide] = None


class IncompleteReason(StrictModel):
    """Why one side of a comparison cannot be considered final yet."""

    side: Literal["left", "right"]
    batch_id: str
    # gaps:              revisions blocked by not-yet-uploaded ancestors;
    # invalid_revisions: revisions in or below a parent cycle.
    reason: Literal["gaps", "invalid_revisions"]
    detail: str
    # Blocked or invalid revisions excluded from the fold (sorted).
    affected_revisions: list[str] = Field(default_factory=list)
    # For reason="gaps": distinct missing parents to backfill (sorted).
    missing_revision_ids: list[str] = Field(default_factory=list)


class CompareResponse(StrictModel):
    """Half-open interval diff of one sensor across two batches."""

    sensor_id: str
    left_batch_id: str
    right_batch_id: str
    # Hashes of each side's folded segments (matching the per-sensor
    # segments_hash of the batch result) for cross-checking.
    left_segments_hash: str
    right_segments_hash: str
    # True when either side has gaps or invalid revisions, so backfilling
    # may still change the diff.
    incomplete: bool
    incomplete_reasons: list[IncompleteReason] = Field(default_factory=list)
    # Contiguous, time-sorted partition of the union coverage span;
    # adjacent intervals merge when kind and both side payloads match.
    intervals: list[CompareInterval] = Field(default_factory=list)


class QualityGateRequest(StrictModel):
    """Quality-gate parameters for one sensor of a batch."""

    sensor_id: SensorId
    # Inclusive allowed-value bounds; at least one is required and a
    # calibrated value exactly on a bound is in spec.
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    # Minimum sustained out-of-bounds duration (nanoseconds) before a
    # candidate run is reported as a violation.
    min_duration_ns: int = Field(ge=1)
    # How withdrawn/uncovered/blocked spans are treated:
    # ignore: not recorded, verdict unaffected;
    # warn:   recorded as gaps, verdict unaffected;
    # fail:   recorded as gaps and any gap fails the gate.
    # Gap spans break out-of-bounds runs and never merge into them,
    # regardless of the policy.
    missing_policy: Literal["ignore", "warn", "fail"] = "warn"

    @model_validator(mode="after")
    def _check_bounds(self) -> "QualityGateRequest":
        if self.min_value is None and self.max_value is None:
            raise ValueError("at least one of min_value and max_value is required")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value must be less than or equal to max_value")
        return self


class ViolationInterval(StrictModel):
    """One sustained out-of-bounds run on the folded timeline."""

    start_ns: int
    end_ns: int
    duration_ns: int
    # Extreme calibrated values observed inside the run.
    observed_min: float
    observed_max: float


class GapInterval(StrictModel):
    """One maximal span the quality gate could not evaluate."""

    start_ns: int
    end_ns: int
    duration_ns: int
    # withdrawn: a tombstone segment wins the fold on the span;
    # uncovered: no grounded segment covers the span;
    # blocked:   a not-yet-grounded revision covers the span (precedence).
    state: Literal["withdrawn", "uncovered", "blocked"]


class QualityGateResponse(StrictModel):
    """Quality-gate verdict for one sensor of a batch."""

    batch_id: str
    sensor_id: str
    # False when any violation was detected, or when missing_policy="fail"
    # and at least one gap was recorded.
    passed: bool
    violations: list[ViolationInterval] = Field(default_factory=list)
    gaps: list[GapInterval] = Field(default_factory=list)
    # Content hash of the scanned sensor's folded segments (matches the
    # per-sensor segments_hash of the batch result).
    segments_hash: str
    # Batch-level hash, identical to the batch result's timeline_hash.
    timeline_hash: str


class IngestResponse(StrictModel):
    status: Literal["accepted", "duplicate", "conflict"]
    idempotency_key: str
    batch_id: str
    sensor_id: str
    revision_id: str
    result: BatchResult


class ResetResponse(StrictModel):
    status: Literal["reset"]
    batches_removed: int
    events_removed: int


class HealthResponse(StrictModel):
    status: Literal["ok"]
    storage: Literal["memory"]
    version: str
