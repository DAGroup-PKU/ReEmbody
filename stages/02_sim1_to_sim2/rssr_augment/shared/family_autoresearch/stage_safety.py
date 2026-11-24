"""Independent bounded safety gate for complete acquisition-stage control plans.

The legacy 4 cm DSL guard remains unchanged.  This new primitive admits at
most 10 cm world-axis displacement of either tool or distal pad relative to
both the parent and frozen source, checks desired joint limits/rates, and
binds every physical-step desired command to the preflight stream.  The frozen
native hard/task judge still decides the outcome after execution.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import mujoco
import numpy as np

from pipeline.crossfamily_stage_v58_seq104 import safety
from pipeline.joint_robot_object_v2_20260923 import runner_core as core


SCHEMA = "rssr.family.bounded_stage_safety.v1"
WORLD_AXIS_BOUND_M = .100
JOINT_RATE_BOUND_RAD_S = 4.0
TOL = 1e-9
HANDS = ("left", "right")


class StageEnvelopeReject(RuntimeError):
    pass


def enforce_points(candidate: np.ndarray, parent: np.ndarray,
                   source: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    for value in (candidate, parent, source):
        if value.shape != (2, 3, 3) or not np.isfinite(value).all():
            raise StageEnvelopeReject("Nonfinite or wrong-shaped bilateral stage FK")
    single = np.abs(candidate-parent)
    total = np.abs(candidate-source)
    for name, value in (("parent", single), ("frozen_source", total)):
        if np.max(value) > WORLD_AXIS_BOUND_M+TOL:
            h, point, axis = np.unravel_index(np.argmax(value), value.shape)
            raise StageEnvelopeReject(
                f"Stage world-axis bound at frame {frame}: {name} {HANDS[h]} "
                f"{safety.POINTS[HANDS[h]][point]} axis {axis} = {value[h,point,axis]:.6f} m")
    return single, total


def _joint_and_rate_audit(model: mujoco.MjModel, times: np.ndarray,
                          candidate: np.ndarray) -> dict:
    worst_margin = math.inf
    for joint in range(model.njnt):
        if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[joint]:
            continue
        adr = int(model.jnt_qposadr[joint])
        lo, hi = map(float, model.jnt_range[joint])
        values = candidate[:, adr]
        margin = float(min(np.min(values-lo), np.min(hi-values)))
        worst_margin = min(worst_margin, margin)
        if margin < -TOL:
            raise StageEnvelopeReject(f"Stage command exceeds joint limit: {model.joint(joint).name}")
    # Free-object qpos stays pinned by the native scope check.  Bound all robot
    # hinge command derivatives; the physical servo still determines tracking.
    hinge = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
             if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    rates = np.abs(np.diff(candidate[:, hinge], axis=0)/np.diff(times)[:, None])
    maximum = float(rates.max(initial=0.))
    if maximum > JOINT_RATE_BOUND_RAD_S+TOL:
        raise StageEnvelopeReject(f"Stage joint command rate {maximum:.4f} rad/s exceeds bound")
    return {"minimum_joint_limit_margin_rad": worst_margin,
            "maximum_joint_rate_rad_s": maximum,
            "joint_rate_bound_rad_s": JOINT_RATE_BOUND_RAD_S}


def precompute(model: mujoco.MjModel, streams: dict, reference_path: Path) -> dict:
    """Audit the complete desired stream at source and physical-step clocks."""
    times = np.asarray(streams["candidate"][0], dtype=float)
    candidate = np.asarray(streams["candidate"][1], dtype=float)
    parent_times, parent = map(np.asarray, streams["parent"][:2])
    source_times, source = map(np.asarray, streams["source"][:2])
    if (times.ndim != 1 or len(times) < 2 or not np.array_equal(times, parent_times)
            or not np.array_equal(times, source_times)
            or not np.all(np.diff(times) > 0)
            or candidate.shape != (len(times), model.nq)
            or parent.shape != candidate.shape or source.shape != candidate.shape
            or not all(np.isfinite(x).all() for x in (times, candidate, parent, source))):
        raise StageEnvelopeReject("Stage candidate/parent/source clock or qpos mismatch")
    joints = _joint_and_rate_audit(model, times, candidate)
    data = mujoco.MjData(model)
    max_single = np.zeros((2, 3, 3), dtype=float)
    max_total = np.zeros_like(max_single)
    dt = float(model.opt.timestep)
    frames = int(np.ceil(float(times[-1])/dt))+1
    parent_xyz = np.empty((frames, 2, 3, 3), dtype=np.float64)
    source_xyz = np.empty_like(parent_xyz)
    for frame in range(frames):
        when = frame*dt
        now = safety.point_positions(model, data, safety._interpolate(times, candidate, when))
        prior = safety.point_positions(model, data, safety._interpolate(times, parent, when))
        original = safety.point_positions(model, data, safety._interpolate(times, source, when))
        parent_xyz[frame], source_xyz[frame] = prior, original
        single, total = enforce_points(now, prior, original, frame)
        max_single = np.maximum(max_single, single)
        max_total = np.maximum(max_total, total)
    reference_path = Path(reference_path)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(reference_path, times_s=times, candidate_qpos=candidate,
                        parent_qpos=parent, source_qpos=source,
                        parent_xyz=parent_xyz, source_xyz=source_xyz,
                        dt_s=np.array(dt))
    return {"schema": SCHEMA, "reference_path": str(reference_path),
            "reference_sha256": core.sha(reference_path),
            "physical_frames": frames, "source_clock_samples": len(times),
            "modified_hands": [hand for i, hand in enumerate(HANDS)
                               if float(np.max(max_single[i])) > 1e-7],
            "stage_world_axis_bound_m": WORLD_AXIS_BOUND_M,
            "max_single_action_world_axis_m": max_single.tolist(),
            "max_total_source_world_axis_m": max_total.tolist(),
            "candidate_target_sha256": hashlib.sha256(np.ascontiguousarray(candidate).tobytes()).hexdigest(),
            "parent_target_sha256": hashlib.sha256(np.ascontiguousarray(parent).tobytes()).hexdigest(),
            "source_target_sha256": hashlib.sha256(np.ascontiguousarray(source).tobytes()).hexdigest(),
            "joint_command_audit": joints}


class RuntimeGuard:
    """Match every executed desired command and enforce the stage envelope."""
    def __init__(self, receipt: dict, audit_dir: Path):
        self.receipt, self.audit_dir = receipt, Path(audit_dir)
        with np.load(receipt["reference_path"], allow_pickle=False) as ref:
            self.times = ref["times_s"].copy()
            self.candidate_qpos = ref["candidate_qpos"].copy()
            self.parent_qpos = ref["parent_qpos"].copy()
            self.parent_xyz = ref["parent_xyz"].copy()
            self.source_xyz = ref["source_xyz"].copy()
            self.dt = float(ref["dt_s"])
        self.modified = tuple(receipt["modified_hands"])
        self.max_single = np.zeros((2, 3, 3), dtype=float)
        self.max_total = np.zeros_like(self.max_single)
        self.first_trigger = None
        self.frames = 0
        self.command_hash = hashlib.sha256()
        self._model = self._data = None

    def attach(self, model: mujoco.MjModel) -> None:
        if (self._model is not None or model.nq != self.candidate_qpos.shape[1]
                or abs(float(model.opt.timestep)-self.dt) > 1e-12):
            raise StageEnvelopeReject("Physical model differs from stage preflight")
        self._model = model
        self._data = mujoco.MjData(model)

    def check(self, data: mujoco.MjData, desired: np.ndarray) -> None:
        if self._model is None:
            raise StageEnvelopeReject("Stage runtime guard not attached")
        frame = int(round(float(data.time)/self.dt))
        if (frame < 0 or frame >= len(self.parent_xyz)
                or abs(float(data.time)-frame*self.dt) > 2e-7):
            raise StageEnvelopeReject("Physical clock differs from stage preflight")
        expected = safety._interpolate(self.times, self.candidate_qpos, float(data.time))
        if (not np.isfinite(desired).all() or desired.shape != expected.shape
                or np.max(np.abs(desired-expected)) > 2e-8):
            raise StageEnvelopeReject("Executed desired control differs from approved stage plan")
        now = safety.point_positions(self._model, self._data, desired)
        try:
            single, total = enforce_points(now, self.parent_xyz[frame],
                                           self.source_xyz[frame], frame)
        except StageEnvelopeReject as exc:
            self.first_trigger = {"frame": frame, "time_s": float(data.time),
                                  "reason": str(exc)}
            raise
        self.max_single = np.maximum(self.max_single, single)
        self.max_total = np.maximum(self.max_total, total)
        self.command_hash.update(np.ascontiguousarray(desired).tobytes())
        self.frames += 1

    def save(self, complete: bool) -> dict:
        result = {"schema": "rssr.family.bounded_stage_runtime_guard.v1",
                  "complete_native_rollout": bool(complete and self.frames > 0
                                                  and self.first_trigger is None),
                  "physical_control_frames": self.frames,
                  "modified_hands": list(self.modified),
                  "first_trigger": self.first_trigger,
                  "stage_world_axis_bound_m": WORLD_AXIS_BOUND_M,
                  "max_single_action_world_axis_m": self.max_single.tolist(),
                  "max_total_source_world_axis_m": self.max_total.tolist(),
                  "executed_desired_target_stream_sha256": self.command_hash.hexdigest(),
                  "reference_sha256": self.receipt["reference_sha256"],
                  "stage_safety_code_sha256": core.sha(Path(__file__)),
                  "executing_state_written_by_guard": False}
        core.write(self.audit_dir / "runtime_guard.json", result)
        return result
