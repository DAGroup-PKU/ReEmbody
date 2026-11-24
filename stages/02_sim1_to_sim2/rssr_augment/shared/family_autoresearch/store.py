"""Atomic single-writer state, budget reservations and append-only event history.

The state and its event log commit in the same atomic file replacement. A crash
cannot commit a budget charge without its reservation, or a solved case without
its evidence. No SQLite/WAL/shared-memory assumptions are made about the PFS.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time

FAMILIES = ("stack_cups", "stack_blocks", "drawer")
DEFAULT_LIMITS = {"full": 300, "short": 150, "api": 600, "wall_seconds": 43200}
DEFAULT_CASE_LIMITS = {"full": 4, "api": 12, "steps": 16}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def pinned_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError("Immutable artifact changed: " + str(path))
    else:
        atomic_json(path, value)


class BudgetExhausted(RuntimeError):
    pass


class CaseBudgetExhausted(BudgetExhausted):
    def __init__(self, case_id, resource, used, limit):
        self.case_id, self.resource, self.used, self.limit = case_id, resource, used, limit
        super().__init__(f"{case_id} {resource} case budget exhausted ({used}/{limit})")


class UncertainOperation(RuntimeError):
    pass


def _event(state, kind, payload):
    events = state["events"]
    item = {"seq": len(events) + 1, "at": utc_now(), "kind": kind,
            "payload": payload, "previous": events[-1]["sha256"] if events else None}
    item["sha256"] = digest(item)
    events.append(item)


def _metadata_case(state, metadata):
    metadata = metadata or {}
    declared = metadata.get("case_id")
    trial = metadata.get("trial")
    inferred = None
    if isinstance(trial, str):
        inferred = state["trials"].get(trial, {}).get("case_id")
        if inferred is None:
            prefix = trial.split("/", 1)[0]
            inferred = prefix if prefix in state["cases"] else None
    if declared is not None and declared not in state["cases"]:
        raise ValueError("Reservation names an unknown case")
    if declared is not None and inferred is not None and declared != inferred:
        raise ValueError("Reservation case_id disagrees with trial ownership")
    return declared or inferred


def _case_usage(state, case_id):
    if case_id not in state["cases"]:
        raise KeyError(case_id)
    usage = {"full": 0, "api": 0,
             "steps": sum(t["case_id"] == case_id for t in state["trials"].values())}
    for reservation in state["reservations"].values():
        kind = reservation["kind"]
        if kind in ("full", "api") and _metadata_case(state, reservation.get("metadata")) == case_id:
            usage[kind] += 1
    return usage


class Store:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.path = self.run / "state.json"

    @contextmanager
    def transaction(self):
        self.run.mkdir(parents=True, exist_ok=True)
        with (self.run / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read() if self.path.exists() else None
            box = [state]
            try:
                yield box
                if box[0] is not None:
                    box[0]["state_sha256"] = digest({k: v for k, v in box[0].items() if k != "state_sha256"})
                    atomic_json(self.path, box[0])
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def initialize(self, manifest, case_ids):
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Duplicate case IDs")
        limits = manifest["limits"]
        if set(limits) != set(DEFAULT_LIMITS):
            raise ValueError("Budget fields differ from required limits")
        if any(type(v) is not int or v < 0 for v in limits.values()):
            raise ValueError("Budgets must be nonnegative integers")
        if limits["wall_seconds"] <= 0:
            raise ValueError("Wall budget must be positive")
        case_limits = manifest.get("case_limits")
        if case_limits is not None:
            if set(case_limits) != set(DEFAULT_CASE_LIMITS):
                raise ValueError("Case budget fields differ from required limits")
            if any(type(v) is not int or v < 0 for v in case_limits.values()):
                raise ValueError("Case budgets must be nonnegative integers")
        pinned_json(self.run / "manifest.json", manifest)
        with self.transaction() as box:
            if box[0] is not None:
                if box[0]["manifest_sha256"] != digest(manifest):
                    raise ValueError("Resume manifest mismatch")
                return
            cases = {}
            for case_id in case_ids:
                family = case_id.rsplit("_ep", 1)[0]
                if family not in FAMILIES:
                    raise ValueError("Unsupported family")
                cases[case_id] = {"family": family, "status": "pending", "visits": 0,
                                  "history": [], "branches": {}, "stable": None,
                                  "pending_trial": None, "blocked_mechanisms": [],
                                  "no_effect": {}, "success": None}
            state = {"schema": "rssr.family.state.v1", "manifest_sha256": digest(manifest),
                     "created_at": utc_now(), "started_epoch": None, "status": "ready",
                     "limits": limits, "spent": {"full": 0, "short": 0, "api": 0},
                     "completed": {"full": 0, "short": 0, "api": 0},
                     "reservations": {}, "cases": cases, "trials": {}, "events": [],
                     "family_cursor": 0}
            _event(state, "initialized", {"cases": len(cases)})
            box[0] = state

    def read(self):
        state = read_json(self.path)
        if state.get("state_sha256") != digest({k: v for k, v in state.items() if k != "state_sha256"}):
            raise ValueError("State snapshot integrity failure")
        if state["manifest_sha256"] != digest(read_json(self.run / "manifest.json")):
            raise ValueError("State manifest identity mismatch")
        return state

    def record(self, kind, payload):
        with self.transaction() as box:
            _event(box[0], kind, payload)

    def set_status(self, status, reason=None):
        with self.transaction() as box:
            box[0]["status"] = status
            _event(box[0], "status", {"status": status, "reason": reason})

    def remaining(self):
        state = self.read()
        result = {k: state["limits"][k] - n for k, n in state["spent"].items()}
        result["wall_seconds"] = max(0, state["limits"]["wall_seconds"] - (
            time.time() - state["started_epoch"] if state["started_epoch"] else 0))
        return result

    def case_usage(self, case_id):
        """Charged attempts, including inherited, failed and uncertain attempts."""
        return _case_usage(self.read(), case_id)

    def case_remaining(self, case_id):
        usage = self.case_usage(case_id)
        limits = read_json(self.run / "manifest.json").get("case_limits")
        return {key: max(0, limits[key] - value) if limits is not None else None
                for key, value in usage.items()}

    def set_case_status(self, case_id, status, reason=None):
        """Close an unresolved worker only after its pending trial is settled."""
        if status not in ("unresolved", "budget_exhausted"):
            raise ValueError("Only unresolved or budget_exhausted are worker terminal statuses")
        with self.transaction() as box:
            case = box[0]["cases"][case_id]
            if case["pending_trial"]:
                raise ValueError("Cannot close a case with a pending trial")
            if case["status"] == "solved" or case.get("success"):
                raise ValueError("Cannot replace a verified case success")
            case.update(status=status, terminal_reason=reason)
            _event(box[0], "case_status", {"case_id": case_id, "status": status, "reason": reason})

    def reserve(self, kind, key, metadata=None):
        """Charge an attempt before execution. Unknown outcomes never retry silently."""
        if kind not in ("api", "full", "short"):
            raise ValueError("Unknown resource")
        with self.transaction() as box:
            state = box[0]
            if key in state["reservations"]:
                old = state["reservations"][key]
                if old["kind"] != kind:
                    raise ValueError("Reservation reused for different resource")
                if old["status"] == "complete":
                    return False
                raise UncertainOperation("Attempt already reserved; inspect saved evidence: " + key)
            case_limits = read_json(self.run / "manifest.json").get("case_limits")
            if case_limits is not None and kind in ("full", "api"):
                case_id = _metadata_case(state, metadata)
                if case_id is None:
                    raise ValueError("Case-budgeted reservations require case_id or trial metadata")
                usage = _case_usage(state, case_id)
                if usage[kind] >= case_limits[kind]:
                    raise CaseBudgetExhausted(case_id, kind, usage[kind], case_limits[kind])
            now = time.time()
            if state["started_epoch"] is not None and (
                    now - state["started_epoch"] >= state["limits"]["wall_seconds"]):
                raise BudgetExhausted("Wall-clock budget exhausted")
            if state["spent"][kind] >= state["limits"][kind]:
                raise BudgetExhausted(kind + " budget exhausted")
            if state["started_epoch"] is None:
                state["started_epoch"] = now
            state["spent"][kind] += 1
            state["reservations"][key] = {"kind": kind, "status": "started",
                                           "at": utc_now(), "metadata": metadata or {}}
            _event(state, "reserved", {"kind": kind, "key": key})
        return True

    def settle(self, key, status, evidence=None):
        if status not in ("complete", "error", "interrupted"):
            raise ValueError("Bad attempt outcome")
        with self.transaction() as box:
            state = box[0]
            item = state["reservations"][key]
            if item["status"] == "complete":
                if status != "complete":
                    raise ValueError("Cannot undo a complete attempt")
                return
            if status == "complete":
                state["completed"][item["kind"]] += 1
            item.update(status=status, evidence=evidence)
            _event(state, "settled", {"key": key, "status": status, "evidence": evidence})

    def start_trial(self, case_id, parent_id=None):
        with self.transaction() as box:
            state = box[0]
            case = state["cases"][case_id]
            if case["pending_trial"]:
                return copy.deepcopy(state["trials"][case["pending_trial"]])
            index = sum(t["case_id"] == case_id for t in state["trials"].values())
            case_limits = read_json(self.run / "manifest.json").get("case_limits")
            if case_limits is not None and index >= case_limits["steps"]:
                raise CaseBudgetExhausted(case_id, "steps", index, case_limits["steps"])
            key = f"{case_id}/trial_{index:04d}"
            trial = {"id": key, "index": index, "case_id": case_id, "phase": "created",
                     "parent_id": parent_id, "at": utc_now()}
            state["trials"][key] = trial
            case.update(pending_trial=key, status="active", visits=case["visits"] + 1)
            _event(state, "trial_started", trial.copy())
            return copy.deepcopy(trial)

    def update_trial(self, key, **fields):
        with self.transaction() as box:
            trial = box[0]["trials"][key]
            trial.update(fields)
            _event(box[0], "trial_updated", {"id": key, **fields})

    def finish_trial(self, key, outcome, branch=None, observation=None):
        with self.transaction() as box:
            state = box[0]
            trial = state["trials"][key]
            if trial.get("phase") == "finished":
                return
            case = state["cases"][trial["case_id"]]
            item = {"trial": key, "status": outcome, **(observation or {})}
            case["history"].append(item)
            case["pending_trial"] = None
            case["status"] = "pending"
            trial.update(phase="finished", outcome=outcome)
            if branch:
                branch = copy.deepcopy(branch)
                branch["id"] = key
                parent = case["branches"].get(trial.get("parent_id"), {})
                if branch["kind"] == "exploration":
                    branch["exploration_root"] = (parent.get("exploration_root", trial["parent_id"])
                                                   if parent.get("kind") == "exploration" else key)
                case["branches"][key] = branch
                if branch["kind"] in ("stable", "success"):
                    case["stable"] = key
                if branch["kind"] == "success":
                    case.update(status="solved", success={
                        "report_path": branch["report_path"],
                        "report_sha256": branch["report_sha256"],
                        "proposal_path": branch["proposal_path"], "trial": key})
                explorations = [(bid, b) for bid, b in case["branches"].items()
                                if b["kind"] == "exploration" and not b.get("retired")]
                for bid, _ in sorted(explorations, key=lambda pair: pair[1]["rank"])[2:]:
                    case["branches"][bid]["retired"] = True
            parent_id = trial.get("parent_id")
            if parent_id and parent_id in case["branches"]:
                parent = case["branches"][parent_id]
                if parent["kind"] == "exploration" and item.get("physical_completed"):
                    root_id = parent.get("exploration_root", parent_id)
                    root = case["branches"][root_id]
                    root["followups"] = root.get("followups", 0) + 1
                    for descendant in case["branches"].values():
                        if descendant.get("exploration_root") == root_id:
                            descendant["followups"] = root["followups"]
                            if root["followups"] >= 4:
                                descendant["retired"] = True
            mechanism = item.get("mechanism")
            if mechanism and item.get("physical_completed"):
                context = str(parent_id) + ":" + mechanism
                case["no_effect"][context] = (case["no_effect"].get(context, 0) + 1
                                                if item.get("no_effect") else 0)
                if case["no_effect"][context] >= 3 and context not in case["blocked_mechanisms"]:
                    case["blocked_mechanisms"].append(context)
            if outcome in ("mechanism_handoff", "no_compatible_baseline"):
                case["status"] = "research_needed"
            _event(state, "trial_finished", {"id": key, "outcome": outcome,
                                              "branch_kind": branch["kind"] if branch else None})

    def validate_events(self):
        previous = None
        for index, event in enumerate(self.read()["events"], 1):
            value = {k: v for k, v in event.items() if k != "sha256"}
            if event["seq"] != index or event["previous"] != previous or digest(value) != event["sha256"]:
                raise ValueError("Event history integrity failure")
            previous = event["sha256"]
        return True
