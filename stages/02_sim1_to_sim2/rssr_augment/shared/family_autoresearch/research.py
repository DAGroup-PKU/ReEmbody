"""Persistent code-research handoffs and budget-preserving campaign revisions.

This module does not synthesize code. A code researcher consumes a durable work
item, implements and tests a family mechanism, then forks an idle campaign. The
new revision inherits its evidence and resource accounting, not a fresh budget.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import fcntl
from pathlib import Path
import shutil
import uuid

from .store import Store, atomic_json, digest, file_sha, read_json, utc_now


def _lineage(run: Path) -> list[Path]:
    result, seen = [], set()
    while True:
        run = run.resolve()
        if run in seen:
            raise ValueError("Campaign revision ancestry contains a cycle")
        seen.add(run)
        result.append(run)
        manifest = read_json(run / "manifest.json")
        ancestry = manifest.get("revision_ancestry")
        if not ancestry:
            return result
        parent = Path(ancestry["parent_run"]).resolve()
        if file_sha(parent / "manifest.json") != ancestry["parent_manifest_file_sha256"]:
            raise ValueError("Parent revision manifest changed")
        run = parent


@contextmanager
def active_revision_guard(run):
    """Hold the whole campaign lineage exclusively while running or forking.

    The registry lives outside immutable run artifacts. Only its active revision
    can consume inherited budget; restarting an ancestor or forking siblings
    cannot create another copy of the same remaining allowance.
    """
    from . import coordinator

    run = Path(run).resolve()
    root = _lineage(run)[-1]
    root_hash = file_sha(root / "manifest.json")
    directory = coordinator.ROOT / "runs/.family_lineages"
    directory.mkdir(parents=True, exist_ok=True)
    pointer_path = directory / (root_hash + ".json")
    with (directory / (root_hash + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Campaign lineage is already active; do not run or fork concurrently") from None
        if pointer_path.exists():
            pointer = read_json(pointer_path)
            if pointer.get("root_manifest_file_sha256") != root_hash or pointer.get("root_run") != str(root):
                raise ValueError("Campaign lineage registry identity changed")
        else:
            if root != run:
                raise ValueError("Revision has no authoritative active lineage registry")
            pointer = {"schema": "rssr.family.active_revision.v1", "root_run": str(root),
                "root_manifest_file_sha256": root_hash, "active_run": str(run),
                "active_manifest_file_sha256": file_sha(run / "manifest.json"), "generation": 0}
            atomic_json(pointer_path, pointer)
        if Path(pointer["active_run"]).resolve() != run:
            raise ValueError("Campaign revision is superseded; use active revision: " + pointer["active_run"])
        if pointer["active_manifest_file_sha256"] != file_sha(run / "manifest.json"):
            raise ValueError("Active revision manifest changed")
        yield pointer_path


def reject_superseded(run) -> None:
    """Check active ownership; callers that execute must hold the full guard."""
    with active_revision_guard(run):
        pass


def list_work_items(run) -> list[dict]:
    """Return unresolved durable work without changing state or requiring an LLM.

    JSON-pointer references retain the exact hypothesis/minimal-test source.
    Researchers can restart from these records after losing conversational state.
    """
    store = Store(run)
    state = store.read()
    store.validate_events()
    lineage = _lineage(store.run)
    items = []
    diagnostics = {
        "review_rejected": ("Clarify the cited hypothesis or evidence and recheck the same compiled control before any new physics.",
                            "Resolve reviewer objections against the saved preflight receipt and raw report."),
        "compile_rejected": ("Correct typed command/schema or frozen bound violation before review.",
                             "Compile one corrected bounded command without a model review or physical replay."),
        "preflight_rejected": ("Change the shared program or its applicability boundary; do not retry the same invalid command.",
                               "Recompile and run the frozen native preflight on a different initial state."),
        "physical_failed": ("Extract the first command-to-actual-to-contact divergence before proposing another mechanism.",
                            "Compare the saved full rollout against its parent at the first failed stage."),
        "invalid_evidence": ("Repair evidence capture and rerun only after source identity is verified.",
                             "Audit saved report, command, scene, guard and hash chain."),
        "tool_error": ("Diagnose the saved tool error without assuming a physical failure.",
                       "Inspect the native process receipt and complete or uncertain reservation."),
    }
    eligible = {"mechanism_handoff", "no_compatible_baseline", *diagnostics}
    for case_id, case in sorted(state["cases"].items()):
        if case["status"] == "solved":
            continue
        outstanding = [r for r in case["history"]
                       if r.get("status") in eligible]
        if not outstanding:
            continue
        row = outstanding[-1]
        trial = row["trial"]
        directory = next((revision / "tasks" / trial for revision in lineage
                          if (revision / "tasks" / trial).is_dir()), None)
        handoff_path = directory / "research_handoff.json" if directory and (directory / "research_handoff.json").is_file() else None
        handoff = read_json(handoff_path) if handoff_path else {}
        proposal = handoff.get("proposal", {})
        mechanism = proposal.get("handoff") or {}
        packet_path = handoff.get("task_packet")
        if packet_path is None and directory and (directory / "task_packet.json").is_file():
            packet_path = str(directory / "task_packet.json")
        source = str(handoff_path) if handoff_path else None
        evidence_paths = ([directory / name for name in
                           ("task_packet.json", "candidate.json", "review_packet.json",
                            "review.json", "result_audit.json")
                           if (directory / name).is_file()] if directory else [])
        for revision in lineage:
            native = revision / "cases" / case_id / f"trial_{int(trial.rsplit('_', 1)[1]):02d}"
            if native.is_dir():
                evidence_paths.extend(path for path in
                    (native / "preflight_result.json", native / "evaluate_result.json") if path.is_file())
                break
        if row.get("report_path") and Path(row["report_path"]).is_file():
            evidence_paths.append(Path(row["report_path"]))
        next_step, default_test = diagnostics.get(row["status"],
            ("Implement and test a bounded family mechanism, then fork this campaign revision; preserve negative evidence.", None))
        items.append({
            "schema": "rssr.family.research_work_item.v1", "case_id": case_id,
            "family": case["family"], "case_status": case["status"], "trial_id": trial,
            "campaign_run": str(store.run), "origin_run": str(directory.parents[2]) if directory else None,
            "task_packet_path": packet_path, "handoff_path": source,
            "handoff_sha256": file_sha(handoff_path) if handoff_path else None,
            "hypothesis": proposal.get("hypothesis"),
            "hypothesis_path": source + "#/proposal/hypothesis" if source else None,
            "minimal_test": mechanism.get("minimal_test"),
            "minimal_test_path": source + "#/proposal/handoff/minimal_test" if source else None,
            "required_evidence": mechanism.get("required_evidence", []),
            "reason": handoff.get("reason", row.get("reason", row.get("details", row["status"]))),
            "latest_outcome": row["status"],
            "diagnostic_evidence_sha256": {str(path): file_sha(path) for path in evidence_paths},
            "negative_history": copy.deepcopy(case["history"]),
            "blocked_mechanisms": list(case["blocked_mechanisms"]),
            "remaining_budget": store.remaining(),
            "next_step": next_step,
        })
        if items[-1]["minimal_test"] is None:
            items[-1]["minimal_test"] = default_test
    return items


def _verify_parent(parent: Store, manifest: dict, state: dict, implementation: dict) -> None:
    parent.validate_events()
    if any(c.get("pending_trial") for c in state["cases"].values()) or any(
            t.get("phase") != "finished" for t in state["trials"].values()):
        raise ValueError("Cannot fork a campaign with unfinished pending trials")
    if any(r.get("status") == "started" for r in state["reservations"].values()):
        raise ValueError("Cannot fork an uncertain resource reservation")
    if state["limits"] != manifest["limits"]:
        raise ValueError("Parent budget identity changed")
    if file_sha(parent.run / "evidence_index.json") != manifest["evidence_index_sha256"]:
        raise ValueError("Parent evidence index changed")
    for path, expected in manifest["source_sha256"].items():
        if file_sha(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    for artifacts in manifest.get("initial_artifact_sha256", {}).values():
        for path, expected in artifacts.items():
            if file_sha(path) != expected:
                raise ValueError("Frozen initial artifact changed: " + path)
    # Code researchers can revise this framework, but not the native judge,
    # physics implementation, shared safety checks or provider configuration.
    family_directory = Path(__file__).resolve().parent
    def external(mapping):
        return {p: h for p, h in mapping.items()
                if not Path(p).resolve().is_relative_to(family_directory)}
    if external(manifest["code_sha256"]) != external(implementation):
        raise ValueError("Frozen native runtime, judge or external configuration changed")
    for case in state["cases"].values():
        for branch in case["branches"].values():
            artifacts = branch.get("output_sha256", {})
            if branch.get("kind") == "success" and not artifacts:
                raise ValueError("Verified success lacks immutable output fingerprints")
            if any(file_sha(p) != h for p, h in artifacts.items()):
                raise ValueError("Parent branch evidence changed")


def fork_revision(parent_run, new_run, *, requeue_cases=None, case_limits=None) -> Store:
    """Create a new code revision with exact inherited evidence and budgets.

    The parent runner must be idle and all its trials settled. An atomic staging
    directory ensures a failed fork never exposes a usable zero-budget-accounting
    campaign. Original manifests, states, events and reports are never rewritten.
    """
    from . import coordinator

    parent = Store(parent_run)
    destination = Path(new_run).resolve()
    if destination == parent.run or destination.exists():
        raise ValueError("Revision destination must be a new directory")
    if destination.parent != coordinator.ROOT / "runs" or not destination.name.startswith("cobot150_family_"):
        raise ValueError("Use a new RSSR/runs/cobot150_family_* revision directory")
    if not parent.path.is_file():
        raise ValueError("Parent campaign is not initialized")
    with active_revision_guard(parent.run) as pointer_path, (parent.run / "runner.lock").open("a") as runner_lock:
        try:
            fcntl.flock(runner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Parent campaign runner is active; checkpoint it before forking") from None
        with (parent.run / "state.lock").open("a") as state_lock:
            fcntl.flock(state_lock, fcntl.LOCK_EX)
            manifest = read_json(parent.run / "manifest.json")
            state = parent.read()
            implementation = coordinator.implementation_identity()
            _verify_parent(parent, manifest, state, implementation)
            _lineage(parent.run)
            requested = set(requeue_cases or [])
            if not requested <= set(state["cases"]):
                raise ValueError("Unknown revision requeue case")
            if any(state["cases"][cid]["status"] not in ("unresolved", "research_needed") for cid in requested):
                raise ValueError("Only unresolved cases can be explicitly requeued; budgets are preserved")
            staging = destination.with_name(destination.name + ".fork-" + uuid.uuid4().hex)
            requeued = sorted(requested | {cid for cid, c in state["cases"].items() if c["status"] == "research_needed"})
            ancestry = {
                "parent_run": str(parent.run), "parent_manifest_file_sha256": file_sha(parent.run / "manifest.json"),
                "parent_state_sha256": state["state_sha256"],
                "root_run": manifest.get("revision_ancestry", {}).get("root_run", str(parent.run)),
                "revision": manifest.get("revision_ancestry", {}).get("revision", 0) + 1,
                "forked_at": utc_now(), "budget_policy": "carry_all_spending_reservations_and_original_start_time",
                "requeued_cases": requeued,
                "inherited_verified_successes": sorted(cid for cid,c in state["cases"].items() if c["status"] == "solved"),
            }
            try:
                fresh = coordinator.initialize(staging, limits=manifest["limits"], profile=manifest["profile"],
                                               index_from=parent.run / "evidence_index.json")
                new_manifest = read_json(staging / "manifest.json")
                required_equal = ("source_sha256", "native_runtime_sha256", "frozen", "profile", "limits",
                                  "case_ids", "evidence_index_sha256", "initial_artifact_sha256")
                for key in required_equal:
                    if new_manifest.get(key) != manifest.get(key):
                        raise ValueError("Revision changed frozen campaign field: " + key)
                new_manifest["case_limits"] = copy.deepcopy(
                    manifest["case_limits"] if case_limits is None else case_limits)
                new_manifest["revision_ancestry"] = ancestry
                atomic_json(staging / "manifest.json", new_manifest)
                carried = copy.deepcopy(state)
                carried["manifest_sha256"] = digest(new_manifest)
                carried["status"] = "ready"
                for cid in requeued:
                    carried["cases"][cid]["status"] = "pending"
                    carried["cases"][cid].pop("terminal_reason", None)
                carried["state_sha256"] = digest({k:v for k,v in carried.items() if k != "state_sha256"})
                atomic_json(staging / "state.json", carried)
                fresh.record("revision_forked", {**ancestry, "new_run": str(destination)})
                fresh.validate_events()
                coordinator.write_summary(fresh)
                # Source/index compatibility was checked after initialization;
                # current code pins are also checked once more before publication.
                coordinator.validate_identity(staging)
                staging.rename(destination)
                pointer = read_json(pointer_path)
                pointer.update(active_run=str(destination),
                               active_manifest_file_sha256=file_sha(destination / "manifest.json"),
                               generation=pointer["generation"] + 1)
                atomic_json(pointer_path, pointer)
            except BaseException:
                if staging.exists():
                    shutil.rmtree(staging)
                raise
    return Store(destination)
