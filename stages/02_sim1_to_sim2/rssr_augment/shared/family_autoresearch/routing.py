"""Measured contact-loss routing; task predicates and their contracts stay fixed.

The rule is deliberately conservative: a prior lift pass alone does not prove
retention, but a contact-free placement near the planned release is not evidence
of a failed grasp either.  This changes the research menu, never the judge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

POLICY = {
    "minimum_sustained_contact_s": .12,
    "minimum_sustained_lift_contact_s": .08,
    "minimum_release_gap_s": .25,
    "latest_loss_fraction_of_hold": .75,
    "minimum_fall_after_loss_m": .015,
    "maximum_sample_gap_s": .05,
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def _sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def _default(report):
    hard = report["objective"]["hard_failures"]
    failed = report["objective"]["task_failures"]
    if hard:
        return "model_health", hard[0]
    for name in failed:
        if name.startswith("lift_with_contact"):
            return "acquisition", name
    for name in ("inside_01", "open_insert_close_03"):
        if name in failed:
            return "articulation", name
    return "placement", failed[0] if failed else "complete"


def _longest_contact_duration(times, mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return max((float(times[end] - times[start]) for start, end in zip(starts, ends)), default=0.)


def _bound_lifts(contract):
    return {row["id"]: (row.get("object"), row.get("hand"))
            for row in contract.get("predicates", []) if row.get("kind") == "lift_with_contact"}


def infer_stage(report: dict, contract: dict, report_path, *, include_all_lifts=False) -> dict:
    """Return deterministic routing and referenced evidence from a saved rollout.

    ``stage_evidence`` is either None or the selected measured loss event. Its
    hand/object/cycle fields guide carry actions without changing target_spec.
    Missing or ambiguous evidence leaves the original routing and records why.
    Call only after the campaign's normal native-result integrity audit.
    """
    stage, target = _default(report)
    path = Path(report_path).resolve()
    result = {"stage": stage, "target_predicate": target, "stage_evidence": None,
              "evidence_refs": [str(path)], "evidence_sha256": {}, "diagnostics": [],
              "routing_policy": dict(POLICY)}
    if not include_all_lifts and stage in ("model_health", "acquisition", "articulation"):
        return result
    try:
        if _read(path) != report:
            raise ValueError("Report argument differs from its saved evidence")
        proposal_path = path.parent.parent / "proposal.json"
        contract_path = path.parent / "task_contract.json"
        rollout_path = path.parent / "rollout.npz"
        proposal, bound_contract = _read(proposal_path), _read(contract_path)
        if _bound_lifts(contract) != _bound_lifts(bound_contract):
            raise ValueError("Source and physical contracts disagree on lift object/hand binding")
        if bound_contract.get("contract_id") != report.get("task_contract_id"):
            raise ValueError("Saved physical contract does not match report")
        scene_ids = [obj["id"] for obj in proposal["scene"]["objects"]]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("Duplicate object IDs in executed scene")
        with np.load(rollout_path, allow_pickle=False) as rollout:
            times = np.asarray(rollout["time_s"], dtype=float)
            contacts = np.asarray(rollout["contacts"])
            positions = np.asarray(rollout["object_position"], dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > POLICY["maximum_sample_gap_s"] + 1e-9):
            raise ValueError("Contact-loss routing requires complete dense monotonic timestamps")
        if (contacts.shape != (len(times), len(scene_ids), 3)
                or positions.shape != (len(times), len(scene_ids), 3)
                or not np.isfinite(contacts).all() or not np.isfinite(positions).all()):
            raise ValueError("Rollout arrays do not match the bound scene object order")
        for i, obj in enumerate(scene_ids):
            final = report.get("final_positions_m", {}).get(obj)
            if final is None or not np.allclose(positions[-1, i], final, rtol=0, atol=1e-9):
                raise ValueError("Rollout object order/terminal positions disagree with report")
        result["evidence_refs"] = [str(p) for p in (path, rollout_path, proposal_path, contract_path)]
        result["evidence_sha256"] = {str(p): _sha(p) for p in (path, rollout_path, proposal_path, contract_path)}
        target_spec = next((row for row in contract.get("predicates", []) if row.get("id") == target), {})
        # The object directly involved in the failed relation owns this carry
        # question. Do not route to a different object's already resolved phase.
        involved = set(target_spec.get("objects", []))
        for key in ("object", "base", "upper"):
            if target_spec.get(key):
                involved.add(target_spec[key])
        if include_all_lifts or target == "complete":
            involved = {row.get("object") for row in contract.get("predicates", [])
                        if row.get("kind") == "lift_with_contact"}
        if not involved:
            result["diagnostics"].append({"reason": "failed_predicate_has_no_bound_object"})
            return result
        evaluated = {row["id"]: row for row in report["task_evaluation"]["predicates"]}
        observed = []
        dt = float(np.max(np.diff(times)))
        for lift in contract.get("predicates", []):
            if lift.get("kind") != "lift_with_contact" or lift.get("object") not in involved:
                continue
            obj, hand = lift.get("object"), lift.get("hand")
            if obj not in scene_ids or hand not in ("left", "right"):
                raise ValueError("Lift predicate lacks an unambiguous scene object/hand binding")
            if evaluated.get(lift["id"], {}).get("passed") is not True:
                continue
            oi, hi = scene_ids.index(obj), ("left", "right").index(hand)
            z = positions[:, oi, 2]
            height = float(lift["height_m"])
            for cycle, event in enumerate(report.get("source_grasp_events", {}).get(hand, [])):
                closed, release = float(event["closed_s"]), float(event["release_s"])
                if not np.isfinite([closed, release]).all() or release <= closed:
                    raise ValueError("Invalid source grasp event")
                if times[-1] < release - dt or times[0] > closed + dt:
                    raise ValueError("Physical trace does not cover the complete source hold interval")
                interval = (times >= closed) & (times <= release)
                touch = interval & (contacts[:, oi, hi] > 0)
                samples = np.flatnonzero(touch)
                if not len(samples):
                    result["diagnostics"].append({"object": obj, "hand": hand, "cycle": cycle,
                                                  "reason": "no_contact_in_this_cycle"})
                    continue
                sustained = _longest_contact_duration(times, touch)
                contact_lift = float(np.max(z[touch] - z[0]))
                lifted_touch = touch & (z - z[0] >= height - 1e-9)
                sustained_lift = _longest_contact_duration(times, lifted_touch)
                if (sustained + 1e-9 < POLICY["minimum_sustained_contact_s"] or contact_lift + 1e-9 < height
                        or sustained_lift + 1e-9 < POLICY["minimum_sustained_lift_contact_s"]):
                    result["diagnostics"].append({"object": obj, "hand": hand, "cycle": cycle,
                         "reason": "sustained_lift_not_established_in_this_cycle",
                         "sustained_contact_s": sustained, "max_contact_lift_m": contact_lift,
                         "sustained_lift_contact_s": sustained_lift})
                    continue
                last = int(samples[-1])
                later = np.flatnonzero(interval & (np.arange(len(times)) > last))
                loss = int(later[0]) if len(later) else None
                gap = max(0., release - float(times[last]))
                fraction = (float(times[last]) - closed) / (release - closed)
                fall = max(0., float(z[last] - np.min(z[later]))) if len(later) else 0.
                metrics = {"object": obj, "hand": hand, "cycle": cycle, "lift_predicate": lift["id"],
                           "closed_s": closed, "release_s": release, "last_contact_s": float(times[last]),
                           "contact_loss_s": float(times[loss]) if loss is not None else None, "gap_s": gap,
                           "loss_fraction_of_hold": fraction, "sustained_contact_s": sustained,
                           "max_contact_lift_m": contact_lift, "drop_after_loss_m": fall,
                           "sustained_lift_contact_s": sustained_lift, "lift_threshold_m": height,
                           "sampling_resolution_s": dt, "evidence_refs": list(result["evidence_refs"]),
                           "evidence_sha256": dict(result["evidence_sha256"])}
                if (loss is not None and gap >= max(POLICY["minimum_release_gap_s"], 3 * dt)
                        and fraction <= POLICY["latest_loss_fraction_of_hold"]
                        and fall >= POLICY["minimum_fall_after_loss_m"]):
                    metrics["reason"] = "sustained_lift_then_early_contact_loss_and_fall"
                    observed.append(metrics)
                else:
                    metrics["reason"] = ("contact_maintained_through_release" if loss is None
                                          else "no_conservative_early_drop_evidence")
                result["diagnostics"].append(metrics)
        if observed and stage == "placement" and target != "complete":
            # Start with the earliest unresolved hold cycle; retaining a later
            # incidental touch must not masquerade as establishing an earlier one.
            chosen = min(observed, key=lambda row: (row["contact_loss_s"], -row["gap_s"], row["object"]))
            result.update(stage="carry", stage_evidence=chosen)
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["diagnostics"].append({"reason": "retention_evidence_unavailable", "detail": str(exc)})
        return result


def retention_progress(before_evidence: dict, after_report: dict, contract: dict,
                       after_report_path) -> list[dict]:
    """Measure added *lifted contact* in the same source cycle, not late touches.

    The parent observation is re-derived from its referenced report and rollout.
    A later terminal contact counts only when lifted contact is maintained through
    the old loss boundary. Falling early and touching again later earns nothing.
    Hard feasibility and branch acceptance remain the coordinator's responsibility.
    """
    try:
        if not isinstance(before_evidence, dict) or before_evidence.get("reason") != "sustained_lift_then_early_contact_loss_and_fall":
            return []
        identity = tuple(before_evidence[key] for key in ("hand", "object", "cycle"))
        if identity[0] not in ("left", "right") or type(identity[2]) is not int:
            return []
        before_paths = [Path(ref) for ref in before_evidence.get("evidence_refs", [])
                        if Path(ref).name == "report.json"]
        if len(before_paths) != 1:
            return []
        for ref, expected in before_evidence.get("evidence_sha256", {}).items():
            if _sha(ref) != expected:
                return []
        before_report = _read(before_paths[0])
        old = infer_stage(before_report, contract, before_paths[0], include_all_lifts=True)
        new = infer_stage(after_report, contract, after_report_path, include_all_lifts=True)

        def event(rows):
            matches = [row for row in rows if tuple(row.get(key) for key in ("hand", "object", "cycle")) == identity
                       and "last_contact_s" in row]
            return matches[0] if len(matches) == 1 else None

        before, after = event(old["diagnostics"]), event(new["diagnostics"])
        if before is None or after is None:
            return []
        for field in ("closed_s", "release_s", "last_contact_s", "max_contact_lift_m", "drop_after_loss_m"):
            if abs(float(before_evidence[field]) - before[field]) > 1e-8:
                return []
        if any(abs(before[key] - after[key]) > 1e-8 for key in ("closed_s", "release_s")):
            return []
        last_gain = after["last_contact_s"] - before["last_contact_s"]
        drop_gain = before["drop_after_loss_m"] - after["drop_after_loss_m"]
        if last_gain < .12 - 1e-9:
            return []
        path = Path(after_report_path).resolve()
        scene = _read(path.parent.parent / "proposal.json")["scene"]
        oi = [obj["id"] for obj in scene["objects"]].index(identity[1])
        hi = ("left", "right").index(identity[0])
        with np.load(path.parent / "rollout.npz", allow_pickle=False) as rollout:
            times = np.asarray(rollout["time_s"], dtype=float)
            contact = np.asarray(rollout["contacts"])[:, oi, hi] > 0
            height = np.asarray(rollout["object_position"], dtype=float)[:, oi, 2]
        retained = contact & (height - height[0] >= after["lift_threshold_m"] - 1e-9)
        extension = np.flatnonzero((times >= before["last_contact_s"] - 1e-9)
                                  & (times <= after["last_contact_s"] + 1e-9))
        if not len(extension) or not retained[extension[0]]:
            return []
        first_absent = next((offset for offset, index in enumerate(extension) if not retained[index]), len(extension))
        if first_absent == 0:
            return []
        retained_through = float(times[extension[first_absent - 1]])
        retained_extension = max(0., retained_through - before["last_contact_s"])
        prolonged = last_gain >= .25 - 1e-9 and retained_extension >= .25 - 1e-9
        reduced_fall = drop_gain >= .015 - 1e-9 and retained_extension >= .12 - 1e-9
        if not (prolonged or reduced_fall):
            return []
        return [{"metric": "retained_lift_contact", "hand": identity[0], "object": identity[1],
                 "cycle": identity[2], "before": before, "after": after,
                 "last_contact_gain_s": last_gain, "retained_extension_s": retained_extension,
                 "retained_through_s": retained_through, "drop_reduction_m": drop_gain,
                 "basis": "extended_lifted_contact" if prolonged else "reduced_drop_with_extended_lifted_contact",
                 "evidence_refs": list(dict.fromkeys([*old["evidence_refs"], *new["evidence_refs"]])),
                 "evidence_sha256": {**old["evidence_sha256"], **new["evidence_sha256"]}}]
    except (OSError, ValueError, KeyError, TypeError):
        return []
