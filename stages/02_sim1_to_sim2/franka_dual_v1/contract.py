"""Frozen RSSR task contract (the one S1 uses) evaluated on a dual-arm Franka rollout.

The contract is compiled from the case's source scene and bindings by the unmodified S1 runtime
(`s1_short_refine_v2/rssr_augment/vendor/runtime/rssr_task_contracts.py`, imported read-only), so
a Franka trajectory and an S1 trajectory of the same case are judged by the same predicates.
Admission additionally requires the Franka hard constraints, >= 20 mm left/right arm clearance in
every frame and a bit-exact independent replay.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[1] / 'rssr_augment/vendor/runtime'
if str(RUNTIME) not in sys.path:
    sys.path.append(str(RUNTIME))
import rssr_task_contracts as _contracts  # noqa: E402

from .placement import case_scene  # noqa: E402


def evaluator_pin() -> dict:
    p = RUNTIME / 'rssr_task_contracts.py'
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def source_scene(case: str) -> dict:
    """The case's source scene; a placed drawer case (`<case>@<placement>`, placement.py) has its objects moved."""
    return case_scene(case)


def compile_case(case: str) -> dict:
    scene = source_scene(case)
    return _contracts.compile_contract(scene['task_id'], scene, scene['bindings'])


def evaluate(case: str, contract: dict, result: dict, hard: dict) -> dict:
    tr = result['trace']
    arrays = dict(time_s=tr['time_s'], object_position=tr['object_position'],
                  object_quaternion=tr['object_quaternion'], contacts=tr['contacts'],
                  object_contacts=tr['object_contacts'], finger_contact_sides=tr['finger_contact_sides'],
                  gripper_opening=tr['gripper_opening'],
                  articulation_open_fraction=tr['articulation_open_fraction'])
    wanted = {k: hard[k] for k in contract['hard_constraints']}
    return _contracts.evaluate(contract, source_scene(case), arrays, wanted)
