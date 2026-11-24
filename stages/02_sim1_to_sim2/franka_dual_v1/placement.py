"""Drawer placement for the dual-arm Franka (2026-11-03): the cabinet and the block moved on the table.

In the 50 source layouts (made for the cobot) the cabinet stands ~0.19 m to one side of the midline with its front
0.675 m ahead of the Franka base line (y = -0.19), and the block on the other side. The block hand then has to put
the block ~0.6 m sideways and ~0.65 m ahead of its own base, at the edge of the Panda's reach (79 programs, 0
accepted, result/franka_dual_luna_v1_20261102_attempt_01).

A placement moves, per case, every cabinet object (cabinet, bottom drawer, handle bar and posts, middle/top fronts
and handles) by one translation and the block by another, both in the table plane, mirrored in x by the side the
source cabinet is on. The per-case differences of the source layouts are kept (a translation, not a fixed pose).
Sizes, yaw, joints, masses and the engineered asset are unchanged; the frozen contract is compiled on the moved
scene (same predicates and thresholds, a different scene digest).

Placement parameters (nominal = the median source case; `s` = +1 when the source cabinet is at +x):
  cabinet_x   cabinet centre, measured from the midline toward the handle hand's side (source nominal 0.194)
  front_dist  cabinet front face ahead of the arm base line (source nominal 0.675)
  block_x     block centre from the midline on the block hand's side (source nominal 0.270)
  block_dist  block centre ahead of the arm base line (source nominal 0.596)

A placed case is named `<source case>@<placement name>`, e.g. drawer_ep000038@center_v1.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import sys as _sys
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from reembody.paths import resolve_source_cases  # noqa: E402

CASES = resolve_source_cases()
BASE_LINE_Y = -0.19  # layout.SELECTED.base_line_y
# medians over the 50 drawer source cases
SRC_CABINET_X = 0.194
SRC_FRONT_DIST = 0.675
SRC_BLOCK_X = 0.270
SRC_BLOCK_DIST = 0.596


@dataclass(frozen=True)
class DrawerPlacement:
    name: str
    cabinet_x: float
    front_dist: float
    block_x: float
    block_dist: float

    def shifts(self, side: float):
        """(cabinet shift, block shift), world xy, for a case whose source cabinet is on side `side` (+1 / -1)."""
        cab = np.array([side * (self.cabinet_x - SRC_CABINET_X), -(self.front_dist - SRC_FRONT_DIST)])
        blk = np.array([-side * (self.block_x - SRC_BLOCK_X), -(self.block_dist - SRC_BLOCK_DIST)])
        return cab, blk


@dataclass(frozen=True)
class WorkspaceShift:
    """Every object of a case (any task) moved by `dy` along y; dy > 0 brings the workspace toward the robot."""
    name: str
    dy: float


PLACEMENTS: dict[str, DrawerPlacement | WorkspaceShift] = {}


def register(p):
    PLACEMENTS[p.name] = p
    return p


def split(case: str):
    """'drawer_ep000038@center_v1' -> ('drawer_ep000038', placement); a plain case -> (case, None)."""
    if '@' not in case:
        return case, None
    src, name = case.split('@', 1)
    if name not in PLACEMENTS:
        raise KeyError(f'unknown placement {name}')
    return src, PLACEMENTS[name]


def source_case(case: str) -> str:
    return split(case)[0]


def _raw_scene(src: str) -> dict:
    return json.loads((CASES / src / 'source_scene.json').read_text())


def place(scene: dict, p):
    """Moved copy of a source scene, and a receipt of the shifts."""
    out = copy.deepcopy(scene)
    if isinstance(p, WorkspaceShift):
        for o in out['objects']:
            o['position_m'] = (np.asarray(o['position_m'], float) + [0.0, p.dy, 0.0]).tolist()
        return out, dict(placement=asdict(p), shift_xy=[0.0, p.dy])
    objs = {o['id']: o for o in out['objects']}
    side = float(np.sign(objs['drawer_cabinet']['position_m'][0])) or 1.0
    cab, blk = p.shifts(side)
    targets = set(out['bindings']['targets'])
    for o in out['objects']:
        shift = blk if o['id'] in targets else cab
        o['position_m'] = (np.asarray(o['position_m'], float) + np.r_[shift, 0.0]).tolist()
    return out, dict(placement=asdict(p), side=side, cabinet_shift_xy=(cab.round(4) + 0.0).tolist(),
                     block_shift_xy=(blk.round(4) + 0.0).tolist(), moved_with_cabinet=sorted(set(objs) - targets),
                     moved_with_block=sorted(targets))


def case_scene(case: str) -> dict:
    src, p = split(case)
    scene = _raw_scene(src)
    return scene if p is None else place(scene, p)[0]


def receipt(case: str):
    src, p = split(case)
    return None if p is None else place(_raw_scene(src), p)[1]


# Selected 2026-11-03 (result/franka_dual_drawer_placement_20261103_attempt_01/sweep): the cabinet 9.4 cm closer to
# the midline (centre 0.10 m from it, toward the handle hand), its front unchanged at 0.675 m; the block 6 cm further
# out (0.33 m from the midline) and 4.6 cm closer (0.55 m ahead of the base line). START_Q bounds the window: the
# handle must be in the handle hand's wrist view at t = 0 (cabinet centre >= ~0.10 toward that hand) and the start
# pose must clear the cabinet (front >= ~0.58 m).
FRANKA_V1 = register(DrawerPlacement('franka_v1', cabinet_x=0.10, front_dist=0.675, block_x=0.33, block_dist=0.55))
