"""FR3 Duo scenes for controller development and synthesis (2026-11-03): one source case under one placement variation.

Robot: FR3 Duo (fr3_duo.build_full: official mount, cover, head and ZED Mini head camera, two FR3v2 arms with position
servos and gravity compensation, a Robotiq 2F-85 on each flange, wrist cameras at the nominal D405 mount), mounting
point 0.20 m behind the near table edge and 0.025 m above the table top (base 0.775 m above the floor, user value;
distance from result/franka_fr3_duo_distance_20261103_attempt_01). Home pose HOME_Q: the camera-alignment home of
result/franka_fr3_duo_calib_20261103_attempt_01 (both hands 0.33 m above the table, 45 deg down).

Variation (the synthesis ranges): starting from one of the 50 source placements of the task,
  * stacking: each target object moved by dx, dy in U(-3, 3) cm and turned by U(-15, 15) deg; the stack region by
    dx, dy in U(-2, 2) cm;
  * drawer: the cabinet with its drawers and handles moved together by dx, dy in U(-2, 2) cm; the block by
    U(-3, 3) cm and U(-15, 15) deg;
  * both arms start at HOME_Q plus U(-0.03, 0.03) rad per joint.
A draw is rejected (and redrawn) when two objects' footprints come within 2 cm (the block within 3 cm of the cabinet
or of the opened drawer's sweep), or an object leaves the table. The zero variation of a source case is `source`.
Splits are separate seed streams: 'dev' (controller revision), 'heldout' (the 50 test initial conditions per task,
never run by the controller during development), 'synth' (amplification).
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import fr3_duo as F
from .placement import CASES

PLACE = F.DuoPlacement(0.20, 0.025)
HOME_Q = {'left': (0.745508, -0.839598, -1.489339, -2.62314, -2.524122, 4.024141, 1.124494),
          'right': (2.404648, 0.789317, -1.642964, -2.631993, 2.476238, 4.049773, -2.637405)}
TASKS = ('stack_cups', 'stack_blocks', 'drawer')
OBJ_SHIFT_M, OBJ_YAW_DEG, REGION_SHIFT_M, CABINET_SHIFT_M, ARM_Q_RAD = 0.03, 15.0, 0.02, 0.02, 0.03
GAP_M, CABINET_GAP_M = 0.02, 0.03
TABLE_X, TABLE_Y = (-0.55, 0.55), (-1.12, -0.40)
SPLIT_SEED = {'dev': 1_000, 'heldout': 2_000, 'synth': 3_000}


@dataclass(frozen=True)
class Variation:
    id: str
    task: str
    source_case: str
    object_shift: dict = field(default_factory=dict)  # id -> [dx, dy, dyaw_rad]
    arm_q_offset: dict = field(default_factory=dict)  # hand -> 7 rad
    layout: str | None = None  # None: PLACE and the source table; duo_site.NAME: the user's photographed site

    def to_json(self):
        d = asdict(self)
        if d['layout'] is None:
            del d['layout']  # records written before layouts existed stay byte-identical
        return d

    @staticmethod
    def from_json(d):
        return Variation(**d)


def source_cases(task):
    return sorted(p.name for p in CASES.iterdir() if p.is_dir() and p.name.rsplit('_ep', 1)[0] == task)


def raw_scene(case):
    return json.loads((CASES / case / 'source_scene.json').read_text())


def apply(scene: dict, v: Variation) -> dict:
    out = copy.deepcopy(scene)
    for o in out['objects']:
        dx, dy, dyaw = v.object_shift.get(o['id'], (0.0, 0.0, 0.0))
        o['position_m'] = (np.asarray(o['position_m'], float) + [dx, dy, 0.0]).tolist()
        if dyaw:
            o['yaw_rad'] = float(o.get('yaw_rad', 0.0) + dyaw)
    return out


def _footprint(o):
    """(centre xy, radius) of a free object or the stack region."""
    s = np.asarray(o['size_m'], float)
    r = s[0] / 2 if o['shape'] == 'cup' else float(np.hypot(s[0], s[1]) / 2)
    return np.asarray(o['position_m'][:2], float), r


def _rect_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(0, np.maximum(lo - p, p - hi))))


def valid(scene: dict) -> bool:
    objs = {o['id']: o for o in scene['objects']}
    b = scene['bindings']
    free = [o for o in scene['objects'] if o['mobility'] == 'free']
    for o in free:
        c, r = _footprint(o)
        if not (TABLE_X[0] + r <= c[0] <= TABLE_X[1] - r and TABLE_Y[0] + r <= c[1] <= TABLE_Y[1] - r):
            return False
    for i, a in enumerate(free):
        for o in free[i + 1:]:
            (ca, ra), (cb, rb) = _footprint(a), _footprint(o)
            if np.linalg.norm(ca - cb) < ra + rb + GAP_M:
                return False
    if b.get('target_region'):
        cr, _ = _footprint(objs[b['target_region']])
        for t in [t for t in b['targets'] if t != _base(scene)]:
            ct, rt = _footprint(objs[t])
            if np.linalg.norm(ct - cr) < rt + 0.05:  # the upper object is not already on the stack spot
                return False
    if 'drawer_cabinet' in objs:
        cab, dr = objs['drawer_cabinet'], objs['bottom_drawer']
        cc, cs = np.asarray(cab['position_m'][:2]), np.asarray(cab['size_m'][:2]) / 2
        lo, hi = cc - cs, cc + cs
        hi = hi + [0.0, dr['joint']['range'][1] + 0.05]  # the drawer opens toward +y (the robot), handle included
        for o in free:
            c, r = _footprint(o)
            if _rect_dist(c, lo, hi) < r + CABINET_GAP_M:
                return False
    return True


def _base(scene):
    """The stack's base object: the first target (the object of the contract's inside_region predicate in all 100
    stacking cases, checked by duo_scene_check)."""
    return scene['bindings']['targets'][0]


def draw(task: str, split: str, k: int) -> Variation:
    """k-th variation of a split (deterministic)."""
    rng = np.random.default_rng([SPLIT_SEED[split], TASKS.index(task), k])
    cases = source_cases(task)
    for attempt in range(1000):
        case = cases[int(rng.integers(len(cases)))]
        scene = raw_scene(case)
        shift = {}
        targets = scene['bindings']['targets']
        if task == 'drawer':
            g = rng.uniform(-CABINET_SHIFT_M, CABINET_SHIFT_M, 2)
            for o in scene['objects']:
                if o['id'] not in targets:
                    shift[o['id']] = [float(g[0]), float(g[1]), 0.0]
        else:
            r = rng.uniform(-REGION_SHIFT_M, REGION_SHIFT_M, 2)
            shift[scene['bindings']['target_region']] = [float(r[0]), float(r[1]), 0.0]
        for t in targets:
            d = rng.uniform(-OBJ_SHIFT_M, OBJ_SHIFT_M, 2)
            shift[t] = [float(d[0]), float(d[1]), float(np.radians(rng.uniform(-OBJ_YAW_DEG, OBJ_YAW_DEG)))]
        arm = {h: rng.uniform(-ARM_Q_RAD, ARM_Q_RAD, 7).round(6).tolist() for h in F.HANDS}
        v = Variation(f'{task}_{split}_{k:04d}', task, case, shift, arm)
        if valid(apply(scene, v)):
            return v
    raise RuntimeError(f'no valid variation for {task} {split} {k}')


def source_variation(case: str) -> Variation:
    return Variation(f'{case}_source', case.rsplit('_ep', 1)[0], case, {}, {})


def split_list(task, split, n):
    return [draw(task, split, k) for k in range(n)]


def home_q(v: Variation, hand):
    if v.layout is None:
        base = np.asarray(HOME_Q[hand])
    else:
        from . import duo_site
        base = duo_site.home_q(v.layout, hand)
    return base + np.asarray(v.arm_q_offset.get(hand, np.zeros(7)))


def place_of(v: Variation):
    if v.layout is None:
        return PLACE
    from . import duo_site
    assert v.layout in duo_site.LAYOUTS, v.layout
    return duo_site.place(v.layout)


def view_size(v: Variation):
    """(width, height) of the camera streams the views are measured at."""
    if v.layout is None:
        return 320, 180
    from . import duo_site
    return duo_site.VIEW_W // 2, duo_site.VIEW_H // 2


def build(v: Variation):
    """(MjModel with a 'home' key, the varied scene dict, scene sha256)."""
    case_dir = CASES / v.source_case
    scene = apply(raw_scene(v.source_case), v)
    spec = F.build_full(case_dir, scene, place_of(v))
    if v.layout is not None:
        from . import duo_site
        # the case's engineered asset pins the source object sizes: the layout's objects replace them in the spec
        scene = duo_site.site_scene(scene, v.layout)
        duo_site.adapt(spec, v.layout, scene)
    m = spec.compile()
    d = mujoco.MjData(m)
    q = d.qpos.copy()
    ctrl = np.zeros(m.nu)
    for h in F.HANDS:
        hq = home_q(v, h)
        for i in range(7):
            q[m.joint(f'{h}_fr3v2_joint{i + 1}').qposadr[0]] = hq[i]
            ctrl[m.actuator(f'{h}_fr3v2_joint{i + 1}').id] = hq[i]
    spec.add_key(name='home', qpos=q, ctrl=ctrl)
    xml = spec.to_xml()
    if v.layout is not None:
        from . import duo_site
        if v.layout in duo_site.S1_ASSET_LAYOUTS:  # the task objects verbatim from the user's S1 scenes
            xml = duo_site.s1_object_xml(xml, v.task)
    m = mujoco.MjModel.from_xml_string(xml)
    return m, scene, hashlib.sha256(xml.encode()).hexdigest(), xml
