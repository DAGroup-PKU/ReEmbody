"""Initial conditions of the FR3 Duo v2 data (2026-11-06; user: "跟S1一样的setting、也需要数据增强、refine轨迹"; v1 on HF is
wrong and replaced). Layout fr3_site_20261105d (duo_site.NAME_V8): the user's site with the desk 15 cm closer, the S1
tasks' objects verbatim, 6 cm to the camera arch (exact), 10 cm between the arms when stacking.

Placement augmentation as S1 (S1 settings: sim2-refine/orchestration/stack_random200_r15/sample_v3.py for cups,
sim2-refine/tasks/stack_blocks/random200_near/sample_near.py for blocks, sim2-refine/tasks/drawer/
new_center_layout_spec_v1.py for the drawer), around the FR3 nominal places (mount frame, x forward, y left):
  * stack_cups: each cup uniform in a 15 cm disk around (0.78, +-0.28), yaw U(-15, 15) deg; the stack region in a 5 cm
    disk around (0.76, 0);
  * stack_blocks: each block uniform in a 4 cm disk around (0.78, +-0.28), yaw U(-15, 15) deg; region 5 cm disk;
  * drawer: the cabinet (with its drawers and handles) moved by U(-1, 1) cm in x and y from (0.84, 0) (front facing the
    robot), the block upright with yaw 0 in a 4 cm disk around (0.76, 0.30);
  * the robot starts exactly at the user's measured start pose (no joint perturbation: user, 2026-11-04).
A draw is redrawn when an object leaves the desk (2 cm margin), two footprints come within 2 cm, the upper object
stands on the stack spot (duo_site.valid_site), or the drawer block is within 3 cm of the cabinet or of the opened
drawer's sweep. Every task uses its source case ep000000 (the S1 objects' sizes; the bindings), as the nominal runs.

    python sample_v2.py   -> ../splits/<task>_<split>.json (list of Variation dicts)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'))
from franka_dual_v1 import duo_scene as DS  # noqa: E402
from franka_dual_v1 import duo_site as SITE  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'splits'
LAYOUT = SITE.NAME_V8
import os as _os
DRAWER_CASE = _os.environ.get('REEMBODY_DRAWER_CASE_DIR', str(Path(__file__).resolve().parents[3] / 'examples/drawer_ep000/01_sim1_reconstructed'))
NOMINAL = {'stack_cups': {'cup_left_base': (0.78, 0.28), 'cup_right_upper': (0.78, -0.28), 'stack_region': (0.76, 0.0)},
           'stack_blocks': {'blue_cube': (0.78, 0.28), 'purple_cube': (0.78, -0.28), 'stack_region': (0.76, 0.0)},
           'drawer': {'drawer_cabinet': (0.84, 0.0), 'red_block': (0.76, 0.30)}}
OBJ_R = {'stack_cups': 0.15, 'stack_blocks': 0.04, 'drawer': 0.04}
REGION_R, YAW_DEG, CAB_JITTER, CAB_GAP = 0.05, 15.0, 0.01, 0.03
SPLITS = {'train': (6_000, 220), 'heldout': (7_000, 50)}  # seed stream, number of draws
TASKS = ('stack_cups', 'stack_blocks', 'drawer')


def disk(rng, c, r):
    a, rho = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.random())
    return np.asarray(c, float) + rho * np.array([np.cos(a), np.sin(a)])


def rect_dist(p, lo, hi):
    return float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - p, p - hi))))


def drawer_ok(v):
    """Block on the desk and >= CAB_GAP from the cabinet footprint and from the opened drawer's sweep."""
    sc = DS.apply(DS.raw_scene(v.source_case), v)
    objs = {o['id']: o for o in sc['objects']}
    cab, dr, blk = objs['drawer_cabinet'], objs['bottom_drawer'], objs['red_block']
    c = np.asarray(cab['position_m'][:2])
    h = np.asarray(cab['size_m'][:2]) / 2
    lo, hi = c - h, c + h
    hi = hi + [0.0, dr['joint']['range'][1] + 0.05]  # the drawer opens toward +y (the robot), handle included
    b = np.asarray(blk['position_m'][:2])
    r = blk['size_m'][0] / 2 * np.sqrt(2)
    pm = SITE.world_to_mount([b[0], b[1], 0.0], LAYOUT)
    return rect_dist(b, lo, hi) >= r + CAB_GAP and SITE.on_desk(pm[:2], r, LAYOUT)


def draw(task, split, k):
    seed, _ = SPLITS[split]
    rng = np.random.default_rng([seed, TASKS.index(task), k])
    case = DRAWER_CASE if task == 'drawer' else DS.source_cases(task)[0]
    for attempt in range(1000):
        centers, yaw = {}, {}
        for oid, c in NOMINAL[task].items():
            if oid == 'drawer_cabinet':
                centers[oid] = tuple(np.asarray(c) + rng.uniform(-CAB_JITTER, CAB_JITTER, 2))
            elif oid == 'stack_region':
                centers[oid] = tuple(disk(rng, c, REGION_R))
            else:
                centers[oid] = tuple(disk(rng, c, OBJ_R[task]))
                yaw[oid] = 0.0 if task == 'drawer' else float(np.radians(rng.uniform(-YAW_DEG, YAW_DEG)))
        v0 = SITE.nominal(task, layout=LAYOUT, case=case, centers=centers, tag=f'v2_{split}_{k:04d}')
        shift = {o: list(s) for o, s in v0.object_shift.items()}
        for o, y in yaw.items():
            shift[o][2] = shift[o][2] + y
        v = DS.Variation(f'{task}_v2_{split}_{k:04d}', task, case, shift, {}, layout=LAYOUT)
        if task == 'drawer':
            if drawer_ok(v):
                return v, attempt
            continue
        sc = SITE.site_scene(DS.apply(DS.raw_scene(case), v), LAYOUT)
        objs = {o['id']: o for o in sc['objects']}
        ok = all(SITE.on_desk(SITE.world_to_mount(objs[i]['position_m'], LAYOUT)[:2], DS._footprint(objs[i])[1], LAYOUT)
                 for i in objs if objs[i]['mobility'] == 'free')
        base = DS._base(sc)
        reg = sc['bindings']['target_region']
        ok = ok and SITE.on_desk(SITE.world_to_mount(objs[reg]['position_m'], LAYOUT)[:2], DS._footprint(objs[base])[1], LAYOUT)
        if ok and SITE.valid_site(sc):
            return v, attempt
    raise RuntimeError(f'no valid draw for {task} {split} {k}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--drawer-cabinet-y', type=float, help='drawer only: nominal cabinet y (mount frame, + = left); '
                                                            'writes drawer_<split>_cy<y>.json')
    ap.add_argument('--drawer-forward', type=float, default=0.0,
                    help='drawer only: move the cabinet and block nominal places this far from the robot (mount x, m; '
                         'user 2026-11-06: "抽屉和方块需要再向后摆放10cm左右"); adds _fx<d> to the file names')
    ap.add_argument('--n', type=int, help='draws per split (default SPLITS)')
    ap.add_argument('--n-train', type=int, help='drawer only: draws in the train split (heldout keeps SPLITS)')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.drawer_cabinet_y is not None:
        fx = a.drawer_forward
        NOMINAL['drawer'] = dict(drawer_cabinet=(NOMINAL['drawer']['drawer_cabinet'][0] + fx, a.drawer_cabinet_y),
                                 red_block=(NOMINAL['drawer']['red_block'][0] + fx, NOMINAL['drawer']['red_block'][1]))
        tag = f'_cy{a.drawer_cabinet_y:+.2f}' + (f'_fx{fx:+.2f}' if fx else '')
        for split, (_, n) in SPLITS.items():
            n = a.n_train if split == 'train' and a.n_train else a.n or n
            rows = [draw('drawer', split, k)[0].to_json() for k in range(n)]
            (OUT / f'drawer_{split}{tag}.json').write_text(json.dumps(rows, indent=1) + '\n')
            print('drawer', split, len(rows), 'cabinet y', a.drawer_cabinet_y, flush=True)
        return
    for task in TASKS:
        for split, (_, n) in SPLITS.items():
            rows, redraws = [], 0
            for k in range(n):
                v, a = draw(task, split, k)
                rows.append(v.to_json())
                redraws += a
            (OUT / f'{task}_{split}.json').write_text(json.dumps(rows, indent=1) + '\n')
            print(task, split, len(rows), 'redraws', redraws, flush=True)


if __name__ == '__main__':
    main()
