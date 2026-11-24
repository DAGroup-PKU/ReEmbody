"""Which fixed grasp per hand (2026-11-07): for each candidate, on the first N placements of a split, plan what the controller
plans before moving: IK of the pre-grasp from the start pose, joint path to it, descent, lift. One hand at a time, the
other at its start pose. No physics; the full check is the pilot's sim2 and refine.

    python scan_grasps.py --task stack_cups --n 40 --out <dir> [--workers 60]
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'glfw')
A = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'))
sys.path.insert(0, str(A / 'code'))
BLOCK_DEG = (0.0, 90.0, 180.0, 270.0)
WALL_DEG = tuple(np.arange(12) * 30.0)


def load(task):
    f = A / 'controllers' / ('duo_stack_cups.py' if task == 'stack_cups' else 'duo_stack_blocks_v2.py')
    spec = importlib.util.spec_from_file_location('scan_controller', str(f))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def one(args):
    task, vj = args
    import arch_away
    from franka_dual_v1 import duo_scene as DS
    from franka_dual_v1.duo_run import Session
    mod = load(task)
    S = Session(DS.Variation.from_json(vj))
    for h in ('left', 'right'):
        arch_away.install(S, h)
    b = S.bindings
    obj = {b['stack_hands']['base']: b['targets'][0], b['stack_hands']['upper']: b['targets'][1]}
    cands = [(d, False) for d in BLOCK_DEG] if task == 'stack_blocks' else [(d, f) for d in WALL_DEG for f in (False, True)]
    out = []
    for hand, o in obj.items():
        for deg, flip in cands:
            t0 = time.time()
            row = dict(variation=vj['id'], hand=hand, deg=deg, flip=flip)
            try:
                g = mod.face_grasp(S, hand, o, deg) if task == 'stack_blocks' else mod.wall_grasp(S, hand, o, deg, flip)
                pre, grasp, R, q = g
                row['q_pre'] = np.round(q, 4).tolist()
                row['move'] = round(float(np.abs(q - S.home[hand]).max()), 3)
                S.joint_plan(hand, q, mod.vj())
                dn = S.plan_line({hand: (grasp, R)}, speed=mod.DESCEND_SPEED, start={hand: q}, label='descend')
                S.plan_line({hand: (grasp + [0, 0, mod.LIFT], R)}, speed=0.08, start={hand: dn[hand][-1]}, label='lift')
                row['ok'] = True
            except Exception as e:  # noqa: BLE001
                row.update(ok=False, why=str(e)[:160])
            row['s'] = round(time.time() - t0, 2)
            out.append(row)
    return out


def clusters(q, gap=0.5):
    """Single-linkage clusters of joint vectors, a gap of more than `gap` rad on some joint separating them."""
    n = len(q)
    lab = list(range(n))

    def find(i):
        while lab[i] != i:
            i = lab[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if np.abs(q[i] - q[j]).max() <= gap:
                lab[find(i)] = find(j)
    c = {}
    for i in range(n):
        c[find(i)] = c.get(find(i), 0) + 1
    return sorted(c.values(), reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=60)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    vs = json.loads((A / 'splits' / f'{a.task}_train.json').read_text())[:a.n]
    with ProcessPoolExecutor(a.workers) as pool:
        rows = [r for rs in pool.map(one, [(a.task, v) for v in vs]) for r in rs]
    (a.out / 'rows.json').write_text(json.dumps(rows) + '\n')
    table = []
    for hand in ('left', 'right'):
        for key in sorted({(r['deg'], r['flip']) for r in rows if r['hand'] == hand}):
            sel = [r for r in rows if r['hand'] == hand and (r['deg'], r['flip']) == key]
            ok = [r for r in sel if r['ok']]
            q = np.array([r['q_pre'] for r in ok]) if ok else np.zeros((0, 7))
            table.append(dict(hand=hand, deg=key[0], flip=key[1], ok=len(ok), n=len(sel), clusters=clusters(q) if len(q) else [],
                              move=round(float(np.median([r['move'] for r in ok])), 2) if ok else None,
                              q_spread=round(float(np.percentile(np.abs(q - np.median(q, axis=0)).max(axis=1), 90)), 2) if len(q) else None,
                              s=round(float(np.mean([r['s'] for r in sel])), 2)))
    (a.out / 'table.json').write_text(json.dumps(table, indent=1) + '\n')
    for t in sorted(table, key=lambda t: (t['hand'], -t['ok'], t['move'] or 9)):
        print(t)


if __name__ == '__main__':
    main()
