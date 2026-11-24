"""One revision round of a task controller on the FR3 Duo (2026-11-03), or a synthesis run of a kept version.

A round snapshots the controller file (controller.py + sha256, the version), runs it on the task's development
placements (duo_scene split 'dev', the same 20 every round so rounds compare) in parallel, and writes
round.json: version, what changed (--change, written by the coding agent before the run), per-run admission and
failures, the admitted fraction, physics steps and wall time; each run's full measurements are in
runs/<variation>/record.json (episode.npz and scene.xml for admitted runs). history.json in the task folder lists
the rounds. The stop rule (applied by the coding agent between rounds): stop when the admitted fraction has not risen
for 3 rounds in a row, or after 12 rounds; keep the version with the highest fraction (earliest on ties).

    python -m franka_dual_v1.duo_rounds --task stack_cups --controller <file> --out <task dir>/round_03 \
        --change "<what this version changes and why>" [--split dev --n 20 --workers 20]
    python -m franka_dual_v1.duo_rounds ... --split source        # the task's 50 source placements
    python -m franka_dual_v1.duo_rounds ... --split heldout --n 50  # only after development has stopped
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

from . import duo_scene as DS  # noqa: E402

DEV_N = 20
FRAMEWORK = ('duo_run.py', 'duo_robot.py', 'duo_scene.py', 'servo.py', 'robot.py', 'fr3_duo.py', 'contract.py',
             'metrics.py', 'duo_gripper_table.json', 'duo_site.py', 'exact_clearance.py')


def framework_sha():
    here = Path(__file__).parent
    return {f: hashlib.sha256((here / f).read_bytes()).hexdigest()[:16] for f in FRAMEWORK}


def variations(task, split, n, start=0):
    if split == 'source':
        return [DS.source_variation(c) for c in DS.source_cases(task)]
    if split.startswith('site_'):
        from . import duo_site
        return duo_site.split_list(task, split, n, start)
    return DS.split_list(task, split, n)


def _job(args):
    controller, vj, out, view = args
    from .duo_run import run_one
    v = DS.Variation.from_json(vj)
    rec = run_one(Path(controller), v, Path(out) / v.id, view=view)
    m = rec.get('measurements', {})
    return dict(variation=v.id, source_case=v.source_case, status=rec['status'], admitted=rec['admitted'],
                fails=rec['fails'], physics_steps=m.get('physics_steps', 0), duration_s=m.get('duration_s'),
                min_arm_clearance_m=m.get('min_arm_clearance_m'), wall_s=rec.get('wall_s'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=DS.TASKS)
    ap.add_argument('--controller', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--change', required=True)
    ap.add_argument('--split', default='dev', choices=('dev', 'source', 'heldout', 'synth', 'site_aug', 'site_heldout', 'site_aug_v2', 'site_aug_v3'))
    ap.add_argument('--start', type=int, default=0, help='first variation index (site_* splits)')
    ap.add_argument('--n', type=int, default=DEV_N)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--no-view', action='store_true', help='skip the camera-visibility measurements (recorded only, '
                                                         'not gates; they need an OpenGL context)')
    ap.add_argument('--variations', type=Path, help='a JSON list of Variation dicts to run instead of --split/--n '
                                                    '(e.g. a data campaign\'s sampled initial conditions)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    ctl = a.out / 'controller.py'
    shutil.copy2(a.controller, ctl)
    sha = hashlib.sha256(ctl.read_bytes()).hexdigest()
    vs = [DS.Variation.from_json(v) for v in json.loads(a.variations.read_text())] if a.variations else \
        variations(a.task, a.split, a.n, a.start)
    assert all(v.task == a.task for v in vs)
    (a.out / 'variations.json').write_text(json.dumps([v.to_json() for v in vs], indent=1) + '\n')
    t0 = time.time()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_job, [(str(ctl), v.to_json(), str(a.out / 'runs'), not a.no_view) for v in vs]))
    n_ok = sum(r['admitted'] for r in rows)
    summary = dict(schema='rssr.fr3_duo.round.v1', task=a.task, view_measurements=not a.no_view,
                   split=str(a.variations) if a.variations else a.split, controller_sha256=sha,
                   framework_sha256=framework_sha(),
                   controller_source=str(a.controller), change=a.change, n=len(rows), admitted=n_ok,
                   admitted_fraction=round(n_ok / len(rows), 4),
                   physics_steps=int(sum(r['physics_steps'] for r in rows)), wall_s=round(time.time() - t0, 1),
                   started_utc=started, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   runs=rows)
    (a.out / 'round.json').write_text(json.dumps(summary, indent=1) + '\n')
    hist = a.out.parent / 'history.json'
    h = json.loads(hist.read_text()) if hist.exists() else []
    h.append(dict(round=a.out.name, split=a.split, controller_sha256=sha, framework_sha256=summary['framework_sha256'],
                  change=a.change, n=len(rows), admitted=n_ok,
                  admitted_fraction=summary['admitted_fraction'], physics_steps=summary['physics_steps'],
                  started_utc=started, finished_utc=summary['finished_utc']))
    hist.write_text(json.dumps(h, indent=1) + '\n')
    print(f'{a.task} {a.out.name}: {n_ok}/{len(rows)} admitted, {summary["physics_steps"]} physics steps, '
          f'{summary["wall_s"]} s')
    reasons = {}
    for r in rows:
        for f in r['fails']:
            k = f.split(':')[0] + ':' + f.split(':')[1][:60] if ':' in f else f[:60]
            reasons[k] = reasons.get(k, 0) + 1
    for k, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f'  {n:3d}  {k}')


if __name__ == '__main__':
    main()
