"""Refine stage for FR3 Duo stack-cups trajectories (2026-11-04): rule-based optimisation of trajectories that already
passed Sim2 admission (real1 -> sim1 -> sim2 -> refine -> real2, sim2-refine/README.md). No LLM is called.

Input: a Sim2 synthesis run of franka_dual_v1.duo_rounds (round.json, controller.py, runs/<variation>/). Each admitted
Sim2 episode is re-run on the same initial scene by the same controller (the run's snapshot) with the refine rules, the
most refined candidate first:
  R1 no detour: the upper hand carries its cup straight to a staging point beside the stack region, instead of first
     out to the waiting pose over its own side (the Sim2 program's outward move);
  R2 overlap: the two arms move at the same time where their joint path is clear together (the upper hand's staging
     while the base hand carries; the base hand's move aside while the upper hand carries);
  R3 speed: line and joint speeds x1.6 (joint speeds capped at 0.72 rad/s, under the real deployment's 0.025 rad per
     30 Hz step; lines keep that per-frame limit), grippers 0.4 s, the hands open while approaching, pre-grasp 7 cm;
  R4 short exit: after the last release the hand only rises (the episode ends with both cups stacked and released).
Candidates in order: R1-R4 at x1.6; R1-R4 at x1.3; R1, R3 (x1.3), R4 without overlap; the upper hand keeps its cup at
the lift pose, R3 (x1.0), R4. The first candidate that passes the full Sim2 admission (contract, hard constraints,
bit-exact independent replay, single grasp per cup, no state-action conflict, >= 2 cm arm clearance, no idle segment)
is the refined episode. If none passes, the Sim2 episode stays as it is and is marked unrefined (rollback). Nothing of
the Sim2 run is modified; every candidate's record is kept.

    t1_retarget/bin/python sim2-refine/tasks/fr3_stack_cups/refine.py --sim2 <sim2 run dir> --out <fresh dir> \
        [--workers 20] [--limit N] [--resume]

--resume continues an interrupted run in the same --out: episodes with runs/<variation>/refine.json are kept, and a
candidate whose record (same options) exists is reused (runs are deterministic: bit-exact replay is an admission gate).
A run that raises outside the controller (e.g. the offscreen renderer of the wrist-visibility measurement failing to
get a context on a busy machine) is retried up to 3 times; such errors are environment failures, not rule failures.
--no-view skips the camera-visibility measurements (recorded only, not admission gates: user, 2026-11-03) when the
local GPU cannot give offscreen contexts; they can be recomputed from episode.npz later.

Output: refine.json (rules, candidates, per episode: Sim2 and refined duration, the candidate used, every candidate's
failures), round.json + variations.json + controller.py in the duo_rounds layout (rows `admitted` = refined), runs/
<variation>/ = the refined episode (record.json, episode.npz, scene.xml, refine.json), candidates/<name>/<variation>/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
SIM = Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))

COMMON = dict(final_aside=False, grip_s=(0.4, 0.4, 0.4), preopen_in_approach=True, pre=0.07)
CANDIDATES = (
    ('r1234_x1.6', dict(upper_wait='stage', overlap=True, speed=1.6, **COMMON)),
    ('r1234_x1.3', dict(upper_wait='stage', overlap=True, speed=1.3, **COMMON)),
    ('r134_x1.3', dict(upper_wait='stage', overlap=False, speed=1.3, **COMMON)),
    ('hold_r34_x1.0', dict(upper_wait='hold', overlap=False, speed=1.0, **COMMON)),
)
RULES = dict(R1='no detour: the upper hand goes straight to a staging point beside the stack region',
             R2='overlap: both arms move at once where their joint path is clear together',
             R3='speed: lines and joints x1.6 (x1.3, x1.0 in later candidates), joints <= 0.72 rad/s, grippers 0.4 s, '
                'opening during the approach, pre-grasp 7 cm',
             R4='short exit: the last hand only rises after its release')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


VIEW = True


def _run(controller, v, d, opts):
    """(record, reused, environment errors) of one candidate run."""
    from franka_dual_v1.duo_run import run_one
    f = d / 'record.json'
    if f.exists():
        rec = json.loads(f.read_text())
        if rec.get('options') == json.loads(json.dumps(opts)):
            return rec, True, []
    errors = []
    for attempt in range(3):
        try:
            return run_one(Path(controller), v, d, options=opts, view=VIEW), False, errors
        except Exception as e:  # noqa: BLE001  (raised outside the controller: environment)
            errors.append(repr(e)[:300])
            time.sleep(30)
    raise RuntimeError(f'{v.id}: environment errors {errors}')


def _refine_one(args):
    global VIEW
    controller, vj, out, sim2_run, VIEW = args
    from franka_dual_v1 import duo_scene as DS
    v = DS.Variation.from_json(vj)
    dst = Path(out) / 'runs' / v.id
    if (dst / 'refine.json').exists():
        return json.loads((dst / 'refine.json').read_text())
    s2 = json.loads((Path(sim2_run) / 'record.json').read_text())
    tried = []
    for name, opts in CANDIDATES:
        d = Path(out) / 'candidates' / name / v.id
        rec, reused, errors = _run(controller, v, d, opts)
        m = rec.get('measurements', {})
        tried.append(dict(candidate=name, admitted=rec['admitted'], duration_s=m.get('duration_s'),
                          fails=[f[:300] for f in rec['fails']], wall_s=rec.get('wall_s'), reused=reused,
                          environment_errors=errors))
        if rec['admitted']:
            break
    win = next((t for t in tried if t['admitted']), None)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(out) / 'candidates' / win['candidate'] / v.id if win else Path(sim2_run)
    for f in ('record.json', 'episode.npz', 'scene.xml'):
        shutil.copy2(src / f, dst / f)
    row = dict(variation=v.id, source_case=v.source_case, refined=win is not None,
               candidate=win['candidate'] if win else None, options=dict(CANDIDATES)[win['candidate']] if win else None,
               sim2_duration_s=s2['measurements']['duration_s'], duration_s=(win or {}).get('duration_s'),
               sim2_record_sha256=sha(Path(sim2_run) / 'record.json'), tried=tried)
    (dst / 'refine.json').write_text(json.dumps(row, indent=1) + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sim2', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--limit', type=int, default=0, help='only the first N admitted Sim2 episodes (0: all)')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--no-view', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=a.resume)
    rnd = json.loads((a.sim2 / 'round.json').read_text())
    ctl = a.out / 'controller.py'
    if not a.resume:
        shutil.copy2(a.sim2 / 'controller.py', ctl)
    assert sha(ctl) == rnd['controller_sha256']
    from franka_dual_v1 import duo_rounds
    fw = duo_rounds.framework_sha()
    order = json.loads((a.sim2 / 'variations.json').read_text())
    ok = {r['variation'] for r in rnd['runs'] if r['admitted']}
    vs = [v for v in order if v['id'] in ok]
    if a.limit:
        vs = vs[:a.limit]
    if a.resume:
        assert json.loads((a.out / 'variations.json').read_text()) == vs, 'resume: different variations'
    else:
        (a.out / 'variations.json').write_text(json.dumps(vs, indent=1) + '\n')
    t0 = time.time()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_refine_one, [(str(ctl), v, str(a.out), str(a.sim2 / 'runs' / v['id']), not a.no_view)
                                           for v in vs]))
    n_ref = sum(r['refined'] for r in rows)
    meta = dict(schema='rssr.fr3_duo.refine.v1', task=rnd['task'], sim2_run=str(a.sim2), sim2_split=rnd['split'],
                controller_sha256=rnd['controller_sha256'], sim2_framework_sha256=rnd['framework_sha256'],
                framework_sha256=fw, refine_code_sha256=sha(__file__), rules=RULES, view_measurements=not a.no_view,
                candidates=[dict(name=n, options=o) for n, o in CANDIDATES],
                sim2_admitted=len(vs), refined=n_ref, rollback=len(vs) - n_ref,
                wall_s=round(time.time() - t0, 1), started_utc=started,
                finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), episodes=rows)
    (a.out / 'refine.json').write_text(json.dumps(meta, indent=1) + '\n')
    # duo_rounds layout for select_site.py: admitted = refined
    (a.out / 'round.json').write_text(json.dumps(dict(
        schema='rssr.fr3_duo.round.v1', task=rnd['task'], split=rnd['split'] + '+refine',
        controller_sha256=rnd['controller_sha256'], framework_sha256=fw, controller_source=str(a.sim2 / 'controller.py'),
        change='refine stage (sim2-refine/tasks/fr3_stack_cups/refine.py) on ' + str(a.sim2), n=len(rows),
        admitted=n_ref, admitted_fraction=round(n_ref / max(len(rows), 1), 4), wall_s=meta['wall_s'],
        started_utc=started, finished_utc=meta['finished_utc'],
        runs=[dict(variation=r['variation'], source_case=r['source_case'], admitted=r['refined'],
                   candidate=r['candidate'], duration_s=r['duration_s'] if r['refined'] else r['sim2_duration_s'],
                   sim2_duration_s=r['sim2_duration_s']) for r in rows]), indent=1) + '\n')
    by = {}
    for r in rows:
        by[r['candidate']] = by.get(r['candidate'], 0) + 1
    print(json.dumps(dict(sim2_admitted=len(vs), refined=n_ref, by_candidate=by, wall_s=meta['wall_s'])))


if __name__ == '__main__':
    main()
