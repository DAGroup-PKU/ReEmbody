"""Refine stage of the FR3 Duo v2 data, all three tasks (2026-11-06): sim2-refine/tasks/fr3_stack_cups/refine.py's
procedure (each admitted Sim2 episode re-run on the same initial scene with the refine rules, most refined candidate
first; the first that passes the full admission is kept, else the Sim2 episode stays unrefined), with per-task
candidates. No LLM is called.
  stack_cups: refine.py's candidates unchanged (R1 no detour, R2 overlap, R3 speed x1.6 / x1.3, R4 short exit);
  stack_blocks: the same rules and candidates (controller duo_stack_blocks_v2.py takes the cups controller's options);
  drawer: R2 overlap (duo_drawer_v2r.py: the block hand moves during the handle hand's approach, pull and push where
          their joint path is clear together), R3 speed (lines and joint moves x1.6 / x1.3, joints <= 0.72 rad/s, 0.5 with
          the block, grippers 0.4 s) and R4 short exit (after the push the handle hand only frees its fingers from the bar);
          the first candidate has all three, the others drop R2 (they run unchanged on duo_drawer_v2r.py).

    t1_retarget/bin/python refine_v2.py --task <task> --sim2 <sim2 run dir> --controller <refine controller> \
        --out <fresh dir> [--workers 30] [--resume]
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
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import refine as R_CUPS  # noqa: E402

DRAWER_COMMON = dict(grip_s=(0.4, 0.4, 0.4), short_exit=True)
CANDIDATES = {
    'stack_cups': list(R_CUPS.CANDIDATES),
    'stack_blocks': list(R_CUPS.CANDIDATES),
    'drawer': [('r234_x1.6', dict(speed=1.6, overlap=True, **DRAWER_COMMON)),
               ('r34_x1.6', dict(speed=1.6, **DRAWER_COMMON)), ('r34_x1.3', dict(speed=1.3, **DRAWER_COMMON)),
               ('r4_x1.0', dict(speed=1.0, **DRAWER_COMMON))],
}
RULES = dict(R_CUPS.RULES, R4_drawer='short exit: after the push the handle hand only frees its fingers from the bar')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _refine_one(args):
    task, controller, vj, out, sim2_run = args
    from franka_dual_v1 import duo_scene as DS
    from franka_dual_v1.duo_run import run_one
    v = DS.Variation.from_json(vj)
    dst = Path(out) / 'runs' / v.id
    if (dst / 'refine.json').exists():
        return json.loads((dst / 'refine.json').read_text())
    s2 = json.loads((Path(sim2_run) / 'record.json').read_text())
    tried = []
    for name, opts in CANDIDATES[task]:
        d = Path(out) / 'candidates' / name / v.id
        f = d / 'record.json'
        rec = json.loads(f.read_text()) if f.exists() else None
        reused = rec is not None and rec.get('options') == json.loads(json.dumps(opts))
        if not reused:
            rec = run_one(Path(controller), v, d, options=opts, view=False)
        m = rec.get('measurements', {})
        tried.append(dict(candidate=name, admitted=rec['admitted'], duration_s=m.get('duration_s'),
                          fails=[x[:300] for x in rec['fails']], wall_s=rec.get('wall_s'), reused=reused))
        if rec['admitted']:
            break
    win = next((t for t in tried if t['admitted']), None)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(out) / 'candidates' / win['candidate'] / v.id if win else Path(sim2_run)
    for fn in ('record.json', 'episode.npz', 'scene.xml'):
        shutil.copy2(src / fn, dst / fn)
    row = dict(variation=v.id, source_case=v.source_case, refined=win is not None,
               candidate=win['candidate'] if win else None,
               options=dict(CANDIDATES[task])[win['candidate']] if win else None,
               sim2_duration_s=s2['measurements']['duration_s'], duration_s=(win or {}).get('duration_s'),
               sim2_record_sha256=sha(Path(sim2_run) / 'record.json'), tried=tried)
    (dst / 'refine.json').write_text(json.dumps(row, indent=1) + '\n')
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True)
    ap.add_argument('--sim2', type=Path, required=True)
    ap.add_argument('--controller', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=30)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--limit', type=int, help='only the first LIMIT admitted Sim2 runs (the selection takes the first refined ones in draw order)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=a.resume)
    ctl = a.out / 'controller.py'
    if not a.resume:
        shutil.copy2(a.controller, ctl)
    rnd = json.loads((a.sim2 / 'round.json').read_text())
    from franka_dual_v1 import duo_rounds
    order = json.loads((a.sim2 / 'variations.json').read_text())
    ok = {r['variation'] for r in rnd['runs'] if r['admitted']}
    vs = [v for v in order if v['id'] in ok]
    vs = vs[:a.limit] if a.limit else vs
    (a.out / 'variations.json').write_text(json.dumps(vs, indent=1) + '\n')
    t0 = time.time()
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(_refine_one, [(a.task, str(ctl), v, str(a.out), str(a.sim2 / 'runs' / v['id'])) for v in vs]))
    n_ref = sum(r['refined'] for r in rows)
    fw = duo_rounds.framework_sha()
    meta = dict(schema='rssr.fr3_duo.refine.v2', task=a.task, sim2_run=str(a.sim2), controller_sha256=sha(ctl),
                sim2_controller_sha256=rnd['controller_sha256'], framework_sha256=fw, refine_code_sha256=sha(__file__),
                rules=RULES, candidates=[dict(name=n, options=o) for n, o in CANDIDATES[a.task]],
                sim2_admitted=len(vs), refined=n_ref, rollback=len(vs) - n_ref, wall_s=round(time.time() - t0, 1),
                started_utc=started, finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), episodes=rows)
    (a.out / 'refine.json').write_text(json.dumps(meta, indent=1) + '\n')
    (a.out / 'round.json').write_text(json.dumps(dict(
        schema='rssr.fr3_duo.round.v1', task=a.task, split=rnd['split'] + '+refine', controller_sha256=sha(ctl),
        framework_sha256=fw, controller_source=str(a.controller), change='refine stage (refine_v2.py) on ' + str(a.sim2),
        n=len(rows), admitted=n_ref, admitted_fraction=round(n_ref / max(len(rows), 1), 4), wall_s=meta['wall_s'],
        started_utc=started, finished_utc=meta['finished_utc'],
        runs=[dict(variation=r['variation'], source_case=r['source_case'], admitted=r['refined'], candidate=r['candidate'],
                   duration_s=r['duration_s'] if r['refined'] else r['sim2_duration_s'],
                   sim2_duration_s=r['sim2_duration_s']) for r in rows]), indent=1) + '\n')
    by = {}
    for r in rows:
        by[r['candidate']] = by.get(r['candidate'], 0) + 1
    print(json.dumps(dict(sim2_admitted=len(vs), refined=n_ref, by_candidate=by, wall_s=meta['wall_s'])))


if __name__ == '__main__':
    main()
