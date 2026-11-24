#!/usr/bin/env python3
"""Render selected v2 episodes on this machine's GPU, next to render_fleet_lean.py on the B200 nodes (2026-11-06).

The B200 GPUs were at about 94 % from other users' jobs (1.1 s per render there; 0.25 s on the local GPU, same scene).
Same renderer, launch settings and verification. This worker takes episodes from the end of the selection and the
scheduler from the start; `local_claim` in an episode's dir and the scheduler's `remote_running.json` keep them apart.
GPU rule as everywhere: start a render only if, over ~60 s of pmon, the SM utilisation of processes that are not ours
stays below 60 % and >= 8 GB is free.

    t1_retarget/bin/python render_local_gpu.py --selection <selection.json> --output <dir> --site-dir <dir> --cup-blend <file> --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

import render_fleet_v2 as RF

HERE = Path(__file__).resolve().parent
import sys as _sys
PY = os.environ.get('REEMBODY_BLENDER_PYTHON', _sys.executable)


def our_pids(output):
    pids = set()
    for p in filter(str.isdigit, os.listdir('/proc')):
        try:
            with open(f'/proc/{p}/cmdline', 'rb') as f:
                argv = f.read().decode('utf8', 'replace').split('\0')
        except OSError:
            continue
        if 'python' in os.path.basename(argv[0]) and any('render_site_v2' in x for x in argv) and any(x.startswith(str(output)) for x in argv):
            pids.add(p)
    return pids


class Adopted:
    """A render started by an earlier worker of ours: success is its receipt, which is written last."""

    def __init__(self, pid, receipt):
        self.pid, self.receipt = pid, receipt

    def poll(self):
        return None if os.path.exists(f'/proc/{self.pid}') else (0 if self.receipt.exists() else 1)


def running_renders(output):
    """pid -> episode of our renders already running here."""
    found = {}
    for p in our_pids(output):
        try:
            with open(f'/proc/{p}/cmdline', 'rb') as f:
                argv = f.read().decode('utf8', 'replace').split('\0')
        except OSError:
            continue
        out = argv[argv.index('--output') + 1]
        if out.endswith('/render_local'):
            found[p] = int(Path(out).parent.name[2:])
    return found


def sample(output):
    """Mean SM of the processes that are not ours over ~60 s, and free memory."""
    text = subprocess.run(['nvidia-smi', 'pmon', '-s', 'u', '-d', '5', '-c', '12'], capture_output=True, text=True, timeout=180).stdout
    mine, foreign = our_pids(output), {}
    for line in text.splitlines():
        x = line.split()
        if len(x) < 4 or x[0].startswith('#') or x[1] == '-' or x[1] in mine:
            continue
        foreign.setdefault(x[1], []).append(0.0 if x[3] == '-' else float(x[3]))
    free = float(subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=60).stdout.split()[0])
    return sum(sum(v) / 12 for v in foreign.values()), free


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--site-dir', type=Path, required=True)
    ap.add_argument('--cup-blend', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    a = ap.parse_args()
    a.stage_root = 'local'
    F = RF.Fleet(a)
    rows = F.sel['episodes']
    has_receipt = lambda ep: (a.output / f'ep{ep:03d}/render/render_receipt.json').exists()
    running, attempts = {}, {}
    for pid, ep in running_renders(a.output).items():
        running[ep] = (Adopted(pid, a.output / f'ep{ep:03d}/render_local/render_receipt.json'), time.time())
    print('adopted', sorted(running), flush=True)
    # claims of an earlier worker of ours whose render is gone
    for ep in range(len(rows)):
        c = a.output / f'ep{ep:03d}/local_claim'
        if c.exists() and ep not in running:
            c.unlink()
    F.log(dict(event='start', scheduler='local', workers=a.workers))

    def start(ep):
        work = F.prepare(ep, rows[ep])
        (work / 'local_claim').write_text(str(os.getpid()))
        # kept apart until this render is the one that counts: a node may still deliver this episode first with its own config
        cfg = work / 'render_local_cfg/config.json'
        cfg.parent.mkdir(exist_ok=True)
        subprocess.run([RF.T1, str(HERE / 'make_config_v2.py'), '--site-dir', str(a.site_dir), '--cup-blend', str(a.cup_blend),
                        '--episode-index', str(ep), '--variation', rows[ep]['variation'], '--out', str(cfg)], check=True)
        tmp = work / 'render_local'
        shutil.rmtree(tmp, ignore_errors=True)
        q = shlex.quote
        cmd = (f'cd {q(str(HERE))} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 RSSR_RENDER_THREADS=4 nice -n 10 {PY} -u render_site_v2.py '
               f'--source {q(str(work / "source"))} --config {q(str(cfg))} --output {q(str(tmp))} --device OPTIX --samples 12 '
               f'2>&1 | tail -c 100000 > {q(str(work / "render_tail.log"))}; exit ${{PIPESTATUS[0]}}')
        running[ep] = (subprocess.Popen(['bash', '-c', cmd], start_new_session=True), time.time())
        F.log(dict(event='launch', slot='local:0', episode=ep, variation=rows[ep]['variation'], remote_dir=str(tmp), scheduler='local'))

    def finish(ep, rc, t0):
        work = a.output / f'ep{ep:03d}'
        tmp = work / 'render_local'
        frames = json.loads((work / 'source/source.json').read_text())['frames']
        ok = False
        if has_receipt(ep):  # a node delivered it first
            shutil.rmtree(tmp, ignore_errors=True)
            (work / 'local_claim').unlink(missing_ok=True)
            F.log(dict(event='discarded', slot='local:0', episode=ep, scheduler='local'))
            return
        if rc == 0:
            (tmp / 'current.png').unlink(missing_ok=True)
            if (work / 'render').exists():
                (work / 'render').rename(work / f'render_stale_{int(time.time())}')
            tmp.rename(work / 'render')
            shutil.copy(work / 'render_local_cfg/config.json', work / 'config.json')
            ok = F.verify(work, frames)
        F.log(dict(event='finished', slot='local:0', episode=ep, returncode=str(rc), accepted=ok, wall_s=round(time.time() - t0), adopted=False, scheduler='local'))
        if ok:
            (work / 'render_device.json').write_text(json.dumps(dict(host='local', gpu=0, device='OPTIX')) + '\n')
        else:
            attempts[ep] = attempts.get(ep, 0) + 1
            if (work / 'render').exists() and rc == 0:
                (work / 'render').rename(work / f'render_failed_{int(time.time())}')
        (work / 'local_claim').unlink(missing_ok=True)

    while True:
        for ep, (proc, t0) in list(running.items()):
            if has_receipt(ep) and isinstance(proc, subprocess.Popen) and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)  # a node delivered it first
            rc = proc.poll()
            if rc is not None:
                del running[ep]
                finish(ep, rc, t0)
        remaining = [ep for ep in range(len(rows)) if not has_receipt(ep)]
        if not remaining and not running:
            break
        try:
            remote = set(json.loads((a.output / 'remote_running.json').read_text()))
            steal = json.loads((a.output / 'steal_candidates.json').read_text())
        except (OSError, ValueError):
            remote, steal = set(), []
        ok_ep = lambda ep: ep not in running and attempts.get(ep, 0) < 3 and not has_receipt(ep) and (Path(rows[ep]['run']) / 'record.json').exists()
        pick = [ep for ep in reversed(remaining) if ep not in remote and ok_ep(ep)] + [ep for ep in steal if ok_ep(ep)]
        free_workers = a.workers - len(running)
        if free_workers > 0 and pick:
            load, free = sample(a.output)
            ok = load < 60 and free >= 8000
            F.log(dict(event='load_check', slot='local:0', foreign_sm_mean_pct=round(load, 1), free_mib=free, our_running_on_gpu=len(running), launch=ok))
            if ok:
                for ep in pick[:free_workers]:
                    try:
                        start(ep)
                    except subprocess.CalledProcessError as e:  # e.g. the scheduler exporting the same episode at this moment
                        attempts[ep] = attempts.get(ep, 0) + 1
                        F.log(dict(event='error', slot='local:0', episode=ep, scheduler='local', error=repr(e)[:400]))
                continue
        time.sleep(20)
    F.log(dict(event='end', scheduler='local'))


if __name__ == '__main__':
    main()
