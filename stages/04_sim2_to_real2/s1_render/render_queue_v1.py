"""Render admitted centred-layout native cases (CPU Cycles, 12 samples) in parallel.

Picks admitted cases from the native batch in case order, up to --limit, and
renders each into <output>/cases/<case_id>/attempt_NN with render_case_v1.
Already-rendered cases are skipped; failed renders keep their attempt dir and
are retried once in a fresh attempt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_case_v1 as RC  # noqa: E402

LOCK = threading.Lock()


def admitted(batch: Path):
    out = {}
    for p in sorted((batch / 'cases').glob('*/attempt_*/batch_result.json')):
        r = json.loads(p.read_text())
        if r.get('accepted'):
            out[r['case_id']] = Path(r['native'])
    return out


def render_one(case_id, native, root):
    case_dir = root / 'cases' / case_id
    for n in range(1, 4):
        att = case_dir / f'attempt_{n:02d}'
        if (att / 'render_case_result.json').exists():
            return json.loads((att / 'render_case_result.json').read_text())
        if att.exists():
            continue
        try:
            return RC.render_case(native, att, device='CPU')
        except Exception as e:  # keep the failed attempt, retry fresh once
            (case_dir / f'attempt_{n:02d}_error.txt').write_text(repr(e))
            if n >= 2:
                raise
    raise RuntimeError('no attempt slot')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--wait', action='store_true', help='keep polling the batch for newly admitted cases')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    done, started = {}, set()

    def job(cid, native):
        try:
            r = render_one(cid, native, a.output)
            ok = True
        except Exception as e:
            r, ok = dict(error=repr(e)), False
        with LOCK:
            done[cid] = dict(ok=ok, **r)
            (a.output / 'status.json').write_text(json.dumps(dict(
                rendered=sorted(k for k, v in done.items() if v['ok']),
                failed={k: v.get('error') for k, v in done.items() if not v['ok']},
                updated=time.time()), indent=1))
            print(cid, ok, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        while True:
            adm = admitted(a.batch)
            for cid in sorted(adm)[:a.limit]:
                if cid not in started:
                    started.add(cid)
                    ex.submit(job, cid, adm[cid])
            if not a.wait or len(started) >= a.limit:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
