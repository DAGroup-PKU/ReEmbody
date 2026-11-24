"""drawer_clearance.measure over a selection's runs (2026-11-07): per episode, the block hand's closest approach to the drawer
and to the cabinet, written to <out>.

    t1_retarget/bin/python drawer_clearance_all.py <selection.json> <out.json> [--workers 6] [--step 2]
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import drawer_clearance as DC


def one(job):
    e, step = job
    r = DC.measure(e['run'], step=step)
    left = {k.split('|')[1]: v for k, v in r.items() if k.startswith('left|')}
    drawer = left['bottom_drawer']
    others = min((v for k, v in left.items() if k != 'bottom_drawer'), key=lambda v: v[0])
    return dict(episode_index=e['episode_index'], variation=e['variation'], left_to_drawer_m=drawer[0], left_to_drawer=drawer[1:],
                left_to_cabinet_m=others[0], left_to_cabinet=others[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('selection')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--step', type=int, default=2)
    a = ap.parse_args()
    eps = json.loads(Path(a.selection).read_text())['episodes']
    with Pool(a.workers) as pool:
        rows = pool.map(one, [(e, a.step) for e in eps], chunksize=1)
    Path(a.out).write_text(json.dumps(dict(selection=a.selection, step=a.step, rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
