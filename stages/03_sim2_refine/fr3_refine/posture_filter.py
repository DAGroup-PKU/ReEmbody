"""Which refined episodes leave a hand's main posture (2026-11-07): from eval_grasps.py's per-episode joints at the grasp, per hand, the
largest single-linkage cluster (a gap of more than 0.5 rad on some joint separates clusters, as in the audit); an episode outside it
for either hand is excluded from the selection, so the released data has one posture per hand.

    python3 posture_filter.py <eval json> <out json>
"""
import json
import sys
from pathlib import Path

import numpy as np


def main_cluster(q, gap=0.5):
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
    roots = [find(i) for i in range(n)]
    best = max(set(roots), key=roots.count)
    return [r == best for r in roots]


def main():
    e = json.loads(Path(sys.argv[1]).read_text())
    excluded = {}
    for hand in ('left', 'right'):
        rows = [(ep['run'], g['joints_at_close']) for ep in e['episodes'] for g in ep['grasps'] if g.get('hand') == hand]
        keep = main_cluster(np.array([q for _, q in rows]))
        for (run, _), k in zip(rows, keep):
            if not k:
                excluded.setdefault(Path(run).name, []).append(hand)
    out = dict(rule='joints at the grasp outside the hand\'s largest cluster (0.5 rad single linkage)', eval=sys.argv[1], excluded=sorted(excluded), hands=excluded)
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1) + '\n')
    print(len(excluded), 'excluded', excluded)


if __name__ == '__main__':
    main()
