"""FR3 v2 joint-limit checks for 30 Hz joint-target trajectories (two arms, 7 joints each).

Limits: franka_description robots/fr3v2_1/joint_limits.yaml (copied next to this file). Position-based velocity limit,
as libfranka applies it to FR3 (inferred from the yaml's velocity_offset / deceleration_limit, not checked on the robot):
    v_up(q) = min(v_max, max(0, -offset + sqrt(max(0, 2 * decel * (q_hi - q)))))
    v_lo(q) = max(-v_max, min(0, offset - sqrt(max(0, 2 * decel * (q - q_lo)))))
A frame's commanded velocity is (q[k] - q[k-1]) * 30 * speed; it must lie in [v_lo, v_up] at both ends of the frame.
The first-generation FR3 datasheet ranges are checked for reference only.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

FPS = 30
YAML = Path(__file__).with_name('fr3v2_1_joint_limits.yaml')
FR3_V1 = [(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007), (-3.0421, -0.1518), (-2.8065, 2.8065),
          (0.5445, 4.5169), (-3.0159, 3.0159)]
HANDS = ('left', 'right')


def load_limits(path=YAML):
    y = path.read_text()
    out = []
    for j in range(1, 8):
        b = y[y.index(f'joint{j}:'):]
        if j < 7:
            b = b[:b.index(f'joint{j + 1}:')]
        g = lambda k: float(re.search(k + r':\s*([-\d.eE]+)', b).group(1))  # noqa: E731
        out.append(dict(lo=g('lower'), hi=g('upper'), v=g('velocity'), off=g('velocity_offset'),
                        dec=g('deceleration_limit')))
    return out


LIM = load_limits()


def v_up(j, q, lim=LIM):
    p = lim[j]
    return min(p['v'], max(0.0, -p['off'] + math.sqrt(max(0.0, 2 * p['dec'] * (p['hi'] - q)))))


def v_lo(j, q, lim=LIM):
    p = lim[j]
    return max(-p['v'], min(0.0, p['off'] - math.sqrt(max(0.0, 2 * p['dec'] * (q - p['lo'])))))


def zero_zone(j, lim=LIM):
    """(q below which motion toward the lower limit is forbidden, q above which motion toward the upper is)."""
    p = lim[j]
    d = p['off'] ** 2 / (2 * p['dec'])
    return p['lo'] + d, p['hi'] - d


def read_csv(path):
    """-> (n, 14) joint targets [left 1..7, right 1..7] including row 0, column names."""
    rows = list(csv.reader(open(path)))
    h = rows[0]
    idx = [h.index(f'{s}_fr3v2_joint{j}') for s in HANDS for j in range(1, 8)]
    return np.array([[float(r[i]) for i in idx] for r in rows[1:]]), h


def from_episode(npz):
    """duo_run episode.npz -> (n+1, 14): measured start state, then the commanded targets."""
    z = np.load(npz)
    st, a = z['state'], z['action']
    start = np.r_[st[0, 0:7], st[0, 16:23]]
    return np.vstack([start, np.c_[a[:, 0:7], a[:, 8:15]]])


def check(Q, speeds=(1.0, 0.5, 0.3)):
    names = [f'{s}_J{j}' for s in HANDS for j in range(1, 8)]
    res = dict(frames=len(Q))
    pos = []
    v1 = []
    for c in range(14):
        j = c % 7
        p = LIM[j]
        m = float(min((Q[:, c] - p['lo']).min(), (p['hi'] - Q[:, c]).min()))
        pos.append((m, names[c]))
        lo1, hi1 = FR3_V1[j]
        ex = float(max((lo1 - Q[:, c]).max(), (Q[:, c] - hi1).max()))
        if ex > 0:
            v1.append((names[c], round(ex, 4)))
    pos.sort()
    res['min_pos_margin_rad'] = round(pos[0][0], 4)
    res['min_pos_margin_joint'] = pos[0][1]
    res['fr3_v1_excess_rad'] = v1
    dq = np.diff(Q, axis=0) * FPS
    res['max_speed_rad_s'] = round(float(np.abs(dq).max()), 3)
    for s in speeds:
        bad = {}
        first = None
        for c in range(14):
            j = c % 7
            for k in range(1, len(Q)):
                v = dq[k - 1, c] * s
                q0, q1 = Q[k - 1, c], Q[k, c]
                if v > min(v_up(j, q0), v_up(j, q1)) + 1e-9 or v < max(v_lo(j, q0), v_lo(j, q1)) - 1e-9:
                    bad[names[c]] = bad.get(names[c], 0) + 1
                    if first is None or k / FPS < first[1]:
                        first = (names[c], round(k / FPS, 3), round(float(q1), 4))
        res[f'pbv_violations_x{s}'] = bad
        if s == speeds[0]:
            res['pbv_first_violation'] = first
    return res


def rate_limit(Q, slack=0.005, scale=0.9):
    """Run the targets through FR3 v2's position-based velocity limits, sequentially from row 0.
    slack: treat the position limits as this much tighter (rad); scale: use this fraction of the allowed speed."""
    lim = [dict(p, lo=p['lo'] + slack, hi=p['hi'] - slack) for p in LIM]
    out = Q.copy()
    for c in range(14):
        j = c % 7
        q = out[0, c]
        for k in range(1, len(Q)):
            hi = q + scale * v_up(j, q, lim) / FPS
            lo = q + scale * v_lo(j, q, lim) / FPS
            t = min(max(Q[k, c], lo), hi)
            # the allowance shrinks as q nears a limit: also respect it at the end of the frame
            for _ in range(30):
                v = (t - q) * FPS
                if scale * v_lo(j, t, lim) - 1e-12 <= v <= scale * v_up(j, t, lim) + 1e-12:
                    break
                t = q + 0.5 * (t - q)
            out[k, c] = q = t
    return out
