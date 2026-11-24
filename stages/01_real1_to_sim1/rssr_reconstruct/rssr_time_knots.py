"""Bounded time-local goal edits on the immutable source action clock."""
import copy
import numpy as np


def validate(value):
    if not isinstance(value, dict) or set(value)-{'left', 'right'}:
        raise ValueError('Time knots must name left/right hands')
    out = copy.deepcopy(value)
    for rows in out.values():
        if not isinstance(rows, list) or len(rows) > 12:
            raise ValueError('Use zero to twelve time knots per hand; [] clears that hand')
        last = -1.
        for row in rows:
            if (not isinstance(row, dict) or set(row)-{'time_s', 'offset_m', 'rotation_degrees'} or
                    'time_s' not in row or not ({'offset_m', 'rotation_degrees'} & set(row))):
                raise ValueError('Each knot requires time_s and at least one correction vector')
            t = row['time_s']
            if type(t) not in (int, float) or not np.isfinite(t) or t < 0 or t <= last:
                raise ValueError('Knot source times must be finite, nonnegative and strictly increasing')
            last = t
            for key, bound in [('offset_m', .12), ('rotation_degrees', 35.)]:
                row.setdefault(key, [0., 0., 0.])
                vector = row[key]
                if (not isinstance(vector, list) or len(vector) != 3 or
                        any(type(v) not in (int, float) for v in vector) or
                        not np.isfinite(vector).all() or np.max(np.abs(vector)) > bound):
                    raise ValueError('Time-knot '+key+' exceeds its fixed per-axis bound')
    return out


def corrections(times, rows):
    times = np.asarray(times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or
            times[0] != 0 or np.any(np.diff(times) <= 0)):
        raise ValueError('Time knots require the unchanged increasing source clock starting at zero')
    rows = validate({'left': rows})['left']
    duration = float(times[-1])
    interior = []
    for row in rows:
        t = row['time_s']
        if t > duration:
            raise ValueError('Time knot outside source episode')
        if t in (0., duration):
            if any(row['offset_m']) or any(row['rotation_degrees']):
                raise ValueError('Episode boundary knot corrections must be zero')
        else:
            interior.append(row)
    kt = np.array([0., *[r['time_s'] for r in interior], duration])
    index = np.clip(np.searchsorted(kt, times, side='right')-1, 0, len(kt)-2)
    u = (times-kt[index])/(kt[index+1]-kt[index])
    weight = (u*u*(3.-2.*u))[:, None]
    result = []
    for key in ('offset_m', 'rotation_degrees'):
        values = np.array([[0., 0., 0.], *[r[key] for r in interior], [0., 0., 0.]])
        result.append((1.-weight)*values[index]+weight*values[index+1])
    return tuple(result)
