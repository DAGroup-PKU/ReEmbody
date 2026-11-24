"""export_replay_v2.py for one run directory that is not in a selection (a pilot), with the block hand's clearance to the cabinet added.

    t1_retarget/bin/python export_replay_run.py <run dir> <out dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

import drawer_clearance as DC
import export_replay_v2 as X


def main():
    run, out = Path(sys.argv[1]), Path(sys.argv[2])
    meta = X.EXP.export(run, out)
    Q, _ = X.L.read_csv(out / 'trajectory_30hz.csv')
    start = np.r_[X.SITE.USER_START_Q['left'], X.SITE.USER_START_Q['right']]
    lim = X.L.check(Q, speeds=(1.0, 0.5))
    v1 = min(float(min((Q[:, c] - X.L.FR3_V1[c % 7][0]).min(), (X.L.FR3_V1[c % 7][1] - Q[:, c]).min())) for c in range(14))
    support_distances, arm_gap = X.v10_checks()
    rec = json.loads((run / 'record.json').read_text())
    report = dict(task='drawer', variation=rec['variation'].get('id'), run=str(run), controller=rec.get('controller'), options=rec.get('options'), duration_s=meta['duration_s'], rows=meta['rows'],
                  start_row_max_abs_diff_rad=float(np.abs(Q[0] - start).max()), pbv_violations_x1=lim['pbv_violations_x1.0'], pbv_violations_x0_5=lim['pbv_violations_x0.5'],
                  min_margin_fr3_v2_rad=lim['min_pos_margin_rad'], min_margin_fr3_v1_rad=round(v1, 4), max_speed_rad_s=lim['max_speed_rad_s'],
                  support=support_distances(out / 'scene.xml', run / 'episode.npz'), arm_gap=arm_gap(out / 'scene.xml', run / 'episode.npz'), exact=X.measure_run(run),
                  cabinet_clearance=DC.measure(run), task_success=rec['measurements']['contract']['task_success'],
                  exits=X.exits(run, rec['variation'].get('layout') or X.SITE.NAME_V8), objects=meta['objects'], stages=meta['stages'])
    (out / 'export_check.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps({k: report[k] for k in ('variation', 'duration_s', 'rows', 'start_row_max_abs_diff_rad', 'pbv_violations_x1', 'pbv_violations_x0_5', 'min_margin_fr3_v1_rad', 'max_speed_rad_s', 'exact', 'task_success', 'exits')}, default=float))
    print('OBJECTS', json.dumps([(o['id'], o['mount_frame'][:2], o['yaw_rad']) for o in report['objects']]))
    print('LEFT', {k: v for k, v in report['cabinet_clearance'].items() if k.startswith('left') and v[0] < 0.08})


if __name__ == '__main__':
    main()
