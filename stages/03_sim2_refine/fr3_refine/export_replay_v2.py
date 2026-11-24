"""Export v2 dataset episodes for replay on the real robot (user, 2026-11-06: "杯子任务给我对应case的轨迹我去replay").

Same files and checks as the v10 export (../../franka_fr3_duo_site_v10_20261105_attempt_01/code/export_all.py and
check_exits.py), for one LeRobot episode of a v2 task: duo_export_replay writes scene.xml, trajectory.json and
trajectory_30hz.csv from the episode's refine run; then, independently of the run's own record:
  * row 0 equals the user's FCI start pose (duo_site.USER_START_Q);
  * FR3 v2 position-based velocity limits at 1x and 0.5x on the CSV, and the margin to the first-generation FR3 ranges;
  * distances to the arch, cart and riser, and the arm gap (contact distances, as in v10), and the exact arch distance
    and arm gap (exact_clearance.measure_run);
  * the exit after each hand's last release (sideways, not back toward the robot).

    t1_retarget/bin/python export_replay_v2.py <task> <episode_index> <out dir>   (task: stack_cups | stack_blocks | drawer)
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'stages/02_sim1_to_sim2'))
from franka_dual_v1 import duo_export_replay as EXP  # noqa: E402
from franka_dual_v1 import duo_site as SITE  # noqa: E402
from franka_dual_v1 import fr3_limits as L  # noqa: E402
from franka_dual_v1.exact_clearance import measure_run  # noqa: E402

V2 = Path(__file__).resolve().parent.parent
V10_EXPORT = V2.parent / 'franka_fr3_duo_site_v10_20261105_attempt_01/code/export_all.py'
SELECTION = {'stack_cups': 'refine/stack_cups', 'stack_blocks': 'refine/stack_blocks', 'drawer': 'refine/drawer_fx10_cy-0.04'}
RELEASES = {'base_release', 'upper_release', 'insert_release', 'handle_release'}


def v10_checks():
    spec = importlib.util.spec_from_file_location('v10_export_all', V10_EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.support_distances, mod.arm_gap


def exits(run, layout):
    """check_exits.py for one run: per hand, its TCP motion in the mount frame from its last release to the end."""
    z = np.load(run / 'episode.npz')
    labels = [str(x) for x in z['labels']]
    rel = [i for i, l in enumerate(labels) if l in RELEASES]
    res = {}
    for k, hand in enumerate(('left', 'right')):
        m = np.array([SITE.world_to_mount(x, layout) for x in z['tcp'][:, 7 * k:7 * k + 3]])
        w = z['action'][:, 7 + 8 * k]
        opens = [i for i in rel if i > 0 and w[i] > w[i - 1] + 1e-4]
        if not opens:
            res[hand] = dict(note='no release by this hand')
            continue
        seg = m[opens[-1]:]
        side = 1.0 if hand == 'left' else -1.0
        res[hand] = dict(last_release_t_s=round(opens[-1] / 30, 2), back_toward_robot_m=round(float(seg[0, 0] - seg[:, 0].min()), 3),
                         out_to_side_m=round(float(side * (seg[-1, 1] - seg[0, 1])), 3),
                         up_m=round(float(seg[-1, 2] - seg[0, 2]), 3),
                         forward_at_release_m=round(float(seg[0, 0]), 3), forward_at_end_m=round(float(seg[-1, 0]), 3))
    return res


def main():
    task, episode, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    sel = json.loads((V2 / SELECTION[task] / 'selection.json').read_text())
    e = next(x for x in sel['episodes'] if x['episode_index'] == episode)
    run = Path(e['run'])
    meta = EXP.export(run, out)  # refuses an existing out dir
    Q, _ = L.read_csv(out / 'trajectory_30hz.csv')
    start = np.r_[SITE.USER_START_Q['left'], SITE.USER_START_Q['right']]
    lim = L.check(Q, speeds=(1.0, 0.5))
    v1 = min(float(min((Q[:, c] - L.FR3_V1[c % 7][0]).min(), (L.FR3_V1[c % 7][1] - Q[:, c]).min())) for c in range(14))
    support_distances, arm_gap = v10_checks()
    rec = json.loads((run / 'record.json').read_text())
    report = dict(task=task, lerobot_episode_index=episode, variation=e['variation'], run=str(run),
                  record_sha256=e['record_sha256'], episode_sha256=e['episode_sha256'],
                  candidate=rec.get('options', {}).get('candidate') if isinstance(rec.get('options'), dict) else None,
                  duration_s=meta['duration_s'], rows=meta['rows'],
                  start_row_max_abs_diff_rad=float(np.abs(Q[0] - start).max()),
                  pbv_violations_x1=lim['pbv_violations_x1.0'], pbv_violations_x0_5=lim['pbv_violations_x0.5'],
                  min_margin_fr3_v2_rad=lim['min_pos_margin_rad'], min_margin_fr3_v1_rad=round(v1, 4),
                  max_speed_rad_s=lim['max_speed_rad_s'],
                  support=support_distances(out / 'scene.xml', run / 'episode.npz'),
                  arm_gap=arm_gap(out / 'scene.xml', run / 'episode.npz'), exact=measure_run(run),
                  exits=exits(run, rec['variation'].get('layout') or SITE.NAME_V8),
                  objects=meta['objects'], stages=meta['stages'])
    (out / 'export_check.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    print(json.dumps({k: report[k] for k in ('variation', 'duration_s', 'rows', 'start_row_max_abs_diff_rad', 'pbv_violations_x1',
                                             'min_margin_fr3_v1_rad', 'max_speed_rad_s', 'exact', 'exits')}, default=float))


if __name__ == '__main__':
    main()
