"""Export an admitted FR3 Duo episode (duo_run record.json + episode.npz + scene.xml) for replay on the real robot.

trajectory_30hz.csv: row 0 is the measured start state (t = 0); row k >= 1 is the 16-value action commanded for frame
k, the target the servos reach at t = k / 30 s (in simulation the target is interpolated linearly at 1200 Hz within
the frame). Columns: time_s, stage (the controller command that produced the row), left/right_fr3v2_joint1..7 (rad,
joint names as franka_description fr3_duo), left/right_gripper_width_m (pad-to-pad target; 0 = close until the force
limit stops it), left/right_robotiq_pos (approximate Robotiq 2F-85 position request 0 open .. 255 closed, linear in
width, for a real gripper driver).
trajectory.json: the start pose, every object's initial pose (world and mount frame; the simulated scene settles
objects before t = 0, so free objects are given as measured at t = 0), stage times, safety numbers and the contract
result from the run's measurements.

    python -m franka_dual_v1.duo_export_replay --run <runs/<variation>> --out <fresh dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import duo_scene as DS
from . import fr3_duo as F
from .fr3_duo_calib import mount_frame

FPS = 30
OPEN_M = 0.085


def robotiq_pos(w):
    return int(round(float(np.clip((OPEN_M - w) / OPEN_M, 0, 1)) * 255))


def export(run: Path, out: Path):
    rec = json.loads((run / 'record.json').read_text())
    if not rec.get('admitted'):
        raise SystemExit(f'{run} was not admitted')
    z = np.load(run / 'episode.npz')
    out.mkdir(parents=True, exist_ok=False)
    v = DS.Variation.from_json(rec['variation'])
    scene = DS.apply(DS.raw_scene(v.source_case), v)
    if v.layout is not None:  # the layout's objects (e.g. the user's 8 x 9 cm cups), as duo_scene.build uses them
        from . import duo_site
        scene = duo_site.site_scene(scene, v.layout)
    st, act, labels = z['state'], z['action'], z['labels']
    cols = (['time_s', 'stage'] + [f'{h}_fr3v2_joint{i}' for h in F.HANDS for i in range(1, 8)] +
            [f'{h}_gripper_width_m' for h in F.HANDS] + [f'{h}_robotiq_pos' for h in F.HANDS])
    start = {h: st[0, 16 * k:16 * k + 7] for k, h in enumerate(F.HANDS)}
    w0 = [OPEN_M, OPEN_M]  # both grippers start fully open
    with (out / 'trajectory_30hz.csv').open('w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        wr.writerow(['0.0000', 'start'] + [f'{x:.6f}' for h in F.HANDS for x in start[h]] +
                    [f'{w:.4f}' for w in w0] + [robotiq_pos(w) for w in w0])
        for k, a in enumerate(act):
            widths = [float(a[7]), float(a[15])]
            wr.writerow([f'{(k + 1) / FPS:.4f}', labels[k]] + [f'{x:.6f}' for x in np.r_[a[0:7], a[8:15]]] +
                        [f'{w:.4f}' for w in widths] + [robotiq_pos(w) for w in widths])
    free = [str(b) for b in z['free_bodies']]
    objs = []
    for o in scene['objects']:
        if o['mobility'] == 'free' and o['id'] in free:
            p = z['objects'][0, free.index(o['id'])]
            pos, quat = p[:3], p[3:7]
            yaw = float(Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')[2])
        else:
            pos, yaw = np.asarray(o['position_m'], float), float(o.get('yaw_rad', 0.0))
        if o['mobility'] == 'static' and o['shape'] not in ('cabinet', 'region'):
            continue
        objs.append(dict(id=o['id'], shape=o['shape'], size_m=o['size_m'], mobility=o['mobility'],
                         world=np.round(pos, 4).tolist(), mount_frame=mount_frame(DS.place_of(v), pos),
                         from_table_edge_m=round(F.TABLE_EDGE_Y - float(pos[1]), 4), from_midline_m=round(float(pos[0]), 4),
                         yaw_rad=round(yaw, 4)))
    m = rec['measurements']
    place = DS.place_of(v)
    p0, _ = place.mount_pose()
    meta = dict(
        schema='rssr.fr3_duo.replay.v1', task=v.task, variation=v.to_json(), controller=rec['controller'],
        fps=FPS, rows=len(act) + 1, duration_s=round(len(act) / FPS, 3), columns=cols,
        start_joints={h: np.round(start[h], 6).tolist() for h in F.HANDS},
        objects=objs, stages=m['stages'],
        placement=dict(mounting_point_world=np.round(p0, 4).tolist(), dist_behind_table_edge_m=place.dist_behind_edge,
                       height_above_table_m=place.height_above_table, table_top_m=F.TABLE_TOP,
                       base_height_above_floor_m=round(F.TABLE_TOP + place.height_above_table, 4)),
        safety=dict(min_arm_clearance_m=m['min_arm_clearance_m'], min_arm_clearance_t=m['min_arm_clearance_t'],
                    min_joint_limit_margin_rad=m['min_joint_limit_margin_rad'],
                    max_joint_speed_rad_s=m['max_joint_speed_rad_s'], max_tracking_err_rad=m['max_tracking_err_rad'],
                    **{k: m[k] for k in ('min_support_clearance_m', 'min_support_clearance_t', 'min_support_clearance_pair',
                                         'fr3_limits') if k in m}),
        contract=dict(task_success=m['contract']['task_success'],
                      predicates={k: p['passed'] for k, p in m['contract']['predicates'].items()}),
        grasps=m['grasps'], replay_bit_exact=m['replay']['bit_exact'])
    (out / 'trajectory.json').write_text(json.dumps(meta, indent=1, default=float) + '\n')
    shutil.copy2(run / 'scene.xml', out / 'scene.xml')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    meta = export(a.run, a.out)
    print(json.dumps({k: meta[k] for k in ('task', 'duration_s', 'safety')}, default=float))


if __name__ == '__main__':
    main()
