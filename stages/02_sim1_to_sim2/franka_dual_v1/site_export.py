"""Export an admitted FR3 Duo site episode for the Blender site renderer (sim2-real2/franka_fr3_duo_site_v1).

Input: a duo_run run directory (record.json, episode.npz, scene.xml) with a site layout (fr3_site_20261103 / _20261104). Output (fresh dir):
  * geometry.json: visual geometry once (export_render.visual_geoms: robot visual meshes, mount, cover, desk, objects);
  * poses.npz: for every 30 Hz frame of the episode (the complete clock, terminal state included) the world pose of
    every exported geom and camera, and of every free object body;
  * cameras.json: head (D455) and wrists (D405) with vertical FOV and the 4:3 stream size of the site;
  * source.json: episode identity, frame count, free objects (shape, size, the body-frame bottom centre of a cup),
    sha256 of every file above and of the inputs.

    python -m franka_dual_v1.site_export --run <runs/<variation>> --out <fresh dir>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from . import duo_scene as DS
from . import duo_site as SITE
from .export_render import geom_record, visual_geoms

CAMS = ('head', 'left_wrist', 'right_wrist')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def export(run: Path, out: Path):
    rec = json.loads((run / 'record.json').read_text())
    if not rec.get('admitted'):
        raise SystemExit(f'{run} was not admitted')
    v = DS.Variation.from_json(rec['variation'])
    if v.layout not in SITE.LAYOUTS:
        raise SystemExit(f'{run}: layout {v.layout}, expected one of {SITE.LAYOUTS}')
    z = np.load(run / 'episode.npz')
    qpos = z['qpos']
    out.mkdir(parents=True, exist_ok=False)
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    d = mujoco.MjData(m)
    geoms = visual_geoms(m)
    scene = {o['id']: o for o in SITE.site_scene(DS.apply(DS.raw_scene(v.source_case), v), v.layout)['objects']}
    free = [str(b) for b in z['free_bodies']]
    bids = [m.body(b).id for b in free]
    cids = [m.camera(c).id for c in CAMS]
    n = len(qpos)
    gpos, gquat = np.zeros((n, len(geoms), 3)), np.zeros((n, len(geoms), 4))
    bpos, bquat = np.zeros((n, len(bids), 3)), np.zeros((n, len(bids), 4))
    cpos, cquat = np.zeros((n, len(cids), 3)), np.zeros((n, len(cids), 4))
    for k in range(n):
        d.qpos[:] = qpos[k]
        mujoco.mj_forward(m, d)
        gpos[k] = d.geom_xpos[geoms]
        for j, g in enumerate(geoms):
            mujoco.mju_mat2Quat(gquat[k, j], d.geom_xmat[g])
        bpos[k], bquat[k] = d.xpos[bids], d.xquat[bids]
        cpos[k] = d.cam_xpos[cids]
        for j, c in enumerate(cids):
            mujoco.mju_mat2Quat(cquat[k, j], d.cam_xmat[c])
    objects = []
    for b, bid in zip(free, bids):
        o = scene[b]
        row = dict(id=b, shape=o['shape'], size_m=o['size_m'],
                   geoms=[g for g in geoms if m.geom_bodyid[g] == bid])
        if o['shape'] == 'cup':
            # bottom centre of the cup in its body frame: lowest point of its geoms at rest
            lows = [m.geom_pos[g][2] - (m.geom_size[g][1] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
                                         else m.geom_size[g][2]) for g in range(m.ngeom) if m.geom_bodyid[g] == bid
                    and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]  # tapered walls (meshes) stand on the bottom disc
            row['bottom_in_body_m'] = [0.0, 0.0, float(min(lows))]
        objects.append(row)
    (out / 'geometry.json').write_text(json.dumps([geom_record(m, g) for g in geoms]))
    np.savez_compressed(out / 'poses.npz', geom_pos=gpos, geom_quat=gquat, body_pos=bpos, body_quat=bquat,
                        cam_pos=cpos, cam_quat=cquat)
    (out / 'cameras.json').write_text(json.dumps(dict(
        names=list(CAMS), fovy_deg=[float(m.cam_fovy[c]) for c in cids],
        stream_size=[SITE.VIEW_W, SITE.VIEW_H], devices=['RealSense D455', 'RealSense D405', 'RealSense D405'])))
    meta = dict(schema='rssr.fr3_duo.site_source.v1', layout=v.layout, variation=v.id, source_case=v.source_case,
                task=v.task, controller=rec['controller'], frames=n, fps=30, duration_s=round((n - 1) / 30, 4),
                objects=objects, labels=[str(x) for x in z['labels']],
                inputs={str(run / f): sha(run / f) for f in ('record.json', 'episode.npz', 'scene.xml')},
                admitted=True, replay_bit_exact=rec['measurements']['replay']['bit_exact'])
    meta['artifacts'] = {f: sha(out / f) for f in ('geometry.json', 'poses.npz', 'cameras.json')}
    (out / 'source.json').write_text(json.dumps(meta, indent=1) + '\n')
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    meta = export(a.run, a.out)
    print(json.dumps({k: meta[k] for k in ('variation', 'frames', 'duration_s')}))


if __name__ == '__main__':
    main()
