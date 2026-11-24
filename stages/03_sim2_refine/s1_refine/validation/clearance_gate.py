"""Independent conservative clearance gate for v3 human-aligned native motion.

Mesh collision geometry is the closed convex hull of EVERY robot mesh, including
visual-only gripper bases. A positive hull distance lower-bounds surface distance
to the original mesh. Camera boxes are explicitly assumed envelopes, not measured
hardware. This module changes no dynamics, qpos, contacts, or old artifacts.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

if __package__:
    from ._shared.paths import FCL_DEPS
else:
    from _shared.paths import FCL_DEPS
DEPS = FCL_DEPS
sys.path.insert(0, str(DEPS))
import fcl

VERSION = 'rssr.v3_human_align.convex_clearance.v1'
REQUIRED_CLEARANCE_M = .020
REQUIRED_CATEGORIES = ('cross_arm', 'gripper_to_opposite_robot', 'wrist_camera_to_opposite_robot')
# Wide axis X, vertical image axis Y, optical axis -Z. Housing extends behind
# the optical plane, from local Z=0 to +40 mm. No calibrated dimensions exist.
CAMERA_BOX_SIZE_M = np.array([.090, .040, .040])
CAMERA_BOX_CENTER_LOCAL_M = np.array([0., 0., .020])


def _side(model, body):
    while body:
        name = model.body(body).name or ''
        if 'astribot_arm_left_' in name or 'astribot_gripper_left_' in name:
            return 'left'
        if 'astribot_arm_right_' in name or 'astribot_gripper_right_' in name:
            return 'right'
        body = int(model.body_parentid[body])
    return None


def _convex_mesh(vertices):
    hull = ConvexHull(vertices)
    # FCL requires outward-wound faces and compact vertex indices.
    used = np.unique(hull.simplices)
    remap = np.full(len(vertices), -1, dtype=int); remap[used] = np.arange(len(used))
    faces = hull.simplices.copy()
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    wrong = np.einsum('ij,ij->i', np.cross(b-a, c-a), hull.equations[:, :3]) < 0
    faces[wrong] = faces[wrong][:, [0, 2, 1]]
    packed = np.column_stack([np.full(len(faces), 3), remap[faces]]).ravel().tolist()
    return fcl.Convex(vertices[used], len(faces), packed)


class ClearanceGate:
    """Reusable FK distance evaluator for planning and native replay.

    ``check(data)`` expects already-forwarded *actual* MuJoCo state. For planning
    one may call ``at_qpos(qpos)`` on this object's private MjData. No simulation
    dynamics are changed. Distances beyond reporting_distance_m are certified
    by bounding spheres and need not invoke GJK.
    """
    def __init__(self, model, required_clearance_m=REQUIRED_CLEARANCE_M,
                 reporting_distance_m=.060):
        self.model = model
        self.data = mujoco.MjData(model)
        self.required = float(required_clearance_m)
        self.report_cap = float(reporting_distance_m)
        self.shapes = []
        self.request = fcl.DistanceRequest(enable_nearest_points=True)
        self._build_shapes()
        self.pairs = []
        for ai, a in enumerate(self.shapes):
            for bi in range(ai+1, len(self.shapes)):
                b = self.shapes[bi]
                category = None
                if a['side'] and b['side'] and a['side'] != b['side']:
                    if a['kind'] == 'camera' or b['kind'] == 'camera':
                        category = 'wrist_camera_to_opposite_robot'
                    elif a['kind'] == 'gripper' or b['kind'] == 'gripper':
                        category = 'gripper_to_opposite_robot'
                    else:
                        category = 'cross_arm'
                elif {a['kind'], b['kind']} == {'gripper', 'table'}:
                    category = 'gripper_to_table'
                elif {a['kind'], b['kind']} == {'camera', 'table'}:
                    category = 'wrist_camera_to_table'
                if category:
                    self.pairs.append((ai, bi, category))
        self.categories = sorted({p[2] for p in self.pairs})
        self._motion_lipschitz = self._build_motion_lipschitz()

    def _build_motion_lipschitz(self):
        """Global conservative point-motion bound for scalar joint paths.

        For each ancestor hinge, radius to any shape point is upper bounded
        by the sum of rigid body offsets, joint-pivot offsets, slide ranges,
        and shape radius. Triangle inequality deliberately overestimates.
        This is a bound for linear scalar-q interpolation, not a claim about
        hardware tracking between samples.
        """
        m = self.model
        out = np.zeros((len(self.shapes), m.nq))
        for si, s in enumerate(self.shapes):
            body = s['body_id']; chain = []
            while body:
                chain.append(body); body = int(m.body_parentid[body])
            if 'geom_id' in s:
                local_radius = np.linalg.norm(m.geom_pos[s['geom_id']]) + s['radius']
            else:
                local_radius = np.linalg.norm(m.cam_pos[s['camera_id']]) + np.linalg.norm(CAMERA_BOX_CENTER_LOCAL_M) + s['radius']
            for depth, ancestor in enumerate(chain):
                # Bodies below the joint body; its translation to parent does
                # not contribute to that joint's rotational lever arm.
                lever = float(local_radius + sum(np.linalg.norm(m.body_pos[b]) for b in chain[:depth]))
                for b in chain[:depth+1]:
                    for j in range(int(m.body_jntadr[b]), int(m.body_jntadr[b]+m.body_jntnum[b])):
                        lever += 2*float(np.linalg.norm(m.jnt_pos[j]))
                        if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_SLIDE):
                            if not m.jnt_limited[j]:
                                raise ValueError('Unbounded critical slide requires explicit validated range')
                            lever += float(np.max(abs(m.jnt_range[j])))
                for j in range(int(m.body_jntadr[ancestor]), int(m.body_jntadr[ancestor]+m.body_jntnum[ancestor])):
                    typ = int(m.jnt_type[j]); qi = int(m.jnt_qposadr[j])
                    if typ == int(mujoco.mjtJoint.mjJNT_HINGE):
                        out[si, qi] = lever
                    elif typ == int(mujoco.mjtJoint.mjJNT_SLIDE):
                        out[si, qi] = 1.
                    else:
                        raise ValueError('Critical free/ball joints require a separate rotation bound')
        return out

    def interval_motion_margin(self, previous_qpos, current_qpos):
        """Worst half-interval distance decrease for each checked category.

        Any scalar-linear interpolated q lies within half the total joint
        variation from one of the endpoints. Sum of body displacement bounds
        gives a distance Lipschitz bound. Subtract from the smaller endpoint
        minimum to certify all of this *defined interpolated* segment.
        """
        displacement = self._motion_lipschitz @ np.abs(np.asarray(current_qpos)-np.asarray(previous_qpos))
        margins = {c:0. for c in self.categories}
        for a,b,c in self.pairs:
            margins[c] = max(margins[c], .5*float(displacement[a]+displacement[b]))
        return margins

    def _build_shapes(self):
        m = self.model
        seen = set()
        for gi in range(m.ngeom):
            body = int(m.geom_bodyid[gi]); side = _side(m, body)
            name = m.geom(gi).name or f'unnamed_geom_{gi}'
            bname = m.body(body).name or ''
            table = 'worktable' in bname or name.startswith('worktable_')
            if not side and not table:
                continue
            typ = int(m.geom_type[gi]); size = m.geom_size[gi].copy()
            if typ == int(mujoco.mjtGeom.mjGEOM_MESH):
                mid = int(m.geom_dataid[gi])
                v = m.mesh_vert[int(m.mesh_vertadr[mid]):int(m.mesh_vertadr[mid]+m.mesh_vertnum[mid])].astype(float)
                # Vertex position/order suffices for a hull; faces are immaterial.
                geom_key = ('hull', hashlib.sha256(v.tobytes()).hexdigest())
                make = lambda v=v: _convex_mesh(v)
                radius = float(np.max(np.linalg.norm(v, axis=1)))
            elif typ == int(mujoco.mjtGeom.mjGEOM_BOX):
                geom_key = ('box', *size); make = lambda s=size: fcl.Box(*(2*s)); radius = float(np.linalg.norm(size))
            elif typ == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                geom_key = ('sphere', size[0]); make = lambda s=size: fcl.Sphere(float(s[0])); radius = float(size[0])
            elif typ == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                geom_key = ('cylinder', *size[:2]); make = lambda s=size: fcl.Cylinder(float(s[0]), float(2*s[1])); radius = float(np.linalg.norm(size[:2]))
            elif typ == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                geom_key = ('capsule', *size[:2]); make = lambda s=size: fcl.Capsule(float(s[0]), float(2*s[1])); radius = float(size[0]+size[1])
            else:
                raise ValueError(f'Unsupported critical geometry {name}: {typ}')
            key = (body, *m.geom_pos[gi], *m.geom_quat[gi], geom_key)
            if key in seen:
                continue
            seen.add(key)
            kind = 'table' if table else ('gripper' if 'astribot_gripper_' in bname else 'arm')
            self.shapes.append(dict(name=name, body=bname, body_id=body, side=side,
                kind=kind, geom_id=gi, radius=radius, object=fcl.CollisionObject(make()),
                collision_enabled=bool(m.geom_contype[gi] or m.geom_conaffinity[gi])))
        for side in ['left', 'right']:
            ci = m.camera(f'{side}_wrist_rgbd').id
            self.shapes.append(dict(name=f'{side}_wrist_camera_assumed_90x40x40mm',
                body=m.body(int(m.cam_bodyid[ci])).name, body_id=int(m.cam_bodyid[ci]),
                side=side, kind='camera', camera_id=ci, radius=float(np.linalg.norm(CAMERA_BOX_SIZE_M/2)),
                object=fcl.CollisionObject(fcl.Box(*CAMERA_BOX_SIZE_M)), collision_enabled=False))

    def _update(self, data):
        centers = []
        for s in self.shapes:
            if 'geom_id' in s:
                gi = s['geom_id']; R = data.geom_xmat[gi].reshape(3,3); pos = data.geom_xpos[gi]
            else:
                ci = s['camera_id']; R = data.cam_xmat[ci].reshape(3,3)
                pos = data.cam_xpos[ci] + R @ CAMERA_BOX_CENTER_LOCAL_M
            centers.append(np.asarray(pos).copy())
            s['object'].setTransform(fcl.Transform(R, pos))
        return np.asarray(centers)

    @staticmethod
    def describe(s):
        return {k:s[k] for k in ['name','body','side','kind','collision_enabled']}

    def check(self, data, detail=True):
        centers = self._update(data)
        minima = {category:dict(distance_m=self.report_cap, exact=False) for category in self.categories}
        # The positive bounding-sphere lower bound is safe for skipping distant pairs.
        for ai, bi, category in self.pairs:
            a, b = self.shapes[ai], self.shapes[bi]
            lower = float(np.linalg.norm(centers[ai]-centers[bi])-a['radius']-b['radius'])
            if lower >= minima[category]['distance_m']:
                continue
            result = fcl.DistanceResult()
            distance = float(fcl.distance(a['object'], b['object'], self.request, result))
            if not np.isfinite(distance):
                raise RuntimeError(f'Nonfinite FCL distance for {a["name"]} and {b["name"]}')
            if distance < minima[category]['distance_m']:
                row = dict(distance_m=distance, exact=True)
                if detail:
                    row.update(first=self.describe(a), second=self.describe(b),
                        nearest_points_m=np.asarray(result.nearest_points).tolist())
                minima[category] = row
        return dict(passed=all(minima[k]['distance_m'] >= self.required for k in REQUIRED_CATEGORIES),
                    required_clearance_m=self.required, required_categories=list(REQUIRED_CATEGORIES), minima=minima)

    def at_qpos(self, qpos, detail=True):
        self.data.qpos[:] = qpos
        # Private FK only: never rebuild contacts/forces in live solver data.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        return self.check(self.data, detail=detail)

    def geometry_manifest(self):
        return dict(schema=VERSION, required_clearance_m=self.required,
            required_categories=list(REQUIRED_CATEGORIES),
            table_clearance_policy='Diagnostic only. User requires 20mm between opposing sides; original native robot/table nonpenetration gate remains mandatory.',
            camera_envelope=dict(size_m=CAMERA_BOX_SIZE_M.tolist(),
                center_in_camera_local_m=CAMERA_BOX_CENTER_LOCAL_M.tolist(),
                calibration='assumed, not measured; optical forward is local -Z'),
            shapes=[self.describe(s) for s in self.shapes], categories=self.categories,
            exclusions=[
                'Task cups are not obstacles in this 20mm gate; cup/table and cup/finger contacts remain subject to original physical task/force checks.',
                'Same-side attached arm/gripper/camera parts are not distance checked; this gate targets opposing sides and gripper/camera to table.',
                'No robot mesh is exempted because its collision bits are zero; visual-only gripper base is included.',
            ],
            mesh_method='Closed convex hulls contain original mesh surfaces; positive hull separation is conservative. Nonconvex gaps may be rejected.',
            interval_motion_bound='For scalar-linear interpolation of consecutive native qpos: sum ancestor hinge angle variation times global triangle-inequality lever radius plus slide variation; half interval bound is subtracted from smaller endpoint category minimum. No hardware path/tracking claim.',
            distance_reporting_cap_m=self.report_cap)


def audit_rollout(native, output, min_clearance=.020):
    native = Path(native); output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    m = mujoco.MjModel.from_xml_path(str(native/'execute/scene.xml'))
    gate = ClearanceGate(m, min_clearance)
    z = np.load(native/'execute/rollout.npz')
    minima = {}; failed = []; start = time.monotonic()
    for ix, t in enumerate(z['time_s']):
        r = gate.at_qpos(z['qpos'][ix])
        for category, row in r['minima'].items():
            if category not in minima or row['distance_m'] < minima[category]['distance_m']:
                minima[category] = dict(row, time_s=float(t), sample_index=ix)
        if not r['passed']:
            failed.append(dict(time_s=float(t), sample_index=ix,
                failed_categories=[k for k in REQUIRED_CATEGORIES if r['minima'][k]['distance_m'] < min_clearance]))
    result = dict(schema=VERSION, native_source=str(native), manifest=gate.geometry_manifest(),
        passed=not failed, samples=len(z['time_s']), max_sample_gap_s=float(np.max(np.diff(z['time_s']))),
        sampled_clock='recorded actual native qpos; FK reconstruction for geometry only',
        continuous_clearance_certified=False,
        limitation='100Hz coarse screen only. An accepted candidate requires independent every-native-step replay and conservative between-step motion margin.',
        minima=minima, failed_sample_count=len(failed), failed_samples=failed,
        wall_seconds=time.monotonic()-start)
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--native', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--min-clearance', type=float, default=.020)
    a = p.parse_args(); r = audit_rollout(a.native, a.output, a.min_clearance)
    print(json.dumps({k:r[k] for k in ['passed','samples','minima','wall_seconds']}, indent=2), flush=True)
