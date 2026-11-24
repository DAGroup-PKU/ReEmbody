"""Read-only, sampled drawer rigid-shell/environment separation admission.

Explicit mj_geomDistance queries include visual geoms with both contact masks 0.
For meshes MuJoCo queries convex hulls: positive distance proves separation of
that mesh, while nonpositive distance is conservative rejection, NOT an exact
triangle penetration depth. Raw zero queries require independent positive AABB
or convex-vs-box projection separation certificates; otherwise they reject.
No dynamics, physics, old labels or task gates change.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

SCHEMA = 'rssr.drawer.shell_environment.sampled_convex_separation.v2'
REQUIRED_MUJOCO = '3.3.7'
AABB_INFLATION_M = 1e-9
DISTANCE_QUERY_CAP_M = 1000.0
SAT_PROJECTION_GUARD_M = 1e-8
AXIS_MIN_NORM = 1e-10


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def xml_dependencies(scene):
    """Pin XML includes and on-disk mesh/texture assets, without changing them."""
    paths = set()
    pending = [scene.resolve(strict=True)]
    while pending:
        xml = pending.pop()
        if xml in paths:
            continue
        paths.add(xml)
        root = ET.parse(xml).getroot()
        compiler = root.find('compiler')
        options = {} if compiler is None else compiler.attrib
        for node in root.iter():
            source = node.get('file')
            if not source:
                continue
            path = Path(source)
            if not path.is_absolute():
                subdir = options.get('meshdir', '') if node.tag == 'mesh' else (
                    options.get('texturedir', '') if node.tag == 'texture' else '')
                path = xml.parent / subdir / path
            path = path.resolve(strict=True)
            if node.tag == 'include':
                pending.append(path)
            else:
                paths.add(path)
    return sorted(paths)


def _body_chain_names(model, body):
    result = []
    while body:
        result.append(model.body(body).name or '')
        body = int(model.body_parentid[body])
    return result


def geom_record(model, gid, category, side=None):
    mesh = int(model.geom_dataid[gid]) if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
    body = int(model.geom_bodyid[gid])
    name = model.geom(gid).name or ''
    body_name = model.body(body).name or ''
    # A stable explanatory alias for unnamed URDF visual geometry; no renaming.
    alias = name or ('visual_' + body_name)
    return dict(geom_id=int(gid), name=name, alias=alias, body_id=body,
                body_name=body_name, category=category, side=side,
                type='mesh' if mesh >= 0 else 'box', mesh_id=mesh,
                mesh_name=(model.mesh(mesh).name or '') if mesh >= 0 else None,
                contype=int(model.geom_contype[gid]),
                conaffinity=int(model.geom_conaffinity[gid]),
                rgba=model.geom_rgba[gid].astype(float).tolist(),
                group=int(model.geom_group[gid]),
                mesh_query_representation='convex_hull' if mesh >= 0 else 'exact_box')


def select_geometry(model):
    shells, environment, excluded_fingers, excluded_invisible = [], [], [], []
    for gid in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[gid])).name or ''
        mesh = int(model.geom_dataid[gid]) if int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_MESH) else -1
        mesh_name = (model.mesh(mesh).name or '') if mesh >= 0 else ''
        text = ' '.join([body_name, model.geom(gid).name or '', mesh_name])
        side = next((s for s in ('left', 'right') if re.search(r'astribot_(?:arm|gripper)_' + s + r'_', text)), None)
        arm = side is not None and mesh >= 0 and (
            re.fullmatch(r'astribot_arm_(?:left|right)_link_\d+', body_name) is not None
            or re.search(r'astribot_arm_(?:left|right)_base_link', mesh_name) is not None)
        base = re.fullmatch(r'astribot_gripper_(?:left|right)_(?:base|body)', body_name) is not None
        finger = 'astribot_gripper_' in body_name and not base
        if finger:
            excluded_fingers.append(dict(geom_id=gid, name=model.geom(gid).name or '', body_name=body_name,
                                         reason='intentional finger/contact geometry is outside this shell-only gate'))
        if arm or base:
            if float(model.geom_rgba[gid, 3]) <= 0:
                excluded_invisible.append(dict(geom_id=gid, name=model.geom(gid).name or '', body_name=body_name,
                                               reason='invisible physical duplicate; original native hard gates retained separately'))
            else:
                if int(model.geom_type[gid]) not in (int(mujoco.mjtGeom.mjGEOM_MESH), int(mujoco.mjtGeom.mjGEOM_BOX)):
                    raise ValueError(f'Unsupported selected rigid shell geometry {gid}')
                shells.append(geom_record(model, gid, 'arm_shell' if arm else 'gripper_base_body', side))
        ancestors = _body_chain_names(model, int(model.geom_bodyid[gid]))
        is_env = any('cabinet' in n or 'drawer' in n or 'worktable' in n for n in ancestors)
        if is_env:
            if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError(f'Expected box for cabinet/drawer/table geometry {gid}')
            environment.append(geom_record(model, gid, 'table' if any('table' in n for n in ancestors) else 'cabinet_drawer_and_handles'))
    # Required native model coverage. Named links 1..7 and the shoulder base
    # visual mesh on the torso must be present on each side; no finger fallback.
    checks = {}
    for side in ('left', 'right'):
        chosen = [g for g in shells if g['side'] == side]
        checks[side + '_visible_links_1_through_7_present'] = all(
            any(g['body_name'] == f'astribot_arm_{side}_link_{j}' for g in chosen) for j in range(1, 8))
        checks[side + '_visible_shoulder_base_present'] = any(
            f'astribot_arm_{side}_base_link' in (g['mesh_name'] or '') for g in chosen)
        checks[side + '_visible_gripper_base_mesh_present'] = any(
            g['category'] == 'gripper_base_body' and g['type'] == 'mesh' for g in chosen)
    checks['table_boxes_present'] = any(g['category'] == 'table' for g in environment)
    checks['cabinet_boxes_present'] = any(g['body_name'] == 'drawer_cabinet' for g in environment)
    checks['moving_bottom_drawer_back_present'] = any(g['name'] == 'bottom_drawer_back' for g in environment)
    checks['no_finger_geometry_selected'] = not bool({g['geom_id'] for g in shells} & {g['geom_id'] for g in excluded_fingers})
    return dict(noncontact_shells=shells, environment_boxes=environment,
                excluded_intentional_finger_contact_geoms=excluded_fingers,
                excluded_invisible_physical_duplicates=excluded_invisible,
                selection_checks=checks,
                selection_rule='All visible arm link/shoulder-base meshes and all visible geoms on the exact gripper base/body rigid body; every cabinet/drawer/handle/worktable descendant box. Finger links and payload excluded.')


class ShellEnvironmentGeometry:
    """FK-only bounding-box branch-and-bound plus explicit convex queries."""
    def __init__(self, model, manifest):
        self.model = model
        self.shells = manifest['noncontact_shells']
        self.environments = manifest['environment_boxes']
        self.records = self.shells + self.environments
        self.ids = np.asarray([g['geom_id'] for g in self.records], dtype=int)
        centers, extents = [], []
        for g in self.records:
            gid = g['geom_id']
            if g['type'] == 'mesh':
                mesh = g['mesh_id']; start = int(model.mesh_vertadr[mesh]); count = int(model.mesh_vertnum[mesh])
                vertices = model.mesh_vert[start:start + count].astype(float)
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
                centers.append((hi + lo) / 2); extents.append((hi - lo) / 2)
            else:
                centers.append(np.zeros(3)); extents.append(model.geom_size[gid].copy())
        self.local_centers = np.asarray(centers)
        self.local_extents = np.asarray(extents)
        self.pairs = np.asarray([(i, len(self.shells) + j) for i in range(len(self.shells)) for j in range(len(self.environments))], dtype=int)
        self.pair_geom_ids = self.ids[self.pairs]

    def _sat_axes(self, mesh, box_rotation_in_mesh):
        if not hasattr(self, '_hulls'):
            self._hulls = {}
        if mesh not in self._hulls:
            start = int(self.model.mesh_vertadr[mesh]); count = int(self.model.mesh_vertnum[mesh])
            vertices = self.model.mesh_vert[start:start + count].astype(float)
            # No Qhull joggle or simplified hull: normals/edges only supply
            # candidate axes. Projection support always uses ALL source verts.
            hull = ConvexHull(vertices)
            edges = np.unique(np.sort(np.concatenate([hull.simplices[:, [0, 1]], hull.simplices[:, [1, 2]], hull.simplices[:, [2, 0]]]), axis=1), axis=0)
            edge_vectors = vertices[edges[:, 1]] - vertices[edges[:, 0]]
            self._hulls[mesh] = vertices, hull.equations[:, :3].copy(), edge_vectors
        vertices, normals, edges = self._hulls[mesh]
        cross_axes = np.cross(edges[:, None, :], box_rotation_in_mesh.T[None, :, :]).reshape(-1, 3)
        axes = np.concatenate([box_rotation_in_mesh.T, normals, cross_axes])
        norms = np.linalg.norm(axes, axis=1)
        axes = axes[np.isfinite(axes).all(axis=1) & (norms > AXIS_MIN_NORM)]
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        # Sign/near-duplicate removal only drops candidate axes. Any retained
        # normalized axis is itself a valid independent separation witness.
        lead = np.argmax(np.abs(axes), axis=1)
        axes *= np.where(axes[np.arange(len(axes)), lead] < 0, -1., 1.)[:, None]
        _, indices = np.unique(np.round(axes, 11), axis=0, return_index=True)
        return vertices, axes[np.sort(indices)]

    def sat_separation_certificate(self, data, shell, env):
        if int(self.model.geom_type[shell]) != int(mujoco.mjtGeom.mjGEOM_MESH):
            return None
        mesh = int(self.model.geom_dataid[shell])
        shell_rotation = data.geom_xmat[shell].reshape(3, 3)
        box_rotation = data.geom_xmat[env].reshape(3, 3)
        relative_rotation = shell_rotation.T @ box_rotation
        box_center = shell_rotation.T @ (data.geom_xpos[env] - data.geom_xpos[shell])
        vertices, axes = self._sat_axes(mesh, relative_rotation)
        box_size = self.model.geom_size[env]
        scale = 1. + max(float(np.max(np.abs(vertices))), float(np.max(np.abs(box_center))), float(np.max(box_size)))
        guard = SAT_PROJECTION_GUARD_M * scale
        best = None
        for begin in range(0, len(axes), 256):
            block = axes[begin:begin + 256]
            projections = vertices @ block.T
            mesh_lo, mesh_hi = projections.min(axis=0), projections.max(axis=0)
            box_mid = block @ box_center
            box_extent = np.abs(block @ relative_rotation) @ box_size
            box_lo, box_hi = box_mid - box_extent, box_mid + box_extent
            raw_gaps = np.maximum(mesh_lo - box_hi, box_lo - mesh_hi)
            j = int(np.argmax(raw_gaps)); bound = float(raw_gaps[j] - 2 * guard)
            if bound > 0 and (best is None or bound > best['separation_lower_bound_m']):
                axis = block[j]
                best = dict(method='convex_mesh_box_projection_axis', separation_lower_bound_m=bound,
                            axis_in_shell_frame=axis.tolist(), axis_in_world=(shell_rotation @ axis).tolist(),
                            mesh_projection_interval_m=[float(mesh_lo[j]), float(mesh_hi[j])],
                            box_projection_interval_m=[float(box_lo[j]), float(box_hi[j])],
                            support_outward_guard_m=guard, raw_axis_gap_m=float(raw_gaps[j]),
                            source_vertices_projected=len(vertices), candidate_axis_count=len(axes),
                            interpretation='Any unit-axis positive interval gap lower-bounds Euclidean separation. This is a certificate lower bound, not exact mesh distance.')
        return best

    def at_forwarded_state(self, data):
        rotations = data.geom_xmat[self.ids].reshape(-1, 3, 3)
        centers = data.geom_xpos[self.ids] + np.einsum('nij,nj->ni', rotations, self.local_centers)
        radii = np.einsum('nij,nj->ni', np.abs(rotations), self.local_extents) + AABB_INFLATION_M
        lo, hi = centers - radii, centers + radii
        a, b = self.pairs.T
        gap = np.maximum(0.0, np.maximum(lo[a] - hi[b], lo[b] - hi[a]))
        lower = np.linalg.norm(gap, axis=1)
        order = np.argsort(lower, kind='stable')
        minimum, worst = float('inf'), None
        rows = []
        for pair_index in order:
            if lower[pair_index] > max(0.0, minimum) + AABB_INFLATION_M:
                break
            shell, env = map(int, self.pair_geom_ids[pair_index])
            fromto = np.zeros(6)
            raw = float(mujoco.mj_geomDistance(self.model, data, shell, env, DISTANCE_QUERY_CAP_M, fromto))
            if not np.isfinite(raw) or raw >= DISTANCE_QUERY_CAP_M:
                raise RuntimeError(f'Invalid/capped explicit geometry distance: {shell}, {env}, {raw}')
            if raw + 1e-7 < lower[pair_index] and raw > 0:
                raise RuntimeError(f'AABB lower bound disagrees with convex query at pair {shell}, {env}')
            bound, certificate = raw, None
            # Raw negatives remain conservative rejections. Only the observed
            # raw-zero degeneration is eligible for an independent certificate.
            if raw == 0:
                if lower[pair_index] > 0:
                    bound = float(lower[pair_index])
                    certificate = dict(method='positive_world_aabb_gap', separation_lower_bound_m=bound,
                                       aabb_inflation_m=AABB_INFLATION_M,
                                       interpretation='Independent enclosing-AABB separation lower bound; raw MuJoCo query remains zero.')
                else:
                    certificate = self.sat_separation_certificate(data, shell, env)
                    if certificate is not None:
                        bound = certificate['separation_lower_bound_m']
            rows.append((int(pair_index), bound, float(lower[pair_index]), fromto.copy(), raw, certificate))
            if bound < minimum:
                minimum, worst = bound, int(pair_index)
        if worst is None:
            raise RuntimeError('No narrowphase pairs queried')
        return minimum, worst, rows


def synthetic_backend_check():
    """Prove this runtime explicitly queries a contype=conaffinity=0 mesh."""
    vertices = ' '.join(str(v) for v in np.asarray([(x, y, z) for x in (-.1, .1) for y in (-.1, .1) for z in (-.1, .1)]).ravel())
    xml = f'<mujoco><asset><mesh name="shell_cube" vertex="{vertices}"/></asset><worldbody><body name="shell"><freejoint/><geom type="mesh" mesh="shell_cube" contype="0" conaffinity="0"/></body><geom type="box" size=".1 .1 .1" pos=".5 0 0"/></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data); mujoco.mj_forward(model, data)
    positive = float(mujoco.mj_geomDistance(model, data, 0, 1, 10, np.zeros(6)))
    initial_contacts = int(data.ncon)
    data.qpos[0] = .45; mujoco.mj_forward(model, data)
    negative = float(mujoco.mj_geomDistance(model, data, 0, 1, 10, np.zeros(6)))
    checks = dict(explicit_mask_zero_mesh_positive_distance=abs(positive - .3) < 1e-7,
                  explicit_mask_zero_mesh_overlap_detected=negative < 0,
                  contact_monitor_has_no_contacts=initial_contacts == 0 and int(data.ncon) == 0)
    return dict(passed=all(checks.values()), checks=checks,
                separated_distance_m=positive, overlapped_convex_signed_distance_m=negative,
                contype=0, conaffinity=0, native_contacts_at_overlap=int(data.ncon),
                interpretation='Synthetic convex cube only: validates explicit queries independent of native contact masks. Negative native visual-mesh values are not exact triangle penetration depths.')


def audit(native, output):
    native = Path(native).resolve(strict=True); output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(f'Fresh output directory required: {output}')
    if mujoco.__version__ != REQUIRED_MUJOCO:
        raise RuntimeError(f'MuJoCo {REQUIRED_MUJOCO} required, got {mujoco.__version__}')
    scene, rollout = native / 'execute/scene.xml', native / 'execute/rollout.npz'
    inputs = sorted(set(xml_dependencies(scene) + [rollout, Path(__file__).resolve()]))
    pins_before = {str(p): sha(p) for p in inputs}
    output.mkdir(parents=True)
    started = time.monotonic(); created = datetime.now(timezone.utc).isoformat()
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    manifest = select_geometry(model)
    if not all(manifest['selection_checks'].values()):
        write_json(output / 'selection_rejection.json', dict(schema=SCHEMA, passed=False, accepted=False, status='rejected', geometry=manifest, input_pins=pins_before))
        raise ValueError('Required shell/environment coverage missing; see selection_rejection.json')
    backend = synthetic_backend_check()
    if not backend['passed']:
        raise RuntimeError('Explicit noncollision visual-mesh backend regression failed')
    with np.load(rollout, allow_pickle=False) as archive:
        times = np.asarray(archive['time_s'], dtype=float).copy()
        qpos = np.asarray(archive['qpos'], dtype=float).copy()
    if times.ndim != 1 or len(times) < 1 or qpos.shape != (len(times), model.nq):
        raise ValueError('Rollout time_s/qpos shapes do not match the native model')
    if not np.isfinite(times).all() or not np.isfinite(qpos).all() or (len(times) > 1 and not np.all(np.diff(times) > 0)):
        raise ValueError('Recorded native clock/state must be finite and strictly increasing')
    geometry = ShellEnvironmentGeometry(model, manifest)
    pair_count = len(geometry.pairs)
    minimums = np.empty(len(times)); worst_pairs = np.empty(len(times), dtype=int)
    narrow_counts = np.zeros(len(times), dtype=int); nonpositive_counts = np.zeros(len(times), dtype=int)
    evaluated_sample, evaluated_pair, evaluated_distance, evaluated_aabb, evaluated_fromto = [], [], [], [], []
    evaluated_raw, evaluated_method, fallback_certificates = [], [], []
    for i, (t, state) in enumerate(zip(times, qpos)):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = state
        data.time = float(t)
        mujoco.mj_forward(model, data)
        minimum, worst, rows = geometry.at_forwarded_state(data)
        minimums[i], worst_pairs[i], narrow_counts[i] = minimum, worst, len(rows)
        nonpositive_counts[i] = sum(row[1] <= 0 for row in rows)
        for pair, distance, aabb, fromto, raw, certificate in rows:
            evaluated_raw.append(raw)
            evaluated_method.append(certificate["method"] if certificate else "raw_mujoco_convex_query")
            if certificate is not None:
                fallback_certificates.append(dict(sample_index=i, time_s=float(t), pair_index=pair, pair_geom_ids=geometry.pair_geom_ids[pair].tolist(), raw_mj_distance_m=raw, **certificate))
            evaluated_sample.append(i); evaluated_pair.append(pair); evaluated_distance.append(distance)
            evaluated_aabb.append(aabb); evaluated_fromto.append(fromto)
        if (i + 1) % 250 == 0:
            print(json.dumps(dict(processed_samples=i + 1, total_samples=len(times), time_s=float(t), sampled_min_separation_certificate_lower_bound_m=float(minimums[:i + 1].min()), elapsed_s=time.monotonic() - started)), flush=True)
    trace = output / 'shell_environment_trace.npz'
    np.savez_compressed(trace, sample_index=np.arange(len(times)), time_s=times,
                        all_pair_min_separation_certificate_lower_bound_m=minimums,
                        worst_pair_index=worst_pairs, worst_pair_geom_ids=geometry.pair_geom_ids[worst_pairs],
                        narrowphase_count=narrow_counts, aabb_positive_separation_skipped_count=pair_count - narrow_counts,
                        nonpositive_convex_pair_count=nonpositive_counts,
                        pair_geom_ids=geometry.pair_geom_ids,
                        evaluated_sample_index=np.asarray(evaluated_sample, dtype=int),
                        evaluated_pair_index=np.asarray(evaluated_pair, dtype=int),
                        evaluated_separation_certificate_lower_bound_m=np.asarray(evaluated_distance),
                        evaluated_raw_mj_distance_m=np.asarray(evaluated_raw),
                        evaluated_certificate_method=np.asarray(evaluated_method),
                        evaluated_aabb_distance_lower_bound_m=np.asarray(evaluated_aabb),
                        evaluated_fromto=np.asarray(evaluated_fromto))
    certificates_path = output / "zero_query_separation_certificates.json"
    write_json(certificates_path, dict(schema=SCHEMA, certificates=fallback_certificates, raw_zero_queries=int(np.sum(np.asarray(evaluated_raw) == 0)), certified_raw_zero_queries=len(fallback_certificates)))
    pins_after = {str(p): sha(p) for p in inputs}
    by_id = {g['geom_id']: g for g in manifest['noncontact_shells'] + manifest['environment_boxes']}

    def detail(index):
        shell, env = map(int, geometry.pair_geom_ids[worst_pairs[index]])
        return dict(sample_index=int(index), time_s=float(times[index]), separation_certificate_lower_bound_m=float(minimums[index]),
                    shell=by_id[shell], environment=by_id[env],
                    interpretation='Minimum of conservative separation evidence over selected pairs; positive may be a convex-query distance or an independent AABB/SAT projection lower bound. Raw values remain in the trace. Nonpositive rejects; never an exact triangle penetration depth.')

    bad = np.flatnonzero(minimums <= 0)
    checks = dict(mujoco_3_3_7=mujoco.__version__ == REQUIRED_MUJOCO,
                  required_geometry_selection_complete=all(manifest['selection_checks'].values()),
                  explicit_mask_zero_visual_mesh_backend_passed=backend['passed'],
                  all_recorded_qpos_samples_checked=len(minimums) == len(times),
                  every_pair_queried_or_certified_by_positive_aabb=True,
                  all_sample_shell_environment_separation_certificates_strictly_positive=bool(np.all(minimums > 0)),
                  input_pins_unchanged=pins_before == pins_after)
    accepted = bool(all(checks.values()))
    report = dict(schema=SCHEMA, status='passed' if accepted else 'rejected', passed=accepted, accepted=accepted,
                  checks=checks, native_case=str(native), created_utc=created,
                  runtime=dict(python=sys.executable, mujoco=mujoco.__version__, elapsed_s=time.monotonic() - started),
                  coverage=dict(source='execute/rollout.npz:time_s,qpos', actual_recorded_native_qpos=True,
                                samples=len(times), start_s=float(times[0]), end_s=float(times[-1]),
                                max_sample_gap_s=float(np.diff(times).max()) if len(times) > 1 else None,
                                model_timestep_s=float(model.opt.timestep), simulation_steps_executed=0,
                                continuous_between_samples_checked=False, every_2000hz_native_step_checked=False,
                                reconstruction='Per record: mj_resetData + recorded qpos + recorded time + mj_forward only; qvel/ctrl and unrecorded integration fields retain reset defaults. Geometry only; no force/contact replay claim.',
                                shell_count=len(manifest['noncontact_shells']), environment_box_count=len(manifest['environment_boxes']),
                                pairs_per_sample=pair_count, total_pairs=len(times) * pair_count,
                                total_narrowphase_queries=int(narrow_counts.sum()),
                                total_aabb_positive_separation_skips=int((pair_count - narrow_counts).sum())),
                  geometry=manifest,
                  method=dict(narrowphase='Explicit mujoco.mj_geomDistance on original model geom IDs, including contype=conaffinity=0 visual geoms; convex hull mesh vs box.',
                              raw_zero_fallback='Positive enclosing AABB gap, otherwise a positive normalized convex-mesh/box SAT projection axis gap with support intervals expanded by an explicit numerical guard. No certificate means rejection; raw negatives never overridden.',
                              sat_projection_guard_base_m=SAT_PROJECTION_GUARD_M,
                              aabb='World enclosure of each compiled local mesh-vertex AABB, transformed with actual FK. Sort pair lower bounds; skip only positive AABB bounds above the current sample minimum.',
                              aabb_inflation_m=AABB_INFLATION_M, distance_query_cap_m=DISTANCE_QUERY_CAP_M,
                              threshold_m=0.0, required_relation='strictly positive separation certificate for every selected pair at every recorded sample',
                              signed_distance_limit='Nonpositive convex-hull distance does not establish original triangle intersection or exact penetration depth; it rejects because separation was not proven.'),
                  all_sample_minimum=detail(int(np.argmin(minimums))),
                  first_nonpositive_sample=detail(int(bad[0])) if len(bad) else None,
                  nonpositive_sample_count=int(len(bad)), nonpositive_evaluated_pair_samples=int(nonpositive_counts.sum()),
                  raw_nonpositive_query_count=int(np.sum(np.asarray(evaluated_raw) <= 0)),
                  raw_zero_query_count=int(np.sum(np.asarray(evaluated_raw) == 0)),
                  raw_zero_certified_separated_count=len(fallback_certificates),
                  zero_query_certificates=dict(path=str(certificates_path), sha256=sha(certificates_path)),
                  backend_regression=backend, input_pins=pins_before,
                  trace=dict(path=str(trace), sha256=sha(trace), per_sample_narrowphase_counts_saved=True,
                             all_sample_pair_minimum_separation_lower_bounds_saved=True, every_evaluated_pair_saved=True, raw_mujoco_values_saved_separately=True),
                  admission_scope='Additional sampled noncontact rigid-shell/environment separation gate only; original native acceptance does not imply this gate passes. Does not relabel native source artifacts.',
                  original_requirements_retained=['Original complete drawer task contract and thresholds', 'Original native hard/physics constraints', 'Intentional finger/grasp contacts and loaded retention audits', 'Opposing arms and grippers >=20mm gate and its own interval/every-step coverage', '15 second duration, state/action alias and exact independent command replay', 'Real2/render/publication admission remains separate'],
                  limitations=['Only the actual recorded qpos instants are checked. This is neither a 2000Hz native-step check nor a continuous motion proof.',
                               'MuJoCo convex hull overlap may reject safe concave mesh configurations. Negative values are not reported as exact visible triangle penetration depth.',
                               'Finger links/pads and payload intentional contact are excluded from this step and remain subject to the original grasp/task/hard gates.',
                               'Independent AABB/SAT fallback values are conservative separation lower bounds, not exact distance. Raw zero query values remain separately recorded.',
                               'Geometry restoration with reset+qpos+forward does not recover original force/contact history or full integration state.'])
    write_json(output / 'shell_environment_receipt.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, required=True, help='Existing native directory containing execute/scene.xml and execute/rollout.npz')
    parser.add_argument('--output', type=Path, required=True, help='Fresh sidecar directory; existing paths are refused')
    args = parser.parse_args()
    result = audit(args.native, args.output)
    print(json.dumps(dict(schema=SCHEMA, accepted=result['accepted'], status=result['status'],
                          samples=result['coverage']['samples'], minimum_separation_lower_bound_m=result['all_sample_minimum']['separation_certificate_lower_bound_m'],
                          output=str(args.output / 'shell_environment_receipt.json'))))
    return 0 if result['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
