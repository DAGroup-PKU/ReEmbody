"""Geometry projection used while preparing source pose evidence."""
from pathlib import Path
import numpy as np
from .common import read
def required_square_fovy(camera_points, margin=.05, minimum=70, maximum=110):
    points = np.asarray(camera_points, dtype=float)
    depth = -points[:, 2]
    if np.any(depth <= 1e-4):
        raise ValueError('Task region falls behind the head camera')
    tangent = float(np.max(np.abs(points[:, :2]) / depth[:, None])) / (1 - 2 * margin)
    angle = max(minimum, float(np.rad2deg(2 * np.arctan(tangent))))
    if angle > maximum:
        raise ValueError(f'Task region needs excessive head FOV {angle:.2f}; inspect camera aim')
    return angle

def fit_head(sim, row, indices):
    import mujoco
    contract = read(Path(row['scene']).parent / 'task_contract.json')
    names = list(dict.fromkeys(contract['bindings']['targets'] + contract['bindings']['containers']))
    bodies = {sim.model.body(name).id for name in names}
    geoms = []
    for gid in range(sim.model.ngeom):
        bid = int(sim.model.geom_bodyid[gid])
        while bid and bid not in bodies:
            bid = int(sim.model.body_parentid[bid])
        if bid in bodies:
            geoms.append(gid)
    if not geoms:
        raise ValueError('No native task geometry for head coverage')
    corners = []
    for gid in geoms:
        size = sim.model.geom_size[gid].copy()
        kind = int(sim.model.geom_type[gid])
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            size[:] = size[0]
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            size = np.array([size[0], size[0], size[1] + (size[0] if kind == mujoco.mjtGeom.mjGEOM_CAPSULE else 0)])
        elif kind != mujoco.mjtGeom.mjGEOM_BOX:
            size[:] = float(sim.model.geom_rbound[gid])
        vertices = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
        corners.append((gid, vertices))
    camera = sim.model.camera('head_rgbd').id
    # Frame the full swept task region instead of including unused table edges
    # behind/under the robot. Use complete task geometry, not body origins.
    cached = []
    for source_frame in indices:
        state = int(np.clip(np.searchsorted(sim.times, float(source_frame) / 24, side='right') - 1, 0, len(sim.times) - 1))
        sim.data.qpos[:] = sim.qpos[state]
        sim.data.qvel[:] = sim.qvel[state]
        sim.data.ctrl[:] = sim.ctrl[state]
        mujoco.mj_forward(sim.model, sim.data)
        world = np.concatenate([vertices @ sim.data.geom_xmat[gid].reshape(3, 3).T + sim.data.geom_xpos[gid]
                                for gid, vertices in corners])
        cached.append((state, world))
    cloud = np.concatenate([points for _, points in cached])
    aim = (cloud.min(axis=0) + cloud.max(axis=0)) / 2
    sim.data.qpos[:] = sim.qpos[0]
    sim.data.qvel[:] = sim.qvel[0]
    sim.data.ctrl[:] = sim.ctrl[0]
    mujoco.mj_forward(sim.model, sim.data)
    from research_20260929.export_success_review_3x3 import camera_quat
    from scipy.spatial.transform import Rotation
    desired = camera_quat(sim.data.cam_xpos[camera], aim)
    world_rotation = Rotation.from_quat(desired[[1, 2, 3, 0]]).as_matrix()
    parent = int(sim.model.cam_bodyid[camera])
    local_rotation = sim.data.xmat[parent].reshape(3, 3).T @ world_rotation
    sim.model.cam_quat[camera] = Rotation.from_matrix(local_rotation).as_quat()[[3, 0, 1, 2]]
    sim.camera_provenance['head_rgbd']['quaternion_in_parent_wxyz'] = sim.model.cam_quat[camera].tolist()
    worst_angle = 70.
    checked = 0
    for state, world in cached:
        sim.data.qpos[:] = sim.qpos[state]
        sim.data.qvel[:] = sim.qvel[state]
        sim.data.ctrl[:] = sim.ctrl[state]
        mujoco.mj_forward(sim.model, sim.data)
        local = (world - sim.data.cam_xpos[camera]) @ sim.data.cam_xmat[camera].reshape(3, 3)
        worst_angle = max(worst_angle, required_square_fovy(local))
        checked += 1
    sim.model.cam_fovy[camera] = worst_angle
    sim.camera_provenance['head_rgbd']['fovy_deg'] = worst_angle
    sim.camera_provenance['head_rgbd']['source'] = 'task_facing_render_optics_fitted_to_saved_rollout'
    return {'all_sampled_task_regions_in_view': True, 'checked_video_frames': checked,
            'task_bodies': names, 'geometry_count': len(geoms), 'includes_full_swept_task_geometry': True,
            'task_region_world_min': cloud.min(axis=0).tolist(), 'task_region_world_max': cloud.max(axis=0).tolist(),
            'head_aim_world_at_initial_state': aim.tolist(),
            'normalized_margin': .05, 'head_fovy_deg': worst_angle,
            'saved_robot_qpos_changed': False, 'wrist_optics_changed': False,
            'source_rollout_sha256': row['rollout_sha256'], 'source_scene_sha256': row['scene_sha256']}

