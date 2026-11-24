"""Task-conditioned initial layout hypotheses from recorded source actions.

These functions only initialize a scene. They neither move objects during a
rollout nor assert that rigid holding or task completion actually occurs.
"""
import itertools
import copy
import numpy as np
from rssr_scene import object_rotation

SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)))


def validate_pose_only_initialization(scene, reference):
    """A versioned initializer may change poses, never geometry/task identities."""
    normalized=copy.deepcopy(scene);changes=[]
    if len(scene['objects'])!=len(reference['objects']):raise ValueError('Object count changed')
    for obj,prior in zip(normalized['objects'],reference['objects']):
        for key,shape in [('position_m',(3,)),('yaw_rad',()),('roll_pitch_rad',(2,))]:
            if key not in obj and key not in prior:continue
            if key not in obj or key not in prior:raise ValueError('Pose representation changed')
            a,b=np.asarray(obj[key],float),np.asarray(prior[key],float)
            if a.shape!=shape or b.shape!=shape or not np.isfinite([a,b]).all():raise ValueError('Invalid pose')
            if not np.array_equal(a,b):changes.append({'object':obj['id'],'field':key,'before':prior[key],'after':obj[key]})
            obj[key]=prior[key]
    if normalized!=reference:raise ValueError('Versioned initializer changed non-pose scene data')
    return changes


def horizontal_axis_for_upright_release(grasp_rotation, release_rotation):
    delta = np.asarray(release_rotation) @ np.asarray(grasp_rotation).T
    axis = delta.T @ np.array([0., 0., 1.])
    length = np.linalg.norm(axis[:2])
    if not np.isfinite(axis).all() or length < 1e-6:
        raise ValueError('Source rotation does not identify a horizontal initial axis')
    horizontal = axis[:2]/length
    yaw = float(np.arctan2(horizontal[0], -horizontal[1]))
    predicted = delta @ np.r_[horizontal, 0.]
    return {'yaw_rad': yaw, 'roll_pitch_rad': [np.pi/2, 0.],
            'predicted_release_tilt_degrees': float(np.rad2deg(np.arccos(np.clip(predicted[2], -1, 1)))),
            'basis': 'closest horizontal axis under source rigid-hold hypothesis',
            'physical_success_asserted': False}


def collision_free_receiver_center(scene, receiver_id, nominal_local_xy,
                                   release_clearance_xy, margin_m=.002, table_bounds_xy=None):
    """Closest rectangle centre fitting release bounds and avoiding initial AABBs.

Conservative object bounds are in the receiver orientation. The admissible
rectangle comes from fixed object sizes and source-predicted release extents.
No obstacle, object size or release clearance is changed when no fit exists.
"""
    objects = {o['id']: o for o in scene['objects']}
    receiver = objects[receiver_id]
    if receiver.get('parent_id') or receiver.get('joint'):
        raise ValueError('Only independent receivers can be fitted')
    rotation = object_rotation(receiver)
    if not np.allclose(rotation[2], [0, 0, 1], atol=1e-6):
        raise ValueError('Receiver must rest upright on a horizontal table')
    nominal = np.asarray(nominal_local_xy, float)
    clearance = np.asarray(release_clearance_xy, float)
    if nominal.shape != (2,) or clearance.shape != (2,) or not np.isfinite([nominal, clearance]).all():
        raise ValueError('Finite 2D layout bounds required')
    if np.any(clearance < 0):
        return {'applied': False, 'reason': 'source release footprints do not fit fixed receiver'}
    low, high = nominal-clearance, nominal+clearance
    center = np.asarray(receiver['position_m']) @ rotation
    half = np.asarray(receiver['size_m'])/2
    blocked = []
    for name, obj in objects.items():
        if name == receiver_id or obj.get('parent_id') == receiver_id or obj.get('support_id') == receiver_id:
            continue
        vertices = ((SIGNS*np.asarray(obj['size_m'])/2) @ object_rotation(obj).T
                    + np.asarray(obj['position_m'])) @ rotation
        a, b = vertices.min(axis=0), vertices.max(axis=0)
        if b[2] <= center[2]-half[2] or a[2] >= center[2]+half[2]:
            continue
        blocked.append((name, a[:2]-half[:2]-margin_m, b[:2]+half[:2]+margin_m))
    # In 2D the closest point in this polygonal feasible set is the nominal
    # point, a projection onto a boundary, or an intersection of boundaries.
    # This also handles receiver yaw relative to the fixed rectangular table.
    constraints=[];lines=[]
    for axis in range(2):
        normal=np.eye(2)[axis]
        constraints.extend([(normal,high[axis]),(-normal,-low[axis])])
        for _, a, b in blocked:
            lines.extend([(normal,a[axis]-1e-8),(normal,b[axis]+1e-8)])
    if table_bounds_xy is not None:
        bounds=np.asarray(table_bounds_xy,float)
        if bounds.shape!=(2,2) or not np.isfinite(bounds).all() or np.any(bounds[0]>=bounds[1]):
            raise ValueError('Finite ordered table XY bounds required')
        extent=(np.abs(rotation)@half)[:2]
        for axis in range(2):
            normal=rotation[axis,:2]
            constraints.extend([(normal,bounds[1,axis]-extent[axis]-margin_m),
                                (-normal,-bounds[0,axis]-extent[axis]-margin_m)])
    lines+=constraints
    candidates=[nominal]
    for normal,bound in lines:
        candidates.append(nominal+normal*(bound-normal@nominal)/(normal@normal))
    for (a,x),(b,y) in itertools.combinations(lines,2):
        matrix=np.array([a,b])
        if abs(np.linalg.det(matrix))>1e-10:candidates.append(np.linalg.solve(matrix,[x,y]))
    choices=[]
    for xy in candidates:
        if any(normal@xy>bound+1e-10 for normal,bound in constraints):continue
        if any(np.all(xy>a+1e-10) and np.all(xy<b-1e-10) for _,a,b in blocked):
            continue
        choices.append(xy)
    if not choices:
        return {'applied': False, 'reason': 'no conservative collision-free release-compatible centre',
                'blocking_objects': [name for name, _, _ in blocked]}
    selected = min(choices, key=lambda x: (float(np.linalg.norm(x-nominal)), *x))
    return {'applied': True, 'selected_local_xy_m': selected.tolist(),
            'nominal_local_xy_m': nominal.tolist(), 'admissible_low_m': low.tolist(),
            'admissible_high_m': high.tolist(), 'clearance_margin_m': margin_m,
            'table_bounds_xy_m':None if table_bounds_xy is None else np.asarray(table_bounds_xy).tolist(),
            'blocking_objects': [name for name, _, _ in blocked],
            'geometry_resized': False, 'physical_success_asserted': False}


def table_bounds_from_model(model):
    """Read the fixed table's actual box/convex-box geometry, never a guessed size."""
    import mujoco
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    gid=model.geom('worktable_top').id
    if model.body_weldid[model.geom_bodyid[gid]]!=0:raise ValueError('Table must be fixed')
    if model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_BOX:
        vertices=SIGNS*model.geom_size[gid]
    elif model.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=model.geom_dataid[gid];start=model.mesh_vertadr[mid];count=model.mesh_vertnum[mid]
        vertices=model.mesh_vert[start:start+count]
    else:raise ValueError('Expected box or convex-box table')
    world=vertices@data.geom_xmat[gid].reshape(3,3).T+data.geom_xpos[gid]
    low,high=world.min(axis=0),world.max(axis=0)
    if not np.all(np.isclose(world,low,atol=1e-7,rtol=0)|np.isclose(world,high,atol=1e-7,rtol=0)):
        raise ValueError('Table is not an axis-aligned rectangular solid')
    return np.array([low[:2],high[:2]])
