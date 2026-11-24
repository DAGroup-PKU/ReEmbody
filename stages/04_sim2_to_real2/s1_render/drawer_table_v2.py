"""In-memory 140 x 78 cm version of the pinned user desk for drawer renders.

Call ``fit_desk`` after opening the original bgfit blend and before creating
native robot, cabinet, block, or camera objects. The helper never saves a blend
or changes a material, light, camera, or background object. The caller must
still pin the original blend and match its physical worktable and registration.
"""
from __future__ import annotations

import math

import bpy
import numpy as np

from environment import DESK_COLLECTION, TOP, bounds, set_table_height


WIDTH_M = 1.40
DEPTH_M = 0.78
BACKBOARD = 'Heightfit / central gray partition'
BACKBOARD_FRONT_Y_M = 0.3652
BACK_GAP_M = 0.001
SOURCE_WIDTH_M = 1.20
SOURCE_DEPTH_M = 0.60
SOURCE_TOP_XY_M = (0.0, 0.0)
TOL_M = 2e-5
ROBOT_CENTER_TOL_M = .005


def _check_close(actual: float, expected: float, description: str, tolerance: float = TOL_M) -> None:
    if not math.isfinite(float(actual)) or abs(float(actual) - expected) > tolerance:
        raise ValueError(f'{description}: expected {expected:.6f} m, measured {actual:.6f} m')


def _box(obj) -> dict:
    low, high = bounds(obj)
    return {
        'min_m': [float(v) for v in low],
        'max_m': [float(v) for v in high],
        'center_m': [float(v) for v in (low + high) / 2],
        'size_m': [float(v) for v in high - low],
    }


def _set_extent_and_center(obj, axis: int, extent: float, center: float) -> None:
    before = _box(obj)
    original = before['size_m'][axis]
    if original <= 0 or extent <= 0:
        raise ValueError(f'Invalid extent for {obj.name}')
    obj.scale[axis] *= extent / original
    bpy.context.view_layer.update()
    measured = _box(obj)['center_m'][axis]
    obj.location[axis] += center - measured
    bpy.context.view_layer.update()
    after = _box(obj)
    _check_close(after['size_m'][axis], extent, f'{obj.name} extent on axis {axis}')
    _check_close(after['center_m'][axis], center, f'{obj.name} center on axis {axis}')


def _move_center(obj, axis: int, center: float) -> None:
    obj.location[axis] += center - _box(obj)['center_m'][axis]


def _materials(objects) -> dict:
    return {obj.name: [material.as_pointer() if material else None for material in obj.data.materials]
            for obj in objects if obj.type in {'MESH', 'CURVE'}}


def registration_matrix(scene, physical_table_center_xy_m) -> np.ndarray:
    """One rigid native-to-Blender transform shared by robot, drawer, and cameras."""
    center = np.asarray(physical_table_center_xy_m, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Physical table center must be a finite XY pair')
    partition = scene.objects.get(BACKBOARD)
    if partition is None:
        raise ValueError('Pinned gray background partition is missing')
    front_y = _box(partition)['min_m'][1]
    _check_close(front_y, BACKBOARD_FRONT_Y_M, 'Background front plane', tolerance=5e-4)
    visual_center_y = front_y - BACK_GAP_M - DEPTH_M / 2
    world = np.eye(4)
    world[:3, :3] = np.diag([-1., -1., 1.])
    world[:2, 3] = center + np.array([0., visual_center_y])
    return world


def fit_desk(scene, *, top_m: float, robot_world_x_m: float,
             physical_table_center_xy_m, physical_table_half_size_xy_m,
             world_registration_matrix) -> dict:
    """Resize the desk and verify native/visual table and robot alignment.

    ``robot_world_x_m`` is the chassis center after the same rigid registration
    used for all native objects and cameras. The visual desk center is X=0;
    the new rear edge is 1 mm in front of the pinned gray partition.
    Raises on an unknown source asset, incomplete supports, or wrong bounds.
    """
    if not all(math.isfinite(float(v)) for v in (top_m, robot_world_x_m)):
        raise ValueError('Top height and registered robot X must be finite')
    if not 0.75 <= float(top_m) <= 0.85:
        raise ValueError('Desk top is outside the approved heightfit interval')
    if bpy.context.window.scene != scene:
        raise ValueError('The requested scene must be the active Blender scene')

    desk = [obj for obj in scene.objects
            if any(collection.name == DESK_COLLECTION for collection in obj.users_collection)]
    if not desk or any(obj.type not in {'MESH', 'CURVE'} for obj in desk):
        raise ValueError('Missing or unexpected editable desk objects')
    desk_names = {obj.name for obj in desk}
    required = {TOP, 'Desk / lower edge band', 'Desk / front modesty crossbar',
                'Desk / crossbar upper folded lip', 'Left / tabletop support',
                'Right / tabletop support', 'Left / rounded sled foot',
                'Right / rounded sled foot', 'Left / upper sliding post',
                'Right / upper sliding post'}
    if not required.issubset(desk_names):
        raise ValueError(f'Desk support parts missing: {sorted(required - desk_names)}')
    if len(desk) != 48:
        raise ValueError(f'Unexpected desk object count: {len(desk)}')
    if any(not name.startswith(('Desk /', 'Left /', 'Right /', 'Crossbar /',
                                'Crank /', 'Desktop /')) for name in desk_names):
        raise ValueError('Unknown editable desk object')

    tabletop = scene.objects[TOP]
    edge = scene.objects['Desk / lower edge band']
    partition = scene.objects.get(BACKBOARD)
    if partition is None:
        raise ValueError('Pinned gray background partition is missing')
    pre_top = _box(tabletop)
    pre_edge = _box(edge)
    pre_partition = _box(partition)
    for axis, expected in enumerate((SOURCE_WIDTH_M, SOURCE_DEPTH_M)):
        _check_close(pre_top['size_m'][axis], expected, f'Source tabletop size axis {axis}')
        _check_close(pre_top['center_m'][axis], SOURCE_TOP_XY_M[axis],
                     f'Source tabletop center axis {axis}')
    _check_close(pre_edge['size_m'][0], 1.198, 'Source edge width')
    _check_close(pre_edge['size_m'][1], 0.598, 'Source edge depth')
    _check_close(pre_partition['min_m'][1], BACKBOARD_FRONT_Y_M,
                 'Background front plane', tolerance=5e-4)
    half_size = np.asarray(physical_table_half_size_xy_m, dtype=float)
    if half_size.shape != (2,) or not np.isfinite(half_size).all() or \
            not np.allclose(half_size, [WIDTH_M / 2, DEPTH_M / 2], atol=TOL_M, rtol=0):
        raise ValueError('Physical table must already be 140 x 78 cm')
    physical_center = np.asarray(physical_table_center_xy_m, dtype=float)
    world = np.asarray(world_registration_matrix, dtype=float)
    if physical_center.shape != (2,) or world.shape != (4, 4) or \
            not np.isfinite(physical_center).all() or not np.isfinite(world).all():
        raise ValueError('Invalid physical table center or rigid registration')
    if abs(float(robot_world_x_m)) > ROBOT_CENTER_TOL_M:
        raise ValueError('Robot chassis is not horizontally centered on the desk')

    before_materials = _materials(desk)
    background = {obj.name: np.array(obj.matrix_world).copy()
                  for obj in scene.objects if obj not in desk}
    height_audit = set_table_height(scene, float(top_m))
    back_y = pre_partition['min_m'][1] - BACK_GAP_M
    center_x = 0.0
    center_y = back_y - DEPTH_M / 2
    expected_world = registration_matrix(scene, physical_center)
    if not np.allclose(world, expected_world, atol=TOL_M, rtol=0):
        raise ValueError('Native-to-Blender registration does not place the whole scene on the desk')
    mapped_center = world @ np.array([*physical_center, 0., 1.])
    if not np.allclose(mapped_center[:2], [center_x, center_y], atol=TOL_M, rtol=0):
        raise ValueError('Physical table center is not mapped to the visual desk center')
    side_delta = (WIDTH_M - SOURCE_WIDTH_M) / 2
    depth_delta = DEPTH_M - SOURCE_DEPTH_M

    # Keep the source tabletop and PVC edge materials. Their original one-mm
    # inset and 18-mm top thickness remain exact after resizing.
    for obj, width, depth in ((tabletop, WIDTH_M, DEPTH_M),
                              (edge, WIDTH_M - .002, DEPTH_M - .002)):
        _set_extent_and_center(obj, 0, width, center_x)
        _set_extent_and_center(obj, 1, depth, center_y)

    for obj in desk:
        if obj in (tabletop, edge):
            continue
        name = obj.name
        box = _box(obj)
        if name.startswith('Left /'):
            _move_center(obj, 0, box['center_m'][0] + center_x - side_delta)
        elif name.startswith(('Right /', 'Crank /')):
            _move_center(obj, 0, box['center_m'][0] + center_x + side_delta)
        elif name.startswith(('Desk / crossbar end seam', 'Crossbar / recessed fastener')):
            sign = -1 if box['center_m'][0] < 0 else 1
            _move_center(obj, 0, box['center_m'][0] + center_x + sign * side_delta)
        else:
            _move_center(obj, 0, box['center_m'][0] + center_x)

        if name in ('Desk / front modesty crossbar', 'Desk / crossbar upper folded lip'):
            _set_extent_and_center(obj, 0, box['size_m'][0] + 2 * side_delta, center_x)
        if name.endswith('tabletop support') or name.endswith('rounded sled foot'):
            _set_extent_and_center(obj, 1, box['size_m'][1] + depth_delta,
                                   box['center_m'][1] + center_y)
        elif 'rubber floor pad' in name:
            # The pads remain near the fore/aft ends of the lengthened feet.
            extra = -depth_delta / 2 if box['center_m'][1] < 0 else depth_delta / 2
            _move_center(obj, 1, box['center_m'][1] + center_y + extra)
        elif name.startswith('Crank /'):
            _move_center(obj, 1, box['center_m'][1] + center_y - depth_delta / 2)
        else:
            _move_center(obj, 1, box['center_m'][1] + center_y)
    bpy.context.view_layer.update()

    top = _box(tabletop)
    band = _box(edge)
    _check_close(top['size_m'][0], WIDTH_M, 'Final tabletop width')
    _check_close(top['size_m'][1], DEPTH_M, 'Final tabletop depth')
    _check_close(top['center_m'][0], center_x, 'Robot/table horizontal alignment')
    _check_close(top['center_m'][1], center_y, 'Final tabletop Y center')
    _check_close(top['max_m'][1], back_y, 'Table/backdrop clearance')
    _check_close(top['max_m'][2], top_m, 'Final tabletop height')
    _check_close(top['size_m'][2], .018, 'Tabletop thickness')
    for axis, expected in enumerate((WIDTH_M - .002, DEPTH_M - .002)):
        _check_close(band['size_m'][axis], expected, f'Edge band size axis {axis}')
        _check_close(band['center_m'][axis], top['center_m'][axis],
                     f'Edge band alignment axis {axis}')
    if _materials(desk) != before_materials:
        raise ValueError('Resizing changed a desk material slot')
    for obj in scene.objects:
        if obj.name in background and not np.array_equal(np.array(obj.matrix_world), background[obj.name]):
            raise ValueError(f'Background geometry moved: {obj.name}')

    supports = {}
    for side in ('Left', 'Right'):
        beam = _box(scene.objects[f'{side} / tabletop support'])
        foot = _box(scene.objects[f'{side} / rounded sled foot'])
        post = _box(scene.objects[f'{side} / upper sliding post'])
        sign = -1 if side == 'Left' else 1
        _check_close(post['center_m'][0], center_x + sign * .66,
                     f'{side} support X center')
        _check_close(beam['size_m'][1], .63, f'{side} beam depth')
        _check_close(foot['size_m'][1], .63, f'{side} foot depth')
        if beam['min_m'][1] < top['min_m'][1] or beam['max_m'][1] > top['max_m'][1]:
            raise ValueError(f'{side} support extends outside the tabletop')
        if foot['max_m'][1] >= pre_partition['min_m'][1]:
            raise ValueError(f'{side} foot intersects the background')
        supports[side.lower()] = {'beam': beam, 'foot': foot, 'post': post}

    return {
        'schema': 'rssr.drawer.user_desk_140x78.v2',
        'source_tabletop': pre_top,
        'top_m': float(top_m),
        'robot_world_x_m': float(robot_world_x_m),
        'physical_table_center_xy_m': physical_center.tolist(),
        'physical_table_half_size_xy_m': half_size.tolist(),
        'world_registration_matrix': world.tolist(),
        'tabletop': top,
        'edge_band': band,
        'background_front_y_m': pre_partition['min_m'][1],
        'background_gap_m': pre_partition['min_m'][1] - top['max_m'][1],
        'supports': supports,
        'desk_object_count': len(desk),
        'desk_material_slots_unchanged': True,
        'background_transforms_unchanged': True,
        'source_blend_saved': False,
        'height_adjustment': height_audit,
    }
