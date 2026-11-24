"""The user's FR3 Duo site (photographed 2026-11-03) and the S1-aligned initial-scene augmentation on it.

Layout `fr3_site_20261103`, fitted by site_fit.py from the user's RealSense D455 head-camera frame (640x480) with the
official FR3 Duo mount geometry (result/franka_fr3_duo_site_20261103_attempt_01/fit/d455_fit.json, 4.0 px rms):
  * desk: the S1 desk model (1.2 x 0.6 m top, white sled feet, crank) under a dark cloth; its near edge 0.47 m in front
    of the mounting point, its centre 0.039 m to the robot's left. The fit gives a 0.73 m top; the sim keeps 0.75 m so
    the mounting point stays 0.025 m above the top with the user's 0.775 m base height;
  * head camera: the D455 on the white arch behind the robot (not on franka_head), 0.063 m behind, 0.009 m right of
    and 0.684 m above the mounting point, 51.8 deg down; 640x480 stream, vertical FOV 65 deg;
  * wrist cameras: the nominal D405 mount of fr3_duo.build_full; 640x480 stream, vertical FOV 58 deg.
The robot, its home pose, servos and Robotiq 2F-85s are unchanged (fr3_duo.build_full).

Augmentation `site_aug` (aligned with S1 v8/v9, sim2-refine/orchestration/stack_random200_r15/sample_v3.py), stacking
tasks: variation k takes source case k mod 50 (object sizes and contract; S1 likewise cycles its source states) and
draws, in the mount frame,
  * each target object uniformly in a 15 cm disk around its centre (the 50 source placements' mean) and its yaw in
    U(-15, 15) deg (S1: 15 cm disks, U(-15, 15) deg);
  * the stack region uniformly in a 5 cm disk around its centre (S1: 5 cm disk);
  * both arms at HOME_Q plus U(-3, 3) deg per joint (S1: U(-3, 3) deg).
A draw is redrawn when an object is not on the desk with 2 cm to spare, two footprints come within 2 cm, or the upper
object already stands on the stack spot (duo_scene rules). S1's stack-centre rule (the head optical axis meets the
table) is not used: the D455 axis meets the desk 4 cm inside its near edge, where the base cup would overhang.

Layout `fr3_site_20261104` (the user's corrections of 2026-11-04) is `fr3_site_20261103` with
  * the start pose of the user's video: both arms at franka_description's `ready` group state (robots/common/
    group_definition.xacro: 0, -pi/4, 0, -3pi/4, 0, pi/2, pi/4), instead of the camera-alignment HOME_Q;
  * the user's cups: two identical tapered paper cups, 8 cm top and 9 cm high (the user's measurement, recorded in
    result/sim2-real2/t1_multiview_asset02_full250_20261016/asset_manifest.json), with S1's tapered collision template
    (sim1-sim2/s1_real78_random200_v2/rssr_augment/measured_s1.py CUP: 5.85 cm bottom, 1.5 mm wall, 60 g, 32 wall
    segments), instead of the source cases' 7.5 / 5.85 cm straight proxies.
Its augmentation split `site_aug_v2` uses the seed stream of `site_aug`.

Layout `fr3_site_20261104b` is `fr3_site_20261104` with the start pose the user read from the robot (FCI, read only,
2026-11-04 16:37:43 +08:00: result/franka_fr3_duo_site_20261104_attempt_01/start_pose_user_20261104.json): it puts the
arms where the user's D455 frame shows them (grippers forward over the far half of the desk), which `ready` does not.
Its split `site_aug_v3` uses the same seed stream.

Layout `fr3_site_20261104c` (site_v6, 2026-11-04 late: result/franka_fr3_duo_site_v6_headfit_20261104_attempt_01) is
`fr3_site_20261104b` re-placed from the user's tape measurements and the head view:
  * desk: the S1 sim desk (user), 1.4 x 0.78 m; the sim keeps its 0.75 m top (real 0.79 m: everything is placed
    relative to the top), centre 0.0074 m to the robot's left;
  * D455 at the measured place (1.74 m above the floor = 0.95 m above the top, 0.64 m behind the desk's near edge), its
    aim fitted to the desk corners in the real D455 frame;
  * robot: mounting point 0.672 m behind the near edge and 0.2167 m above the top (0.237 m above the 0.77 m tray: an
    unknown frame raises the arms), by a silhouette fit of the robot to the real D455 frame with the URDF mount
    angles; it also matches the user's two side photos (not used in the fit) and puts the start-pose grippers
    mid-desk, as the user said;
  * collision objects behind and under the arms (none of them in the earlier layouts): the cart tray, a block standing
    in for the frame between tray and mount, and the white arch that carries the D455 (legs beside the shoulders,
    estimated from the surround video); SUPPORT_CLEARANCE_M is the arm-to-arch clearance planning and admission keep;
  * tasks placed on the near part of the desk the arms reach (CENTERS_V4), objects and start pose as fr3_site_20261104b.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import fr3_duo as F

NAME = 'fr3_site_20261103'
NAME_V2 = 'fr3_site_20261104'
NAME_V3 = 'fr3_site_20261104b'
NAME_V4 = 'fr3_site_20261104c'
# site_v6 with the user's factory URDF robot (2026-11-05): Robotiq not rotated on the flange, on a 17 mm adapter,
# arm bases 0.9 mm off v0_3 (fr3_duo.FACTORY_*); everything else as NAME_V4
NAME_V5 = 'fr3_site_20261105a'
# NAME_V5 with the desk 0.15 m closer to the robot (2026-11-05, user: objects 15 cm farther from the desk's near edge,
# else the desk 15 cm closer; with the objects moved, top-down grasps at 0.93 m forward have no IK solution,
# result/franka_fr3_duo_site_v8_20261105_attempt_01/logs/reach_plus15.log). Robot, D455 and supports as NAME_V5; the
# objects keep their place in the mount frame and so stand 15 cm farther from the near edge. Stacking tasks keep
# ARM_GAP_M between the two arms (user: "叠放任务时左右臂会碰到一起，加一个10cm的安全距离").
NAME_V6 = 'fr3_site_20261105b'
# NAME_V6 with 6 cm to the camera arch (legs and arc) instead of 5, measured exactly (EXACT_LAYOUTS) (2026-11-05: the
# user first asked for 10 cm, "拱架立柱也保证10cm安全距离"; on the estimated arch the measured start pose is already 9.47 cm
# from the right post, then "限制改成6cm"); the same site, robot and objects
NAME_V7 = 'fr3_site_20261105c'
# NAME_V7 with the task objects taken verbatim from the user's S1 task scenes (2026-11-05, user: "你可以直接复用S1对应任务的
# 资产包括抽屉、方块、杯子"): the <body> elements (and their meshes) of S1_ASSET_DIR/<task>/objects.xml replace the built
# ones in the scene XML; only each object's root position on the FR3 desk is the FR3 layout's (s1_object_xml)
NAME_V8 = 'fr3_site_20261105d'
LAYOUTS = (NAME, NAME_V2, NAME_V3, NAME_V4, NAME_V5, NAME_V6, NAME_V7, NAME_V8)
SITE_V6 = (NAME_V4, NAME_V5, NAME_V6, NAME_V7, NAME_V8)  # the site_v6 desk, D455, supports and joint ranges
S1_ASSET_LAYOUTS = (NAME_V8,)
from pathlib import Path as _Path
import sys as _sys
_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from reembody.paths import resolve_s1_task_assets  # noqa: E402

S1_ASSET_DIR = str(resolve_s1_task_assets())
DESK_CLOSER_V6 = 0.15
# left/right arm clearance (m) planning and admission keep, per layout and task (default duo_run.ARM_CLEAR planning,
# servo.ARM_CLEARANCE_M admission); the two arms' links 0 and 1 on the shared mount are left out as before
ARM_GAP_M = {(NAME_V6, 'stack_cups'): 0.10, (NAME_V6, 'stack_blocks'): 0.10,
             (NAME_V7, 'stack_cups'): 0.10, (NAME_V7, 'stack_blocks'): 0.10,
             (NAME_V8, 'stack_cups'): 0.10, (NAME_V8, 'stack_blocks'): 0.10}
PLACE = F.DuoPlacement(0.47, 0.025)
DESK_SIZE = (1.2, 0.6)  # width (robot left-right) x depth
DESK_LEFT_M = 0.039  # desk centre to the robot's left of the mounting point
PLACE_V4 = F.DuoPlacement(0.672, 0.2167)
PLACE_V5 = F.DuoPlacement(0.672, 0.2167, mount_origin=F.FACTORY_MOUNT_ORIGIN, gripper_yaw=F.FACTORY_GRIPPER_YAW,
                          gripper_offset=F.FACTORY_GRIPPER_OFFSET, coupling=F.FACTORY_COUPLING)
PLACE_V6 = F.DuoPlacement(0.672 - DESK_CLOSER_V6, 0.2167, mount_origin=F.FACTORY_MOUNT_ORIGIN,
                          gripper_yaw=F.FACTORY_GRIPPER_YAW, gripper_offset=F.FACTORY_GRIPPER_OFFSET,
                          coupling=F.FACTORY_COUPLING)
DESK_SIZE_V4, DESK_LEFT_M_V4 = (1.4, 0.78), 0.0074
D455_IN_MOUNT_V4 = np.array([0.0316, -0.0224, 0.7333])
D455_R_IN_MOUNT_V4 = np.array([[0.042218, -0.804994, 0.591779],
                               [-0.999094, -0.037218, 0.020649],
                               [0.005402, -0.592114, -0.805836]])
# mount frame (x forward, y left, z up from the mounting point): the tray top 0.237 m below the mounting point, the
# cart under it from 0.42 m behind to 0.12 m in front, +-0.25 m; the stand-in frame block; the arch tube (radius
# 0.018 m) in the plane x = -0.0054, legs at y = +-0.42 from the tray up to z = 0.2933, a half circle above
SUPPORT_V4 = dict(tray_z=-0.2367, cart_x=(-0.42, 0.12), cart_halfwidth=0.25, riser_x=(-0.11, 0.11), riser_y=(-0.14, 0.14),
                  mount_bottom_z=-0.0308, arch_x=-0.0054, arch_radius=0.42, arch_centre_z=0.2933, arch_tube=0.018)
SUPPORT_CLEARANCE_M = 0.05
SUPPORT_CLEARANCE_BY_LAYOUT = {NAME_V7: 0.06, NAME_V8: 0.06}  # layouts that keep more than SUPPORT_CLEARANCE_M to the arch
# layouts whose planning and admission measure the arch and arm-to-arm distances exactly (exact_clearance: dense
# surface samples against the arch capsules / the other arm), not with MuJoCo's mesh contact distances, which read
# 10.46 cm for a pose 8.89 cm from the arch (2026-11-05)
EXACT_LAYOUTS = (NAME_V7, NAME_V8)


def support_clearance_m(layout):
    """Arm-to-arch clearance planning and admission keep on a layout."""
    return SUPPORT_CLEARANCE_BY_LAYOUT.get(layout, SUPPORT_CLEARANCE_M)
D455_IN_MOUNT = np.array([-0.062908, -0.008954, 0.684186])
# camera axes (x right, y down, z forward) as columns, in the mount frame (x forward, y left, z up)
D455_R_IN_MOUNT = np.array([[0.002722, -0.786273, 0.617874],
                            [-0.999946, 0.004072, 0.009586],
                            [-0.010053, -0.617866, -0.786219]])
HEAD_FOVY_DEG, WRIST_FOVY_DEG = 65.0, 58.0
VIEW_W, VIEW_H = 640, 480
SPLIT_SEED = {'site_aug': 4_000, 'site_heldout': 5_000, 'site_aug_v2': 4_000, 'site_aug_v3': 4_000, 'site_aug_v4': 4_000}
SPLIT_LAYOUT = {'site_aug': NAME, 'site_heldout': NAME, 'site_aug_v2': NAME_V2, 'site_aug_v3': NAME_V3,
                'site_aug_v4': NAME_V4}
USER_START_Q = {'left': (-0.102979, 0.206048, 0.52927, -1.546546, -1.072101, 3.007822, 0.046065),
                'right': (-0.142175, 0.411645, -0.352853, -1.34792, 0.220946, 2.820929, 0.610911)}
READY_Q = (0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4)  # franka_description group_state `ready`
CUP = dict(top_diameter_m=0.08, bottom_diameter_m=0.0585, height_m=0.09, wall_m=0.0015, mass_kg=0.06, segments=32)
CENTERS = {'stack_cups': {'cup_left_base': (0.645, 0.238), 'cup_right_upper': (0.532, -0.270),
                          'stack_region': (0.528, -0.017)}}
OBJ_RADIUS_M, REGION_RADIUS_M, YAW_DEG, ARM_DEG, EDGE_MARGIN_M = 0.15, 0.05, 15.0, 3.0, 0.02
# NAME_V4 task placements (mount frame xy): on the near part of the desk the arms reach from 0.672 m behind its edge
# (reach_scan of result/franka_fr3_duo_site_v6_tasks_20261104_attempt_01: top-down grasps reach only 0.70 - 0.85 m
# forward, i.e. 3 - 18 cm into the desk; left hand y >= -0.2, right hand y <= 0.2). Drawer: cabinet centre; its
# front face 0.025 m inside the near edge, the bottom handle bar 0.695 m forward, opening toward the robot.
CENTERS_V4 = {'stack_cups': {'cup_left_base': (0.78, 0.28), 'cup_right_upper': (0.78, -0.28), 'stack_region': (0.76, 0.0)},
              'stack_blocks': {'blue_cube': (0.78, 0.28), 'purple_cube': (0.78, -0.28), 'stack_region': (0.76, 0.0)},
              'drawer': {'drawer_cabinet': (0.85, 0.03), 'red_block': (0.76, 0.40)}}
# NAME_V4 joint ranges for planning: the first-generation FR3 datasheet ranges (narrower than FR3 v2's) less
# V4_RANGE_INSET, so that with the planner's own 0.025 rad margin every commanded joint keeps >= 0.05 rad to them
FR3_V1_RANGES = [(-2.7437, 2.7437), (-1.7837, 1.7837), (-2.9007, 2.9007), (-3.0421, -0.1518), (-2.8065, 2.8065),
                 (0.5445, 4.5169), (-3.0159, 3.0159)]
V4_RANGE_INSET = 0.03


def place(layout=NAME):
    return {NAME_V4: PLACE_V4, NAME_V5: PLACE_V5, NAME_V6: PLACE_V6, NAME_V7: PLACE_V6, NAME_V8: PLACE_V6}.get(layout, PLACE)


def desk(layout=NAME):
    """(width, depth) and the desk centre's offset to the robot's left of the mounting point."""
    return (DESK_SIZE_V4, DESK_LEFT_M_V4) if layout in SITE_V6 else (DESK_SIZE, DESK_LEFT_M)


def mount_to_world(p_mount, layout=NAME):
    p0, R0 = place(layout).mount_pose()
    return p0 + R0 @ np.asarray(p_mount, float)


def world_to_mount(p_world, layout=NAME):
    p0, R0 = place(layout).mount_pose()
    return R0.T @ (np.asarray(p_world, float) - p0)


def desk_box(layout=NAME):
    """(centre xyz, half sizes) of the desk top in the world (the sim top is 0.05 m thick, like the source table)."""
    (w, dep), left = desk(layout)
    c = mount_to_world([place(layout).dist_behind_edge + dep / 2, left, 0.0], layout)
    return np.array([c[0], c[1], F.TABLE_TOP - 0.025]), np.array([w / 2, dep / 2, 0.025])


def head_pose_world(layout=NAME):
    p0, R0 = place(layout).mount_pose()
    if layout in SITE_V6:
        return p0 + R0 @ D455_IN_MOUNT_V4, R0 @ D455_R_IN_MOUNT_V4
    return p0 + R0 @ D455_IN_MOUNT, R0 @ D455_R_IN_MOUNT


def add_supports(spec, layout):
    """NAME_V4: cart tray, stand-in frame and camera arch as static collision geoms (bodies site_cart, site_riser,
    site_support_arch; the arch tube as capsules between 24 points of its path)."""
    p0, R0 = place(layout).mount_pose()
    S = SUPPORT_V4
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, R0.reshape(9))
    rgba = [0.85, 0.85, 0.83, 1.0]

    def box(name, lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        b = spec.worldbody.add_body(name=name, pos=p0 + R0 @ ((lo + hi) / 2), quat=q0)
        b.add_geom(name=name + '_box', type=mujoco.mjtGeom.mjGEOM_BOX, size=(hi - lo) / 2, rgba=rgba)
    box('site_cart', [S['cart_x'][0], -S['cart_halfwidth'], S['tray_z'] - 0.03], [S['cart_x'][1], S['cart_halfwidth'], S['tray_z']])
    box('site_riser', [S['riser_x'][0], S['riser_y'][0], S['tray_z']], [S['riser_x'][1], S['riser_y'][1], S['mount_bottom_z']])
    arch = spec.worldbody.add_body(name='site_support_arch', pos=p0, quat=q0)
    r, zc, x = S['arch_radius'], S['arch_centre_z'], S['arch_x']
    path = [(x, r, S['tray_z']), (x, r, zc)] + [(x, r * np.cos(t), zc + r * np.sin(t)) for t in np.linspace(0, np.pi, 21)[1:-1]] \
        + [(x, -r, zc), (x, -r, S['tray_z'])]
    for i, (a, b) in enumerate(zip(path[:-1], path[1:])):
        arch.add_geom(name=f'site_support_arch_{i:02d}', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[S['arch_tube'], 0, 0],
                      fromto=list(a) + list(b), rgba=rgba)
    return spec


def home_q(layout, hand):
    """Start joints of a layout's arms (before the per-variation offset)."""
    from . import duo_scene as DS
    if layout in (NAME_V3, *SITE_V6):
        return np.asarray(USER_START_Q[hand], float)
    return np.asarray(READY_Q if layout == NAME_V2 else DS.HOME_Q[hand], float)


def site_scene(scene, layout):
    """The varied source scene with the layout's objects: on NAME_V2 both cups are the user's 8 x 9 cm cups (S1's
    template), standing on the table at their drawn xy and yaw."""
    if layout not in (NAME_V2, NAME_V3, *SITE_V6):
        return scene
    import copy
    out = copy.deepcopy(scene)
    for o in out['objects']:
        if o['shape'] == 'cup':
            o['size_m'] = [CUP['top_diameter_m'], CUP['top_diameter_m'], CUP['height_m']]
            o['wall_m'], o['mass_kg'] = CUP['wall_m'], CUP['mass_kg']
            o['position_m'] = [o['position_m'][0], o['position_m'][1], F.TABLE_TOP + CUP['height_m'] / 2 + 0.0003]
            o['evidence'] = 'User cups: identical tapered paper cups, 8 cm top, 9 cm high; S1 collision template.'
    return out


def cup_wall_radius(z_from_bottom):
    """Outer radius of the tapered cup wall at a height above its bottom."""
    return (CUP['bottom_diameter_m'] + z_from_bottom / CUP['height_m'] * (CUP['top_diameter_m'] - CUP['bottom_diameter_m'])) / 2


def tapered_cups(spec, scene):
    """Replace each cup's straight 16-box wall with S1's tapered template (measured_s1.add_cup_geometry): a bottom
    disc and 32 convex wall segments, keeping the cup's contact parameters and colour."""
    n, h, t = CUP['segments'], CUP['height_m'], CUP['wall_m']
    for o in scene['objects']:
        if o['shape'] != 'cup':
            continue
        body = spec.body(o['id'])
        body.pos = o['position_m']
        old = [g for g in spec.geoms if g.name.startswith(o['id'] + '_')]
        ref = next(g for g in old if '_wall_' in g.name)
        attrs = dict(contype=ref.contype, conaffinity=ref.conaffinity, condim=ref.condim, friction=list(ref.friction),
                     solref=list(ref.solref), solimp=list(ref.solimp), rgba=list(ref.rgba))
        for g in old:
            spec.delete(g)
        body.add_geom(name=o['id'] + '_bottom', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      size=[CUP['bottom_diameter_m'] / 2, t / 2, 0], pos=[0, 0, -h / 2 + t / 2], mass=CUP['mass_kg'] * 0.2,
                      **attrs)
        for i in range(n):
            angles = (2 * np.pi * i / n, 2 * np.pi * (i + 1) / n)
            verts = []
            for z in (-h / 2, h / 2):
                outer = cup_wall_radius(z + h / 2)
                for r in (outer, outer - t):
                    for a in angles:
                        verts += [r * np.cos(a), r * np.sin(a), z]
            mesh = spec.add_mesh(name=f"{o['id']}_wall_{i:02d}_mesh", uservert=verts)
            body.add_geom(name=f"{o['id']}_wall_{i:02d}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                          mass=CUP['mass_kg'] * 0.8 / n, **attrs)
    return spec


def adapt(spec, layout=NAME, scene=None):
    """Desk and D455 of the site on an fr3_duo.build_full spec built with place(layout) (and the NAME_V2+ cups;
    NAME_V4: the supports)."""
    c, h = desk_box(layout)
    body = spec.body('worktable')
    assert np.allclose(body.quat, [1, 0, 0, 0])
    c = c - np.asarray(body.pos)  # the desk geoms are in the worktable body's frame
    top = spec.geom('worktable_top')
    top.pos, top.size = c, h
    for i, (sx, sy) in enumerate(((-1, 1), (-1, -1), (1, 1), (1, -1))):
        g = spec.geom(f'worktable_leg_{i}')
        g.pos = [c[0] + sx * (h[0] - 0.04), c[1] + sy * (h[1] - 0.05), g.pos[2]]
    # no franka_head on this robot: the D455 sits on the arch behind it
    spec.delete(spec.geom('franka_head'))
    pos, R = head_pose_world(layout)
    cam = spec.camera('head')
    cam.pos = pos
    cam.fovy = HEAD_FOVY_DEG
    cam.alt.xyaxes = list(np.r_[R[:, 0], -R[:, 1]])  # MuJoCo: x right, y up, looking along -z
    for hand in F.HANDS:
        spec.camera(f'{hand}_wrist').fovy = WRIST_FOVY_DEG
    spec.visual.global_.offwidth = VIEW_W
    spec.visual.global_.offheight = VIEW_H
    if layout in (NAME_V2, NAME_V3, *SITE_V6):
        tapered_cups(spec, scene)
    if layout in SITE_V6:
        add_supports(spec, layout)
        for hand in F.HANDS:
            for j, (lo, hi) in enumerate(FR3_V1_RANGES, 1):
                spec.joint(f'{hand}_fr3v2_joint{j}').range = [lo + V4_RANGE_INSET, hi - V4_RANGE_INSET]
    return spec


def on_desk(xy_mount, r, layout=NAME):
    (w, dep), left = desk(layout)
    d0 = place(layout).dist_behind_edge
    lo = (d0 + r + EDGE_MARGIN_M, left - w / 2 + r + EDGE_MARGIN_M)
    hi = (d0 + dep - r - EDGE_MARGIN_M, left + w / 2 - r - EDGE_MARGIN_M)
    return lo[0] <= xy_mount[0] <= hi[0] and lo[1] <= xy_mount[1] <= hi[1]


def _disk(rng, center, r):
    a, rho = rng.uniform(0, 2 * np.pi), r * np.sqrt(rng.random())
    return np.asarray(center) + rho * np.array([np.cos(a), np.sin(a)])


def draw(task, split, k):
    from . import duo_scene as DS
    rng = np.random.default_rng([SPLIT_SEED[split], DS.TASKS.index(task), k])
    layout = SPLIT_LAYOUT[split]
    centers = (CENTERS_V4 if layout in SITE_V6 else CENTERS)[task]
    cases = DS.source_cases(task)
    case = cases[k % len(cases)]
    scene = DS.raw_scene(case)
    objs = {o['id']: o for o in scene['objects']}
    for attempt in range(1000):
        shift, placed = {}, {}
        for oid, c in centers.items():
            o = objs[oid]
            r = REGION_RADIUS_M if o['mobility'] == 'static' else OBJ_RADIUS_M
            xy = _disk(rng, c, r)
            w = mount_to_world([xy[0], xy[1], 0.0], layout)
            yaw = 0.0 if o['mobility'] == 'static' else float(np.radians(rng.uniform(-YAW_DEG, YAW_DEG)))
            shift[oid] = [float(w[0] - o['position_m'][0]), float(w[1] - o['position_m'][1]),
                          float(yaw - o.get('yaw_rad', 0.0)) if o['mobility'] != 'static' else 0.0]
            placed[oid] = xy
        arm = {h: np.radians(rng.uniform(-ARM_DEG, ARM_DEG, 7)).round(6).tolist() for h in F.HANDS}
        v = DS.Variation(f'{task}_{split}_{k:04d}', task, case, shift, arm, layout=SPLIT_LAYOUT[split])
        sc = site_scene(DS.apply(scene, v), v.layout)
        sobjs = {o['id']: o for o in sc['objects']}  # the layout's object sizes
        base = DS._base(sc)  # the base object will stand on the region
        ok = all(on_desk(placed[i], DS._footprint(sobjs[i])[1], layout) for i in placed if sobjs[i]['mobility'] == 'free')
        ok = ok and on_desk(placed[scene['bindings']['target_region']], DS._footprint(sobjs[base])[1], layout)
        if ok and valid_site(sc):
            return v
    raise RuntimeError(f'no valid site variation for {task} {split} {k}')


def valid_site(scene):
    """duo_scene.valid without its table box (the desk test is on_desk)."""
    from . import duo_scene as DS
    objs = {o['id']: o for o in scene['objects']}
    free = [o for o in scene['objects'] if o['mobility'] == 'free']
    for i, a in enumerate(free):
        for o in free[i + 1:]:
            (ca, ra), (cb, rb) = DS._footprint(a), DS._footprint(o)
            if np.linalg.norm(ca - cb) < ra + rb + DS.GAP_M:
                return False
    b = scene['bindings']
    cr, _ = DS._footprint(objs[b['target_region']])
    for t in [t for t in b['targets'] if t != DS._base(scene)]:
        ct, rt = DS._footprint(objs[t])
        if np.linalg.norm(ct - cr) < rt + 0.05:
            return False
    return True


def split_list(task, split, n, start=0):
    return [draw(task, split, k) for k in range(start, start + n)]


def nominal(task, layout=NAME_V4, case=None, centers=None, tag='nominal'):
    """The layout's nominal placement of a task as a Variation (no random draw, both arms exactly at the start pose):
    stacking objects and the region centred at CENTERS_V4 with zero yaw; the drawer cabinet (with its drawers and
    handles) moved as one so that its centre is at CENTERS_V4['drawer']['drawer_cabinet'], the block at its centre."""
    from . import duo_scene as DS
    case = case or DS.source_cases(task)[0]
    scene = DS.raw_scene(case)
    centers = centers or CENTERS_V4[task]
    targets = scene['bindings']['targets']
    shift = {}

    def to(o, xy):
        w = mount_to_world([xy[0], xy[1], 0.0], layout)
        return [float(w[0] - o['position_m'][0]), float(w[1] - o['position_m'][1])]
    objs = {o['id']: o for o in scene['objects']}
    if task == 'drawer':
        g = to(objs['drawer_cabinet'], centers['drawer_cabinet'])
        for o in scene['objects']:
            if o['id'] not in targets:
                shift[o['id']] = g + [0.0]
    else:
        r = scene['bindings']['target_region']
        shift[r] = to(objs[r], centers[r]) + [0.0]
    for t in targets:
        shift[t] = to(objs[t], centers[t]) + [float(-objs[t].get('yaw_rad', 0.0))]
    return DS.Variation(f'{task}_{layout}_{tag}', task, case, shift, {}, layout=layout)


def s1_object_xml(xml, task):
    """The scene XML with every task-object <body> replaced, in place (the state vector keeps its layout), by the S1
    <body> of the same name from S1_ASSET_DIR/<task>/objects.xml, and the S1 meshes in place of same-named ones. Only
    the root body's pos / quat stay the FR3 scene's (where the object stands on this desk)."""
    import xml.etree.ElementTree as ET
    from pathlib import Path
    root = ET.fromstring(xml)
    s1 = ET.parse(Path(S1_ASSET_DIR) / task / 'objects.xml').getroot()
    world = root.find('worldbody')
    bodies = {b.get('name'): b for b in s1.find('worldbody').findall('body')}
    done = []
    for i, b in enumerate(list(world)):
        if b.tag == 'body' and b.get('name') in bodies:
            new = bodies[b.get('name')]
            new.set('pos', b.get('pos', '0 0 0'))
            if b.get('quat'):
                new.set('quat', b.get('quat'))
            elif new.get('quat'):
                del new.attrib['quat']
            world.remove(b)
            world.insert(i, new)
            done.append(b.get('name'))
    assert sorted(done) == sorted(bodies), (task, done, list(bodies))
    asset = root.find('asset')
    meshes = {m.get('name'): m for m in s1.find('asset').findall('mesh')}
    for m in list(asset.findall('mesh')):
        if m.get('name') in meshes:
            asset.remove(m)
    for m in meshes.values():
        asset.append(m)
    return ET.tostring(root, encoding='unicode')
