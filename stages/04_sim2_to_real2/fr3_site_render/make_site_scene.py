"""Build the Blender scene of the user's FR3 Duo site (S1 method: sim2-real2/s1_heightfit_v2/make_scene.py).

As for S1, the room is world-fixed procedural geometry shared by all cameras, built from the user's desk asset and
its materials (the S1 runtime desk scene, unchanged on disk), placed by the site fit and the surround video:
  * desk: the S1 desk model (the same white-legged 1.2 x 0.6 m crank desk is in the video), top at the sim's 0.75 m,
    turned so its front (crank side) faces the robot, centre at the fitted position (sim world frame of
    franka_dual_v1.duo_site: mounting point at (0, 0.12, 0.775), robot facing -y, robot left = +x);
  * a charcoal cotton cloth over the top, draped 7 cm over every edge;
  * the S1 grey carpet tiles over the whole floor;
  * the white pillar behind the desk (black skirting), the potted plant on a red bin at its right corner (up to site_v3), a glass
    curtain wall with mullions on the robot's left, a dark open ceiling with linear LED strips, office desks, chairs,
    monitors and an orange sofa behind the robot, the white workbench at its left;
  * the robot cart (aluminium frame, black controller box, e-stop) under the mount, and the white arch carrying the
    D455 behind the robot.
Robot, cameras and task objects are not in this file; the renderer adds them per episode. Lights are named area lights
(the per-episode lighting recipe moves and scales them, as in S1); the LED strips are emissive meshes.

    native_photo/bin/python make_site_scene.py --source <S1 desk_scene_bgfit_v3_runtime.blend> \
        --parameters site_parameters.json --output <fresh dir>/site_scene.blend
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from s1_environment import set_table_height, lights as light_rows  # noqa: E402

DESK_COLLECTION = '01 Desk - editable geometry'
CARPET = 'Floor / low pile gray carpet fiber'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def material(name, color, roughness=0.5, metallic=0.0, coat=0.0, sheen=0.0, emission=None, strength=0.0, bump=None,
             specular=0.5):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    s = nt.nodes['Principled BSDF']
    s.inputs['Base Color'].default_value = (*color, 1)
    s.inputs['Roughness'].default_value = roughness
    s.inputs['Metallic'].default_value = metallic
    s.inputs['Coat Weight'].default_value = coat
    s.inputs['Sheen Weight'].default_value = sheen
    s.inputs['Sheen Tint'].default_value = (0.5, 0.5, 0.5, 1)
    s.inputs['Specular IOR Level'].default_value = specular
    if emission is not None:
        s.inputs['Emission Color'].default_value = (*emission, 1)
        s.inputs['Emission Strength'].default_value = strength
    if bump:
        scale, amount = bump
        tc = nt.nodes.new('ShaderNodeTexCoord')
        noise = nt.nodes.new('ShaderNodeTexNoise')
        noise.inputs['Scale'].default_value = scale
        noise.inputs['Detail'].default_value = 6.0
        b = nt.nodes.new('ShaderNodeBump')
        b.inputs['Strength'].default_value = amount
        nt.links.new(tc.outputs['Object'], noise.inputs['Vector'])
        nt.links.new(noise.outputs['Fac'], b.inputs['Height'])
        nt.links.new(b.outputs['Normal'], s.inputs['Normal'])
    return m


def carpet_multiply(mat, scale):
    """Fixed colour multiply after the carpet's image texture (S1: 'Heightfit gray neutral value multiply')."""
    tree = mat.node_tree
    shader = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
    sock = shader.inputs['Base Color']
    node = tree.nodes.new('ShaderNodeMixRGB')
    node.name = 'Site carpet colour multiply'
    node.blend_type = 'MULTIPLY'
    node.inputs[0].default_value = 1.0
    link = sock.links[0] if sock.links else None
    if link:
        tree.links.new(link.from_socket, node.inputs[1])
    else:
        node.inputs[1].default_value = sock.default_value
    node.inputs[2].default_value = (*scale, 1)
    tree.links.new(node.outputs[0], sock)


def mesh_object(collection, name, verts, faces, mat, smooth=False):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(map(float, v)) for v in verts], [], faces)
    me.update()
    if smooth:
        for p in me.polygons:
            p.use_smooth = True
    o = bpy.data.objects.new(name, me)
    collection.objects.link(o)
    me.materials.append(mat)
    return o


def box(collection, name, lo, hi, mat):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = [(x0, y0, z0), (x0, y0, z1), (x0, y1, z0), (x0, y1, z1), (x1, y0, z0), (x1, y0, z1), (x1, y1, z0), (x1, y1, z1)]
    f = [(4, 6, 7, 5), (0, 1, 3, 2), (2, 3, 7, 6), (0, 4, 5, 1), (1, 5, 7, 3), (0, 2, 6, 4)]
    return mesh_object(collection, name, v, f, mat)


def cylinder(collection, name, center, radius, height, mat, n=40, radius_top=None):
    cx, cy, z0 = center
    rt = radius if radius_top is None else radius_top
    v = [(cx + radius * math.cos(2 * math.pi * k / n), cy + radius * math.sin(2 * math.pi * k / n), z0) for k in range(n)]
    v += [(cx + rt * math.cos(2 * math.pi * k / n), cy + rt * math.sin(2 * math.pi * k / n), z0 + height) for k in range(n)]
    f = [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)] + [tuple(range(n))[::-1], tuple(range(n, 2 * n))]
    o = mesh_object(collection, name, v, f, mat, smooth=False)
    for p in o.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return o


def tube_path(collection, name, points, radius, mat, n=16):
    """A round tube through the polyline `points`."""
    pts = [np.asarray(p, float) for p in points]
    verts, faces = [], []
    for i, p in enumerate(pts):
        t = (pts[min(i + 1, len(pts) - 1)] - pts[max(i - 1, 0)])
        t /= np.linalg.norm(t)
        a = np.cross(t, [0, 0, 1.0]) if abs(t[2]) < 0.9 else np.cross(t, [1.0, 0, 0])
        a /= np.linalg.norm(a)
        b = np.cross(t, a)
        for k in range(n):
            ang = 2 * math.pi * k / n
            verts.append(p + radius * (math.cos(ang) * a + math.sin(ang) * b))
    for i in range(len(pts) - 1):
        for k in range(n):
            faces.append((i * n + k, i * n + (k + 1) % n, (i + 1) * n + (k + 1) % n, (i + 1) * n + k))
    return mesh_object(collection, name, verts, faces, mat, smooth=True)


def cloth(collection, name, center, size, top_z, drape, mat, n=48):
    """Cloth over the desk: a top sheet and four drapes, gently rounded over the edges, with slight sag."""
    cx, cy = center
    hx, hy = size[0] / 2 + 0.006, size[1] / 2 + 0.006
    r = 0.012  # fold radius over the edge
    # profile across one edge: (outward offset, z) from the top sheet over the fold to the hem
    prof = [(-0.0, top_z)] + [(r * math.sin(a), top_z - r + r * math.cos(a)) for a in np.linspace(0.2, math.pi / 2, 5)]
    prof += [(r + 0.002 * k / 4, top_z - r - (drape - r) * k / 4) for k in range(1, 5)]
    verts, faces = [], []
    # top sheet grid
    xs, ys = np.linspace(-hx, hx, n), np.linspace(-hy, hy, n // 2)
    for y in ys:
        for x in xs:
            verts.append((cx + x, cy + y, top_z))
    nx, ny = len(xs), len(ys)
    for j in range(ny - 1):
        for i in range(nx - 1):
            faces.append((j * nx + i, j * nx + i + 1, (j + 1) * nx + i + 1, (j + 1) * nx + i))
    # drapes along each side: rows of the profile, outward normal (ox, oy)
    sides = [((xs, -hy), (0, -1)), ((xs, hy), (0, 1)), ((ys, -hx), (-1, 0)), ((ys, hx), (1, 0))]
    for (line, off), (ox, oy) in sides:
        base = len(verts)
        for (dout, z) in prof:
            for t in line:
                x, y = (t, off) if oy else (off, t)
                verts.append((cx + x + ox * dout, cy + y + oy * dout, z))
        m = len(line)
        for j in range(len(prof) - 1):
            for i in range(m - 1):
                faces.append((base + j * m + i, base + j * m + i + 1, base + (j + 1) * m + i + 1, base + (j + 1) * m + i))
    o = mesh_object(collection, name, verts, faces, mat, smooth=True)
    sol = o.modifiers.new('thickness', 'SOLIDIFY')
    sol.thickness = 0.0015
    return o


def plant(collection, name, base, height, leaf_mat, rng):
    """Tall broad-leaf plant: elongated leaf blades fanning out of the pot."""
    objs = []
    for k in range(22):
        az = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.15, 0.9)
        length = rng.uniform(0.35, 0.6)
        h0 = base[2] + rng.uniform(0.0, height - length)
        d = np.array([math.cos(az) * math.sin(tilt), math.sin(az) * math.sin(tilt), math.cos(tilt)])
        side = np.cross(d, [0, 0, 1.0])
        side = side / (np.linalg.norm(side) + 1e-9)
        p0 = np.array([base[0], base[1], h0])
        w = rng.uniform(0.08, 0.14)
        v, f = [], []
        for s in range(7):
            u = s / 6
            c = p0 + d * length * u - np.array([0, 0, 0.12 * u * u * tilt])
            half = w * math.sin(math.pi * min(u * 1.1, 1.0)) / 2 + 0.003
            v += [c - side * half, c + side * half]
        for s in range(6):
            f.append((2 * s, 2 * s + 1, 2 * s + 3, 2 * s + 2))
        objs.append(mesh_object(collection, f'{name} leaf {k}', v, f, leaf_mat, smooth=True))
    return objs


def join_static(scene):
    """All mesh objects of the room and desk into one object (modifiers applied, materials kept): the renderer syncs
    every object before every frame, and ~500 static objects dominated that time."""
    meshes = [o for o in scene.objects if o.type == 'MESH']
    dg = bpy.context.evaluated_depsgraph_get()
    for o in meshes:
        if o.modifiers or o.data.users > 1:
            me = bpy.data.meshes.new_from_object(o.evaluated_get(dg))
            o.modifiers.clear()
            o.data = me
    with bpy.context.temp_override(active_object=meshes[0], selected_editable_objects=meshes, selected_objects=meshes):
        bpy.ops.object.join()
    meshes[0].name = 'Site static geometry (joined)'
    return len(meshes)


def build(scene, P):
    rng = np.random.default_rng(20261103)
    view_layer = bpy.context.view_layer
    # ---- remove the S1 room: everything but the desk collection and the carpet material users
    removed = []
    keep = {o.name for o in bpy.data.collections[DESK_COLLECTION].all_objects}
    tile = next(o for o in scene.objects if o.name.startswith('Carpet tile / 0 0'))
    for o in list(scene.objects):
        if o.name in keep or o == tile:
            continue
        removed.append(o.name)
        bpy.data.objects.remove(o, do_unlink=True)
    col = bpy.data.collections.new('10 FR3 Duo site (fr3_site_20261103)')
    scene.collection.children.link(col)
    # ---- desk: height, then turned and moved as one rigid set
    table = set_table_height(scene, P['desk_top_m'])
    if P.get('desk_model_scale_xy'):  # site_v5: the 1.2 x 0.6 m model stretched to the S1 sim desk (1.4 x 0.78 m)
        top = scene.objects['Desk / 1200 x 600 x 18 mm top']
        pts = [top.matrix_world @ Vector(v) for v in top.bound_box]
        cx, cy = (min(p.x for p in pts) + max(p.x for p in pts)) / 2, (min(p.y for p in pts) + max(p.y for p in pts)) / 2
        sx, sy = P['desk_model_scale_xy']
        S = Matrix.Translation((cx, cy, 0)) @ Matrix.Diagonal((sx, sy, 1.0, 1.0)) @ Matrix.Translation((-cx, -cy, 0))
        for o in bpy.data.collections[DESK_COLLECTION].all_objects:
            if o.parent is None:
                o.matrix_world = S @ o.matrix_world
        view_layer.update()
        pts = [top.matrix_world @ Vector(v) for v in top.bound_box]
        table.update(width_m=max(p.x for p in pts) - min(p.x for p in pts), depth_m=max(p.y for p in pts) - min(p.y for p in pts),
                     model_scale_xy=[sx, sy])
    T = Matrix.Translation((P['desk_centre_xy_m'][0], P['desk_centre_xy_m'][1], 0)) @ Matrix.Rotation(math.radians(P['desk_yaw_deg']), 4, 'Z')
    for o in bpy.data.collections[DESK_COLLECTION].all_objects:
        if o.parent is None:
            o.matrix_world = T @ o.matrix_world
    view_layer.update()
    # ---- carpet: the S1 tile (its material and UVs) repeated over the floor
    tile_pts = np.array([tuple(tile.matrix_world @ Vector(v)) for v in tile.bound_box])
    tsize = float(tile_pts[:, 0].max() - tile_pts[:, 0].min())
    tc = (tile_pts.max(0) + tile_pts.min(0)) / 2
    (x0, x1), (y0, y1) = P['floor_x_m'], P['floor_y_m']
    tiles = 0
    for ix, x in enumerate(np.arange(x0 + tsize / 2, x1, tsize)):
        for iy, y in enumerate(np.arange(y0 + tsize / 2, y1, tsize)):
            t = tile.copy()
            t.name = f'Site carpet tile {ix} {iy}'
            t.location = tile.location + Vector((x - tc[0], y - tc[1], 0))
            # tiles are laid in alternating directions, as carpet tiles are
            t.rotation_euler[2] = tile.rotation_euler[2] + (math.pi / 2 if (ix + iy) % 2 else 0)
            col.objects.link(t)
            tiles += 1
    bpy.data.objects.remove(tile, do_unlink=True)
    carpet_multiply(bpy.data.materials[CARPET], P['carpet_rgb_scale'])
    mats = {k: bpy.data.materials[v] for k, v in dict(paint='Room / off white paint', glass='Window / dark green glass reflection',
                                                     grid='Window / dark aluminum grid', laminate='Desktop / warm white polished laminate',
                                                     ivory='Frame / ivory fine powder coat', rubber='Feet / dark rubber contact pads').items()}
    mats.update(
        cloth=material('Site / charcoal cotton cloth', P['cloth_rgb'], 0.92, sheen=P['cloth_sheen'], bump=(160.0, 0.12),
                       specular=P['cloth_specular']),
        skirting=material('Site / black skirting', (0.02, 0.02, 0.02), 0.6),
        ceiling=material('Site / dark open ceiling', (0.035, 0.036, 0.04), 0.9),
        led=material('Site / LED strip diffuser', (1, 1, 1), 0.5, emission=(1.0, 0.97, 0.92), strength=P['led_strength']),
        alu=material('Site / anodized aluminium extrusion', (0.62, 0.63, 0.64), 0.35, metallic=0.85),
        black=material('Site / black powder coat', (0.025, 0.025, 0.027), 0.55),
        white_plastic=material('Site / white plastic', (0.8, 0.8, 0.78), 0.35, coat=0.2),
        yellow=material('Site / safety yellow', (0.75, 0.55, 0.02), 0.4),
        red=material('Site / signal red', (0.55, 0.02, 0.02), 0.4),
        bin=material('Site / red bin', (0.5, 0.03, 0.04), 0.45),
        pot=material('Site / white ceramic pot', (0.75, 0.75, 0.73), 0.3, coat=0.4),
        leaf=material('Site / plant leaf', (0.03, 0.14, 0.03), 0.55, sheen=0.2),
        sofa=material('Site / orange sofa fabric', (0.75, 0.12, 0.02), 0.85, sheen=0.5, bump=(220.0, 0.1)),
        chair=material('Site / black mesh chair', (0.02, 0.02, 0.022), 0.75),
        screen=material('Site / monitor screen', (0.01, 0.012, 0.015), 0.15, emission=(0.35, 0.45, 0.6), strength=0.6),
        d455=material('Site / D455 housing', (0.12, 0.12, 0.13), 0.4, metallic=0.6))
    created = []

    def add(o):
        created.append(o.name)
        return o
    c, s = P['desk_centre_xy_m'], P['desk_size_m']
    add(cloth(col, 'Site / desk cloth', c, s, P['desk_top_m'] + 0.001, P['cloth_drape_m'], mats['cloth']))
    # ---- pillar behind the desk, with skirting; plant on a red bin at its right corner
    (px0, px1), (py0, py1), ph = P['pillar_x_m'], P['pillar_y_m'], P['ceiling_z_m']
    add(box(col, 'Site / white pillar', (px0, py0, 0), (px1, py1, ph), mats['paint']))
    add(box(col, 'Site / pillar skirting front', (px0 - 0.004, py1, 0), (px1 + 0.004, py1 + 0.008, 0.08), mats['skirting']))
    add(box(col, 'Site / pillar skirting right', (px0 - 0.008, py0, 0), (px0, py1 + 0.004, 0.08), mats['skirting']))
    add(box(col, 'Site / pillar skirting left', (px1, py0, 0), (px1 + 0.008, py1 + 0.004, 0.08), mats['skirting']))
    if P.get('plant_xy_m') is not None:  # site_v4: none (not in the user's D455 frame, 2026-11-04)
        bx, by = P['plant_xy_m']
        add(box(col, 'Site / red bin', (bx - 0.15, by - 0.15, 0), (bx + 0.15, by + 0.15, 0.48), mats['bin']))
        add(cylinder(col, 'Site / plant pot', (bx, by, 0.48), 0.13, 0.30, mats['pot'], radius_top=0.15))
        created += [o.name for o in plant(col, 'Site / plant', (bx, by, 0.72), 1.15, mats['leaf'], rng)]
    # ---- curtain wall on the robot's left: glass, mullions, transom
    wx = P['window_x_m']
    (wy0, wy1) = P['room_y_m']
    add(box(col, 'Site / curtain wall glass', (wx, wy0, 0), (wx + 0.012, wy1, ph), mats['glass']))
    for k, y in enumerate(np.arange(wy0, wy1 + 1e-6, P['mullion_spacing_m'])):
        add(box(col, f'Site / mullion {k}', (wx - 0.05, y - 0.025, 0), (wx, y + 0.025, ph), mats['grid']))
    for k, z in enumerate((0.12, 2.35)):
        add(box(col, f'Site / transom {k}', (wx - 0.05, wy0, z - 0.03), (wx, wy1, z + 0.03), mats['grid']))
    # far walls (white), ceiling and LED strips
    (rx0, rx1) = P['room_x_m']
    add(box(col, 'Site / far wall right', (rx0 - 0.1, wy0, 0), (rx0, wy1, ph), mats['paint']))
    add(box(col, 'Site / far wall front', (rx0, wy0 - 0.1, 0), (wx, wy0, ph), mats['paint']))
    add(box(col, 'Site / far wall back', (rx0, wy1, 0), (wx, wy1 + 0.1, ph), mats['paint']))
    add(box(col, 'Site / ceiling', (rx0, wy0, ph), (wx, wy1, ph + 0.05), mats['ceiling']))
    for k, (x, y0_, y1_) in enumerate(P['led_strips']):
        add(box(col, f'Site / LED strip {k}', (x - 0.025, y0_, ph - 0.03), (x + 0.025, y1_, ph - 0.005), mats['led']))
    # ---- office behind and left of the robot
    for k, (x, y, w, d) in enumerate(P['office_desks']):
        add(box(col, f'Site / office desk {k} top', (x - w / 2, y - d / 2, 0.73), (x + w / 2, y + d / 2, 0.755), mats['laminate']))
        for sx in (-1, 1):
            add(box(col, f'Site / office desk {k} leg {sx}', (x + sx * (w / 2 - 0.05) - 0.02, y - d / 2 + 0.05, 0), (x + sx * (w / 2 - 0.05) + 0.02, y + d / 2 - 0.05, 0.73), mats['ivory']))
        add(box(col, f'Site / office desk {k} monitor', (x - 0.27, y + d / 2 - 0.12, 0.85), (x + 0.27, y + d / 2 - 0.10, 1.17), mats['screen']))
        add(box(col, f'Site / office desk {k} monitor stand', (x - 0.02, y + d / 2 - 0.10, 0.755), (x + 0.02, y + d / 2 - 0.08, 0.86), mats['black']))
        cy_ = y - d / 2 - 0.35
        add(box(col, f'Site / chair {k} seat', (x - 0.25, cy_ - 0.25, 0.42), (x + 0.25, cy_ + 0.25, 0.5), mats['chair']))
        add(box(col, f'Site / chair {k} back', (x - 0.23, cy_ - 0.3, 0.5), (x + 0.23, cy_ - 0.24, 1.1), mats['chair']))
        add(cylinder(col, f'Site / chair {k} post', (x, cy_, 0.06), 0.025, 0.36, mats['black'], n=12))
    sx_, sy_ = P['sofa_xy_m']
    add(box(col, 'Site / orange sofa seat', (sx_ - 0.9, sy_ - 0.45, 0.0), (sx_ + 0.9, sy_ + 0.45, 0.42), mats['sofa']))
    add(box(col, 'Site / orange sofa back', (sx_ - 0.9, sy_ + 0.25, 0.42), (sx_ + 0.9, sy_ + 0.45, 0.8), mats['sofa']))
    wbx, wby = P['workbench_xy_m']
    add(box(col, 'Site / workbench top', (wbx - 0.35, wby - 0.6, 0.86), (wbx + 0.35, wby + 0.6, 0.885), mats['laminate']))
    for sx in (-1, 1):
        for sy in (-1, 1):
            add(box(col, f'Site / workbench leg {sx} {sy}', (wbx + sx * 0.3 - 0.02, wby + sy * 0.55 - 0.02, 0), (wbx + sx * 0.3 + 0.02, wby + sy * 0.55 + 0.02, 0.86), mats['ivory']))
    add(box(col, 'Site / workbench case', (wbx - 0.2, wby - 0.15, 0.885), (wbx + 0.1, wby + 0.12, 0.96), mats['black']))
    for name, lo, hi in P.get('floor_boxes', []):
        add(box(col, 'Site / ' + name, lo, hi, mats['black']))
    # ---- robot cart under the mount and the arch with the D455 (mount frame -> world: x_w = y_m, y_w = 0.12 - x_m)
    m0 = np.array(P['mount_world_m'])

    def mw(p):
        return (m0[0] + p[1], m0[1] - p[0], m0[2] + p[2])
    cz = P.get('cart_top_z_m', m0[2])  # site_v6: the tray (0.77 m) is below the mounting point; earlier sites: equal
    (cfx, cbx), cw = P['cart_x_m'], P['cart_halfwidth_m']
    for sxm in (-1, 1):
        for xm in (cfx, cbx):
            lo, hi = mw((xm, sxm * cw, -cz)), mw((xm, sxm * cw, -0.01))
            add(box(col, f'Site / cart post {xm:+.2f} {sxm}', (min(lo[0], hi[0]) - 0.02, min(lo[1], hi[1]) - 0.02, 0.08), (max(lo[0], hi[0]) + 0.02, max(lo[1], hi[1]) + 0.02, cz - 0.01), mats['alu']))
    for z in (0.12, cz - 0.03):
        a, b = mw((cbx, -cw, 0)), mw((cfx, cw, 0))
        add(box(col, f'Site / cart rail z{z:.2f}', (min(a[0], b[0]) - 0.02, min(a[1], b[1]) - 0.02, z - 0.02), (max(a[0], b[0]) + 0.02, max(a[1], b[1]) + 0.02, z), mats['alu']))
    a, b = mw((cbx + 0.03, -cw + 0.03, 0)), mw((cfx - 0.03, cw - 0.03, 0))
    add(box(col, 'Site / controller box', (min(a[0], b[0]), min(a[1], b[1]), 0.14), (max(a[0], b[0]), max(a[1], b[1]), 0.62), mats['black']))
    e = mw((cbx + 0.06, cw - 0.04, 0))  # e-stop on the cart's left rear corner
    add(box(col, 'Site / e-stop box', (e[0] - 0.04, e[1] - 0.04, cz - 0.03), (e[0] + 0.04, e[1] + 0.04, cz + 0.05), mats['yellow']))
    add(cylinder(col, 'Site / e-stop button', (e[0], e[1], cz + 0.05), 0.022, 0.02, mats['red'], n=20))
    for sxm in (-1, 1):
        for xm in (cfx, cbx):
            w = mw((xm, sxm * cw, 0))
            add(cylinder(col, f'Site / cart wheel {xm:+.2f} {sxm}', (w[0], w[1], 0), 0.04, 0.08, mats['black'], n=16))
    ax, ar, az = P['arch_x_m'], P['arch_radius_m'], P['d455_mount_m'][2] - P['arch_radius_m'] - 0.02
    pts = [mw((ax, ar * math.cos(t), az + ar * math.sin(t))) for t in np.linspace(0, math.pi, 41)]
    pts = [mw((ax, ar, cz - m0[2]))] + pts + [mw((ax, -ar, cz - m0[2]))]
    if P.get('riser_mount_frame_m'):  # site_v6: placeholder for the user's frame between the tray and the mount
        (rx0, rx1), (ry0, ry1) = P['riser_mount_frame_m']
        a, b = mw((rx0, ry0, cz - m0[2])), mw((rx1, ry1, -0.0308))
        add(box(col, 'Site / riser frame (placeholder)', (min(a[0], b[0]), min(a[1], b[1]), a[2]),
                (max(a[0], b[0]), max(a[1], b[1]), b[2]), mats['alu']))
    add(tube_path(col, 'Site / camera arch', pts, 0.018, mats['white_plastic']))
    dp = mw(P['d455_mount_m'])
    add(box(col, 'Site / D455 housing', (dp[0] - 0.062, dp[1] - 0.005, dp[2] + 0.012), (dp[0] + 0.062, dp[1] + 0.024, dp[2] + 0.04), mats['d455']))
    # ---- lights: named area lights over the site (the lighting recipe perturbs these); a dim world
    for o in [o for o in scene.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    for name, pos, target, power, size in P['lights']:
        data = bpy.data.lights.new(name, 'AREA')
        data.shape = 'RECTANGLE'
        data.size, data.size_y = size
        data.energy = power
        o = bpy.data.objects.new(name, data)
        col.objects.link(o)
        o.location = pos
        o.rotation_euler = (Vector(target) - Vector(pos)).to_track_quat('-Z', 'Y').to_euler()
    world = scene.world
    for n in world.node_tree.nodes:
        if n.type == 'BACKGROUND':
            n.inputs['Strength'].default_value = P['world_strength']
    scene.view_settings.exposure = P['exposure_ev']
    view_layer.update()
    joined = join_static(scene)
    view_layer.update()
    return dict(table=table, removed_s1_room_objects=len(removed), carpet_tiles=tiles, created_objects=len(created),
                joined_static_mesh_objects=joined, created=created)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--parameters', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    source, out = a.source.resolve(), a.output.resolve()
    assert not out.exists()
    src_sha = sha(source)
    P = json.loads(a.parameters.read_text())
    bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
    scene = bpy.data.scenes['Scene']
    bpy.context.window.scene = scene
    for o in scene.objects:
        o.update_tag()
    bpy.context.view_layer.update()
    info = build(scene, P)
    out.parent.mkdir(parents=True, exist_ok=True)
    # image files travel with the scene: copied unchanged to <out dir>/textures, referenced relatively
    deps = {}
    (out.parent / 'textures').mkdir(exist_ok=True)
    for im in bpy.data.images:
        if im.source != 'FILE' or not im.filepath or not im.users:
            continue
        src = Path(bpy.path.abspath(im.filepath, library=im.library)).resolve()
        dst = out.parent / 'textures' / src.name
        dst.write_bytes(src.read_bytes())
        im.filepath = '//textures/' + src.name
        deps['textures/' + src.name] = sha(dst)
    bpy.ops.wm.save_as_mainfile(filepath=str(out), compress=True, relative_remap=False)
    (out.parent / 'site_parameters.json').write_bytes(a.parameters.read_bytes())  # frozen copy used by make_config
    receipt = dict(schema='rssr.fr3_duo.site_scene.v1', source=str(source), source_sha256=src_sha,
                   source_unchanged=sha(source) == src_sha, parameters=str(a.parameters), parameters_sha256=sha(a.parameters),
                   code_sha256=sha(__file__), derived_blend=str(out), derived_sha256=sha(out), asset_dependencies=deps,
                   blender_version=bpy.app.version_string, lights=light_rows(scene),
                   **{k: v for k, v in info.items() if k != 'created'}, created=info['created'])
    out.with_suffix('.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('derived_sha256', 'carpet_tiles', 'created_objects', 'removed_s1_room_objects')}))


if __name__ == '__main__':
    main()
