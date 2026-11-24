"""Render one admitted FR3 Duo site episode (site_export.py source) in the site scene: head (D455), left and right
wrist (D405), 30 fps, the episode's complete clock.

S1 conventions kept (s1_stack_cups_random300_r15_extra200_v2/renderer.py): Cycles, 12 samples with denoising, one
seeded lighting recipe per episode shared by all frames and cameras (existing area lights moved by <= 0.3 m and
scaled, one or two added area lights), per-camera exposure offsets, raw frames black-padded to 224 x 224, H.264
CRF 18 per camera, receipt with sha256 of every input and output. Differences: the raw frame is 224 x 168 (the
site cameras stream 4:3, 640 x 480; S1's are 16:9, 224 x 126); robot and objects come from the MuJoCo visual geometry
of the episode; each sim cup is drawn with the S1 measured paper cup (kraft outer, rolled lip), scaled to the sim
cup's rim diameter and height, at the sim cup body's pose.

    native_photo/bin/python render_site.py --source <export> --config <config.json> --output <fresh dir> \
        [--device OPTIX|CPU --samples 12 --start 0 --stop N --preview]
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from PIL import Image

RAW_W, RAW_H, OUT = 224, 168, 224
PAD = (OUT - RAW_H) // 2
CUP_RIM_R, CUP_H = 0.0372, 0.0848  # S1 measured paper cup: rolled-lip radius, height


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def principled(name, rgba, roughness, metallic=0.0, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    s = m.node_tree.nodes['Principled BSDF']
    s.inputs['Base Color'].default_value = (*rgba[:3], 1)
    s.inputs['Roughness'].default_value = roughness
    s.inputs['Metallic'].default_value = metallic
    s.inputs['Coat Weight'].default_value = coat
    return m


def robot_material(rec, cache):
    """FR3 / 2F-85 / mount materials from the MuJoCo colour: white lacquer, black satin, aluminium."""
    r, g, b = rec['rgba'][:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum > 0.7:
        key, args = 'robot white lacquer', ((0.80, 0.80, 0.79), 0.28, 0.0, 0.35)
    elif lum < 0.2:
        key, args = 'robot black satin', ((0.018, 0.018, 0.02), 0.5, 0.0, 0.1)
    else:
        key, args = 'robot brushed aluminium', ((r, g, b), 0.35, 0.8, 0.0)
    if key not in cache:
        cache[key] = principled('Site robot / ' + key, *args)
    return cache[key]


def mesh_from_record(rec, mat, collection):
    t, s = rec['type'], rec['size']
    if t == 'mesh':
        me = bpy.data.meshes.new(rec['name'])
        me.from_pydata(rec['verts'], [], rec['faces'])
        me.update()
        for p in me.polygons:
            p.use_smooth = len(rec['faces']) > 200
        ob = bpy.data.objects.new(rec['name'], me)
        collection.objects.link(ob)
    else:
        if t == 'box':
            bpy.ops.mesh.primitive_cube_add(size=2.0)
            ob = bpy.context.active_object
            ob.scale = s
        elif t in ('cylinder', 'capsule'):
            bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=s[0], depth=2 * s[1])
            ob = bpy.context.active_object
        elif t == 'sphere':
            bpy.ops.mesh.primitive_uv_sphere_add(radius=s[0])
            ob = bpy.context.active_object
        else:
            raise ValueError(t)
        for c in ob.users_collection:
            c.objects.unlink(ob)
        collection.objects.link(ob)
        ob.name = rec['name']
    ob.data.materials.append(mat)
    ob.rotation_mode = 'QUATERNION'
    ob['mj_body'] = rec['body']
    return ob


class SiteScene:
    def __init__(self, source, config, samples, device):
        self.src = Path(source)
        self.meta = json.loads((self.src / 'source.json').read_text())
        for f, h in self.meta['artifacts'].items():
            assert sha(self.src / f) == h, f
        self.cfg = json.loads(Path(config).read_text())
        blend = Path(self.cfg['site_blend'])
        assert sha(blend) == self.cfg['site_blend_sha256']
        for rel, h in self.cfg['site_dependencies'].items():
            assert sha(blend.parent / rel) == h, rel
        bpy.ops.wm.open_mainfile(filepath=str(blend), load_ui=False)
        self.scene = sc = bpy.data.scenes['Scene']
        bpy.context.window.scene = sc
        self.base_exposure = float(self.cfg.get('exposure_ev', sc.view_settings.exposure))
        geometry = json.loads((self.src / 'geometry.json').read_text())
        self.pose = dict(np.load(self.src / 'poses.npz'))
        cams = json.loads((self.src / 'cameras.json').read_text())
        col = bpy.data.collections.new('20 Episode robot and objects')
        sc.collection.children.link(col)
        cup_geoms = {g for o in self.meta['objects'] if o['shape'] == 'cup' for g in o['geoms']}
        cache = {}
        object_geoms = {g for o in self.meta['objects'] for g in o['geoms']}
        self.geoms = []
        for j, rec in enumerate(geometry):
            if rec['id'] in cup_geoms or rec['name'].startswith('worktable'):
                continue  # cups: S1 paper cups below; desk: the site desk model
            if rec['id'] in object_geoms:
                mat = principled('Site object / ' + rec['name'], rec['rgba'], 0.5)
            else:
                mat = robot_material(rec, cache)
            self.geoms.append((j, mesh_from_record(rec, mat, col)))
        # S1 measured paper cups, one assembly per sim cup
        cup_blend = Path(self.cfg['cup_blend'])
        assert sha(cup_blend) == self.cfg['cup_blend_sha256']
        self.cups = []
        cups = [(i, o) for i, o in enumerate(self.meta['objects']) if o['shape'] == 'cup']
        for k, (bi, o) in enumerate(cups):
            names = ['left_paper_cup', 'left_paper_cup_rolled_lip'] if k % 2 == 0 else ['right_paper_cup', 'right_paper_cup_rolled_lip']
            with bpy.data.libraries.load(str(cup_blend), link=False) as (avail, loaded):
                loaded.objects = names
            parts = []
            for ob in loaded.objects:
                ob.name = f'Site cup {o["id"]} / {ob.name}'
                col.objects.link(ob)
                parts.append(ob)
            pts = np.array([tuple(ob.matrix_basis @ Vector(v)) for ob in parts for v in ob.bound_box])
            lo, hi = pts.min(0), pts.max(0)
            bottom = Vector(((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]))
            r_sim, h_sim = o['size_m'][0] / 2, o['size_m'][2]
            S = Matrix.Diagonal((r_sim / CUP_RIM_R, r_sim / CUP_RIM_R, h_sim / (hi[2] - lo[2]), 1.0))
            local = Matrix.Translation(Vector(o['bottom_in_body_m'])) @ S @ Matrix.Translation(-bottom)
            self.cups.append((bi, [(ob, local @ ob.matrix_basis.copy()) for ob in parts]))
        # Cycles bakes the transform into a mesh used by one object only, so a moving single-user mesh has its BVH
        # rebuilt every render; a second user (out of view, 50 m under the floor) makes it an instance that moves by
        # its transform alone
        anchors = bpy.data.collections.new('21 Instance anchors (50 m below the floor)')
        sc.collection.children.link(anchors)
        for ob in [o for _, o in self.geoms] + [o for _, parts in self.cups for o, _ in parts]:
            if ob.type == 'MESH' and ob.data.users == 1 and not os.environ.get('RSSR_NO_ANCHORS'):
                a = bpy.data.objects.new(ob.name + ' / anchor', ob.data)
                a.location = (0.0, 0.0, -50.0)
                anchors.objects.link(a)
        # cameras: vertical FOV, 4:3 raw frame
        self.cams = []
        for name, fovy in zip(cams['names'], cams['fovy_deg']):
            cd = bpy.data.cameras.new('Site cam ' + name)
            cd.sensor_fit = 'VERTICAL'
            cd.angle_y = math.radians(fovy)
            cd.clip_start = 0.005
            cd.clip_end = 60
            co = bpy.data.objects.new('Site cam ' + name, cd)
            co.rotation_mode = 'QUATERNION'
            sc.collection.objects.link(co)
            self.cams.append((name, co))
        self.camera_names = cams['names']
        sc.render.engine = 'CYCLES'
        sc.cycles.samples = samples
        sc.cycles.use_denoising = True
        sc.cycles.use_adaptive_sampling = True
        sc.cycles.adaptive_threshold = 0.04
        sc.render.use_persistent_data = True
        # robot and objects move every frame: keep per-object BVHs and update instance transforms only
        sc.cycles.debug_bvh_type = os.environ.get('RSSR_BVH', 'DYNAMIC_BVH')
        sc.render.threads_mode = 'FIXED'
        sc.render.threads = int(os.environ.get('RSSR_RENDER_THREADS', 4))
        if device == 'OPTIX':
            p = bpy.context.preferences.addons['cycles'].preferences
            p.compute_device_type = 'OPTIX'
            p.get_devices()
            used = []
            for dv in p.devices:
                dv.use = dv.type == 'OPTIX'
                if dv.use:
                    used.append(dv.name)
            assert len(used) == 1, f'expose exactly one GPU, got {used}'
            sc.cycles.device = 'GPU'
        else:
            sc.cycles.device = 'CPU'
            sc.cycles.denoising_use_gpu = False
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = RAW_W, RAW_H, 100
        sc.render.image_settings.file_format = 'PNG'
        sc.render.image_settings.color_mode = 'RGB'
        sc.render.film_transparent = False
        self.light_records = self.apply_lighting(self.cfg.get('lighting_recipe'))

    def apply_lighting(self, recipe):
        lights = {o.name: o for o in self.scene.objects if o.type == 'LIGHT'}
        if recipe:
            assert set(recipe) == {'seed', 'lights', 'added_lights'}
            assert {v['name'] for v in recipe['lights']} == set(lights)
            for v in recipe['lights']:
                o = lights[v['name']]
                assert 0.6 <= v['power_scale'] <= 1.8 and 0.7 <= v['size_scale'] <= 1.4
                assert np.max(np.abs(np.array(v['position_m']) - np.array(o.location))) <= 0.3 + 1e-8
                o.data = o.data.copy()
                o.location = v['position_m']
                o.rotation_euler = (Vector(v['target_m']) - o.location).to_track_quat('-Z', 'Y').to_euler()
                o.data.energy *= v['power_scale']
                o.data.color = v['color']
                o.data.size *= v['size_scale']
                if o.data.shape == 'RECTANGLE':
                    o.data.size_y *= v['size_scale']
            assert 1 <= len(recipe['added_lights']) <= 2
            for v in recipe['added_lights']:
                assert v['name'].startswith('site_added_') and v['name'] not in lights and 0 < v['power_w'] <= 100
                data = bpy.data.lights.new(v['name'], 'AREA')
                data.shape = 'DISK'
                data.energy = v['power_w']
                data.color = v['color']
                data.size = v['size_m']
                o = bpy.data.objects.new(v['name'], data)
                self.scene.collection.objects.link(o)
                o.location = v['position_m']
                o.rotation_euler = (Vector(v['target_m']) - o.location).to_track_quat('-Z', 'Y').to_euler()
        rows = []
        for o in self.scene.objects:
            if o.type == 'LIGHT':
                rows.append(dict(name=o.name, position_m=list(o.location), energy_w=o.data.energy, color=list(o.data.color)))
        return rows

    def set_frame(self, k):
        P = self.pose
        for j, ob in self.geoms:
            ob.location = P['geom_pos'][k, j]
            ob.rotation_quaternion = P['geom_quat'][k, j]
        for bi, parts in self.cups:
            T = Matrix.Translation(Vector(P['body_pos'][k, bi])) @ Quaternion(P['body_quat'][k, bi]).to_matrix().to_4x4()
            for ob, local in parts:
                ob.matrix_world = T @ local
        for i, (name, co) in enumerate(self.cams):
            co.location = P['cam_pos'][k, i]
            co.rotation_quaternion = P['cam_quat'][k, i]

    def render(self, k, ci, path):
        self.set_frame(k)
        name, co = self.cams[ci]
        off = float(self.cfg.get('camera_exposure_offset_ev', {}).get(name, 0.0))
        assert -3 <= off <= 3
        if self.scene.camera != co:  # assigned only on a change: any assignment re-syncs the whole scene
            self.scene.camera = co
            self.scene.view_settings.exposure = self.base_exposure + off
        self.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        return Image.open(path).convert('RGB').copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=('OPTIX', 'CPU'), default='OPTIX')
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--stop', type=int)
    ap.add_argument('--preview', type=int, nargs='*', help='only these frames, PNGs at 4x raw size, no video')
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    assert not (a.output / 'render_receipt.json').exists()
    t0 = time.monotonic()
    inputs = {str(a.source / 'source.json'): sha(a.source / 'source.json'), str(a.config): sha(a.config)}
    S = SiteScene(a.source, a.config, a.samples, a.device)
    n = S.meta['frames']
    if a.preview is not None:
        sc = S.scene
        sc.render.resolution_percentage = 400
        for k in a.preview or [0]:
            for ci, name in enumerate(S.camera_names):
                S.render(k, ci, a.output / f'preview_{k:05d}_{name}.png')
        bpy.ops.wm.save_as_mainfile(filepath=str(a.output / 'preview_scene.blend'), compress=True)
        print(json.dumps(dict(preview=str(a.output), elapsed_s=time.monotonic() - t0)))
        return
    frames = list(range(a.start, min(a.stop or n, n)))
    import shutil as _shutil; ffmpeg = os.environ.get('REEMBODY_FFMPEG') or os.environ.get('RSSR_FFMPEG') or _shutil.which('ffmpeg') or 'ffmpeg'
    enc = [subprocess.Popen([ffmpeg, '-v', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size',
                             f'{OUT}x{OUT}', '-framerate', '30', '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
                             '-crf', '18', '-pix_fmt', 'yuv420p', '-threads', '1', str(a.output / f'{c}.mp4')],
                            stdin=subprocess.PIPE) for c in S.camera_names]
    try:
        # one pass per camera: within a pass only the robot and the objects move between renders
        for ci, name in enumerate(S.camera_names):
            for count, k in enumerate(frames):
                raw = S.render(k, ci, a.output / 'current.png')
                assert raw.size == (RAW_W, RAW_H)
                im = Image.new('RGB', (OUT, OUT), (0, 0, 0))
                im.paste(raw, (0, PAD))
                enc[ci].stdin.write(np.asarray(im).tobytes())
                if k in (frames[0], frames[-1]):
                    im.save(a.output / f'frame_{k:06d}_{name}.png')
                if count % 10 == 9 or count == len(frames) - 1:
                    (a.output / 'progress.json').write_text(json.dumps(dict(
                        renders_completed=ci * len(frames) + count + 1, renders_expected=3 * len(frames),
                        camera=name, elapsed_s=time.monotonic() - t0)))
            enc[ci].stdin.close()
            assert enc[ci].wait(timeout=120) == 0
    except BaseException:
        for e in enc:
            if e.poll() is None:
                e.kill()
                e.wait()
        raise
    assert all(sha(p) == h for p, h in inputs.items())
    receipt = dict(schema='rssr.fr3_duo.site_render.v1', source=str(a.source), variation=S.meta['variation'],
                   frames=frames, fps=30, resolution=[OUT, OUT], raw_resolution=[RAW_W, RAW_H],
                   raster_policy=f'raw {RAW_W}x{RAW_H} (4:3, the site cameras stream 640x480); black-pad {PAD}px top/bottom to {OUT}x{OUT}',
                   cameras=S.camera_names, samples=a.samples, device=a.device, blender=bpy.app.version_string,
                   site_blend=S.cfg['site_blend'], site_blend_sha256=S.cfg['site_blend_sha256'],
                   cup_blend_sha256=S.cfg['cup_blend_sha256'], lighting_recipe=S.cfg.get('lighting_recipe'),
                   camera_exposure_offset_ev=S.cfg.get('camera_exposure_offset_ev'), lights=S.light_records,
                   input_pins=inputs, elapsed_s=time.monotonic() - t0,
                   artifacts={p.name: sha(p) for p in a.output.iterdir() if p.suffix in ('.png', '.mp4') and p.name != 'current.png'})
    (a.output / 'render_receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(dict(rendered=str(a.output), frames=len(frames), elapsed_s=time.monotonic() - t0)))


if __name__ == '__main__':
    main()
