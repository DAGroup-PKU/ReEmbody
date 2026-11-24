"""Centred-layout drawer renderer (copy of s1_drawer_v1/renderer_quaternion_drawer_v9b_uniform_upright.py,
unchanged there).  Same blend, appearance recipe, lights, materials, raster and
padding; cameras are the SDK nominal (image-up) mounts carried by the native
scene; geometry checks bind the centred cabinet layout.

Original docstring: Uniform cabinet v9b for newly verified upright-block native motion.

Render upright 224x126 content at the calibrated reference aspect, then place
it in a 224x224 RGB frame with 49 black pixels above and below. The pinned
desk/background blend is adjusted in memory; one rigid registration covers all
native geometry and sensor cameras. Phone photos guide appearance only: the S1
camera calibration remains attached to the native robot FK.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 's1_drawer_v1'))
import bpy
from mathutils import Matrix,Vector
from camera import camera_adjustment,fovy
from environment import apply_overrides,apply_style_fit
from drawer_table_v2 import fit_desk,registration_matrix
import numpy as np
from PIL import Image

APPEARANCE_SHA256 = '06e97d4c51d5afa786ac970b68809250e6e85c5984a25d1c5dcf0f9396d4c0c7'
PROVENANCE_VISUAL_PREVIEW_SHA256 = '2768b717b9879585885aea759675034ca1d17d7b7fde0949c156a0d4fc53f231'
APPEARANCE_PATH = Path(__file__).resolve().parents[1] / 's1_drawer_v1' / 'drawer_appearance_v9b_uniform_cabinet_upright.json'
VISUAL_FRONTS = {'bottom_drawer_back', 'middle_drawer_front_geom',
                 'top_drawer_front_geom', 'lowest_drawer_front_geom'}
VISUAL_HANDLES = {'bottom_handle_bar_geom', 'middle_handle_geom',
                  'top_handle_geom', 'lowest_handle_geom'}
HIDDEN_NATIVE_HANDLE_POSTS = {'bottom_handle_post_a_geom', 'bottom_handle_post_b_geom'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def transform(position,rotation):
    a=np.eye(4);a[:3,:3]=np.asarray(rotation).reshape(3,3);a[:3,3]=position;return Matrix(a.tolist())
def material(name,color,roughness,metallic=0):
    m=bpy.data.materials.new(name);m.use_nodes=True;s=m.node_tree.nodes.get('Principled BSDF')
    s.inputs['Base Color'].default_value=(*color,1);s.inputs['Roughness'].default_value=roughness;s.inputs['Metallic'].default_value=metallic;return m


def u_handle_mesh(appearance):
    """A render-only rounded U at the native handle pose, open toward its drawer face."""
    span=float(appearance['handle_half_span_m']);y0=float(appearance['handle_mount_y_m'])
    y1=float(appearance['handle_bar_y_m']);radius=float(appearance['handle_tube_radius_m'])
    path=[(-span,y0),(-span,y1*.40),(-span*.96,y1*.78),(-span*.86,y1),
          (span*.86,y1),(span*.96,y1*.78),(span,y1*.40),(span,y0)]
    vertices=[];faces=[];sides=10
    for i,(x,y) in enumerate(path):
        a=np.array(path[max(0,i-1)],float);b=np.array(path[min(len(path)-1,i+1)],float)
        tangent=b-a;tangent/=np.linalg.norm(tangent)
        normal=np.array([-tangent[1],tangent[0]])
        for j in range(sides):
            theta=2*math.pi*j/sides
            vertices.append((x+radius*normal[0]*math.cos(theta),
                             y+radius*normal[1]*math.cos(theta),radius*math.sin(theta)))
    for i in range(len(path)-1):
        for j in range(sides):
            n=(j+1)%sides;faces.append((i*sides+j,i*sides+n,(i+1)*sides+n,(i+1)*sides+j))
    faces += [tuple(reversed(range(sides))),tuple((len(path)-1)*sides+j for j in range(sides))]
    return vertices,faces


def native_mesh(spec,geo,mat,scene,appearance):
    k=spec['key'];name=spec['geom_name'];mesh=bpy.data.meshes.new('S1_'+name)
    custom=False
    if name in VISUAL_HANDLES:
        vertices,faces=u_handle_mesh(appearance);custom=True
    else:
        vertices=np.asarray(geo[k+'_v'],float).copy()
        faces=geo[k+'_f'].tolist()
        if name in VISUAL_FRONTS:
            native_height=float(np.ptp(vertices[:,2]));target=float(appearance['front_height_m'])
            assert abs(native_height-.072181180202)<2e-5 and .082<target<.085
            vertices[:,2]*=target/native_height
            if name=='bottom_drawer_back':
                native_width=float(np.ptp(vertices[:,0]));target_width=float(appearance['front_width_m'])
                assert abs(native_width-.291818181818)<2e-5
                vertices[:,0]*=target_width/native_width
                vertices[:,1]+=float(appearance['moving_front_local_y_shift_m'])
            custom=True
        vertices=vertices.tolist()
    mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new('S1_'+spec['geom_name'],mesh);obj.rotation_mode='QUATERNION';scene.collection.objects.link(obj);mesh.materials.append(mat)
    normal=geo[k+'_n']
    if custom:
        for p in mesh.polygons:p.use_smooth=name in VISUAL_HANDLES
    elif len(normal):
        for p in mesh.polygons:p.use_smooth=True
        mesh.normals_split_custom_set(normal.reshape(-1,3).tolist())
    if name in HIDDEN_NATIVE_HANDLE_POSTS:obj.hide_render=True
    return obj


def visual_only_box(scene,name,center_native,size_native,material_obj,world_matrix):
    """A Blender-only façade trim; it has no source mesh or MuJoCo collision."""
    hx,hy,hz=(float(v)/2 for v in size_native)
    vertices=[(x,y,z) for x in (-hx,hx) for y in (-hy,hy) for z in (-hz,hz)]
    faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);scene.collection.objects.link(obj)
    mesh.materials.append(material_obj)
    obj.matrix_world=world_matrix@transform(center_native,np.eye(3))
    return obj


def validate_native_task_geometry(meta, geo, pose, contract):
    """Reject a source whose actual t0 meshes differ from the new user geometry."""
    objects = meta['objects']
    shell = [i for i, spec in enumerate(objects)
             if spec['body_name'] == 'drawer_cabinet' and
             spec['geom_name'].startswith('drawer_cabinet_')]
    if len(shell) != 5:
        raise ValueError('Expected the five corrected cabinet shell geoms')
    block = [i for i, spec in enumerate(objects)
             if spec['geom_name'] == 'red_block_geom' and spec['body_name'] == 'red_block']
    if len(block) != 1:
        raise ValueError('Expected one corrected pink task block')

    def points(index):
        spec = objects[index]
        vertices = np.asarray(geo[spec['key'] + '_v'], dtype=float)
        rotation = np.asarray(pose['geom_rotation'][0, index], dtype=float).reshape(3, 3)
        position = np.asarray(pose['geom_position'][0, index], dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or \
                rotation.shape != (3, 3) or not np.isfinite(rotation).all() or \
                position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError('Invalid exported task mesh or native pose')
        return vertices @ rotation.T + position

    cabinet_points = np.concatenate([points(i) for i in shell])
    cabinet_low = cabinet_points.min(axis=0)
    cabinet_high = cabinet_points.max(axis=0)
    cabinet_size = cabinet_high - cabinet_low
    cabinet_center = (cabinet_low + cabinet_high) / 2
    expected = contract['cabinet']
    if not np.allclose(cabinet_size, expected['size_xyz_m'], atol=2e-4, rtol=0):
        raise ValueError('Cabinet shell is not 35 x 26.5 x 35 cm (or not yaw 0)')
    if np.any(np.abs(cabinet_center[:2] - np.asarray(expected['nominal_center_xy_m'])) >
              float(expected['placement_jitter_uniform_m']) + 2e-4):
        raise ValueError('Cabinet is not centred on the robot forward axis within the declared jitter')
    side = 'center'
    rear_gap = float(cabinet_low[1] - contract['table']['rear_edge_y_m'])

    expected_tiers = ('lowest_drawer_front', 'bottom_drawer', 'middle_drawer_front', 'top_drawer_front')
    expected_handles = ('lowest_handle', 'bottom_handle_bar', 'middle_handle', 'top_handle')
    expected_z = np.asarray([.84405, .93155, 1.01905, 1.10655], dtype=float)
    if contract['cabinet'].get('tier_order_from_bottom') != list(expected_tiers) or \
            contract['cabinet'].get('tier_handle_alignment') != 'centered' or \
            not np.isclose(contract['cabinet'].get('tier_pitch_m', -1), .0875, atol=1e-8):
        raise ValueError('Four equal cabinet tiers are not contract-pinned')
    tier_centers = []
    handle_centers = []
    for tier, handle in zip(expected_tiers, expected_handles):
        ids = [i for i, spec in enumerate(objects) if spec['body_name'] == tier]
        handle_ids = [i for i, spec in enumerate(objects) if spec['body_name'] == handle]
        if not ids or not handle_ids:
            raise ValueError(f'Missing four-tier front or handle: {tier}, {handle}')
        tier_points = np.concatenate([points(i) for i in ids])
        handle_points = np.concatenate([points(i) for i in handle_ids])
        tier_centers.append(float((tier_points[:, 2].min()+tier_points[:, 2].max())/2))
        handle_centers.append(float((handle_points[:, 2].min()+handle_points[:, 2].max())/2))
    if not np.allclose(tier_centers, expected_z, atol=2e-4, rtol=0) or \
            not np.allclose(np.diff(tier_centers), .0875, atol=2e-4, rtol=0) or \
            not np.allclose(handle_centers, tier_centers, atol=2e-4, rtol=0):
        raise ValueError('Four equal native drawer tiers or centered handles differ from contract')

    block_index = block[0]
    block_spec = objects[block_index]
    block_vertices = np.asarray(geo[block_spec['key'] + '_v'], dtype=float)
    block_size = np.ptp(block_vertices, axis=0)
    if not np.allclose(block_size, contract['block']['edge_m'], atol=2e-5, rtol=0):
        raise ValueError('Pink cube is not 5 cm on each local axis')
    if not np.allclose(block_spec['rgba'], contract['block']['rgba_linear'], atol=2e-5, rtol=0):
        raise ValueError('Pink cube material differs from the stack-block reference')
    block_center = np.asarray(pose['geom_position'][0, block_index], dtype=float)
    block_radius = float(np.linalg.norm(block_center[:2] - np.asarray(contract['block']['region_center_xy_m'])))
    if block_radius > float(contract['block']['center_region_radius_m']) + 5e-4:
        raise ValueError('Pink cube center is outside the centred-layout block disk')
    tabletop_z = float(contract['table']['top_z_m'])
    block_bottom = float(points(block_index)[:, 2].min())
    if abs(block_bottom - tabletop_z) > .005:
        raise ValueError('Pink cube does not rest on the corrected tabletop')
    return {
        'cabinet_side': side,
        'cabinet_min_m': cabinet_low.tolist(),
        'cabinet_max_m': cabinet_high.tolist(),
        'cabinet_center_m': cabinet_center.tolist(),
        'cabinet_size_m': cabinet_size.tolist(),
        'cabinet_rear_clearance_m': rear_gap,
        'tier_centers_z_m': tier_centers,
        'handle_centers_z_m': handle_centers,
        'block_local_size_m': block_size.tolist(),
        'block_rgba_linear': list(block_spec['rgba']),
        'block_center_native_m': block_center.tolist(),
        'block_center_disk_radius_m': block_radius,
        'block_bottom_native_z_m': block_bottom,
    }


class AssetScene:
    def __init__(self,source,config,samples=64,device='OPTIX',width=224,height=224,reference=False):
        self.source=Path(source);self.meta=json.loads((self.source/'source.json').read_text());self.cfg=json.loads(Path(config).read_text());self.reference=reference
        assert sha(APPEARANCE_PATH)==APPEARANCE_SHA256,'Visual-only appearance recipe changed'
        self.appearance=json.loads(APPEARANCE_PATH.read_text())
        assert self.appearance['schema']=='rssr.drawer.appearance.visual_only.v9b_uniform_cabinet_upright'
        assert self.cfg['appearance_source']==str(APPEARANCE_PATH) and self.cfg['appearance_source_sha256']==APPEARANCE_SHA256
        geometry=self.cfg.get('drawer_table_geometry')
        if self.cfg.get('schema')!='rssr.drawer.render_config.center_layout_sdk_cameras.v1' or not isinstance(geometry,dict) or \
                geometry.get('revision')!='user_desk_140x78_robotbase04_center_layout_v1':
            raise ValueError('Corrected drawer geometry config is required')
        contract_path=Path(geometry['geometry_contract'])
        if sha(contract_path)!=geometry['geometry_contract_sha256']:
            raise ValueError('Drawer geometry contract changed')
        contract=json.loads(contract_path.read_text())
        if contract.get('schema')!='rssr.drawer.center_layout_contract.v1' or \
                not np.allclose(contract['table']['center_xy_m'],self.meta['table_center_m'][:2],atol=2e-5,rtol=0) or \
                not np.allclose(np.asarray(contract['table']['size_xy_m'])/2,self.meta['table_half_size_m'][:2],atol=2e-5,rtol=0):
            raise ValueError('Render source does not match the corrected physical table contract')
        base_shift=contract['robot_base_initial_state_change']['translation_native_m']
        calibration=self.meta['sensor_calibration']
        if not np.allclose(base_shift,[0,.04,0],atol=1e-8,rtol=0) or \
                not np.allclose(calibration.get('native_base_translation_m',[]),base_shift,atol=1e-8,rtol=0) or \
                calibration.get('relative_camera_mounts_unchanged') is not True:
            raise ValueError('Source lacks the verified 4 cm native S1 base translation')
        for n,h in self.meta['artifacts'].items():assert sha(self.source/n)==h
        self.pose=dict(np.load(self.source/'poses.npz',allow_pickle=False));self.geo=np.load(self.source/'geometry.npz',allow_pickle=False)
        self.native_geometry_audit=validate_native_task_geometry(self.meta,self.geo,self.pose,contract)
        front_names={spec['geom_name'] for spec in self.meta['objects'] if not spec['is_robot']}
        assert VISUAL_FRONTS<=front_names and VISUAL_HANDLES<=front_names
        # In this source the moving drawer's visible +Y façade is named
        # bottom_drawer_back; bottom_drawer_front sits at the rear of the tray.
        def world_y(name):
            i=next(i for i,s in enumerate(self.meta['objects']) if s['geom_name']==name)
            s=self.meta['objects'][i];v=np.asarray(self.geo[s['key']+'_v']);r=self.pose['geom_rotation'][0,i].reshape(3,3)
            return float(np.mean((v@r.T+self.pose['geom_position'][0,i])[:,1]))
        assert world_y('bottom_drawer_back')>world_y('bottom_drawer_front')+.15
        assert abs(world_y('bottom_drawer_back')-world_y('middle_drawer_front_geom'))<.02
        asset=Path(self.cfg['asset_blend']);assert sha(asset)==self.cfg['asset_sha256']
        for relative,digest in self.cfg.get('asset_dependencies',{}).items():
            assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
            assert sha(asset.parent/relative)==digest,'Environment texture dependency changed'
        bpy.ops.wm.open_mainfile(filepath=str(asset),load_ui=False);self.scene=bpy.data.scenes[self.cfg.get('asset_scene','Scene')];bpy.context.window.scene=self.scene
        for obj in self.scene.objects:obj.update_tag()
        self.scene.frame_set(self.scene.frame_current);bpy.context.view_layer.update()
        self.asset_view_settings=dict(view_transform=self.scene.view_settings.view_transform,look=self.scene.view_settings.look,
            exposure_ev=self.scene.view_settings.exposure,gamma=self.scene.view_settings.gamma)
        old_cups=[]  # No cup assets are loaded for a drawer task.
        self.demo_removed=[]
        top=float(self.meta['table_center_m'][2]+self.meta['table_half_size_m'][2])
        self.table_baseline=float(self.cfg['table_baseline_m']);height_delta=self.cfg.get('height_delta_m',[-.05,.05])
        assert self.table_baseline+height_delta[0]-1e-6<=top<=self.table_baseline+height_delta[1]+1e-6,'Physical table outside configured baseline height augmentation range'
        self.world=registration_matrix(self.scene,self.meta['table_center_m'][:2])
        if 'world_registration_matrix' in self.cfg:
            self.world=np.asarray(self.cfg['world_registration_matrix'],float)
        chassis=[i for i,spec in enumerate(self.meta['objects'])
                 if spec['is_robot'] and spec['body_name']=='chassis_base']
        if len(chassis)!=1:raise ValueError('Expected one chassis-base visual geometry')
        robot_native=self.pose['geom_position'][0,chassis[0]]
        robot_world_x=float(-robot_native[0]+self.world[0,3])
        self.table_height_audit=fit_desk(self.scene,top_m=top,robot_world_x_m=robot_world_x,
            physical_table_center_xy_m=self.meta['table_center_m'][:2],
            physical_table_half_size_xy_m=self.meta['table_half_size_m'][:2],
            world_registration_matrix=self.world)
        self.table_top=top
        hidden=[]
        for name in ('Adjacent desk / cropped top','Adjacent desk / foot'):
            adjacent=self.scene.objects.get(name)
            if adjacent is None:
                raise ValueError('Pinned source blend lacks the adjacent-desk appearance object')
            adjacent.hide_render=True
            hidden.append(name)
        apply_overrides(self.scene,self.cfg.get('environment_overrides',{}))
        self.style_audit=apply_style_fit(self.scene,self.cfg.get('style_fit',{}))
        self.style_audit['adjacent_desk_hidden_from_drawer_views']=hidden
        self.base_exposure_ev=float(self.scene.view_settings.exposure)
        self.static_objects=[o for o in self.scene.objects if o.type!='CAMERA' and o not in old_cups]
        self.material_pins=self.material_inventory()
        # The single native registration was checked against the physical and
        # visual table centers before adding the cabinet, block, and cameras.
        assert self.world.shape==(4,4) and np.allclose(self.world[3],[0,0,0,1])
        self.world_m=Matrix(self.world.tolist())
        white=material('S1_native_white',[.70,.71,.72],.30);steel=material('S1_native_gripper_metal',[.42,.44,.46],.36,.78);black=material('S1_native_finger_pad',[.012,.014,.016],.72)
        self.native_objects=[]
        material_recipe=self.cfg.get('native_render_material_overrides',{})
        if set(material_recipe)!={'cabinet_linear_rgb_multiplier','block_linear_rgb_multiplier','cabinet_roughness','block_roughness'}:
            raise ValueError('Explicit native drawer appearance recipe is required')
        cabinet_gain=np.asarray(material_recipe['cabinet_linear_rgb_multiplier'],float)
        block_gain=np.asarray(material_recipe['block_linear_rgb_multiplier'],float)
        if cabinet_gain.shape!=(3,) or block_gain.shape!=(3,) or not np.isfinite(cabinet_gain).all() or \
                not np.isfinite(block_gain).all() or np.any(cabinet_gain<.4) or np.any(cabinet_gain>1.2) or \
                np.any(block_gain<.4) or np.any(block_gain>1.2):
            raise ValueError('Invalid drawer material color calibration')
        self.material_recipe=material_recipe
        body_rgb=np.asarray(self.appearance['cabinet_body_linear_rgb'],float)
        assert body_rgb.shape==(3,) and np.isfinite(body_rgb).all() and np.all((body_rgb>=0)&(body_rgb<=1))
        shell_material=None;top_material=None
        for i,spec in enumerate(self.meta['objects']):
            if spec['geom_name'].startswith('worktable'):continue
            body=spec['body_name']
            if spec['is_robot']:mat=white if 'gripper' not in body else steel if 'base' in body else black
            else:
                color=np.asarray(spec['rgba'][:3],float)
                is_block=spec['geom_name']=='red_block_geom'
                is_cabinet=spec['geom_name'].startswith('drawer_cabinet_') or \
                    'drawer' in body and 'handle' not in body
                if is_block:color=color*block_gain
                elif spec['geom_name'] in VISUAL_HANDLES:
                    color=np.asarray(self.appearance['handle_linear_rgb'],float)
                elif is_cabinet:
                    # Every shell, drawer and front is the same cabinet paint.
                    # Face illumination may differ, but material color may not.
                    color=body_rgb.copy()
                mat=material('Drawer_native_'+spec['geom_name'],color.tolist(),
                             float(material_recipe['block_roughness'] if is_block else
                                   material_recipe['cabinet_roughness'] if is_cabinet else .65))
                if spec['geom_name']=='drawer_cabinet_xplus':shell_material=mat
                if spec['geom_name']=='drawer_cabinet_top':top_material=mat
            self.native_objects.append((i,spec,native_mesh(spec,self.geo,mat,self.scene,self.appearance)))
        assert shell_material is not None and top_material is not None
        c=np.asarray(self.native_geometry_audit['cabinet_center_m'],float)
        max_y=float(self.native_geometry_audit['cabinet_max_m'][1])
        trim_width=float(self.appearance['visual_front_trim_width_m'])
        trim_depth=float(self.appearance['visual_front_trim_depth_m'])
        y=max_y+float(self.appearance['visual_front_trim_y_from_cabinet_front_m'])
        self.visual_front_trim=[]
        for sign in (-1,1):
            x=c[0]+sign*(float(self.appearance['front_width_m'])/2+trim_width/2)
            self.visual_front_trim.append(visual_only_box(self.scene,'Drawer_v9_front_trim_'+str(sign),
                (x,y,c[2]),(trim_width,trim_depth,.339),shell_material,self.world_m))
        bridge_y=max_y+float(self.appearance['visual_top_bridge_y_from_cabinet_front_m'])
        bridge_depth=float(self.appearance['visual_top_bridge_depth_m'])
        assert bridge_y-bridge_depth/2 <= max_y and bridge_y+bridge_depth/2 >= y
        self.visual_front_trim.append(visual_only_box(self.scene,'Drawer_v9_continuous_top_front_lip',
            (c[0],bridge_y,self.native_geometry_audit['cabinet_max_m'][2]-.004),
            (.35,bridge_depth,.008),shell_material,self.world_m))
        edge_width=float(self.appearance['visual_top_edge_strip_width_m'])
        edge_raise=float(self.appearance['visual_top_edge_strip_raise_m'])
        assert 0<edge_width<=.007 and 0<edge_raise<=.001
        for sign in (-1,1):
            self.visual_front_trim.append(visual_only_box(self.scene,'Drawer_v9_top_side_edge_'+str(sign),
                (c[0]+sign*(.175-edge_width/2),c[1],
                 self.native_geometry_audit['cabinet_max_m'][2]+edge_raise),
                (edge_width,.265,2*edge_raise),top_material,self.world_m))
        eave_height=float(self.appearance['visual_side_eave_height_m'])
        eave_outset=float(self.appearance['visual_side_eave_outset_m'])
        assert .005<=eave_height<=.015 and 0<eave_outset<=.001
        for sign in (-1,1):
            self.visual_front_trim.append(visual_only_box(self.scene,'Drawer_v9_top_side_eave_'+str(sign),
                (c[0]+sign*.175,c[1],
                 self.native_geometry_audit['cabinet_max_m'][2]-eave_height/2),
                (2*eave_outset,.265,eave_height),shell_material,self.world_m))
        calibrated=self.meta['sensor_calibration']['same_robot_sensor_calibration_applicable']
        alignment=self.cfg.get('camera_alignment',{}) if calibrated else {}
        if isinstance(alignment,str):alignment=json.loads(Path(alignment).read_text())
        self.camera_specs=alignment.get('cameras',{})
        for name,native_fov in zip(self.meta['camera_names'],self.meta['camera_fovy_deg']):
            spec=self.camera_specs.get(name,{})
            if 'reference_fovy_deg' in spec:
                assert abs(float(native_fov)-float(spec['reference_fovy_deg']))<1e-7,('Native camera FOV differs from frozen aligned mount',name,native_fov,spec['reference_fovy_deg'])
        self.camera_deltas=[camera_adjustment(self.camera_specs.get(n,{})) for n in self.meta['camera_names']]
        self.camera_fov=[fovy(self.camera_specs.get(n,{}),v,True) for n,v in zip(self.meta['camera_names'],self.meta['camera_fovy_deg'])]
        self.sensor_calibration_inherited=bool(calibrated)
        self.style_audit.update(apply_style_fit(self.scene,self.cfg.get('style_fit',{})))
        self.cameras=[]
        for name,vertical_fov in zip(self.meta['camera_names'],self.camera_fov):
            c=bpy.data.cameras.new('S1_'+name);o=bpy.data.objects.new('S1_'+name,c);o.rotation_mode='QUATERNION';self.scene.collection.objects.link(o)
            c.sensor_fit='VERTICAL';c.sensor_height=24;c.lens=12/math.tan(math.radians(vertical_fov)/2);c.clip_start=.005;c.clip_end=100;self.cameras.append(o)
        self.scene.render.engine='CYCLES';self.scene.cycles.samples=samples;self.scene.cycles.use_denoising=True
        self.scene.cycles.use_adaptive_sampling=True;self.scene.cycles.adaptive_threshold=.04
        self.scene.render.use_persistent_data=True;self.scene.render.threads_mode='FIXED';self.scene.render.threads=int(self.cfg.get('cpu_threads',8))
        if device=='OPTIX':
            p=bpy.context.preferences.addons['cycles'].preferences;p.compute_device_type='OPTIX';p.get_devices();selected=[]
            for d in p.devices:
                d.use=d.type=='OPTIX'
                if d.use:selected.append(d.name)
            assert len(selected)==1,'Expose exactly one idle GPU'
            self.scene.cycles.device='GPU'
        else:self.scene.cycles.device='CPU';self.scene.cycles.denoising_use_gpu=False
        self.scene.render.resolution_x=width;self.scene.render.resolution_y=height;self.scene.render.resolution_percentage=100
        self.scene.render.image_settings.file_format='PNG';self.scene.render.image_settings.color_mode='RGB';self.scene.render.film_transparent=False
        static_transforms={o.name:np.array(o.matrix_world).copy() for o in self.static_objects if o.type!='LIGHT'}
        self.apply_lighting(self.cfg.get('lighting_recipe'))
        assert all(np.array_equal(np.array(self.scene.objects[n].matrix_world),m) for n,m in static_transforms.items()),'Lighting API moved static environment geometry'
        self.set_frame(0)

    def material_inventory(self):
        result={}
        for obj in self.static_objects:
            if obj.type!='MESH':continue
            for mat in obj.data.materials:
                if mat is not None and mat.name not in result:
                    # Serialized shader graphs are retained within the immutable
                    # input blend; record material identity without rewriting it.
                    result[mat.name]=dict(nodes=len(mat.node_tree.nodes) if mat.use_nodes else 0,
                        diffuse_color=list(mat.diffuse_color),use_nodes=mat.use_nodes)
        return result

    def apply_lighting(self,recipe):
        lights=[o for o in self.scene.objects if o.type=='LIGHT']
        baseline_light_count=len(lights)
        self.light_records=[]
        if recipe:
            if set(recipe)!={'lights','added_lights','seed'}:raise ValueError('Only lights may be augmented')
            lookup={o.name:o for o in lights};assert set(lookup)=={v['name'] for v in recipe['lights']}
            for v in recipe['lights']:
                if set(v)!={'name','position_m','target_m','power_scale','color','size_scale'}:raise ValueError('Invalid lighting field')
                o=lookup[v['name']];assert .6<=v['power_scale']<=1.8 and .7<=v['size_scale']<=1.4
                pos=np.array(v['position_m'],float);target=np.array(v['target_m'],float)
                assert pos.shape==target.shape==(3,) and np.isfinite(pos).all() and np.isfinite(target).all()
                assert np.max(abs(pos-np.array(o.location)))<=.3+1e-8
                assert np.linalg.norm(target-pos)>.01
                assert np.allclose(v['color'],list(o.data.color),atol=1e-6),'Lighting augmentation changed a fitted baseline light color'
                o.data=o.data.copy();o.location=pos;o.rotation_euler=(Vector(v['target_m'])-o.location).to_track_quat('-Z','Y').to_euler()
                o.data.energy*=v['power_scale'];o.data.color=v['color'];o.data.size*=v['size_scale']
                if o.data.shape=='RECTANGLE':o.data.size_y*=v['size_scale']
        if recipe:
            for v in recipe['added_lights']:
                assert set(v)=={'name','position_m','target_m','power_w','color','size_m'}
                assert v['name'].startswith('s1_heightfit_added_') and v['name'] not in lookup
                assert 0<float(v['power_w'])<=100 and len(v['position_m'])==len(v['target_m'])==3
                data=bpy.data.lights.new(v['name'],'AREA');data.shape='DISK';data.energy=float(v['power_w']);data.color=v['color'];data.size=float(v['size_m'])
                o=bpy.data.objects.new(v['name'],data);self.scene.collection.objects.link(o);o.location=v['position_m']
                o.rotation_euler=(Vector(v['target_m'])-o.location).to_track_quat('-Z','Y').to_euler();lights.append(o)
            assert 1<=len(recipe['added_lights'])<=2
        assert baseline_light_count>=3
        assert len(lights)==baseline_light_count+(len(recipe['added_lights']) if recipe else 0)
        for o in lights:
            direction=o.rotation_euler.to_matrix()@Vector((0,0,-1))
            self.light_records.append(dict(name=o.name,position_m=list(o.location),target_m=list(o.location+direction*2),
                energy_w=o.data.energy,color=list(o.data.color),size_m=[o.data.size,o.data.size_y if o.data.shape=='RECTANGLE' else o.data.size]))
        assert self.material_inventory()==self.material_pins,'Lighting API changed non-light materials'

    def set_frame(self,index):
        for gi,spec,o in self.native_objects:o.matrix_world=self.world_m@transform(self.pose['geom_position'][index,gi],self.pose['geom_rotation'][index,gi])
        for i,o in enumerate(self.cameras):o.matrix_world=self.world_m@transform(self.pose['camera_position'][index,i],self.pose['camera_rotation'][index,i])@Matrix(self.camera_deltas[i].tolist())
        bpy.context.view_layer.update()

    def projection_error(self,index):
        from bpy_extras.object_utils import world_to_camera_view
        self.set_frame(index);w,h=self.scene.render.resolution_x,self.scene.render.resolution_y;worst=0.
        checked=excluded=0;pose_error=0.
        for ci,camera in enumerate(self.cameras):
            native=np.array(transform(self.pose['camera_position'][index,ci],self.pose['camera_rotation'][index,ci]))
            aligned=native@self.camera_deltas[ci];C=aligned[:3,3];R=aligned[:3,:3]
            expected_matrix=self.world@aligned;actual_matrix=np.array(camera.matrix_world)
            pose_error=max(pose_error,float(np.max(abs(expected_matrix-actual_matrix))))
            assert pose_error<1e-6,'Aligned fixed mount no longer follows native camera kinematics'
            measured_fovy=math.degrees(2*math.atan(camera.data.sensor_height/(2*camera.data.lens)))
            assert abs(measured_fovy-self.camera_fov[ci])<1e-5,'Frozen aligned FOV changed'
            f=h/(2*np.tan(np.deg2rad(self.camera_fov[ci])/2))
            for pi in self.meta['task_object_indices']:
                point=self.pose['geom_position'][index,pi]
                native_object=next((o for gi,spec,o in self.native_objects if gi==pi),None)
                assert native_object is not None,'Task geometry was not rendered'
                expected_object=self.world_m@transform(point,self.pose['geom_rotation'][index,pi])
                assert np.max(abs(np.array(native_object.matrix_world)-np.array(expected_object)))<1e-6,'Rendered drawer/block rigid pose changed'
                q=R.T@(point-C)
                if -q[2]<=camera.data.clip_start:excluded+=1;continue
                expected=np.array([w/2+f*q[0]/-q[2],h/2-f*q[1]/-q[2]])
                if not (0<=expected[0]<w and 0<=expected[1]<h):excluded+=1;continue
                world_point=self.world_m@Vector(point)
                uv=world_to_camera_view(self.scene,camera,world_point);actual=np.array([uv.x*w,(1-uv.y)*h])
                worst=max(worst,float(np.max(abs(actual-expected))));checked+=1
        # Visibility is reported explicitly; diagnostic views are not moved to fit task objects.
        assert worst<.003,('Native in-frame camera projection changed',worst)
        self.projection_scope=dict(in_frame_points_checked=checked,behind_or_offscreen_centers_excluded=excluded,
            all_camera_and_task_rigid_pose_matrices_checked=True,maximum_camera_matrix_error=pose_error,
            native_rigid_rotation_storage='QUATERNION',
            fov_tolerance_degrees=1e-5,in_frame_pixel_tolerance=.003)
        return worst

    def render(self,index,camera,path):
        self.set_frame(index)
        for gi,spec,obj in self.native_objects:obj.visible_camera=not(camera==0 and spec['head_occluder'])
        self.scene.camera=self.cameras[camera]
        name=self.meta['camera_names'][camera];response=self.cfg.get('camera_response',{}).get(name,{})
        offset=float(self.camera_specs.get(name,{}).get('exposure_offset_ev',response.get('exposure_offset_ev',0)))
        assert -3<=offset<=3;self.scene.view_settings.exposure=self.base_exposure_ev+offset
        self.scene.render.filepath=str(path);bpy.ops.render.render(write_still=True)
        return Image.open(path).convert('RGB').copy()

    def render_external_photo_diagnostic(self,index,path):
        """External-camera visual audit only; never an S1 observation or training view."""
        self.set_frame(index)
        for gi,spec,obj in self.native_objects:obj.visible_camera=not spec['is_robot']
        data=bpy.data.cameras.new('Drawer_v8_EXTERNAL_DIAGNOSTIC')
        camera=bpy.data.objects.new('Drawer_v8_EXTERNAL_DIAGNOSTIC',data)
        self.scene.collection.objects.link(camera)
        camera.location=(-0.08,-0.91,1.46)
        target=Vector((-0.08,0.10,0.98))
        camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
        data.sensor_fit='HORIZONTAL';data.sensor_width=36
        data.lens=36/(2*math.tan(math.radians(70)/2))
        data.clip_start=.005;data.clip_end=100
        self.scene.camera=camera
        self.scene.render.resolution_x=640;self.scene.render.resolution_y=480
        self.scene.view_settings.exposure=self.base_exposure_ev
        self.scene.render.filepath=str(path)
        bpy.ops.render.render(write_still=True)
        return dict(position_visual_world_m=list(camera.location),target_visual_world_m=list(target),
                    horizontal_fov_deg=70,resolution=[640,480],robot_hidden_for_external_photo_comparison=True,
                    training_camera=False)


def gpu_guard(args):
    """Admit one renderer on the named GPU, with local start leases.

    Shared mode only coexists with this same renderer and the same explicit
    share group. Other users' compute processes always block admission. The
    lock/lease closes the race before a second Cycles process allocates VRAM.
    """
    import atexit
    import ctypes
    import fcntl
    import re
    if args.device == 'CPU':
        assert not os.environ.get('CUDA_VISIBLE_DEVICES'), 'CPU mode must hide CUDA devices'
        return {'device':'CPU'}
    assert args.gpu_uuid and args.gpu_pci and args.gpu_index is not None
    assert re.fullmatch(r'[A-Za-z0-9_-]{4,80}', args.share_group or '') if args.allow_shared_gpu else not args.share_group
    assert 1 <= args.max_renderers_per_gpu <= 4
    assert 4096 <= args.reserve_mib <= 32768 and 4096 <= args.min_free_mib <= 131072
    assert 0 <= args.max_util_percent <= 100
    lease_root=Path('/tmp/rssr_drawer_v22_gpu_leases')/args.gpu_uuid
    lease_root.mkdir(parents=True,exist_ok=True)
    proc=Path('/proc')/str(os.getpid())
    start_ticks=proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
    lease_path=lease_root/(str(os.getpid())+'_'+start_ticks+'.json')
    with (lease_root/'admission.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        row=subprocess.check_output(['nvidia-smi','--id='+str(args.gpu_index),
            '--query-gpu=uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu,name',
            '--format=csv,noheader,nounits'],text=True).strip().split(',')
        uuid,pci,total,memory,free,util,name=[x.strip() for x in row]
        assert uuid==args.gpu_uuid and pci.lower()==args.gpu_pci.lower(),'NVML identity changed'
        assert name in ('NVIDIA B20Z','NVIDIA B200') or (name=='NVIDIA RPBZZZ6' and uuid in (
            'GPU-254e779d-602d-8448-b69e-06c9722de0ef')),'Device outside authorized drawer rendering resources'
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],text=True).splitlines()
        gpu_apps=[]
        for line in apps:
            columns=[part.strip() for part in line.split(',')]
            if len(columns)>=2 and columns[0]==uuid and columns[1].isdigit():
                gpu_apps.append(int(columns[1]))
        active=[]
        for path in lease_root.glob('*.json'):
            try:
                item=json.loads(path.read_text())
                owner=Path('/proc')/str(item['pid'])
                still=owner.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==str(item['start_ticks'])
            except (OSError,ValueError,KeyError,json.JSONDecodeError):
                still=False
            if still:active.append(item)
            else:path.unlink(missing_ok=True)
        if args.allow_shared_gpu:
            assert all(item.get('share_group')==args.share_group and
                       item.get('renderer')==Path(__file__).name for item in active), 'Another render group owns GPU lease'
            assert len(active)<args.max_renderers_per_gpu,'Per-GPU renderer cap reached'
            assert int(util)<=args.max_util_percent,'GPU utilization exceeds shared render cap'
            for pid in gpu_apps:
                command=Path('/proc').joinpath(str(pid),'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                assert Path(__file__).name in command and ('--share-group '+args.share_group) in command, \
                    'GPU has a compute process outside the authorized render group'
            pending=sum(int(item['reserve_mib']) for item in active if int(item['pid']) not in gpu_apps)
            assert int(free)-pending >= max(args.min_free_mib,args.reserve_mib), 'Insufficient unreserved GPU memory'
        else:
            assert int(memory)<100 and int(util)==0 and not gpu_apps and not active, 'GPU now occupied; no render started'
        lease={'pid':os.getpid(),'start_ticks':start_ticks,'renderer':Path(__file__).name,
               'share_group':args.share_group,'gpu_uuid':uuid,'gpu_pci':pci,'reserve_mib':args.reserve_mib,
               'started_unix':time.time()}
        lease_path.write_text(json.dumps(lease,sort_keys=True)+'\n')
        fcntl.flock(lock,fcntl.LOCK_UN)
    def release():
        try:
            current=json.loads(lease_path.read_text())
            if current.get('pid')==os.getpid() and current.get('start_ticks')==start_ticks:
                lease_path.unlink(missing_ok=True)
        except (OSError,ValueError):
            pass
    atexit.register(release)
    cuda=ctypes.CDLL('libcuda.so.1');assert cuda.cuInit(0)==0
    count=ctypes.c_int();assert cuda.cuDeviceGetCount(ctypes.byref(count))==0 and count.value==1,'Expose exactly one CUDA device'
    dev=ctypes.c_int();assert cuda.cuDeviceGet(ctypes.byref(dev),0)==0
    bus=ctypes.create_string_buffer(32);assert cuda.cuDeviceGetPCIBusId(bus,32,dev)==0
    normalized=lambda b:b.lower().split(':')[-2:]
    assert normalized(bus.value.decode())==normalized(pci),'CUDA/NVML PCI mismatch'
    return {'device':'OPTIX','nvml_uuid':uuid,'nvml_pci':pci,'cuda_pci':bus.value.decode(),
        'nvml_index':args.gpu_index,'memory_total_mib':int(total),'memory_before_mib':int(memory),
        'memory_free_before_mib':int(free),'utilization_before':int(util),
        'compute_apps_before':gpu_apps,'existing_render_leases':len(active),
        'share_group':args.share_group,'allow_shared_gpu':args.allow_shared_gpu,
        'max_renderers_per_gpu':args.max_renderers_per_gpu,'lease':str(lease_path)}


def main():
    import socket
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('render_config_drawer_140x78_equal4_base04_upright_v6.json'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preview',action='store_true',help='Diagnostic frame0 only; never a success episode')
    parser.add_argument('--external-photo-view',action='store_true',help='Additional diagnostic external-camera still, never a training view')
    parser.add_argument('--device',choices=['CPU','OPTIX'],default='OPTIX')
    parser.add_argument('--gpu-uuid');parser.add_argument('--gpu-index',type=int);parser.add_argument('--gpu-pci')
    parser.add_argument('--allow-shared-gpu',action='store_true')
    parser.add_argument('--share-group')
    parser.add_argument('--max-renderers-per-gpu',type=int,default=2)
    parser.add_argument('--reserve-mib',type=int,default=12000)
    parser.add_argument('--min-free-mib',type=int,default=20000)
    parser.add_argument('--max-util-percent',type=int,default=95)
    parser.add_argument('--samples',type=int,default=12)
    parser.add_argument('--width',type=int,default=224);parser.add_argument('--height',type=int,default=126)
    parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int)
    args=parser.parse_args();meta=json.loads((args.source/'source.json').read_text())
    if args.external_photo_view and not args.preview:raise ValueError('External camera is diagnostic only')
    config_path=str(args.config.resolve(strict=True))
    if meta.get('source_pins',{}).get(config_path)!=sha(config_path):
        raise ValueError('Source export did not pin the corrected drawer render config')
    if not args.preview:
        assert meta['native_accepted'] and not meta['diagnostic_only'],'Production requires successful native drawer task and drawer QA'
        assert meta['sensor_calibration']['same_robot_sensor_calibration_applicable'],'Production requires verified S1 camera calibration inheritance'
        assert (args.width,args.height)==(224,126) and args.samples==12
    assert not args.output.exists(),'Fresh render output required; never overwrite an old attempt'
    args.output.mkdir(parents=True)
    inputs={str(args.source/'source.json'):sha(args.source/'source.json'),str(args.config.resolve()):sha(args.config),
            str(Path(__file__).resolve()):sha(__file__),str(APPEARANCE_PATH.resolve()):sha(APPEARANCE_PATH)}
    for name in ('camera.py','environment.py','drawer_table_v2.py'):
        p=Path(__file__).resolve().parents[1]/'s1_drawer_v1'/name;inputs[str(p.resolve())]=sha(p)
    contract=Path(json.loads(args.config.read_text())['drawer_table_geometry']['geometry_contract'])
    inputs[str(contract.resolve())]=sha(contract)
    inputs.update(meta['source_pins'])
    frames=[0] if args.preview else list(range(args.start,min(args.stop or meta['frames'],meta['frames'])))
    assert frames and min(frames)>=0
    guard=gpu_guard(args)
    proc=Path('/proc')/str(os.getpid());stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
    owner={'schema':'rssr.drawer.renderer_owner.v1','pid':os.getpid(),'start_ticks':stat[19],'cmdline_sha256':hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest(),'hostname':socket.gethostname(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'started_at':time.time(),'gpu_guard':guard,'diagnostic_only':args.preview,'source':str(args.source.resolve()),'source_sha256':sha(args.source/'source.json')}
    (args.output/'renderer_owner.json').write_text(json.dumps(owner,indent=2)+'\n')
    started=time.monotonic();scene=AssetScene(args.source,args.config,args.samples,args.device,args.width,args.height,reference=True)
    errors=[];encoders=[];ffmpeg=os.environ.get('REEMBODY_FFMPEG') or os.environ.get('RSSR_FFMPEG') or __import__('shutil').which('ffmpeg') or 'ffmpeg'
    if not args.preview:
        for camera in meta['camera_names']:
            encoders.append(subprocess.Popen([ffmpeg,'-v','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size','224x224','-framerate','30','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-threads','1',str(args.output/(camera+'.mp4'))],stdin=subprocess.PIPE))
    try:
        for number,index in enumerate(frames):
            errors.append(scene.projection_error(index));panels=[]
            for ci,name in enumerate(meta['camera_names']):
                raw=scene.render(index,ci,args.output/'current.png')
                assert raw.size==(224,126),'Native content must be 224x126'
                image=Image.new('RGB',(224,224),(0,0,0));image.paste(raw,(0,49));panels.append(image)
                if args.preview:image.save(args.output/(name+'.png'))
                else:
                    encoders[ci].stdin.write(np.asarray(image).tobytes())
                    if index in (frames[0],frames[-1]):image.save(args.output/f'frame_{index:06d}_{name}.png')
            if args.preview:
                joint=Image.new('RGB',(3*224,224))
                for i,image in enumerate(panels):joint.paste(image,(i*224,0))
                joint.save(args.output/'three_views.png')
            (args.output/'progress.json').write_text(json.dumps({'frames_completed':number+1,'frames_expected':len(frames),'native_frame_index':index,'elapsed_s':time.monotonic()-started})+'\n')
        for encoder in encoders:
            encoder.stdin.close();assert encoder.wait(timeout=60)==0,'Video encoder failed'
    finally:
        for encoder in encoders:
            if encoder.poll() is None:encoder.kill();encoder.wait()
    external_camera=None
    if args.external_photo_view:
        external_camera=scene.render_external_photo_diagnostic(0,args.output/'external_photo_diagnostic.png')
    assert all(sha(path)==digest for path,digest in inputs.items()),'Render input changed'
    assert sha(scene.cfg['asset_blend'])==scene.cfg['asset_sha256']
    for relative,digest in scene.cfg.get('asset_dependencies',{}).items():assert sha(Path(scene.cfg['asset_blend']).parent/relative)==digest
    f=args.height/(2*np.tan(np.deg2rad(scene.camera_fov)/2))
    receipt={'schema':'rssr.drawer.render.v9b.uniform_cabinet_upright','task':'Put_the_block_in_the_drawer.','diagnostic_only':args.preview,'native_accepted':meta['native_accepted'],
             'renderer_revision':'center_layout_sdk_cameras_v1_from_v9b_uniform_cabinet_upright','visual_preview_source_sha256':PROVENANCE_VISUAL_PREVIEW_SHA256,
             'success_episode':bool(not args.preview and meta['native_accepted']),'frames':frames,'fps':30,'resolution':[224,224],
             'raw_resolution':[args.width,args.height],
             'raster_policy':'source/reference FOV; upright raw 224x126 16:9; black-pad 49px top/bottom to 224x224 RGB',
             'source':str(args.source.resolve()),'native_scene':meta['native_scene'],'native_scene_sha256':meta['native_scene_sha256'],
             'all_drawer_cabinet_and_redblock_geometry_preserved':False,'native_source_geometry_and_collision_preserved':True,
             'visual_drawer_geometry_replaced':True,'visual_appearance_config':str(APPEARANCE_PATH.resolve()),
             'visual_appearance_config_sha256':APPEARANCE_SHA256,'external_photo_diagnostic':external_camera,
             'native_geometry_count':len(meta['objects']),
             'task_object_indices':meta['task_object_indices'],'camera_names':meta['camera_names'],
             'sensor_calibration_inherited_from_v3':scene.sensor_calibration_inherited,'calibration_comparison':meta['sensor_calibration'],
             'camera_fovy_deg':scene.camera_fov,
             'raw_camera_intrinsics_K':[[[float(x),0,args.width/2],[0,float(x),args.height/2],[0,0,1]] for x in f],
             'camera_intrinsics_K':[[[float(x),0,112],[0,float(x),112],[0,0,1]] for x in f],
             'projection_error_px_max':max(errors),'projection_validation_scope':scene.projection_scope,
             'world_registration_matrix':scene.world.tolist(),'table_height_m':scene.table_top,'table_height_audit':scene.table_height_audit,
             'native_geometry_audit':scene.native_geometry_audit,
             'native_base_translation_m':meta['sensor_calibration']['native_base_translation_m'],
             'head_camera_native_t0_m':scene.pose['camera_position'][0,0].tolist(),
             'head_camera_visual_t0_m':(scene.world@np.r_[scene.pose['camera_position'][0,0],1])[:3].tolist(),
             'native_render_material_overrides':scene.material_recipe,
             'lighting_fixed_across_frames_and_cameras':True,'lights':scene.light_records,'style_audit':scene.style_audit,
             'input_pins':inputs,'gpu_guard':guard,'owner':owner,'elapsed_s':time.monotonic()-started,
             'artifacts':{p.name:sha(p) for p in args.output.iterdir() if p.suffix in ('.png','.mp4')}}
    (args.output/'render_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'rendered':str(args.output),'frames':len(frames),'diagnostic_only':args.preview,'sensor_calibration_inherited':scene.sensor_calibration_inherited,'elapsed_s':receipt['elapsed_s']}))


if __name__=='__main__':
    main()
