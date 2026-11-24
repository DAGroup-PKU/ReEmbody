"""Centred-layout drawer export (copy of s1_drawer_v1/export_source_drawer_v3.py, unchanged there).

Identical read-only FK export; the camera check verifies the SDK nominal
(image-up) head/wrist mounts written into the native scene instead of the
photo-fitted v3 mounts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 's1_drawer_v1'))
from action_columns import desired_columns
from source_clock import complete_source_clock

CAMERAS = ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def triangles(polygons):
    return np.asarray([(p[0], p[i], p[i+1]) for p in polygons for i in range(1, len(p)-1)], dtype=np.int32)


def geometry(model, gid):
    kind = int(model.geom_type[gid]); size = model.geom_size[gid]
    normals = np.empty((0, 3, 3))
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[gid]); start = int(model.mesh_vertadr[mid]); count = int(model.mesh_vertnum[mid])
        vertices = model.mesh_vert[start:start+count].astype(float).copy()
        start = int(model.mesh_faceadr[mid]); count = int(model.mesh_facenum[mid]); faces = model.mesh_face[start:start+count].copy()
        start_n = int(model.mesh_normaladr[mid]); count_n = int(model.mesh_normalnum[mid]); ix = model.mesh_facenormal[start:start+count]
        if count_n and np.all(ix >= 0) and np.all(ix < count_n):
            normals = model.mesh_normal[start_n:start_n+count_n][ix].astype(float).copy()
        return vertices, faces, normals
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = np.asarray([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], float)*size
        return vertices, triangles([(4,6,7,5),(0,1,3,2),(2,3,7,6),(0,4,5,1),(1,5,7,3),(0,2,6,4)]), normals
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        count = 64; theta = np.arange(count)*2*np.pi/count
        vertices = np.asarray([(size[0]*np.cos(a),size[0]*np.sin(a),z) for z in [-size[1],size[1]] for a in theta])
        faces = [tuple(range(count-1,-1,-1)),tuple(range(count,2*count))]+[(i,(i+1)%count,(i+1)%count+count,i+count) for i in range(count)]
        return vertices, triangles(faces), normals
    raise ValueError(f'Unsupported visible native geometry {model.geom(gid).name}: {kind}; no geometry is silently dropped')


def camera_chains(scene):
    result = {}
    def visit(node, chain):
        chain = chain + [{'body': dict(node.attrib), 'joints': [dict(x.attrib, tag=x.tag) for x in node if x.tag in ('joint','freejoint')]}]
        for camera in node.findall('camera'):
            if camera.get('name') in CAMERAS:
                result[camera.get('name')] = {'ancestors': chain, 'camera': dict(camera.attrib)}
        for child in node.findall('body'):
            visit(child, chain)
    root = ET.parse(scene).getroot()
    require(not root.findall('include'), 'Need a fully expanded scene XML')
    visit(root.find('worldbody'), [])
    require(set(result) == set(CAMERAS), 'Three native cameras are required')
    return result


def calibration_comparison(scene, config):
    """Verify the native scene carries the SDK nominal camera mounts and the 4 cm base shift."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from layout_common_v1 import sdk_cameras
    sdk = sdk_cameras()
    pins = {}
    contract_path = Path(config['drawer_table_geometry']['geometry_contract'])
    require(sha(contract_path) == config['drawer_table_geometry']['geometry_contract_sha256'],
            'Centred drawer geometry contract changed')
    contract = read(contract_path)
    require(np.allclose(contract['robot_native_base_position_m'], [0., .04, .097], atol=1e-8, rtol=0),
            'Drawer must use the contract-pinned 4 cm native base shift')
    pins[str(contract_path.resolve())] = sha(contract_path)
    actual = camera_chains(scene)
    cameras = {}
    for name in CAMERAS:
        cam = actual[name]['camera']
        parent = actual[name]['ancestors'][-1]['body'].get('name')
        pos = np.fromstring(cam.get('pos', ''), sep=' ')
        quat = np.fromstring(cam.get('quat', ''), sep=' ')
        same = bool(parent == sdk[name]['parent'] and np.allclose(pos, sdk[name]['pos'], atol=1e-9) and
                    np.allclose(quat, sdk[name]['quat'], atol=1e-9) and
                    abs(float(cam.get('fovy')) - sdk[name]['fovy']) < 1e-9)
        cameras[name] = {'native_mount_matches_sdk_nominal_image_up': same, 'parent_body': parent,
                         'native': actual[name], 'sdk_nominal': sdk[name]}
    ok = all(x['native_mount_matches_sdk_nominal_image_up'] for x in cameras.values())
    return {'same_robot_sensor_calibration_applicable': ok, 'policy': config['calibration_policy'],
            'cameras': cameras, 'input_pins': pins, 'world_scene_registration_is_separate': True,
            'task_specific_camera_refit_performed': False, 'native_base_translation_m': [0., .04, 0.],
            'relative_camera_mounts_unchanged': ok, 'camera_source': 'astribot_sdk_kinematic_xml_nominal_image_roll180'}


def export(args):
    require(mujoco.__version__ == '3.3.7', 'Export requires original native MuJoCo3.3.7')
    native = args.native.resolve(); scene = (args.scene or native/'execute/scene.xml').resolve(strict=True)
    config = read(args.config); pins = {str(scene): sha(scene), str(args.config.resolve()): sha(args.config), str(Path(__file__).resolve()): sha(__file__)}
    for helper in ('action_columns.py','source_clock.py'):
        path = Path(__file__).resolve().parents[1]/'s1_drawer_v1'/helper; pins[str(path.resolve())] = sha(path)
    xml=ET.parse(scene).getroot();compiler=xml.find('compiler')
    for asset in xml.findall('asset/*'):
        for key,value in asset.attrib.items():
            if not key.startswith('file'):continue
            path=Path(value)
            directory={'mesh':'meshdir','texture':'texturedir'}.get(asset.tag)
            if not path.is_absolute():path=scene.parent/(compiler.get(directory,'') if compiler is not None and directory else '')/path
            path=path.resolve(strict=True);pins[str(path)]=sha(path)
    model = mujoco.MjModel.from_xml_path(str(scene)); data = mujoco.MjData(model)
    require(model.nu == 22, 'New measured S1 requires22 native actuators')
    accepted = False; qa = None;initial_basis='actual_recorded_native_rollout'
    if args.initial_state:
        path = args.initial_state.resolve(strict=True); pins[str(path)] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            state = {k: z[k].copy() for k in z.files}
        initial_basis='reference_initial_pose_FK_diagnostic' if bool(state.get('reference_initial_pose_fk_diagnostic',False)) else 'recorded_initial_snapshot_diagnostic'
        q = state['qpos']; q = q[0] if q.ndim == 2 else q
        v = state.get('qvel', np.zeros(model.nv)); v = v[0] if v.ndim == 2 else v
        ctrl = state.get('ctrl', np.zeros(model.nu)); ctrl = ctrl[0] if ctrl.ndim == 2 else ctrl
        raw = {'qpos': q[None], 'qvel': v[None], 'ctrl': ctrl[None], 'time_s': np.asarray([0.])}
        require(q.shape == (model.nq,) and v.shape == (model.nv,) and ctrl.shape == (model.nu,), 'Initial snapshot shapes differ from the native model')
        commands = None; video = queries = np.asarray([0.]); si = ci = np.asarray([0])
    else:
        receipt_path = native/'receipt.json'; qa_path = args.qa_receipt
        require(qa_path is not None, 'Production export requires explicit drawer QA receipt')
        qa_path = qa_path.resolve(strict=True); receipt = read(receipt_path); qa = read(qa_path)
        require(receipt.get('accepted') is True and receipt.get('native',{}).get('task_goal_met') is True, 'Native drawer task did not succeed')
        require(receipt.get('replay',{}).get('exact') is True and receipt.get('replay',{}).get('every_step_hard_trace_exact') is True, 'Exact native replay/hard trace proof missing')
        require(qa.get('accepted', qa.get('passed')) is True and qa.get('checks') and all(v is True for v in qa['checks'].values()), 'Drawer-specific QA did not pass')
        bound = qa.get('native_case', qa.get('native_root', qa.get('native')))
        require(isinstance(bound,str) and Path(bound).resolve() == native, 'QA must bind this exact native attempt')
        for p in (receipt_path,qa_path,native/'execute/rollout.npz',native/'issued_robot_commands.npz'):
            pins[str(p)] = sha(p)
        for p,h in qa.get('input_pins',qa.get('pins',{})).items():
            require(sha(p) == h, 'QA input changed: '+p); pins[p] = h
        with np.load(native/'execute/rollout.npz',allow_pickle=False) as z:
            raw = {k:z[k].copy() for k in ('qpos','qvel','ctrl','time_s')}
        with np.load(native/'issued_robot_commands.npz',allow_pickle=False) as z:
            commands = {k:z[k].copy() for k in z.files}
        video,si,ci,queries = complete_source_clock(raw['time_s'],commands['time_s'],float(raw['time_s'][-1]),fps=30)
        require(si[-1] == len(raw['time_s'])-1, 'Terminal physical state missing')
        accepted = True
    for key,width in (('qpos',model.nq),('qvel',model.nv),('ctrl',model.nu)):
        require(raw[key].shape == (len(raw['time_s']),width) and np.isfinite(raw[key]).all(), 'Invalid actual state array: '+key)
    require(float(raw['time_s'][0]) == 0., 'Initial native physical time must be zero')
    contract_path = args.task_contract or scene.parent/'task_contract.json'; contract = read(contract_path) if contract_path.exists() else None
    if contract_path.exists(): pins[str(contract_path)] = sha(contract_path)
    require(contract and contract.get('bindings',{}).get('articulation'), 'Explicit drawer task articulation required')
    articulation = contract['bindings']['articulation']; articulation = articulation if isinstance(articulation,str) else articulation['joint_name']
    joint = model.joint(articulation)
    require(model.jnt_type[joint.id] == mujoco.mjtJoint.mjJNT_SLIDE, 'Drawer articulation must be the native slide joint')
    ids=[]; specs=[]; arrays={}
    for gid in range(model.ngeom):
        name=model.geom(gid).name or f'geom_{gid}'; body=model.body(int(model.geom_bodyid[gid])).name or ''
        if model.geom_group[gid] >= 3 or model.geom_rgba[gid,3] == 0 or model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        vertices,faces,normals=geometry(model,gid); key=f'g{len(specs):03d}'
        arrays.update({key+'_v':vertices,key+'_f':faces,key+'_n':normals})
        specs.append({'key':key,'geom_id':gid,'geom_name':name,'body_name':body,'rgba':model.geom_rgba[gid].tolist(),
                      'head_occluder':any(x in body for x in ('astribot_head_','astribot_torso_','chassis_base','wheel_')),
                      'is_robot':body.startswith(('astribot_','arm_','chassis_','wheel_'))})
        ids.append(gid)
    task_indices=[i for i,s in enumerate(specs) if not s['is_robot'] and not s['geom_name'].startswith('worktable')]
    require(task_indices and any('drawer' in specs[i]['body_name'] for i in task_indices), 'Drawer cabinet geometry was not exported')
    robot_names=[model.joint(i).name for i in range(model.njnt) if (model.joint(i).name or '').startswith('astribot_') and model.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE)]
    require(len(robot_names) == 32, 'Expected32 named measured S1 robot states')
    qa_robot=[int(model.joint(n).qposadr[0]) for n in robot_names]
    issued = commands['qpos'] if commands is not None else raw['qpos']
    columns,action_addresses=desired_columns(model,issued)
    require(len(columns) == 22 and all(c['name'].startswith('astribot_') for c in columns), 'Action22 must contain robot targets only')
    poses={k:[] for k in ('geom_position','geom_rotation','camera_position','camera_rotation')}; cams=[model.camera(n).id for n in CAMERAS]
    for i in si:
        data.qpos[:]=raw['qpos'][i];data.qvel[:]=raw['qvel'][i];data.ctrl[:]=raw['ctrl'][i];data.time=float(raw['time_s'][i]);mujoco.mj_forward(model,data)
        for k,value in [('geom_position',data.geom_xpos[ids]),('geom_rotation',data.geom_xmat[ids]),('camera_position',data.cam_xpos[cams]),('camera_rotation',data.cam_xmat[cams])]:poses[k].append(value.copy())
    calibration=calibration_comparison(scene,config);pins.update(calibration['input_pins'])
    if accepted:require(calibration['same_robot_sensor_calibration_applicable'], 'Native camera chain differs from approved S1; production calibration not transferable')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'geometry.npz',**arrays)
    np.savez_compressed(out/'poses.npz',video_time_s=video,state_source_index=si,command_source_index=ci,state_time_s=raw['time_s'][si],**{k:np.asarray(v) for k,v in poses.items()})
    features={'video_time_s':video,'physical_query_time_s':queries,'state_source_index':si,'command_source_index':ci,'state_time_s':raw['time_s'][si],
              'full_qpos':raw['qpos'][si],'full_qvel':raw['qvel'][si],'ctrl':raw['ctrl'][si],
              'scalar_joint_names':np.asarray(robot_names),'scalar_joint_qpos':raw['qpos'][si][:,qa_robot],
              'joint_names':np.asarray([c['name'] for c in columns]),'qpos':issued[ci][:,action_addresses],
              'command_time_s':commands['time_s'][ci] if commands is not None else np.asarray([0.])}
    np.savez_compressed(out/'actions_30fps.npz',**features)
    data.qpos[:]=raw['qpos'][0];mujoco.mj_forward(model,data);top=model.geom('worktable_top').id
    meta={'schema':'rssr.drawer.render_source.v1','task':'Put_the_block_in_the_drawer.','task_family':'drawer','native_attempt':str(native),
          'native_scene':str(scene),'native_scene_sha256':sha(scene),'source_pins':pins,'native_accepted':accepted,'diagnostic_only':not accepted,
          'frames':len(video),'fps':30,'duration_s':float(raw['time_s'][-1]),'raster':[224,224],'objects':specs,'task_object_indices':task_indices,
          'camera_names':CAMERAS,'camera_fovy_deg':model.cam_fovy[cams].tolist(),'sensor_calibration':calibration,
          'scalar_joint_names':robot_names,'action_columns':columns,'task_contract':contract,'drawer_joint_name':articulation,
          'drawer_slide_initial_m':float(raw['qpos'][0,int(joint.qposadr[0])]),'drawer_slide_final_m':float(raw['qpos'][-1,int(joint.qposadr[0])]),
          'table_center_m':data.geom_xpos[top].tolist(),'table_half_size_m':model.geom_size[top].tolist(),
          'complete_terminal_state_included':True,'causal_actions_bound_to_rendered_state':True,'mujoco_version':mujoco.__version__,
          'physics_step_calls':0,'object_or_robot_state_edited':False,'initial_pose_basis':initial_basis,
          'ctrl_is_actual_recorded_native':initial_basis!='reference_initial_pose_FK_diagnostic',
          'initial_reference_is_not_actual_native_t0':initial_basis=='reference_initial_pose_FK_diagnostic',
          'qa_receipt':str(args.qa_receipt.resolve()) if args.qa_receipt else None,
          'artifacts':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    require(all(sha(p)==h for p,h in pins.items()), 'Input changed during read-only export')
    (out/'source.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'source':str(out),'frames':len(video),'native_accepted':accepted,'camera_mounts_match_sdk_nominal':calibration['same_robot_sensor_calibration_applicable']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--scene',type=Path)
    parser.add_argument('--task-contract',type=Path,help='Explicit compiled contract; diagnostic may use a separately generated identical contract')
    parser.add_argument('--initial-state',type=Path,help='Immutable qpos/qvel/ctrl NPZ; exports diagnostic frame0 only')
    parser.add_argument('--qa-receipt',type=Path,help='Required drawer QA for a complete successful trajectory')
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('render_config.json'))
    parser.add_argument('--output',type=Path,required=True)
    export(parser.parse_args())
