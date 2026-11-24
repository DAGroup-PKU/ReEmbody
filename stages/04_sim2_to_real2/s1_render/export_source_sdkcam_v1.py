"""Copy of s1_stack_cups_random300_r15_extra200_v2/export_source.py (sha256 34831c93dd595d7fdfa9cf0c46e830c5913362437249ff817bf7c5a49dfe4e62).

Export logic is identical except:
  * the accepted-source gate reads `case_result.json` of a
    sim2-refine/visual_rules/run_case_visual_v1.py candidate (accepted under the
    visual-rule admission: strict judge, every-step hard constraints, bit-exact
    official replay, object gates, agent alias / 20 mm clearance / shell-table);
  * `--wrist sdk` replaces both wrist cameras of the native scene model by the
    nominal Astribot SDK S1v2 kinematic.xml mounts before camera poses are
    computed (same parent body arm_*_tool_cam, fovy 57). The head camera is the
    v8 mount. Physics and robot/cup states are untouched (camera pose only).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"stages/02_sim1_to_sim2"))
from rssr_render.source_clock import complete_source_clock
from rssr_augment.action_columns import desired_columns

import mujoco
import numpy as np
from source_arrays import one_frame_vector

CAMERAS = ['head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def triangles(faces):
    return np.array([(p[0], p[i], p[i+1]) for p in faces for i in range(1, len(p)-1)], dtype=np.int32)


def geometry(m, gid):
    kind = int(m.geom_type[gid]); size = m.geom_size[gid]
    normals = np.empty((0, 3, 3))
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[gid]); a = int(m.mesh_vertadr[mid]); n = int(m.mesh_vertnum[mid])
        v = m.mesh_vert[a:a+n].astype(float).copy()
        a = int(m.mesh_faceadr[mid]); n = int(m.mesh_facenum[mid]); f = m.mesh_face[a:a+n].copy()
        na = int(m.mesh_normaladr[mid]); nn = int(m.mesh_normalnum[mid]); ix = m.mesh_facenormal[a:a+n]
        if nn and np.all(ix >= 0) and np.all(ix < nn):
            normals = m.mesh_normal[na:na+nn][ix].astype(float).copy()
        return v, f, normals
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        v = np.array([[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],
                      [1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]], float)*size
        return v, triangles([(4,6,7,5),(0,1,3,2),(2,3,7,6),(0,4,5,1),(1,5,7,3),(0,2,6,4)]), normals
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        n=64; t=np.arange(n)*2*np.pi/n
        v=np.array([(size[0]*np.cos(a),size[0]*np.sin(a),z) for z in [-size[1],size[1]] for a in t])
        f=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]+[(i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n)]
        return v,triangles(f),normals
    raise ValueError(f'Unsupported visible native geometry {gid}: {kind}')


# Astribot SDK S1v2 kinematic.xml wrist mounts (parent arm_*_tool_cam at pos 0 -0.15 0),
# from pi0.5/operations/camera_mount_alignment_20261025_attempt_01/camera/SDK_S1v2_camera_nodes.json
SDK_WRIST = {
    'left_wrist_rgbd': ([0.011420996159648984, 0.058882000224149605, 0.044260118849329295],
                        [0.703108425353722, -0.7109134990656752, -0.009037420129074155, -0.012604129490710232], 57.0),
    'right_wrist_rgbd': ([0.00772, 0.05592, 0.04319],
                         [-0.7130607009975457, 0.7006910271198059, 0.013493708973317153, 0.019860539389038687], 57.0),
}
SDK_SOURCE = 'astribot_sdk_s1v2_camera_nodes.json'


def apply_wrist(m, wrist):
    if wrist == 'native':
        return dict(kind='native_scene', fovy_deg={n: float(m.cam_fovy[m.camera(n).id]) for n in SDK_WRIST})
    assert wrist == 'sdk'
    record = {}
    for name, (pos, quat, fovy) in SDK_WRIST.items():
        c = m.camera(name).id
        assert m.body(int(m.cam_bodyid[c])).name == name.replace('_wrist_rgbd', '').join(['arm_', '_tool_cam'])
        assert np.allclose(m.body_pos[m.cam_bodyid[c]], [0, -0.15, 0])
        record[name] = dict(replaced_pos=m.cam_pos[c].tolist(), replaced_quat_wxyz=m.cam_quat[c].tolist(),
                            replaced_fovy_deg=float(m.cam_fovy[c]), pos=pos, quat_wxyz=quat, fovy_deg=fovy,
                            parent_body=m.body(int(m.cam_bodyid[c])).name)
        m.cam_pos[c] = pos
        q = np.asarray(quat, float) / np.linalg.norm(quat)
        # 180 deg roll about the optical axis (q * (0,0,0,1)): the raw SDK-convention image has the
        # table at the top; the v8 block pipeline (SDK-like wrist mounts) shows the same views rotated
        # 180 deg, and v8 cup wrist views are upright. Image-orientation convention, not a measured mount.
        w, x, y, z = q
        m.cam_quat[c] = [-z, y, -x, w]
        record[name]['image_roll_180_about_optical_axis'] = True
        record[name]['rendered_quat_wxyz'] = m.cam_quat[c].tolist()
        m.cam_fovy[c] = fovy
    return dict(kind='astribot_sdk_s1v2_kinematic_xml_nominal', source=SDK_SOURCE, cameras=record,
                image_orientation='rotated 180 deg (table at bottom) to match v8 cup/block views; unverified on a real S1 wrist frame',
                verified_against_real_wrist_frame=False)


def export(source, output, baseline=False, fps=30, diagnostic=False, wrist='native'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    source=Path(source)
    scene=source/'scene.xml' if baseline else source/'execute/scene.xml'
    if baseline and not scene.exists():scene=source/'mobile_model_preflight/scene.xml'
    pins={str(scene):sha(scene)}
    if baseline:
        z=dict(np.load(source/'initial_state.npz',allow_pickle=False)) if (source/'initial_state.npz').exists() else {'qpos':np.load(source/'initial_qpos.npy',allow_pickle=False)}
        initial_model=mujoco.MjModel.from_xml_path(str(scene))
        if 'qvel' not in z:z['qvel']=np.zeros(initial_model.nv)
        initial_qpos=one_frame_vector(z['qpos'],initial_model.nq,'initial qpos');initial_qvel=one_frame_vector(z['qvel'],initial_model.nv,'initial qvel')
        recorded={'time_s':np.array([0.]),'qpos':initial_qpos[None], 'qvel':initial_qvel[None]}
        initial_path=source/('initial_state.npz' if (source/'initial_state.npz').exists() else 'initial_qpos.npy');pins[str(initial_path)]=sha(initial_path)
        commands=None;receipt=None
    else:
        if diagnostic:
            receipt=json.loads((source/'case_result.json').read_text())
            assert receipt['accepted'] is True and all(receipt['checks'].values())
            assert receipt['schema']=='rssr.visual_rules.case_result.v1'
        else:
            receipt=json.loads((source/'receipt.json').read_text())
            assert receipt['accepted'] and receipt['native']['task_goal_met']
            assert receipt['replay']['exact'] and receipt['replay']['every_step_hard_trace_exact']
        recorded=dict(np.load(source/'execute/rollout.npz',allow_pickle=False))
        commands=dict(np.load(source/'issued_robot_commands.npz',allow_pickle=False))
        for p in [source/('execute/report.json' if diagnostic else 'receipt.json'),source/'execute/rollout.npz',source/'issued_robot_commands.npz']:
            pins[str(p)]=sha(p)
        if diagnostic:pins[str(source/'case_result.json')]=sha(source/'case_result.json')
    m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m)
    wrist_model=apply_wrist(m,wrist)
    d.qpos[:]=recorded['qpos'][0];d.qvel[:]=recorded['qvel'][0];mujoco.mj_forward(m,d)
    arrays={};objects=[]
    for gid in range(m.ngeom):
        body=m.body(int(m.geom_bodyid[gid])).name or ''
        if m.geom_group[gid]==3 or m.geom_rgba[gid,3]==0 or body.startswith('cup_') or m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        v,f,n=geometry(m,gid);key=f'g{len(objects):03d}'
        arrays[key+'_v']=v;arrays[key+'_f']=f;arrays[key+'_n']=n
        objects.append(dict(key=key,geom_id=gid,geom_name=m.geom(gid).name or f'geom_{gid}',body_name=body,
            rgba=m.geom_rgba[gid].tolist(),head_occluder=any(s in body for s in ['astribot_head_','astribot_torso_','chassis_base','wheel_'])))
    ids=[o['geom_id'] for o in objects];cams=[m.camera(c).id for c in CAMERAS]
    cup_names=['cup_left_base','cup_right_upper'];cups=[m.body(n).id for n in cup_names]
    ts=recorded['time_s'];duration=float(ts[-1])
    if baseline:
        vt=np.array([0.]);ri=np.array([0]);ci=np.array([0]);queries=np.array([0.])
    else:
        vt,ri,ci,queries=complete_source_clock(ts,commands['time_s'],duration,fps)
    poses={n:[] for n in ['geom_position','geom_rotation','camera_position','camera_rotation','cup_position','cup_rotation']}
    for i in ri:
        d.qpos[:]=recorded['qpos'][i];d.qvel[:]=recorded['qvel'][i];d.time=float(ts[i]);mujoco.mj_forward(m,d)
        for name,value in [('geom_position',d.geom_xpos[ids]),('geom_rotation',d.geom_xmat[ids]),
            ('camera_position',d.cam_xpos[cams]),('camera_rotation',d.cam_xmat[cams]),
            ('cup_position',d.xpos[cups]),('cup_rotation',d.xmat[cups])]:poses[name].append(value.copy())
    np.savez_compressed(output/'geometry.npz',**arrays)
    np.savez_compressed(output/'poses.npz',video_time_s=vt,physical_query_time_s=queries,state_source_index=ri,command_source_index=ci,state_time_s=ts[ri],**{k:np.array(v) for k,v in poses.items()})
    columns,qa=desired_columns(m,commands['qpos'] if commands is not None else recorded['qpos'])
    joint_names=[c['name'] for c in columns]
    qa=[int(m.joint(n).qposadr[0]) for n in joint_names];va=[int(m.joint(n).dofadr[0]) for n in joint_names]
    scalar_names=[m.joint(i).name for i in range(m.njnt) if m.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE) and m.joint(i).name.startswith('astribot_')]
    scalar_qa=[int(m.joint(n).qposadr[0]) for n in scalar_names];scalar_va=[int(m.joint(n).dofadr[0]) for n in scalar_names]
    if commands is not None:
        assert commands['qvel'].shape==commands['qpos'].shape,'Issued desired velocity uses qpos coordinates (not MuJoCo dofs)'
        assert recorded['qvel'].shape[1]==m.nv
        np.savez_compressed(output/'actions_30fps.npz',video_time_s=vt,physical_query_time_s=queries,state_source_index=ri,command_source_index=ci,state_time_s=ts[ri],
            command_time_s=commands['time_s'][ci],joint_names=np.array(joint_names),
            qpos=commands['qpos'][ci][:,qa],qvel=commands['qvel'][ci][:,qa],
            actual_qpos=recorded['qpos'][ri][:,qa],actual_qvel=recorded['qvel'][ri][:,va],event=commands['event'][ci],
            scalar_joint_names=np.array(scalar_names),scalar_joint_qpos=recorded['qpos'][ri][:,scalar_qa],scalar_joint_qvel=recorded['qvel'][ri][:,scalar_va],
            full_qpos=recorded['qpos'][ri],full_qvel=recorded['qvel'][ri],full_command_qpos=commands['qpos'][ci],full_command_qvel=commands['qvel'][ci])
        assert ri[-1]==len(ts)-1 and np.max(queries-ts[ri]) <= np.max(np.diff(ts))+1e-8
        assert np.all(commands['time_s'][ci] <= ts[ri]+1e-10)
    top=m.geom('worktable_top').id
    d.qpos[:]=recorded['qpos'][0];mujoco.mj_forward(m,d)
    meta=dict(schema='rssr.s1_heightfit.render_source.v1',source=str(source),source_pins=pins,
        baseline_diagnostic=baseline,postprocess_diagnostic=diagnostic,
        native_accepted=None if baseline else True,visual_rule_case_admitted=bool(diagnostic),wrist_camera_model=wrist_model,frames=len(vt),fps=fps,duration_s=duration,
        raster=[224,224],content_raster=[224,126],resize_policy='native upright 16:9 render at 224x126; black-pad 49 pixels above and below to 224x224',
        native_scene=str(scene),native_scene_sha256=sha(scene),native_attempt=str(source),native_receipt_sha256=None if baseline or diagnostic else sha(source/'receipt.json'),
        scalar_joint_names=scalar_names,controlled_joint_names=joint_names,action_columns=columns,desired_velocity_layout='qpos_coordinates',actual_velocity_layout='mujoco_dof_coordinates',complete_terminal_state_included=True,causal_actions_bound_to_rendered_state=True,
        camera_names=CAMERAS,camera_fovy_deg=m.cam_fovy[cams].tolist(),objects=objects,cup_body_names=cup_names,
        cup_model=dict(top_diameter_m=.08,bottom_diameter_m=.0585,height_m=.09,wall_m=.0015),
        table_center_m=d.geom_xpos[top].tolist(),table_half_size_m=m.geom_size[top].tolist(),
        axes_and_camera_mounts_copied_from_native=True,object_and_camera_states_copied_from_native=True,
        task_regions_not_rendered=True,hardware_sdk_ready=False,mujoco_version=mujoco.__version__)
    meta['artifacts']={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    assert all(sha(p)==h for p,h in pins.items())
    (output/'source.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'exported':str(output),'frames':len(vt),'native_geoms':len(objects)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',action='store_true');p.add_argument('--diagnostic',action='store_true',help='source is a visual-rule case (case_result.json gate)')
    p.add_argument('--wrist',choices=['native','sdk'],default='sdk')
    a=p.parse_args();export(a.source,a.output,a.baseline,diagnostic=a.diagnostic,wrist=a.wrist)
