#!/usr/bin/env python3
"""Run one exact-source, genuine Luna FR3 candidate with read-only step geometry.

This launcher preserves run_one's original physics and judges. The only runtime
hooks observe the primary Executor after its original step_audit, and verify its
initial model/state. Independent replay uses the unmodified original auditor.
No retry, source replacement, source-options override, rendering or upload.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True

HERE=Path(__file__).resolve().parent
PREPARED=HERE/'prepared_sources.json'
API_HELPER=HERE.parent/'trajectory_v3/luna_api.py'
TASKS={'cup':'stack_cups','block':'stack_blocks','draw':'drawer'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def verify_pins(pins):
    for path,digest in pins.items():
        if not Path(path).is_file() or sha(path)!=digest:
            raise ValueError('Pinned input missing or changed: '+path)


def load_api_helper():
    spec=importlib.util.spec_from_file_location('trajectory_luna_bound_helper',API_HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parse_patches(source):
    """Read only an explicit literal API declaration; no host patch invention."""
    result=dict(declared=False,valid=False,patches=[],errors=[],binding='AST literal GAZE_PATCHES in the exact API program')
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GAZE_PATCHES' for t in node.targets):
            if result['declared']:
                result['errors'].append('Multiple GAZE_PATCHES assignments');continue
            result['declared']=True
            try:rows=ast.literal_eval(node.value)
            except Exception:
                result['errors'].append('GAZE_PATCHES is not a static literal');continue
            if not isinstance(rows,list) or not rows or len(rows)>32:
                result['errors'].append('GAZE_PATCHES must contain1..32 literal dictionaries');continue
            for index,row in enumerate(rows):
                try:
                    if not isinstance(row,dict) or set(row)!={'hand','object','point_object_m','start_s','end_s'}:
                        raise ValueError('Expected hand/object/point_object_m/start_s/end_s')
                    if row['hand'] not in ('left','right') or not isinstance(row['object'],str) or not row['object']:
                        raise ValueError('Invalid hand or object')
                    import math
                    point=row['point_object_m']
                    if not isinstance(point,list) or len(point)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in point):
                        raise ValueError('Invalid finite local3D point')
                    if any(type(row[k]) not in (int,float) or not math.isfinite(row[k]) for k in ('start_s','end_s')) or not 0<=row['start_s']<row['end_s']<=20:
                        raise ValueError('Invalid explicit patch time interval')
                    result['patches'].append(dict(row,index=index))
                except ValueError as error:result['errors'].append(f'Patch{index}: {error}')
    result['valid']=result['declared'] and bool(result['patches']) and not result['errors']
    return result


def preflight(args):
    helper=load_api_helper()
    api=args.api_request.resolve(strict=True)
    api=api if api.is_dir() else api.parent
    strategy=helper.verify_response(api)
    code=args.controller.resolve(strict=True).read_bytes()
    if not isinstance(strategy.get('controller_source'),str) or code!=strategy['controller_source'].encode('utf-8'):
        raise ValueError('Controller bytes differ from genuine verified API controller_source')
    prepared=read(args.prepared_sources)
    if prepared.get('schema')!='rssr.franka_v3.exact_prepared_sources.v1' or not prepared.get('all_sources_reproducible'):
        raise ValueError('Exact300-source preparation must be complete and reproducible')
    if prepared.get('source_prefix')!='150case_data_enhance_blender_franka_v2' or prepared.get('output_prefix')!='150case_data_enhance_blender_franka_v3':
        raise ValueError('Prepared input/output scope differs from the requested Franka release')
    matches=[r for r in prepared['episodes'] if r['output_task']==args.task and r['source_episode_index']==args.episode_index]
    if len(matches)!=1:raise ValueError('Task and episode must select exactly one exact source')
    row=matches[0]
    if row['variation']['task']!=TASKS[args.task] or not row['build_check'].get('reproducible') or not all(row['build_check']['checks'].values()):
        raise ValueError('Prepared variation or its initial model/state verification failed')
    pins=dict(prepared['framework_pins'])
    pins.update({str(args.prepared_sources.resolve()):sha(args.prepared_sources),str(API_HELPER):sha(API_HELPER),
                 str(args.controller.resolve()):sha(args.controller),str(Path(__file__).resolve()):sha(__file__),
                 str(HERE/'audit_grasp_camera.py'):sha(HERE/'audit_grasp_camera.py')})
    for name in ('request.json','response.json','raw_provider_response.json'):
        pins[str(api/name)]=sha(api/name)
    for pathkey,shakey in [('source_native_scene','source_native_scene_local_sha256'),
                          ('source_native_episode','source_native_episode_sha256'),
                          ('source_native_record','source_native_record_sha256')]:pins[row[pathkey]]=row[shakey]
    verify_pins(pins)
    source_record=read(row['source_native_record'])
    if source_record['variation']!=row['variation'] or not source_record.get('admitted'):
        raise ValueError('Exact source variation or source admission changed')
    # Bind the exact requested source to the original actual API evidence.
    request=read(api/'request.json')
    evidence=None
    for item in request['actual_sdk_kwargs'].get('input',[]):
        if item.get('role')=='user' and isinstance(item.get('content'),str):
            try:evidence=json.loads(item['content'])
            except json.JSONDecodeError:pass
    evidence_rows=evidence.get('actual_exact_source_variations',evidence.get('source_cases',[])) if isinstance(evidence,dict) else []
    bound=[r for r in evidence_rows if isinstance(r,dict) and r.get('source_id')==row['source_id']]
    if len(bound)!=1 or bound[0].get('variation')!=row['variation'] or \
            bound[0].get('source_native_run')!=row['source_native_run'] or \
            bound[0].get('source_native_scene_local_sha256')!=row['source_native_scene_local_sha256']:
        raise ValueError('This source was not exactly bound in the actual API request evidence')
    explicit_episode_pin=bound[0].get('source_native_episode_sha256')
    if explicit_episode_pin is not None and explicit_episode_pin!=row['source_native_episode_sha256']:
        raise ValueError('Explicit API evidence source episode hash differs from the exact prepared source')
    patches=parse_patches(strategy['controller_source'])
    response=read(api/'response.json')
    return prepared,row,pins,patches,dict(folder=str(api),model=response['provider_model'],
        provider_response_id=response['provider_response_id'],request_sha256=response['request_sha256'],
        controller_source_json_pointer=response['raw_arguments_json_pointer']+'#controller_source',
        controller_sha256=hashlib.sha256(code).hexdigest(),api_strategy_task=strategy.get('task'),
        exact_source_evidence_binding='source_id + complete Variation + native run + local scene hash',
        source_episode_hash_binding=('explicit actual API request evidence and exact prepared manifest'
                                    if explicit_episode_pin is not None else 'exact prepared manifest; compact actual API evidence omits episode hash'),
        source_episode_sha256=row['source_native_episode_sha256'])


class StepGeometry:
    """Primary actual physics clock and actual-state FK, never a policy hook."""
    def __init__(self,session,row,patches):
        import numpy as np
        import mujoco
        self.np,self.mj=np,mujoco
        self.session,self.executor,self.model=session,session.ex,session.m
        self.source_id=row['source_id']
        self.scratch=mujoco.MjData(self.model)
        self.hands=('left','right');self.camera_names=('head','left_wrist','right_wrist')
        self.cids=[self.model.camera(n).id for n in self.camera_names]
        self.tids=[self.model.site(h+'_tcp').id for h in self.hands]
        self.object_names=list(session.ids)
        self.bids=[self.model.body(n).id for n in self.object_names]
        self.patch_declaration=patches
        self.patch_validation=[]
        self.runtime_bindings={};self.runtime_events=[];self.runtime_errors=[];self.runtime_current={};self.runtime_last_hash=None
        self.rows={k:[] for k in ('time_s','qpos','qvel','camera_position_m','camera_rotation_world','tcp_position_m',
                                  'tcp_rotation_world','object_position_m','object_rotation_world','patch_world_m','patch_active',
                                  'runtime_patch_world_m','runtime_patch_point_object_m','runtime_patch_object_index',
                                  'runtime_patch_active','runtime_declaration_sha256')}
        self.observe_errors=[];self.primary_step_calls=0
        self.observe(session.d)
        # Validate declared surface points once on the original state/model.
        from audit_grasp_camera import nearest_object_surface
        for patch in patches['patches']:
            entry=dict(index=patch['index'],object=patch['object'],surface_distance_m=None,valid=False)
            if patch['object'] not in self.object_names:
                entry['reason']='Object is not a native scene task object'
            else:
                bi=self.model.body(patch['object']).id
                world=self.scratch.xpos[bi]+self.scratch.xmat[bi].reshape(3,3)@np.array(patch['point_object_m'])
                nearest=nearest_object_surface(self.model,self.scratch,patch['object'],world)
                if nearest:
                    entry.update(surface_distance_m=nearest['distance_from_contact_tcp_m'],nearest_surface=nearest,
                                 valid=nearest['distance_from_contact_tcp_m']<=.005,
                                 qualification='5mm surface-distance tolerance is engineering diagnostic only')
                else:entry['reason']='No supported actual collision surface'
            self.patch_validation.append(entry)

    def capture_runtime_declaration(self,now,*,before_step=False):
        """Read API's per-variation object patches without selecting any target.

        Existing hand/object local coordinates cannot be changed post-hoc;
        deactivate/reactivate is recorded separately and does not revise them.
        """
        raw=getattr(self.session,'gaze_patches',None)
        try:
            text=json.dumps(raw,sort_keys=True,separators=(',',':'),allow_nan=False)
        except Exception as error:
            self.runtime_errors.append(dict(time_s=float(now),error='Non-JSON gaze_patches: '+str(error)))
            self.runtime_current={};return
        digest=hashlib.sha256(text.encode()).hexdigest()
        if digest==self.runtime_last_hash:return
        self.runtime_last_hash=digest
        event=dict(time_s=float(now),before_command_frame_step=before_step,primary_physics_steps_already_executed=self.primary_step_calls,
                   declaration_sha256=digest,raw_declaration=json.loads(text),errors=[])
        current={}
        if raw is not None:
            if not isinstance(raw,dict) or set(raw)-set(self.hands):event['errors'].append('Expected hand-indexed gaze_patches dictionary')
            else:
                for hand,row in raw.items():
                    try:
                        if not isinstance(row,dict) or set(row)!={'object','point_object_m','active'}:
                            raise ValueError('Expected object/point_object_m/active')
                        if row['object'] not in self.object_names or type(row['active']) is not bool:
                            raise ValueError('Invalid source-bound task object or active flag')
                        point=self.np.asarray(row['point_object_m'],float)
                        if point.shape!=(3,) or not self.np.isfinite(point).all():raise ValueError('Invalid finite3D object patch')
                        key=hand+':'+row['object']
                        if key not in self.runtime_bindings:
                            self.runtime_bindings[key]=dict(source_id=self.source_id,hand=hand,object=row['object'],point_object_m=point.tolist(),
                                first_declared_s=float(now),declared_before_first_native_step=bool(before_step and self.primary_step_calls==0))
                        elif self.runtime_bindings[key]['point_object_m']!=point.tolist():
                            raise ValueError('Patch coordinates changed after first object binding; no post-hoc repair permitted')
                        current[hand]=dict(row,binding_key=key)
                    except Exception as error:event['errors'].append(hand+': '+type(error).__name__+':'+str(error))
        self.runtime_current=current
        self.runtime_events.append(event)
        if event['errors']:self.runtime_errors.append(dict(time_s=float(now),errors=event['errors']))

    def observe(self,actual):
        np,mj=self.np,self.mj
        d=self.scratch
        d.qpos[:]=actual.qpos;d.qvel[:]=actual.qvel;d.time=float(actual.time)
        mj.mj_kinematics(self.model,d);mj.mj_camlight(self.model,d)
        now=float(actual.time)
        self.capture_runtime_declaration(now)
        if self.rows['time_s'] and now<=self.rows['time_s'][-1]:raise ValueError('Actual physics observation clock is not strictly increasing')
        values=dict(time_s=now,qpos=actual.qpos.copy(),qvel=actual.qvel.copy(),
            camera_position_m=d.cam_xpos[self.cids].copy(),camera_rotation_world=d.cam_xmat[self.cids].reshape(-1,3,3).copy(),
            tcp_position_m=d.site_xpos[self.tids].copy(),tcp_rotation_world=d.site_xmat[self.tids].reshape(-1,3,3).copy(),
            object_position_m=d.xpos[self.bids].copy(),object_rotation_world=d.xmat[self.bids].reshape(-1,3,3).copy())
        points=[];active=[]
        for patch in self.patch_declaration['patches']:
            if patch['object'] in self.object_names:
                bi=self.model.body(patch['object']).id
                points.append(d.xpos[bi]+d.xmat[bi].reshape(3,3)@np.array(patch['point_object_m']))
                active.append(patch['start_s']<=now<=patch['end_s'])
            else:points.append(np.full(3,np.nan));active.append(False)
        values['patch_world_m']=np.asarray(points).reshape(-1,3)
        values['patch_active']=np.asarray(active,bool)
        runtime_world=[];runtime_local=[];runtime_object=[];runtime_active=[]
        for hand in self.hands:
            row=self.runtime_current.get(hand)
            if row:
                oi=self.object_names.index(row['object']);bi=self.bids[oi];local=np.array(row['point_object_m'])
                runtime_world.append(d.xpos[bi]+d.xmat[bi].reshape(3,3)@local)
                runtime_local.append(local);runtime_object.append(oi);runtime_active.append(row['active'])
            else:
                runtime_world.append(np.full(3,np.nan));runtime_local.append(np.full(3,np.nan));runtime_object.append(-1);runtime_active.append(False)
        values.update(runtime_patch_world_m=np.array(runtime_world),runtime_patch_point_object_m=np.array(runtime_local),
                      runtime_patch_object_index=np.array(runtime_object,int),runtime_patch_active=np.array(runtime_active,bool),
                      runtime_declaration_sha256=self.runtime_last_hash or '')
        for key,value in values.items():self.rows[key].append(value)

    def finish(self,root,record):
        np=self.np
        arrays={k:np.asarray(v) for k,v in self.rows.items()}
        arrays.update(camera_names=np.array(self.camera_names),object_names=np.array(self.object_names),hand_names=np.array(self.hands))
        trace=root/'every_step_geometry.npz'
        np.savez_compressed(trace,**arrays)
        t=arrays['time_s'];step=float(self.model.opt.timestep)
        report=dict(schema='rssr.fr3_duo.actual_every_step_geometry.v1',trace=str(trace),trace_sha256=sha(trace),
            primary_executor_step_audit_calls=self.primary_step_calls,recorded_states=len(t),
            initial_time_s=float(t[0]),final_actual_time_s=float(t[-1]),physics_step_s=step,
            original_step_audit_called_before_observation=True,independent_replay_observation_not_substituted=True,
            robot_only_simulation_unmodified=True,observations_from_separate_mjdata_fk=True,
            no_mj_step_or_force_reconstruction_in_observer=True,
            every_step_clock_contiguous=len(t)>1 and bool(np.allclose(np.diff(t),step,rtol=0,atol=max(1e-10,step*1e-6))),
            observer_errors=self.observe_errors,patch_declaration=self.patch_declaration,
            runtime_patch_bindings=self.runtime_bindings,runtime_patch_declaration_events=self.runtime_events,
            runtime_patch_declaration_errors=self.runtime_errors,
            patch_surface_diagnostics=self.patch_validation,full_requested_gaze_admission=None,
            full_requested_topdown_admission=None,
            interpretation='Measured geometry only. Gaze or topdown qualification requires an independently fixed active-grasp interval and approved criteria; missing explicit API patch never passes.',
            centroid_gaze={},declared_patch_gaze={})
        from audit_grasp_camera import angle,stats,camera_projection
        for hi,hand in enumerate(self.hands):
            ci=self.camera_names.index(hand+'_wrist')
            camera=arrays['camera_position_m'][:,ci];R=arrays['camera_rotation_world'][:,ci]
            forward=-R[:,:,2]
            tilt=angle(arrays['tcp_rotation_world'][:,hi,:,2],np.array([0.,0.,-1.]))
            for oi,obj in enumerate(self.object_names):
                angles=angle(forward,arrays['object_position_m'][:,oi]-camera)
                report['centroid_gaze'][hand+':'+obj]=dict(error_deg=stats(angles),initial_error_deg=float(angles[0]),
                    final_error_deg=float(angles[-1]),tcp_approach_angle_to_down_deg=stats(tilt),diagnostic_only=True)
        for pi,patch in enumerate(self.patch_declaration['patches']):
            hi=self.hands.index(patch['hand']);ci=self.camera_names.index(patch['hand']+'_wrist')
            active=arrays['patch_active'][:,pi]
            valid=active & np.isfinite(arrays['patch_world_m'][:,pi]).all(axis=1)
            if valid.any():
                R=arrays['camera_rotation_world'][:,ci];camera=arrays['camera_position_m'][:,ci]
                angles=angle(-R[:,:,2],arrays['patch_world_m'][:,pi]-camera)
                uv,depth=camera_projection(arrays['patch_world_m'][:,pi],camera,R,float(self.model.cam_fovy[self.cids[ci]]))
                report['declared_patch_gaze'][str(pi)]=dict(declaration=patch,samples=int(valid.sum()),
                    sampled_start_s=float(t[np.flatnonzero(valid)[0]]),sampled_end_s=float(t[np.flatnonzero(valid)[-1]]),
                    angular_error_deg=stats(angles[valid]),in_raw4by3_sensor_fov_fraction=float(np.mean((np.max(abs(uv[valid]),axis=1)<=1)&(depth[valid]>.005))),
                    diagnostic_only=True,continuous_between_physics_steps_not_claimed=True)
            else:report['declared_patch_gaze'][str(pi)]=dict(declaration=patch,samples=0,status='unknown')
        report['runtime_declared_patch_gaze']={}
        for hi,hand in enumerate(self.hands):
            active=arrays['runtime_patch_active'][:,hi]&np.isfinite(arrays['runtime_patch_world_m'][:,hi]).all(axis=1)
            if not active.any():
                report['runtime_declared_patch_gaze'][hand]=dict(status='unknown',samples=0,reason='No explicitly active valid API runtime object patch')
                continue
            ci=self.camera_names.index(hand+'_wrist');R=arrays['camera_rotation_world'][:,ci];camera=arrays['camera_position_m'][:,ci]
            angles=angle(-R[:,:,2],arrays['runtime_patch_world_m'][:,hi]-camera)
            uv,depth=camera_projection(arrays['runtime_patch_world_m'][:,hi],camera,R,float(self.model.cam_fovy[self.cids[ci]]))
            report['runtime_declared_patch_gaze'][hand]=dict(samples=int(active.sum()),angular_error_deg=stats(angles[active]),
                first_active_s=float(t[np.flatnonzero(active)[0]]),last_active_s=float(t[np.flatnonzero(active)[-1]]),
                in_raw4by3_sensor_fov_fraction=float(np.mean((np.max(abs(uv[active]),axis=1)<=1)&(depth[active]>.005))),
                diagnostic_only=True,no_complete_gaze_admission_claim=True)
        write_new(root/'every_step_geometry_receipt.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=TASKS,required=True)
    p.add_argument('--episode-index',type=int,required=True)
    p.add_argument('--controller',type=Path,required=True)
    p.add_argument('--api-request',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared-sources',type=Path,default=PREPARED)
    p.add_argument('--preflight-only',action='store_true',help='Verify source/API/code pins only; no build, controller import or physics')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('Fresh trial output required: '+str(args.output))
    prepared,row,pins,patches,api=preflight(args)
    if args.preflight_only:
        print(json.dumps(dict(preflight=True,source_id=row['source_id'],api=api,patches=patches,output_created=False,physics_steps=0)))
        return 0
    import numpy as np
    import mujoco
    sys.path.insert(0,str(Path(prepared['framework_root']).parent))
    from franka_dual_v1 import duo_run as DR,duo_scene as DS
    from franka_dual_v1.servo import Executor
    if mujoco.__version__!='3.3.7':raise ValueError('Native runtime must remain MuJoCo3.3.7')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    controller=args.output/'controller.py'
    controller.write_bytes(args.controller.read_bytes())
    pins[str(controller)]=sha(controller)
    write_new(args.output/'source.json',row)
    write_new(args.output/'launch.json',dict(schema='rssr.franka_v3.luna_native_trial.v1',at_utc=datetime.now(timezone.utc).isoformat(),
        task=args.task,source_episode_index=args.episode_index,source_id=row['source_id'],api=api,
        command=sys.argv,controller=str(controller),input_pins=pins,
        source_options_preserved_as_provenance=row['source_options'],options_override_passed_to_run_one=False,
        policy_options='Exact API module OPTIONS, without host overrides',original_native_gates_unchanged=True,
        extra_duration_gate=dict(motion_le_s=18.,full_physics_le_s=20.,original_settle_s=float(DR.SETTLE_S)),
        no_retry=True,no_resampling=True,no_render=True,no_upload=True,workers=1))
    recorder=None;session_ref=None;initial_proof=None;record=None;error=None
    old_init,old_audit,old_step=DR.Session.__init__,Executor.step_audit,DR.Session._step
    def init_observed(session,variation):
        nonlocal recorder,session_ref,initial_proof
        old_init(session,variation)
        session_ref=session
        with np.load(row['source_native_episode'],allow_pickle=False) as z:
            checks=dict(actual_variation_exact=variation.to_json()==row['variation'],
                actual_model_xml_exact=session.xml==Path(row['source_native_scene']).read_text(),
                initial_qpos_exact=bool(np.array_equal(session.d.qpos,z['qpos'][0])),
                initial_qvel_exact=bool(np.array_equal(session.d.qvel,z['qvel'][0])),
                initial_time_exact=float(session.d.time)==float(z['time'][0])==0.)
        initial_proof=dict(checks=checks,accepted=all(checks.values()),model_scene_sha256=hashlib.sha256(session.xml.encode()).hexdigest())
        write_new(args.output/'initial_state_receipt.json',initial_proof)
        (args.output/'initial_scene.xml').write_text(session.xml)
        if not initial_proof['accepted']:raise ValueError('Actual new native initialization differs from this exact source')
        recorder=StepGeometry(session,row,patches)
    def audit_observed(executor,data,audit):
        old_audit(executor,data,audit)
        if recorder is not None and executor is recorder.executor:
            recorder.primary_step_calls+=1
            try:recorder.observe(data)
            except Exception as err:
                recorder.observe_errors.append(dict(time_s=float(data.time),error=type(err).__name__+':'+str(err)))
                raise
    def step_observed(session,action,label):
        if recorder is not None and session is recorder.session:
            recorder.capture_runtime_declaration(float(session.d.time),before_step=True)
        return old_step(session,action,label)
    start=time.monotonic()
    DR.Session.__init__,Executor.step_audit,DR.Session._step=init_observed,audit_observed,step_observed
    try:
        verify_pins(pins)
        variation=DS.Variation.from_json(row['variation'])
        record=DR.run_one(controller,variation,args.output/'native',view=False,save_episode=True)
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        write_new(args.output/'runner_error.json',error)
    finally:
        DR.Session.__init__,Executor.step_audit,DR.Session._step=old_init,old_audit,old_step
    geometry=None
    if recorder is not None:geometry=recorder.finish(args.output,record)
    verify_pins(pins)
    motion=None
    ep=args.output/'native/episode.npz'
    if ep.is_file():
        with np.load(ep,allow_pickle=False) as z:motion=float(z['time'][-1]-z['time'][0])
    full=None if geometry is None else geometry['final_actual_time_s']-geometry['initial_time_s']
    original_ok=bool(record and record.get('admitted'))
    clock_ok=bool(geometry and geometry['every_step_clock_contiguous'] and not geometry['observer_errors'] and
                  geometry['recorded_states']==geometry['primary_executor_step_audit_calls']+1)
    duration_ok=motion is not None and full is not None and 0<motion<=18.+1e-8 and 0<full<=20.+1e-8
    artifacts={}
    for item in args.output.rglob('*'):
        if item.is_file():artifacts[str(item.relative_to(args.output))]=sha(item)
    result=dict(schema='rssr.franka_v3.luna_native_trial_receipt.v1',source_id=row['source_id'],api=api,
        source_episode_index=args.episode_index,output_task=args.task,controller_sha256=sha(controller),
        original_native_admitted=original_ok,original_native_fails=None if not record else record.get('fails'),
        full_physics_duration_s=full,recorded_motion_duration_s=motion,original_settle_s=float(DR.SETTLE_S),
        checks=dict(exact_source_initialization=bool(initial_proof and initial_proof['accepted']),
                    original_all_gates_admitted=original_ok,actual_every_step_observation_clock=clock_ok,
                    motion_le_18s_and_full_physics_le_20s=bool(duration_ok),runner_exception_absent=error is None),
        native_and_duration_accepted=bool(original_ok and clock_ok and duration_ok and error is None),
        full_user_gaze_requirement_verified=None,full_user_topdown_requirement_verified=None,
        publication_admitted=False,candidate_only=True,no_source_dropped=True,no_retry_performed=True,
        input_pins=pins,artifacts=artifacts,wall_s=time.monotonic()-start)
    write_new(args.output/'trial_receipt.json',result)
    print(json.dumps({k:result[k] for k in ('source_id','original_native_admitted','original_native_fails',
        'full_physics_duration_s','recorded_motion_duration_s','native_and_duration_accepted','publication_admitted')}))
    return 0 if result['native_and_duration_accepted'] else 1


if __name__=='__main__':
    raise SystemExit(main())
