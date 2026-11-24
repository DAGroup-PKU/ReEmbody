"""Piper source EEF replay; recorded Cartesian actions are never joint angles."""
import hashlib
import json
import pathlib
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

REVISION = 'source_eef_to_piper_joints_v1'


def frame_contract(config):
    from common import read
    contract=read(pathlib.Path(config['model_root']) / 'source/scene/task_frame_contract.json')
    world=np.asarray(contract['world_from_task'],float)
    for side,shift in config.get('source_base_world_shift_m',{}).items():
        if side not in ('left','right'):raise ValueError('Unknown source side')
        shift=np.asarray(shift,float)
        if shift.shape!=(3,) or not np.isfinite(shift).all() or np.max(abs(shift))>.15:
            raise ValueError('Source registration exceeds declared 15 cm reconstruction bound')
        base=np.asarray(contract['chains'][side]['task_from_source_base'],float)
        base[:3,3]+=world[:3,:3].T@shift
        contract['chains'][side]['task_from_source_base']=base.tolist()
    return contract


def commands(config, model, source):
    from common import read, write
    root = pathlib.Path(config['import_case'])
    contract = frame_contract(config)
    source_hash = hashlib.sha256((root / 'source_action_state.npz').read_bytes()).hexdigest()
    identity = dict(revision=REVISION, source_sha256=source_hash, contract=contract,
                    piper_joint_ranges={model.joint(j).name:model.jnt_range[j].tolist()
                        for j in range(model.njnt) if model.joint(j).name.startswith(('left_', 'right_'))})
    tag = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    cache = root / 'source_eef_ik' / tag
    cache.mkdir(parents=True, exist_ok=True)
    times = np.asarray(source['timestamp']) - source['timestamp'][0]
    raw = np.asarray(source['action'])
    # Imported RSSR data uses RIGHT then LEFT. The HF import manifest records
    # the explicit conversion from the HF LEFT then RIGHT feature order.
    q = np.tile(model.qpos0, (len(times), 1))
    grip = {s:np.clip(raw[:,g] / 1.05, 0, 1) for s,g in [('left',13),('right',6)]}
    data = mujoco.MjData(model)
    if (cache / 'trajectory.npz').exists():
        with np.load(cache / 'trajectory.npz') as saved:
            arms = {s:saved[s].copy() for s in ('left','right')}
        receipt = read(cache / 'receipt.json')
    else:
        started=time.monotonic(); arms={}; diagnostics={}
        world=np.asarray(contract['world_from_task'])
        for side,offset in [('left',7),('right',0)]:
            names=[f'{side}_joint{i}' for i in range(1,7)]
            qa=np.array([model.joint(n).qposadr[0] for n in names])
            va=np.array([model.joint(n).dofadr[0] for n in names])
            joint_ids=[model.joint(n).id for n in names]
            lo,hi=model.jnt_range[joint_ids].T
            lo=lo+1e-4; hi=hi-1e-4
            body=model.body(side+'_link6').id
            base=world@np.asarray(contract['chains'][side]['task_from_source_base'])
            rotations=Rotation.from_euler('xyz',raw[:,offset+3:offset+6]).as_matrix()
            goals_p=raw[:,offset:offset+3]@base[:3,:3].T+base[:3,3]
            goals_r=base[:3,:3]@rotations
            previous=np.clip(np.zeros(6),lo,hi); path=[]; errors=[]
            jp=np.zeros((3,model.nv)); jr=np.zeros_like(jp)
            def residual(k):
                mujoco.mj_forward(model,data)
                ep=goals_p[k]-data.xpos[body]
                eo=Rotation.from_matrix(goals_r[k]@data.xmat[body].reshape(3,3).T).as_rotvec()
                return np.r_[ep,.12*eo]
            def fit(seed,k):
                data.qpos[qa]=np.clip(seed,lo,hi)
                for _ in range(100):
                    e=residual(k); cost=float(e@e)
                    if np.linalg.norm(e[:3])<2e-5 and np.linalg.norm(e[3:])<2e-5:break
                    mujoco.mj_jacBody(model,data,jp,jr,body)
                    J=np.vstack((jp[:,va],.12*jr[:,va]))
                    delta=J.T@np.linalg.solve(J@J.T+1e-7*np.eye(6),e)
                    delta*=min(1.,.12/max(1e-12,np.max(abs(delta))))
                    old=data.qpos[qa].copy(); accepted=False
                    for scale in (1.,.5,.25,.125):
                        data.qpos[qa]=np.clip(old+scale*delta,lo,hi)
                        new=residual(k)
                        if float(new@new)<cost-1e-15:accepted=True;break
                    if not accepted:data.qpos[qa]=old;break
                e=residual(k)
                return float(e@e),data.qpos[qa].copy(),[float(np.linalg.norm(e[:3])),float(np.linalg.norm(e[3:])/.12)]
            for k in range(len(times)):
                candidates=[fit(previous,k)]
                if candidates[0][2][0]>.001 or candidates[0][2][1]>.02:
                    for seed in (np.zeros(6),(lo+hi)/2,np.array([0.,.6,-.7,0.,.1,0.])):
                        candidates.append(fit(seed,k))
                        if candidates[-1][2][0]<.0001 and candidates[-1][2][1]<.001:break
                _,previous,error=min(candidates,key=lambda x:x[0])
                path.append(previous.copy());errors.append(error)
            arms[side]=np.asarray(path); errors=np.asarray(errors)
            diagnostics[side]={'position_error_max_m':float(errors[:,0].max()),
                'orientation_error_max_rad':float(errors[:,1].max()),
                'position_error_p95_m':float(np.percentile(errors[:,0],95)),
                'max_joint_step_rad':float(np.max(abs(np.diff(arms[side],axis=0))))}
        np.savez_compressed(cache/'trajectory.npz',time_s=times,**arms)
        receipt=dict(identity=identity,tracking=diagnostics,wall_s=time.monotonic()-started,
                     source_action_modified=False,source_frame='recorded EEF interpreted as Piper link6 flange',
                     calibration_status='inherited estimated base transforms, not hardware calibration')
        write(cache/'receipt.json',receipt)
    for side in ('left','right'):
        qa=[model.joint(f'{side}_joint{i}').qposadr[0] for i in range(1,7)]
        q[:,qa]=arms[side]
        q[:,model.joint(f'{side}_joint7').qposadr[0]]=.0475*grip[side]
        q[:,model.joint(f'{side}_joint8').qposadr[0]]=-.0475*grip[side]
    config['_source_eef_receipt']=str(cache/'receipt.json')
    return times,q,grip
