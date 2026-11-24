"""Conservative joint-observation alias gate on the exact export sample clock.

This gate rejects a defined near-state/future-chunk pattern. It does not claim
neural feature equality, invalid action labels, or elimination of all partial
observability. No simulated or rendered data is modified.
"""
from pathlib import Path
import argparse, hashlib, json, sys
import mujoco
import numpy as np
from scipy.spatial.distance import cdist

if __package__:
 from ._shared.paths import PROJECT_ROOT, SOURCE_CLOCK_ROOT, SHARED_RUNTIME_ROOT, STATEALIAS_REFERENCE_PACKAGE
else:
 from _shared.paths import PROJECT_ROOT, SOURCE_CLOCK_ROOT, SHARED_RUNTIME_ROOT, STATEALIAS_REFERENCE_PACKAGE
REPO=PROJECT_ROOT
BLENDER=SOURCE_CLOCK_ROOT
sys.path.insert(0,str(SHARED_RUNTIME_ROOT))
sys.path.insert(0,str(BLENDER))
from rssr_render.source_clock import complete_source_clock

VERSION='rssr.v3_human_align.statealias_gate.v2'
PACKAGE=STATEALIAS_REFERENCE_PACKAGE

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(native,output):
 native=Path(native).resolve();output=Path(output).resolve()
 if output.exists():raise FileExistsError(output)
 model=mujoco.MjModel.from_xml_path(str(native/'execute/scene.xml'))
 info=json.loads((PACKAGE/'meta/info.json').read_text())
 sn=info['features']['observation.state']['names'][0];an=info['features']['action']['names'][0]
 assert len(sn)==32 and len(an)==22
 sa=[int(model.joint(n).qposadr[0]) for n in sn];aa=[int(model.joint(n).qposadr[0]) for n in an]
 z=np.load(native/'execute/rollout.npz');cmd=np.load(native/'issued_robot_commands.npz')
 vt,si,ci,queries=complete_source_clock(z['time_s'],cmd['time_s'],float(z['time_s'][-1]),fps=30)
 state=z['qpos'][si][:,sa].astype(np.float32).astype(float)
 action=cmd['qpos'][ci][:,aa].astype(np.float32).astype(float)
 assert state.shape==(len(vt),32) and action.shape==(len(vt),22)
 assert np.isfinite(state).all() and np.isfinite(action).all()
 dist=cdist(state,state,'chebyshev');gaps=vt[None,:]-vt[:,None]
 sensitivity={};strict=[]
 for gap in [0.5,1.0]:
  for threshold in [0.5,1.0,2.0]:
   for horizon in [0.2,0.5,1.0,2.0]:
    steps=int(round(horizon*30));valid=(gaps>=gap-1e-8)&(dist<=np.deg2rad(threshold))
    valid[-steps:,:]=False;valid[:,-steps:]=False
    ii,jj=np.nonzero(valid);conflicts=[]
    for begin in range(0,len(ii),512):
     a,b=ii[begin:begin+512],jj[begin:begin+512]
     offsets=np.arange(steps+1)
     delta=action[a[:,None]+offsets]-action[b[:,None]+offsets]
     peaks=np.max(abs(delta),axis=(1,2));rms=np.sqrt(np.mean(delta*delta,axis=(1,2)))
     for k in np.nonzero(peaks>np.deg2rad(5))[0]:
      ia,ib=int(a[k]),int(b[k]);conflicts.append(dict(frame_a=ia,frame_b=ib,time_a_s=float(vt[ia]),time_b_s=float(vt[ib]),state_max_difference_deg=float(np.rad2deg(dist[ia,ib])),future_chunk_max_target_difference_deg=float(np.rad2deg(peaks[k])),future_chunk_rms_target_difference_deg=float(np.rad2deg(rms[k]))))
    key=f'gap_ge_{gap}s__state_le_{threshold}deg__future_{horizon}s'
    sensitivity[key]=dict(near_state_pairs=len(ii),diverging_chunk_pairs=len(conflicts),examples=conflicts[:8])
    if gap==0.5 and threshold==0.5 and horizon==1.0:strict=conflicts
 unique,counts=np.unique(state.astype(np.float32),axis=0,return_counts=True)
 checks=dict(exact_training_export_clock_used=True,full_32_state_and_22_absolute_joint_action_used=True,
  no_nonadjacent_near_state_with_diverging_1s_chunk=len(strict)==0,all_scanned_values_finite=True)
 inputs=[native/'execute/scene.xml',native/'execute/rollout.npz',native/'issued_robot_commands.npz',PACKAGE/'meta/info.json',Path(__file__),BLENDER/'rssr_render/source_clock.py',BLENDER/'rssr_render/common.py']
 result=dict(schema=VERSION,accepted=all(checks.values()),checks=checks,native_case=str(native),
  strict_definition=dict(state_max_difference_deg=0.5,min_time_separation_s=0.5,future_horizon_s=1.0,action_difference_threshold_deg=5.0,fps=30,future_action_samples='all 30fps labels including observation instant'),
  frames=len(vt),strict_conflicting_pair_count=len(strict),strict_examples=strict[:12],exact_float32_duplicate_state_groups=int(np.sum(counts>1)),
  sensitivity=sensitivity,limitations=['Proximity in joint state is a conservative proxy; images can disambiguate some pairs.','No claim of equal policy representations or complete removal of partial observability.','Actual checkpoint normalization, history wrapper and action horizon were not supplied.','Sensitivity probes are reported separately from strict rejection criteria.'],
  input_pins={str(p):sha(p) for p in inputs})
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.native,a.output);print(json.dumps(dict(accepted=r['accepted'],conflicting_pairs=r['strict_conflicting_pair_count'],output=str(a.output))))
