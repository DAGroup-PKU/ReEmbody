"""Read-only first-grasp diagnostics; never execute commands or alter evidence.

The existing task judge is intentionally unchanged. This adds stricter acquisition
diagnostics, with explicit sampling limits, using saved native states and commands.
"""
import concurrent.futures as cf
import collections
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import time
import numpy as np
import mujoco

SOURCE = Path(os.environ.get('RSSR_PROJECT_ROOT', Path(__file__).resolve().parents[3])) / 'result/sim1-sim2/s1_heightfit_3x1_20261017' 
OUT = Path(__file__).parent
HANDS = ('left', 'right')
OBJECTS = ('cup_left_base', 'cup_right_upper')
CRITERIA = {
    'sampling': 'All saved native rollout states, nominal 100 Hz; no interpolation of contact.',
    'force_reconstruction': 'MuJoCo mj_forward at saved qpos/qvel/ctrl. Solver warmstart is unavailable; forces are diagnostic reconstructions, not exact replay evidence.',
    'distal_mapping': 'Exact same physical distal bodies as native session._observation; do not trust legacy rollout name-only finger_contact_sides.',
    'loaded_pad_minimum_n': 0.0001,
    'bilateral_acquisition_duration_s': 0.1,
    'lift_above_initial_m': 0.025,
    'lift_hold_duration_s': 0.5,
    'maximum_contact_gap_s': 0.1,
    'command_reopen_delta_rad': 0.12,
    'actual_reopen_delta_m': 0.008,
    'command_reclose_delta_rad': 0.06,
    'slip_drop_m': 0.015,
    'relative_object_motion_review_m': 0.015,
    'first_intent': 'First target/hand contact or grasp action, excluding initial SDK settle and prior approach touches.',
    'cycle': 'Persistent contact/grasp/hold actions remain one closure. A new cycle requires explicit target-hand open intent (approach/retreat/release), commanded reopening >=0.12rad accompanied by >=8mm physical jaw opening, followed by a new commanded close >=0.06rad.',
    'first_success': 'First intentional close obtains >=0.1s loaded bilateral distal contact, reaches 25mm lift in same close, and holds lifted/contacted for >=0.5s; no reopening/reapproach before proof.',
    'retention': 'Separately report >0.1s loaded-contact losses or relative object motion before first intentional release; this does not replace the unchanged task acceptance judge.',
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def runs(t, mask):
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    dt = float(np.median(np.diff(t)))
    return [(int(a), int(b), float(t[b-1]-t[a]+dt))
            for a,b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def first_run(t, mask, duration):
    return next((a for a,b,d in runs(t, mask) if d >= duration-1e-6), None)


def summarize_hand(t, obj_position, pad_force, finger_local, opening, cmd_angle,
                   event_kind, event_obj, event_hand, hi, packets):
    obj, hand = OBJECTS[hi], HANDS[hi]
    intended = (event_obj == obj) & (event_hand == hand)
    acquisition = intended & np.isin(event_kind, ['contact', 'grasp'])
    open_intent = intended & np.isin(event_kind, ['approach', 'retreat', 'release'])
    indices = np.flatnonzero(acquisition)
    if not len(indices):
        return {'object':obj,'hand':hand,'status':'no_intentional_acquisition','needs_optimization':True}
    start = int(indices[0]); dt = float(np.median(np.diff(t)))
    release_indices = np.flatnonzero(intended & (event_kind == 'release') & (np.arange(len(t)) >= start))
    release = int(release_indices[0]) if len(release_indices) else len(t)
    end = release
    loaded = (pad_force[:, hi] > CRITERIA['loaded_pad_minimum_n']).all(axis=-1)
    unilateral = (pad_force[:, hi] > CRITERIA['loaded_pad_minimum_n']).sum(axis=-1) == 1
    lift = obj_position[:, hi, 2] - obj_position[0, hi, 2]
    # A genuine reopening must be both commanded and physically observed.
    starts=[start]; opens=[]; phase='closed'; peak=cmd_angle[start,hi]
    peak_opening=opening[start,hi]; trough=peak; reopen_opening=peak_opening
    for i in range(start+1,end):
        angle=cmd_angle[i,hi]; aperture=opening[i,hi]
        if phase=='closed':
            if angle>peak:
                peak=angle;peak_opening=aperture
            if (open_intent[i] and peak-angle >= CRITERIA['command_reopen_delta_rad'] and
                    aperture-peak_opening >= CRITERIA['actual_reopen_delta_m']):
                opens.append(i);phase='open';trough=angle;reopen_opening=aperture
        else:
            trough=min(trough,angle);reopen_opening=max(reopen_opening,aperture)
            if acquisition[i] and angle-trough>=CRITERIA['command_reclose_delta_rad']:
                starts.append(i);phase='closed';peak=angle;peak_opening=aperture
    first_end=opens[0] if opens else end
    scope=(np.arange(len(t))>=start)&(np.arange(len(t))<first_end)
    acquired=first_run(t,loaded&scope,CRITERIA['bilateral_acquisition_duration_s'])
    lifted=first_run(t,loaded&scope&(lift>=CRITERIA['lift_above_initial_m']),CRITERIA['lift_hold_duration_s'])
    # Retreat/reapproach after closing and before stable lift is recovery even
    # if a software command happened to preserve the jaw opening.
    proof_end=lifted if lifted is not None else first_end
    recovery=(intended & np.isin(event_kind,['retreat','approach']) &
              (np.arange(len(t))>start)&(np.arange(len(t))<proof_end))
    recovery_spans=[(a,b,d) for a,b,d in runs(t,recovery) if d>=.10]
    first_success=lifted is not None and not recovery_spans
    qualified=first_run(t,loaded&(lift>=CRITERIA['lift_above_initial_m'])&
                        (np.arange(len(t))>=start)&(np.arange(len(t))<end),.1)
    loss_spans=[];slip_spans=[];max_relative_motion=0.
    if qualified is not None:
        retain=(np.arange(len(t))>=qualified)&(np.arange(len(t))<end)
        loss_spans=[(a,b,d) for a,b,d in runs(t,(~loaded)&retain) if d>CRITERIA['maximum_contact_gap_s']+1e-6]
        for a,b,dur in loss_spans:
            before=max(qualified,a-5);after=min(end-1,b+50)
            drop=float(np.max(obj_position[before:a+1,hi,2])-np.min(obj_position[a:after+1,hi,2]))
            relative=float(np.max(np.linalg.norm(finger_local[a:after+1,hi]-finger_local[before,hi],axis=1)))
            if drop>=CRITERIA['slip_drop_m'] and relative>=CRITERIA['relative_object_motion_review_m']:
                slip_spans.append({'start_s':float(t[a]),'end_s':float(t[b-1]+dt),'drop_m':drop,'relative_motion_m':relative})
        max_relative_motion=float(np.max(np.linalg.norm(finger_local[qualified:end,hi]-finger_local[qualified,hi],axis=1))) if end>qualified else 0.
    # Only native observation data are read from packets; no model requests.
    evidence=[]
    for po in packets:
        if t[start]-.02<=po['time_s']<=(t[end] if end<len(t) else t[-1]):
            ff=po.get('finger_normal_force_n',{}).get(obj,{}).get(hand)
            if ff is not None:
                evidence.append({'time_s':po['time_s'],'finger_contact':po['finger_contact'][obj][hand],
                                 'finger_normal_force_n':ff,'opening_m':po['gripper_opening_m'][hand],
                                 'object_z_m':po['object_position_m'][obj][2],
                                 'support':po['support_contact'][obj]})
    if first_success:status='first_close_acquire_lift_hold_pass'
    elif acquired is None:
        status='first_close_unilateral' if np.any(unilateral&scope) else 'first_close_empty_or_no_loaded_bilateral'
    else:status='first_close_no_stable_lift_or_recovery'
    return {
        'object':obj,'hand':hand,'status':status,'first_close_success':bool(first_success),
        'first_intent_s':float(t[start]),'first_reopen_s':float(t[opens[0]]) if opens else None,
        'first_release_intent_s':float(t[release]) if release<len(t) else None,
        'intent_close_count':len(starts),'intent_close_start_s':[float(t[i]) for i in starts],
        'commanded_reopen_count':len(opens),'commanded_reopen_s':[float(t[i]) for i in opens],
        'first_close_minimum_aperture_m':float(np.min(opening[start:first_end,hi])) if first_end>start else None,
        'first_close_maximum_lift_m':float(np.max(lift[start:first_end])) if first_end>start else None,
        'first_loaded_bilateral_s':float(t[acquired]) if acquired is not None else None,
        'first_25mm_lift_held_500ms_start_s':float(t[lifted]) if lifted is not None else None,
        'first_close_longest_loaded_bilateral_s':max((d for a,b,d in runs(t,loaded&scope)),default=0.),
        'recovery_before_first_proof':[{'start_s':float(t[a]),'end_s':float(t[b-1]+dt)} for a,b,d in recovery_spans],
        'contact_loss_before_release':[{'start_s':float(t[a]),'end_s':float(t[b-1]+dt),'duration_s':d} for a,b,d in loss_spans],
        'slip_candidates':slip_spans,'max_object_motion_in_gripper_frame_m':max_relative_motion,
        'relative_motion_review':max_relative_motion>CRITERIA['relative_object_motion_review_m'],
        'needs_optimization':bool(not first_success or len(starts)>1 or recovery_spans or slip_spans),
        'native_packet_force_evidence':evidence,
    }


def process(receipt_path):
    receipt_path=Path(receipt_path);a=receipt_path.parent;r=json.loads(receipt_path.read_text())
    model=mujoco.MjModel.from_xml_path(str(a/'execute/scene.xml'));data=mujoco.MjData(model)
    z=np.load(a/'execute/rollout.npz');t=z['time_s'];q=z['qpos'];v=z['qvel'];ctrl=z['ctrl'];opening=z['gripper_opening'];pos=z['object_position']
    commands=np.load(a/'issued_robot_commands.npz');ct=commands['time_s'];cq=commands['qpos'];idx=np.clip(np.searchsorted(ct,t),0,len(ct)-1)
    qa=[int(model.joint(f'astribot_gripper_{h}_joint_L1').qposadr[0]) for h in HANDS];cmd_angle=cq[idx][:,qa]
    del cq,commands
    history=r['event_history'];ends=np.asarray([e['time_s'] for e in history]);ei=np.minimum(np.searchsorted(ends,t,side='right'),len(history)-1)
    event_kind=np.asarray([history[i]['event_id'].split(':')[0] for i in ei]);event_obj=np.asarray([history[i].get('selected_object_id') or '' for i in ei]);event_hand=np.asarray([history[i].get('selected_hand') or '' for i in ei])
    object_bodies={model.body(obj).id:oi for oi,obj in enumerate(OBJECTS)}
    distal={};tip_geoms=[];hand_bodies=[]
    for hi,h in enumerate(HANDS):
        tips=[model.geom(f'astribot_gripper_{h}_geom_{s}').id for s in ('L112','R112')]
        tip_geoms.append(tips);hand_bodies.append(model.body(f'astribot_gripper_{h}_base').id)
        for side,g in enumerate(tips):distal[int(model.geom_bodyid[g])]=(hi,side)
    forces=np.zeros((len(t),2,2));finger_local=np.zeros((len(t),2,3));contact=np.zeros((len(t),2,2),bool)
    for k in range(len(t)):
        data.qpos[:]=q[k];data.qvel[:]=v[k];data.ctrl[:]=ctrl[k];data.time=t[k];mujoco.mj_forward(model,data)
        for hi in range(2):
            center=data.geom_xpos[tip_geoms[hi]].mean(axis=0)
            rot=data.xmat[hand_bodies[hi]].reshape(3,3)
            finger_local[k,hi]=rot.T@(pos[k,hi]-center)
        for ci,c in enumerate(data.contact):
            if c.dist>0:continue
            for geom,other in ((int(c.geom1),int(c.geom2)),(int(c.geom2),int(c.geom1))):
                oi=object_bodies.get(int(model.geom_bodyid[geom]));ds=distal.get(int(model.geom_bodyid[other]))
                if oi is not None and ds is not None and oi==ds[0]:
                    hi,side=ds;contact[k,hi,side]=True;f=np.zeros(6);mujoco.mj_contactForce(model,data,ci,f)
                    forces[k,hi,side]=max(forces[k,hi,side],max(0.,float(f[0])))
    packets=[]
    for packet in sorted((a/'decisions').glob('*/packet.json')):
        packets.append(json.loads(packet.read_text())['observation'])
    diagnostics=[summarize_hand(t,pos,forces,finger_local,opening,cmd_angle,event_kind,event_obj,event_hand,hi,packets) for hi in range(2)]
    legacy=np.stack([z['finger_contact_sides'][:,hi,hi,:] for hi in range(2)],axis=1).astype(bool)
    result={'group':a.parents[2].name,'case_id':r['case_id'],'accepted':r['accepted'],
            'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
            'scene_sha256':digest(a/'execute/scene.xml'),'hands':diagnostics,
            'needs_optimization':any(h['needs_optimization'] for h in diagnostics),
            'legacy_finger_side_samples_disagree':int(np.any(contact!=legacy,axis=(1,2)).sum()),
            'rollout_samples':len(t),'task_required_failures':r['native']['required_failures']}
    stem=f"{result['group']}_{r['case_id']}"
    (OUT/'cases').mkdir(exist_ok=True)
    (OUT/'cases'/f'{stem}.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'cases'/f'{stem}_diagnostics.npz',time_s=t,pad_force_n=forces,
                        distal_contact=contact,object_in_finger_frame_m=finger_local,
                        desired_gripper_angle_rad=cmd_angle,event_kind=event_kind,
                        event_object=event_obj,event_hand=event_hand)
    return result


def summarize_saved(receipt_path):
    """Refine diagnostic criteria using unchanged, previously derived FK/forces."""
    receipt_path=Path(receipt_path);a=receipt_path.parent
    stem=f"{a.parents[2].name}_{a.parent.name}"
    report_path=OUT/'cases'/f'{stem}.json'
    r=json.loads(report_path.read_text())
    assert r['receipt_sha256']==digest(receipt_path)
    assert r['scene_sha256']==digest(a/'execute/scene.xml')
    z=np.load(OUT/'cases'/f'{stem}_diagnostics.npz')
    native=np.load(a/'execute/rollout.npz')
    packets=[json.loads(p.read_text())['observation']
             for p in sorted((a/'decisions').glob('*/packet.json'))]
    r['hands']=[summarize_hand(z['time_s'],native['object_position'],z['pad_force_n'],
                z['object_in_finger_frame_m'],native['gripper_opening'],
                z['desired_gripper_angle_rad'],z['event_kind'],z['event_object'],
                z['event_hand'],hi,packets) for hi in range(2)]
    r['needs_optimization']=any(h['needs_optimization'] for h in r['hands'])
    report_path.write_text(json.dumps(r,indent=2)+'\n')
    return r


def main():
    receipts=sorted(SOURCE.glob('physical_??/cases/*/attempt_00/receipt.json'))
    assert len(receipts)==150
    (OUT/'criteria.json').write_text(json.dumps(CRITERIA,indent=2)+'\n')
    results=[];errors=[];started=time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        worker=summarize_saved if '--summarize-saved' in sys.argv else process
        jobs={pool.submit(worker,str(p)):p for p in receipts}
        for future in cf.as_completed(jobs):
            try:results.append(future.result())
            except Exception as exc:errors.append({'receipt':str(jobs[future]),'error':repr(exc)})
            if (len(results)+len(errors))%10==0:
                print(json.dumps({'done':len(results),'errors':len(errors),'seconds':round(time.monotonic()-started,1)}),flush=True)
    results.sort(key=lambda r:(r['group'],r['case_id']))
    rows=[]
    for r in results:
        for h in r['hands']:
            rows.append({'group':r['group'],'case_id':r['case_id'],'accepted':r['accepted'],
                         **{k:h.get(k) for k in ['hand','status','needs_optimization','first_close_success','first_intent_s','intent_close_count','commanded_reopen_count','first_close_minimum_aperture_m','first_close_maximum_lift_m','first_loaded_bilateral_s','first_25mm_lift_held_500ms_start_s','first_close_longest_loaded_bilateral_s','max_object_motion_in_gripper_frame_m','relative_motion_review']},
                         'contact_loss_count':len(h.get('contact_loss_before_release',[])),
                         'slip_candidate_count':len(h.get('slip_candidates',[])),
                         'recovery_before_proof_count':len(h.get('recovery_before_first_proof',[])),
                         'report_path':str(OUT/'cases'/f"{r['group']}_{r['case_id']}.json")})
    with (OUT/'all_150_hands.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'schema':'rssr.first_grasp_readonly_audit.v2','source_root':str(SOURCE),
             'cases_expected':150,'cases_audited':len(results),'errors':errors,
             'cases_needing_optimization':sum(r['needs_optimization'] for r in results),
             'accepted_cases_needing_optimization':sum(r['accepted'] and r['needs_optimization'] for r in results),
             'hand_status_counts':dict(collections.Counter(row['status'] for row in rows)),
             'hands_with_multiple_closes':sum((row['intent_close_count'] or 0)>1 for row in rows),
             'hands_with_slip_candidates':sum(row['slip_candidate_count']>0 for row in rows),
             'cases_with_legacy_contact_disagreement':sum(r['legacy_finger_side_samples_disagree']>0 for r in results),
             'wall_seconds':time.monotonic()-started,'criteria_path':str(OUT/'criteria.json'),
             'audit_script_sha256':digest(__file__)}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (OUT/'all_150_cases.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
