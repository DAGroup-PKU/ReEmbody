"""Summarize observed object motion before closure; never infer its cause."""
import numpy as np


def preclosure_motion(arrays,scene,hand,cycle,events):
    times=np.asarray(arrays['time_s'])
    event=events[cycle]
    start=float(events[cycle-1]['release_s']) if cycle else float(times[0])
    close=float(event['closed_s'])
    indices=np.flatnonzero((times>=start-1e-9)&(times<=close+1e-9))
    if not len(indices):
        return {'hand':hand,'cycle':cycle,'available':False,'reason':'No saved states in this source interval'}
    positions=arrays['object_position'];side=('left','right').index(hand)
    particles=set(scene.get('bindings',{}).get('particles',[]))
    objects={}
    for j,obj in enumerate(scene['objects']):
        if obj['id'] in particles:continue
        delta=positions[indices,j]-positions[indices[0],j]
        magnitude=np.linalg.norm(delta,axis=1)
        moving=np.flatnonzero(magnitude>.002)
        contact=np.flatnonzero(arrays['contacts'][indices,j,side].astype(bool))
        objects[obj['id']]={
            'displacement_at_close_world_m':delta[-1].tolist(),
            'max_displacement_from_interval_start_m':float(magnitude.max()),
            'first_displacement_gt_2mm_s':float(times[indices[moving[0]]]) if len(moving) else None,
            'first_this_hand_contact_s':float(times[indices[contact[0]]]) if len(contact) else None,
            'actual_opening_at_first_this_hand_contact_m':float(arrays['gripper_opening'][indices[contact[0]],side]) if len(contact) else None,
        }
    return {'hand':hand,'cycle':cycle,'available':True,'interval_start_s':start,
            'source_closed_s':close,'first_observed_s':float(times[indices[0]]),
            'last_observed_s':float(times[indices[-1]]),'objects':objects,
            'omitted_particle_count':len(particles),
            'semantics':'Saved states from previous source release (first cycle: rollout start) through closure. Motion/contact association is observational, not a cause or success predicate. Free particles are omitted from this summary only.'}
