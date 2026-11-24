"""Unlearned IK seeds from model defaults/limits, with fixed S1 waist constraints."""
import numpy as np

POLICIES=('model_default','joint_midpoint','random')


def initial_guess(default,low,high,policy,rng):
    low,high=np.asarray(low,float),np.asarray(high,float)
    base=np.asarray(default,float)
    if policy not in POLICIES or base.shape!=low.shape or low.shape!=high.shape or len(low)<4:
        raise ValueError('Invalid declared IK initialization')
    if not np.isfinite(np.r_[base,low,high]).all() or np.any(low>high):
        raise ValueError('Finite valid IK bounds required')
    if policy=='random':q=rng.uniform(low,high)
    elif policy=='joint_midpoint':q=(low+high)/2
    else:q=base.copy()
    q=np.clip(q,low,high)
    waist_low=max(low[0],-.28403-high[1]);waist_high=min(high[0],-.28403-low[1])
    if waist_low>waist_high or not low[2]<=.28834<=high[2] or not low[3]<=-.00123<=high[3]:
        raise ValueError('Model bounds incompatible with declared waist constraints')
    # This clamps an IK seed, never an executed state or measured trajectory.
    q[0]=rng.uniform(waist_low,waist_high) if policy=='random' else np.clip(q[0],waist_low,waist_high)
    q[1]=-.28403-q[0];q[2:4]=[.28834,-.00123]
    return q
