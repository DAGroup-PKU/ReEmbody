"""Bounded target command edits; source event times and physics stay immutable."""
import copy, math
import numpy as np

def validate(value):
    if not isinstance(value,dict) or set(value)-{'left','right'}:
        raise ValueError('release_opening must map left/right to event lists')
    out=copy.deepcopy(value)
    for side,items in out.items():
        if not isinstance(items,list) or len(items)>16:raise ValueError('At most 16 release cycles per hand')
        seen=set()
        for x in items:
            if not isinstance(x,dict) or set(x)!={'cycle','opening_floor','ramp_s'}:raise ValueError('release opening requires cycle, opening_floor, ramp_s')
            c=x['cycle']
            if type(c) is not int or c<0 or c in seen:raise ValueError('Unique nonnegative source cycle required')
            seen.add(c)
            for k,lo,hi in [('opening_floor',0.,1.),('ramp_s',.1,1.)]:
                v=x[k]
                if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError('release opening outside bounds: '+k)
    return out

def apply(time_s,source,patch):
    from rssr_physical_feedback import grasp_events
    ts=np.asarray(time_s,float);out={s:np.array(g,float,copy=True) for s,g in source.items()}
    for side,items in validate(patch).items():
        cycles=grasp_events(ts,source[side])
        for x in items:
            c=x['cycle']
            if c>=len(cycles):raise ValueError('Release cycle missing in immutable source')
            start=cycles[c]['release_s']
            if c+1<len(cycles):end=cycles[c+1]['closed_s']
            else:end=float(ts[-1])+x['ramp_s']
            # Smoothly enter after release and leave before next source closure.
            a=np.clip((ts-start)/x['ramp_s'],0.,1.)
            b=np.clip((end-ts)/x['ramp_s'],0.,1.)
            weight=(a*a*(3-2*a))*(b*b*(3-2*b))
            target=np.maximum(np.asarray(source[side]),x['opening_floor'])
            candidate=np.asarray(source[side])+weight*(target-np.asarray(source[side]))
            out[side]=np.maximum(out[side],candidate)
    return out
