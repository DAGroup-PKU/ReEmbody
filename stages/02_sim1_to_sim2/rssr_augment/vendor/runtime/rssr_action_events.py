"""Smooth bounded corrections tied to every source grasp/release cycle."""
import copy
import numpy as np
from rssr_physical_feedback import grasp_events


def merge_patch(base,patch):
    """Patch named source cycles, retaining untouched fields and other cycles.

    Omitted fields in a newly introduced cycle mean zero correction. Existing
    corrections require explicit zero vectors to reset. No values are inferred
    from an episode's success or from a previous case.
    """
    out=validate(base)
    if not isinstance(patch,dict) or set(patch)-{'left','right'}:
        raise ValueError('Event controls must name robot hands')
    fields=('grasp_offset_m','release_offset_m','grasp_rotation_degrees','release_rotation_degrees')
    for side,rows in patch.items():
        if not isinstance(rows,list) or not 1<=len(rows)<=16:
            raise ValueError('Event patch requires one to sixteen source cycles per hand')
        current={row['cycle']:row for row in out.get(side,[])};seen=set()
        for row in rows:
            if not isinstance(row,dict) or 'cycle' not in row or set(row)-{'cycle',*fields}:
                raise ValueError('Event patch requires a cycle and only known correction fields')
            cycle=row['cycle']
            if type(cycle) is not int or not 0<=cycle<16 or cycle in seen:
                raise ValueError('Invalid/duplicate source cycle index')
            seen.add(cycle)
            entry=copy.deepcopy(current.get(cycle,dict(cycle=cycle,**{key:[0.,0.,0.] for key in fields})))
            entry.update(copy.deepcopy(row));current[cycle]=entry
        out[side]=[current[cycle] for cycle in sorted(current)]
    return validate(out)


def validate(value):
    if not isinstance(value,dict) or set(value)-{'left','right'}:raise ValueError('Event controls must name robot hands')
    out=copy.deepcopy(value)
    for side,rows in out.items():
        if not isinstance(rows,list) or len(rows)>16:raise ValueError('At most 16 source cycles per hand')
        cycles=set()
        for row in rows:
            fields={'cycle','grasp_offset_m','release_offset_m','grasp_rotation_degrees','release_rotation_degrees'}
            if not isinstance(row,dict) or set(row)!=fields:raise ValueError('Each event needs a cycle and four explicit 3-vectors')
            if type(row['cycle']) is not int or not 0<=row['cycle']<16 or row['cycle'] in cycles:raise ValueError('Invalid/duplicate source cycle index')
            cycles.add(row['cycle'])
            for key in fields-{'cycle'}:
                a=np.asarray(row[key],float);limit=.08 if key.endswith('_m') else 35.
                if a.shape!=(3,) or not np.isfinite(a).all() or np.abs(a).max()>limit:raise ValueError('Event correction exceeds fixed tool boundary')
    return out


def corrections(time_s,command,rows):
    times=np.asarray(time_s,float);cycles=grasp_events(times,command)
    translation=np.zeros((len(times),3));rotation=np.zeros_like(translation)
    validate({'left':rows})
    for row in rows:
        i=row['cycle']
        if i>=len(cycles):raise ValueError('Requested source grasp cycle does not exist')
        event=cycles[i];close,release=event['closed_s'],event['release_s']
        prior=cycles[i-1]['release_s'] if i else float(times[0])
        following=cycles[i+1]['closed_s'] if i+1<len(cycles) else float(times[-1])
        start=max(float(times[0]),min(close-.75,(prior+close)/2))
        end=min(float(times[-1]),max(release+.75,(release+following)/2))
        # Midpoint boundaries keep adjacent cycle windows disjoint.
        if i:start=max(start,(prior+close)/2)
        if i+1<len(cycles):end=min(end,(release+following)/2)
        for out,field in [(translation,'offset_m'),(rotation,'rotation_degrees')]:
            anchors={start:np.zeros(3),close:np.asarray(row['grasp_'+field]),release:np.asarray(row['release_'+field])}
            if end>release:anchors[end]=np.zeros(3)
            keys=sorted(anchors)
            for a,b in zip(keys,keys[1:]):
                mask=(times>=a)&(times<=b);u=(times[mask]-a)/(b-a);w=(u*u*(3-2*u))[:,None]
                out[mask]=(1-w)*anchors[a]+w*anchors[b]
    return translation,rotation
