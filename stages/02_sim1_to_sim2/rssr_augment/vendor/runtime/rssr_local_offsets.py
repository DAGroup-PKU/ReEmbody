"""Bounded Agent-chosen goal displacements in the declared grasp frame."""
import copy
import numpy as np


def validate(value):
    if not isinstance(value,dict) or not value or set(value)-{'left','right'}:
        raise ValueError('Local grasp offsets require named left/right vectors')
    for vector in value.values():
        if (not isinstance(vector,list) or len(vector)!=3 or
            any(type(x) not in (int,float) for x in vector) or
            not np.isfinite(vector).all() or np.max(np.abs(vector))>.16):
            raise ValueError('Local grasp offset must be a finite 3-vector, each axis within .16m')
    return copy.deepcopy(value)


def goals(reference_position,world_offsets,site_rotations,site_from_grasp,local_offset=None):
    """Shift the IK site, rotating the optional displacement with its goal pose.

    All existing world/phase/event and new local displacements share the same
    original .16 m per-world-axis bound. No contact point or value is chosen.
    """
    delta=np.asarray(world_offsets,float).copy()
    if local_offset is not None:
        local=np.asarray(local_offset,float);conversion=np.asarray(site_from_grasp,float)
        if (local.shape!=(3,) or not np.isfinite(local).all() or np.max(abs(local))>.16 or
            conversion.shape!=(3,3) or not np.isfinite(conversion).all() or
            not np.allclose(conversion.T@conversion,np.eye(3),atol=1e-6,rtol=0) or
            not np.isclose(np.linalg.det(conversion),1.,atol=1e-6,rtol=0)):
            raise ValueError('Finite local displacement and proper declared grasp rotation required')
        if np.any(local):
            delta=delta+np.asarray(site_rotations)@conversion@local
    if np.max(np.abs(delta))>.16+1e-10:
        raise ValueError('Combined local, world and event displacement exceeds .16m')
    return np.asarray(reference_position)+delta
