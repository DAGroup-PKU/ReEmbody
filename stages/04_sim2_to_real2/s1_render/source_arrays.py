"""Normalize exactly one initial native state without changing its coordinates."""
import numpy as np


def one_frame_vector(value,width,name):
    result=np.asarray(value)
    if result.shape==(1,width):result=result[0]
    if result.shape!=(width,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain exactly one finite state of width {width}, got {result.shape}')
    return result.copy()
