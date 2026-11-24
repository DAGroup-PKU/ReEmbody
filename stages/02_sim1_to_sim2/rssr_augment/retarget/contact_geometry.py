"""Scene-generic proximity measurement for a gripper contact event.

The measured point is the midpoint of the two finger pads. A surface can be
near that point while the center of a finite-sized object is several cm away.
This measurement only makes an event eligible; physical contact and the native
task predicates still decide whether it worked.
"""
from __future__ import annotations

import numpy as np


CONTACT_START_CLEARANCE_M = .020


def contact_clearance_m(obs: dict, object_id: str, hand: str,
                        target_offset_m=(0., 0., 0.)) -> float:
    point = np.asarray(obs['finger_center_m'][hand], dtype=float)
    center = (np.asarray(obs['object_position_m'][object_id], dtype=float) +
              np.asarray(target_offset_m, dtype=float))
    direct = float(np.linalg.norm(point - center))
    geometry = obs.get('object_geometry', {}).get(object_id, {})
    shape = geometry.get('shape')
    if shape not in ('box', 'cup', 'cylinder'):
        return direct
    size = np.asarray(geometry.get('size_m'), dtype=float)
    rotation = np.asarray(obs.get('object_rotation_world', {}).get(object_id),
                          dtype=float)
    if (size.shape != (3,) or rotation.shape != (3, 3) or
            not np.isfinite(size).all() or not np.isfinite(rotation).all() or
            np.any(size <= 0)):
        return direct
    local = rotation.T @ (point - center)
    if shape == 'box':
        outside = np.maximum(np.abs(local) - size / 2, 0.)
        return float(np.linalg.norm(outside))
    radial = max(float(np.linalg.norm(local[:2])) - float(max(size[:2]) / 2), 0.)
    vertical = max(abs(float(local[2])) - float(size[2] / 2), 0.)
    return float(np.hypot(radial, vertical))
