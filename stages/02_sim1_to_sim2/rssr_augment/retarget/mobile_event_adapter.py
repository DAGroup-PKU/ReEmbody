"""Offer robot capability events from measured contacts and copied state.

The S1 adapter proposes options only. It does not select a task strategy,
author a RuleAgent rule, drive physics, or count a task success.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from rssr_augment.shared.family_autoresearch.store import digest

from .events import EventDefinition
from .executor import desired_rotation
from .mobility_planner import BASE_NAMES, propose_reposition


def compile_reposition_events(model, obs: dict,
                              task_events: list[EventDefinition],
                              feasible_events: list[EventDefinition]):
    """Bind copied-state base candidates to current, contract-feasible pickups.

    The capability module is intentionally independent of task-family names.
    The caller must keep its certificate, original state hash, and candidate
    model hash with the run. Later states need a fresh proposal; each event
    expires under the immutable guard if its arm/base start pose moves.
    """
    qpos = np.asarray(obs['robot_qpos'], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError('Robot capability scene differs from live observation')
    groups = {}
    for event in feasible_events:
        action = event.action
        if action.event == 'approach':
            groups.setdefault((action.hand, action.object_id), event)
    proposals, evidence = [], []
    for (hand, obj), first in groups.items():
        if obj not in obs['object_position_m']:
            raise ValueError('Feasible approach lacks measured object')
        center = np.asarray(obs['object_position_m'][obj], dtype=float)
        points = {}
        for event in task_events:
            action = event.action
            if (action.event == 'approach' and action.hand == hand and
                    action.object_id == obj and action.recovery_mode != 'overhead'):
                points[event.event_id] = center + np.asarray(
                    action.target_offset_m, dtype=float)
        geometry = obs.get('object_geometry', {}).get(obj, {})
        if geometry.get('mobility') == 'free':
            half = np.asarray(geometry['size_m'], dtype=float) / 2.
            points['near_surface'] = center + np.array([0., half[1] + .005, 0.])
            points['near_surface_above'] = points['near_surface'] + \
                np.array([0., 0., .015])
        # Keep the physical contact menu bounded and reproducible. The
        # planner is an adapter, so RuleAgent still chooses an offered ID.
        points = dict(list(points.items())[:7])
        result = propose_reposition(
            model, qpos, hand, points, desired_rotation(obj, obs))
        if not result['path_valid_candidates']:
            result = propose_reposition(
                model, qpos, hand, points, desired_rotation(obj, obs),
                max_path_checks=48, search_profile='expanded')
        certificate = digest({'source_qpos': qpos.tolist(),
                              'contact_points': {k: v.tolist()
                                                 for k, v in points.items()},
                              'planner_output': result})
        evidence.append({'hand': hand, 'object_id': obj,
                         'source_event_id': first.event_id,
                         'source_qpos_sha256': digest(qpos.tolist()),
                         'certificate_sha256': certificate,
                         'contact_points_world_m': {k: v.tolist()
                                                    for k, v in points.items()},
                         'planner_output': result})
        for rank, candidate in enumerate(result['path_valid_candidates']):
            order = tuple(BASE_NAMES.index(name)
                          for name in candidate['path']['order'])
            if sorted(order) != [0, 1, 2]:
                raise ValueError('Capability path has invalid base axis order')
            action = replace(
                first.action, event='reposition_base', target_id=None,
                target_offset_m=(0., 0., 0.), gripper_fraction=None,
                recovery_mode=None, max_speed_m_s=.08,
                base_joint_target=tuple(float(v) for v in
                                        candidate['base_joint_target']),
                base_axis_order=order,
                planned_base_qpos=tuple(float(v) for v in
                                        obs['base_joint_qpos']),
                planned_arm_qpos=tuple(float(v) for v in
                                       obs['arm_joint_qpos'][hand]),
                planned_object_position_m=tuple(float(v) for v in center),
                base_certificate_sha256=certificate)
            proposals.append(EventDefinition(
                f'reposition_base:{obj}:{hand}:candidate{rank}', action,
                'copied-state reachability proposal for contact region '
                f"{candidate['contact_region']}; source certificate "
                f'{certificate[:16]}; static path table and other scene '
                'contacts at most 3 mm',
                'base progress stalls, unexpected contact, or hard margin worsens',
                first.source_predicate))
    return proposals, evidence
