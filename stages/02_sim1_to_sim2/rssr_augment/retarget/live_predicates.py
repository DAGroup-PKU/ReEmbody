"""Measured runtime margins using frozen predicate geometry and time windows.

These observations guide decisions. Only the complete native rollout report and
independent replay certify success. Historical events latch; final relations
are continuously remeasured and can regress.
"""
from collections import deque
import itertools
import numpy as np

SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)))
RELATIONS = {'inside_region', 'stable_supported_on', 'cup_nested', 'inside'}


class LivePredicates:
    def __init__(self, contract, scene):
        self.contract = contract
        self.objects = {o['id']: o for o in scene['objects']}
        self.initial = None
        self.history = deque()
        self.latched = set()
        self.protected = set()
        self.notified_lost = set()
        self.open_seen = set()
        self.insert_seen = set()
        self.previous_inside = {}
        self.base_times = {}
        self.stack_times = {}
        self.last_rows = {}
        self.last_sample_time = None
        self.max_window = max([1.0] + [float(p.get('window_s', p.get('hold_s', .2)))
                                      for p in contract['predicates']])

    def _corners(self, obj, target, obs):
        pos, rot = obs['object_position_m'], obs['object_rotation_world']
        size = np.asarray(self.objects[obj]['size_m'])
        if (self.contract.get('geometry_evaluation') == 'sphere_analytic_v1' and
                self.objects[obj].get('shape') in ('sphere', 'grain')):
            center = rot[target].T @ (pos[obj] - pos[target])
            return center + SIGNS * size[0] / 2
        world = (SIGNS * size / 2) @ rot[obj].T + pos[obj]
        return (world - pos[target]) @ rot[target]

    def relation(self, pred, obs):
        obj = pred['object']; kind = pred['kind']
        target = pred.get('region', pred.get('support', pred.get('container')))
        corners = self._corners(obj, target, obs)
        size = np.asarray(self.objects[target]['size_m'])
        relative = obs['object_rotation_world'][target].T @ (
            obs['object_position_m'][obj] - obs['object_position_m'][target])
        if kind == 'inside_region':
            margin = float(np.min(size[:2]/2 + pred.get('tolerance_m', .002)
                                  - np.abs(corners[:, :2]).max(axis=0)))
            return margin, {'xy_clearance_m': margin}
        if kind == 'stable_supported_on':
            xy = float(np.min(size[:2]/2 - pred['center_margin_m'] - np.abs(relative[:2])))
            height = float(pred['z_tolerance_m'] - abs(corners[:, 2].min()-size[2]/2))
            return min(xy, height), {'center_clearance_m': xy, 'height_clearance_m': height}
        if kind == 'cup_nested':
            upper = np.asarray(self.objects[obj]['size_m'])
            wall = self.objects[target]['wall_m']
            radial = float(size[0]/2-wall-upper[0]/2+pred['radial_tolerance_m']
                           - np.linalg.norm(relative[:2]))
            depth = size[2]/2 - (relative[2]-upper[2]/2)
            floor = float(size[2]-wall-depth + pred.get('floor_penetration_tolerance_m', 0.))
            overlap = float(depth-pred['minimum_depth_m'])
            axis = obs['object_rotation_world'][target].T @ obs['object_rotation_world'][obj][:, 2]
            tilt = float(pred['maximum_tilt_degrees']-np.degrees(np.arccos(np.clip(axis[2],-1,1))))
            # Convert degrees to radians only to combine signs. Preserve units
            # in individual measurements; a positive scalar requires all four.
            return min(radial, floor, overlap, np.radians(tilt)), {
                'radial_clearance_m': radial, 'floor_clearance_m': floor,
                'overlap_clearance_m': overlap, 'tilt_clearance_degrees': tilt}
        if kind == 'inside':
            wall = self.objects[target].get('wall_m', 0.)
            side_clearance = size[:2]/2-wall+.002-np.abs(corners[:, :2]).max(axis=0)
            xy = float(np.min(side_clearance))
            measurements = {
                'horizontal_clearance_m': xy,
                'center_offset_target_local_m': relative.astype(float).tolist(),
                'vertical_bottom_clearance_m': float(
                    corners[:, 2].min()+size[2]/2-wall+.004),
                'vertical_top_clearance_m': float(
                    size[2]/2+.003-corners[:, 2].max()),
            }
            if self.objects[target].get('shape') in ('cup','bowl'):
                xy = float(size[0]/2-wall+.002 - np.linalg.norm(corners[:, :2],axis=1).max())
                measurements['horizontal_clearance_m'] = xy
                measurements['radial_clearance_m'] = xy
            else:
                measurements['axis_clearance_target_local_m'] = side_clearance.astype(float).tolist()
            return min(xy, measurements['vertical_bottom_clearance_m'],
                       measurements['vertical_top_clearance_m']), measurements
        raise ValueError('Unsupported live geometry '+kind)

    def _window(self, now, seconds):
        complete = now-self.initial['time_s'] >= seconds-1e-8
        return [r for r in self.history if r['time'] >= now-seconds-1e-9], complete

    def __call__(self, obs):
        if self.initial is None:
            self.initial = {'time_s': obs['time_s'], 'positions': {
                k:v.copy() for k,v in obs['object_position_m'].items()}}
        now = obs['time_s']
        samples = {}
        measured = {}
        for p in self.contract['predicates']:
            pid, kind = p['id'], p['kind']; obj = p.get('object')
            if kind in RELATIONS:
                samples[pid], measured[pid] = self.relation(p, obs)
            elif kind == 'initial_outside':
                if self.initial['time_s'] == now:
                    if self.relation({'kind':'inside','object':obj,'container':p['container']},obs)[0]<0:
                        self.latched.add(pid)
                samples[pid]=1. if pid in self.latched else -1.
            elif kind == 'lift_with_contact':
                lift = float(obs['object_position_m'][obj][2]-self.initial['positions'][obj][2])
                contact = obs.get('hand_contact', {}).get(obj, {}).get(p['hand'],
                    any(obs['finger_contact'][obj][p['hand']]))
                samples[pid] = lift-p['height_m'] if contact else -p['height_m']
                measured[pid] = {'lift_m':lift,'hand_contact':contact}
                if samples[pid] >= 0: self.latched.add(pid)
            elif kind == 'object_contact':
                samples[pid] = float(p['support'] in obs['support_contact'][obj])
            elif kind == 'released':
                samples[pid] = float(not any(obs.get('hand_contact', {}).get(name, {}).get(hand, any(pads))
                                            for name in p['objects']
                                            for hand,pads in obs['finger_contact'][name].items()))
            elif kind == 'stable':
                samples[pid] = 0.
            elif kind == 'open_insert_close':
                fraction = float(obs['articulation_fraction'][p['joint']])
                if fraction >= p['open_fraction']: self.open_seen.add(pid)
                inside = self.relation({'kind':'inside','object':obj,'container':p['container']},obs)[0]>=0
                if inside and not self.previous_inside.get(pid, False) and now>self.initial['time_s']:
                    if pid in self.open_seen and fraction>=p['open_fraction']:
                        self.insert_seen.add(pid)
                self.previous_inside[pid] = inside
                samples[pid] = p['closed_fraction']-fraction if pid in self.insert_seen else -1.
                measured[pid] = {'opened':pid in self.open_seen,'inserted_while_open':pid in self.insert_seen,
                                 'open_fraction':fraction}
            elif kind == 'base_placed_before_stack':
                base = p['base'];upper=p['upper']
                placed = self.relation({'kind':'inside_region','object':base,'region':p['region']},obs)[0]>=0
                if placed and 'worktable' in obs['support_contact'][base]:self.base_times.setdefault(pid, now)
                support_pred = next((q for q in self.contract['predicates']
                                     if q['kind'] in ('cup_nested','stable_supported_on') and q['object']==upper), None)
                if support_pred and self.relation(support_pred,obs)[0]>=0 and base in obs['support_contact'][upper]:
                    self.stack_times.setdefault(pid, now)
                samples[pid] = 1. if pid in self.base_times and pid in self.stack_times and self.base_times[pid]<self.stack_times[pid] else -1.
                if samples[pid]>=0:self.latched.add(pid)
            else:
                raise ValueError('No live semantics for frozen predicate '+kind)
        # Match the native report's 100 Hz observation windows. Instantaneous
        # geometry and protected-relation loss still update each physics step.
        if self.last_sample_time is not None and now-self.last_sample_time < .01-1e-8:
            rows = {pid: dict(row, instantaneous_margin=float(samples[pid]))
                    for pid,row in self.last_rows.items()}
            for pid in self.latched:
                rows[pid].update(signed_margin=1., passed=True)
            self.last_rows = rows
            obs['live_predicates'] = rows
            return obs
        self.last_sample_time = now
        self.history.append({'time': now, 'samples': samples,
                             'positions': {k:v.copy() for k,v in obs['object_position_m'].items()}})
        while len(self.history)>1 and self.history[1]['time'] < now-self.max_window-1e-9:
            self.history.popleft()
        rows={}
        for p in self.contract['predicates']:
            pid,kind=p['id'],p['kind']
            seconds=float(p.get('window_s',p.get('hold_s',.2)))
            window,complete=self._window(now,seconds)
            if kind in ('lift_with_contact','base_placed_before_stack','initial_outside'):
                margin=1. if pid in self.latched else samples[pid];complete=True
            elif kind=='stable':
                positions=np.array([[r['positions'][k] for k in p['objects']] for r in window])
                motion=float(np.linalg.norm(np.ptp(positions,axis=0),axis=-1).max())
                margin=p['max_motion_m']-motion;measured[pid]={'maximum_motion_m':motion}
            elif kind=='object_contact':
                margin=float(np.mean([r['samples'][pid] for r in window]))-p['minimum_fraction']
            elif kind=='released':margin=min(r['samples'][pid] for r in window)-.5
            else:margin=min(r['samples'][pid] for r in window)
            rows[pid]={'kind':kind,'signed_margin':float(margin),'instantaneous_margin':float(samples[pid]),
                       'window_complete':complete,'window_s':seconds,
                       'passed':bool(complete and margin>=0),'measurements':measured.get(pid,{})}
        self.last_rows=rows
        obs['live_predicates']=rows
        return obs

    def protect(self, predicate_id, obs):
        row=obs['live_predicates'].get(predicate_id)
        if row and row['kind'] in RELATIONS and row['passed']:
            self.protected.add(predicate_id)

    def lost_protected(self, obs):
        lost = {pid for pid in self.protected
                if obs['live_predicates'][pid]['instantaneous_margin'] < -1e-6}
        self.notified_lost.intersection_update(lost)
        newly_lost = lost - self.notified_lost
        self.notified_lost.update(lost)
        return sorted(newly_lost)
