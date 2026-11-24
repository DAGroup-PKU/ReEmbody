"""One continuous, one-pass physical drawer candidate, with no task retries.

Only native robot commands are produced. Phase time limits reject a candidate;
they do not certify reaching an event or warp/crop a longer trajectory.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np

STOP_S = 14.965
VERSION = 'rssr.sim2_refine.drawer.one_pass.v2'
# Unused time is available to later phases. The final native one-second windows
# remain reserved even when a preceding stage consumes its whole allowance.
PHASES = (
 ('approach_handle', 'approach', 'handle', 3.0),
 ('contact_handle', 'contact', 'handle', .35),
 ('grasp_handle', 'grasp', 'handle', .15),
 ('open_drawer', 'open_drawer', 'handle', 2.6),
 ('approach_payload', 'approach', 'payload', 1.4),
 ('contact_payload', 'contact', 'payload', .35),
 ('grasp_payload', 'grasp', 'payload', .15),
 ('lift_payload', 'lift', 'payload', .6),
 ('insert_payload', 'transport', 'payload', 1.4),
 ('release_payload', 'release', 'payload', 1.0),
 ('close_drawer', 'close_drawer', 'handle', 2.3),
 ('release_handle', 'release', 'handle', .665),
)

class RefineRejected(RuntimeError):
    pass

class DrawerRefineAgent:
    def __init__(self, session, case, output, atomic_json):
        from rssr_augment.retarget.events import catalog
        from rssr_augment.retarget.capability_compiler import choose_articulation_feature
        if __package__:
            from .contract import derive_bindings
        else:
            from contract import derive_bindings
        self.session, self.case, self.output, self.write = session, case, output, atomic_json
        self.contract = case['contract']
        self.bindings = derive_bindings(self.contract, case['scene'], choose_articulation_feature)
        self.events = catalog(self.contract, case['scene'])
        self.history = []
        self.assignment = None
        self.initial = None
        self.event_rows = []
        self.failure = None
        reserve = max(self.bindings.final_stability_s, self.bindings.final_release_s)
        if reserve < 1. or abs(sum(row[3] for row in PHASES) + reserve - STOP_S) > 1e-9:
            raise ValueError('Native final window differs from the declared short schedule')
        self.reserve = reserve

    def _event(self, kind, role):
        b = self.bindings
        obj = getattr(b, role)
        hand = self.assignment[role]
        options = [e for e in self.events if e.action.event == kind
                   and e.action.object_id == obj and e.action.hand == hand
                   and e.action.recovery_mode is None]
        if kind in ('open_drawer', 'close_drawer', 'transport', 'lift'):
            options = [e for e in options if e.action.target_id == b.container]
        if not options:
            raise RefineRejected('native_catalogue_missing:' + kind + ':' + role)
        # The explicit drawer insert catalogue is preferred to a generic
        # relational pickup duplicate, with no hand switching or recovery path.
        options.sort(key=lambda e: (not e.event_id.startswith('insert:'), e.event_id))
        event = options[0]
        speed = .1 if kind == 'approach' else .08 if kind in ('lift', 'transport') else .06 if kind in ('open_drawer', 'close_drawer') else event.action.max_speed_m_s
        # DrawerSession applies its declared 6cm/s loaded drawer cap; other native contact-loaded caps are retained.
        return replace(event, action=replace(event.action, max_speed_m_s=speed))

    def _persist(self, obs):
        self.write(self.output/'drawer_refine_progress.json', {
            'schema': VERSION, 'time_s': float(obs['time_s']), 'assignment': self.assignment,
            'bindings': self.bindings.packet(), 'history': self.history,
            'failure_reason': self.failure, 'no_object_state_overrides': True,
            'no_reference_action_replay': True, 'native_final_window_s': self.reserve})

    def _safety(self, obs):
        hard = obs.get('immutable_hard', {})
        if hard.get('measurements_valid') is not True or hard.get('all_hard_passed') is not True:
            return 'native_hard_failure'
        clear = hard.get('drawer_cross_side_clearance')
        if not clear or clear.get('sampled_passed') is not True:
            return 'cross_side_clearance_below_20mm_or_missing'
        return None

    def run(self, obs):
        from rssr_augment.retarget.events import signed_margin
        from rssr_augment.retarget.immutable_rule_guard import guard
        if __package__:
            from .contract import select_hands
        else:
            from contract import select_hands
        self.initial = obs
        self.assignment = select_hands(self.bindings, self.contract, obs)
        cumulative_budget = 0.
        for index, (name, kind, role, allowance) in enumerate(PHASES):
            cumulative_budget += allowance
            event = self._event(kind, role)
            start = float(obs['time_s'])
            deadline = min(cumulative_budget, STOP_S-self.reserve)
            allowed, reasons = guard(event, obs, self.contract, self.events)
            if not allowed:
                self.failure = 'native_guard_rejected:' + name + ':' + ','.join(reasons)
                self._persist(obs)
                raise RefineRejected(self.failure)
            def test(current):
                safety = self._safety(current)
                if safety:
                    return safety
                if kind in ('lift', 'transport', 'open_drawer', 'close_drawer'):
                    pads = current['finger_contact'][event.action.object_id][event.action.hand]
                    if not all(pads):
                        return 'contact_lost'
                margin = float(signed_margin(event, current, self.initial, self.contract))
                return 'event_reached' if math.isfinite(margin) and margin >= 0. else None
            reason = test(obs)
            while reason is None and not self.session.done and float(obs['time_s']) < deadline - 1e-8:
                duration = min(.25, deadline-float(obs['time_s']))
                obs, reason = self.session.advance(event.action, duration, test, action_epoch=index)
                if reason == 'duration':
                    reason = test(obs)
            reason = reason or 'phase_deadline'
            row = {'phase': name, 'event_id': event.event_id, 'event': kind,
                'hand': event.action.hand, 'object_id': event.action.object_id,
                'target_id': event.action.target_id, 'start_s': start,
                'end_s': float(obs['time_s']), 'deadline_s': deadline,
                'stop_reason': reason, 'signed_margin': float(signed_margin(event, obs, self.initial, self.contract)),
                'gripper_opening_m': dict(obs['gripper_opening_m']),
                'articulation_fraction': dict(obs['articulation_fraction'])}
            self.history.append(row)
            self._persist(obs)
            if reason != 'event_reached':
                self.failure = name + ':' + reason
                self._persist(obs)
                raise RefineRejected(self.failure)
        # Same physical scene continues until the declared stop; native released
        # and stable predicates measure their complete original final windows.
        self.hold_to_end(obs)
        return self.session.observe()

    def hold_to_end(self, obs):
        from rssr_augment.retarget.session import EventAction
        hand = self.assignment['handle'] if self.assignment else 'left'
        hold = EventAction(event='hold', hand=hand)
        while not self.session.done:
            duration = min(.25, max(.0005, STOP_S-float(obs['time_s'])))
            obs, _ = self.session.advance(hold, duration)
        return obs
