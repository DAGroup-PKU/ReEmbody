"""Reserve RuleAgent reviews for measured progress and later task stalls.

This observer reads the native predicate packet and target contact state. It
never chooses a physical event, edits a rule, or writes simulator state.
"""
from __future__ import annotations

from math import isfinite


class ProgressEpoch:
    def __init__(self, contract: dict, *, stall_s: float = 8.,
                 max_stall_reviews: int = 12):
        self.target_ids = {str(row[key]) for row in contract['predicates']
                           for key in ('object', 'container') if row.get(key)}
        self.stall_s = float(stall_s)
        self.max_stall_reviews = int(max_stall_reviews)
        self.initial: dict[str, float] = {}
        self.best: dict[str, float] = {}
        self.current: dict[str, float] = {}
        self.contact_keys: set[str] = set()
        self.last_progress_s: float | None = None
        self.last_review_s = -1e9
        self.stall_reviews = 0
        self.milestones: list[dict] = []

    def observe(self, obs: dict) -> dict:
        now = float(obs['time_s'])
        if not isfinite(now):
            raise ValueError('Non-finite physical observation time')
        if self.last_progress_s is None:
            self.last_progress_s = now
        current: dict[str, float] = {}
        for name, row in obs['live_predicates'].items():
            if row.get('passed'):
                continue
            margin = float(row['instantaneous_margin'])
            if isfinite(margin):
                current['predicate:'+name] = margin
            # Continuous mechanism progress can precede a final ordered
            # native predicate. It is measured, never inferred from task ID.
            for key, value in row.get('measurements', {}).items():
                if key.endswith('_fraction') and isinstance(value, (int,float)):
                    fraction = float(value)
                    if isfinite(fraction):
                        current['fraction:'+name+':'+key] = fraction
        for object_id in self.target_ids:
            contacts = obs.get('finger_contact', {}).get(object_id, {})
            if not any(len(sides) == 2 and all(sides)
                       for sides in contacts.values()):
                continue
            position = obs.get('object_position_m', {}).get(object_id)
            if position is not None and len(position) == 3:
                z = float(position[2])
                if isfinite(z):
                    current['target_lift:'+object_id] = z
        fresh = []
        for key, value in current.items():
            if key not in self.best:
                self.initial[key] = self.best[key] = value
                continue
            start = self.initial[key]
            threshold = (.08 if key.startswith('fraction:') else
                         .025 if key.startswith('target_lift:') else
                         max(.005, .1*abs(start)))
            if value - self.best[key] >= threshold:
                self.best[key] = value
                fresh.append({'signal': key, 'value': round(value, 6)})
        for object_id in self.target_ids:
            for hand, sides in obs.get('finger_contact', {}).get(
                    object_id, {}).items():
                key = object_id+':'+hand
                if key not in self.contact_keys and len(sides) == 2 and all(sides):
                    self.contact_keys.add(key)
                    fresh.append({'signal': 'bilateral_target_contact:'+key,
                                  'value': True})
        if fresh:
            self.last_progress_s = now
            self.milestones.append({'time_s': now, 'changes': fresh})
        self.current = current
        return self.snapshot(now)

    def stall_review_due(self, now: float, revisions: int,
                         task_complete: bool) -> bool:
        if (task_complete or revisions < 24 or revisions >= 48 or
                self.stall_reviews >= self.max_stall_reviews or
                self.last_progress_s is None or
                now-self.last_progress_s < self.stall_s or
                now-self.last_review_s < self.stall_s):
            return False
        self.last_review_s = now
        self.stall_reviews += 1
        return True

    def snapshot(self, now: float) -> dict:
        return {'schema': 'rssr.progress_epoch.v1',
                'seconds_since_measured_progress': (
                    None if self.last_progress_s is None else
                    round(float(now-self.last_progress_s), 4)),
                'last_progress_s': self.last_progress_s,
                'stall_reviews': self.stall_reviews,
                'current_signals': {key: round(value, 6)
                                    for key, value in self.current.items()},
                'best_signals': {key: round(value, 6)
                                 for key, value in self.best.items()},
                'recent_milestones': self.milestones[-6:]}
