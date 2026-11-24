"""Vectorized broad phase for the frozen conservative clearance implementation.

The same closed hulls and exact FCL distance requests are used. Only bounding
sphere screening and the conservative motion-radius arithmetic are batched.
"""
import numpy as np
if __package__:
    from .clearance_gate import ClearanceGate, REQUIRED_CATEGORIES, fcl
else:
    from clearance_gate import ClearanceGate, REQUIRED_CATEGORIES, fcl


class NativeClearanceGate(ClearanceGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pair_a = np.asarray([p[0] for p in self.pairs], dtype=int)
        self.pair_b = np.asarray([p[1] for p in self.pairs], dtype=int)
        self.pair_radius = np.asarray([self.shapes[a]['radius']+self.shapes[b]['radius'] for a,b,_ in self.pairs])
        self.category_masks = {c:np.asarray([p[2] == c for p in self.pairs]) for c in self.categories}

    def check(self, data, detail=True):
        centers = self._update(data)
        minima = {c:dict(distance_m=self.report_cap, exact=False) for c in self.categories}
        lower = np.linalg.norm(centers[self.pair_a]-centers[self.pair_b], axis=1)-self.pair_radius
        for pi in np.flatnonzero(lower < self.report_cap):
            ai, bi, category = self.pairs[pi]
            if lower[pi] >= minima[category]['distance_m']:
                continue
            a, b = self.shapes[ai], self.shapes[bi]
            result = fcl.DistanceResult()
            distance = float(fcl.distance(a['object'], b['object'], self.request, result))
            if not np.isfinite(distance):
                raise RuntimeError(f'Nonfinite FCL distance for {a["name"]} and {b["name"]}')
            if distance < minima[category]['distance_m']:
                row = dict(distance_m=distance, exact=True)
                if detail:
                    row.update(first=self.describe(a), second=self.describe(b),
                        nearest_points_m=np.asarray(result.nearest_points).tolist())
                minima[category] = row
        return dict(passed=all(minima[k]['distance_m'] >= self.required for k in REQUIRED_CATEGORIES),
                    required_clearance_m=self.required, required_categories=list(REQUIRED_CATEGORIES), minima=minima)

    def interval_motion_margin(self, previous_qpos, current_qpos):
        displacement = self._motion_lipschitz @ np.abs(np.asarray(current_qpos)-np.asarray(previous_qpos))
        pair_bound = .5*(displacement[self.pair_a]+displacement[self.pair_b])
        return {c:float(np.max(pair_bound[mask], initial=0.)) for c,mask in self.category_masks.items()}
