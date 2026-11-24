"""Exact cross-side geometry gate, omitting redundant table diagnostics.

User's 20mm requirement is cross-side only. Original actual-solver native hard
trace remains the mandatory robot/table penetration evidence. This evaluator
uses exactly the same cross-side pairs and hull distances as the full gate.
"""
import numpy as np
if __package__:
    from .clearance_fast import NativeClearanceGate
    from .clearance_gate import REQUIRED_CATEGORIES
else:
    from clearance_fast import NativeClearanceGate
    from clearance_gate import REQUIRED_CATEGORIES


class CrossSideNativeGate(NativeClearanceGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = [p for p in self.pairs if p[2] in REQUIRED_CATEGORIES]
        self.categories = sorted(REQUIRED_CATEGORIES)
        self.pair_a = np.asarray([p[0] for p in self.pairs], dtype=int)
        self.pair_b = np.asarray([p[1] for p in self.pairs], dtype=int)
        self.pair_radius = np.asarray([self.shapes[a]['radius']+self.shapes[b]['radius'] for a,b,_ in self.pairs])
        self.category_masks = {c:np.asarray([p[2] == c for p in self.pairs]) for c in self.categories}

    def geometry_manifest(self):
        manifest = super().geometry_manifest()
        manifest['table_clearance_policy'] = 'No 20mm table requirement and no duplicate table distance diagnostics here. Original exact native hard robot/table penetration checks remain mandatory.'
        manifest['pair_scope'] = 'Identical full-model cross-side pairs, including wrist camera assumed envelopes; table pairs alone omitted.'
        return manifest
