"""Rank remaining physical violations before geometric task tie-breaks.

Success predicates and failure counts remain authoritative. Within equal counts,
use measured, dimensionless violation excess; never reward a deeper joint-limit
violation merely because the final object is closer to a container centre.
"""
import copy
import math

REVISION = 'physical_hard_deficit_tiebreak_v2'
# Same native-unit resolution already required for measured repair hypotheses.
# Quantization is for candidate ordering only, never the success threshold.
RESOLUTION = .0001
FIELDS = {
    'robot_table_penetration_le_3mm': ('max_robot_table_penetration_m', .003),
    'limited_joints_within_2mrad_or_2mm': ('max_joint_limit_violation_rad', .002),
    'waist_coupling_within_1mrad': ('max_waist_sum_residual_rad', .001),
    'gripper_coupling_within_2mrad': ('model_health.max_coupling_residual_rad', .002),
}


def with_hard_progress(objective, report):
    result = copy.deepcopy(objective)
    if result.get('hard_progress', {}).get('revision') == REVISION:
        return result
    rank = result['rank']
    if len(rank) != 3:
        raise ValueError('Expected original failure-count and residual ranking')
    terms = {}
    raw_terms = {}
    for name in result['hard_failures']:
        if name not in FIELDS:
            terms[name] = 1.  # Discrete failure, no invented continuous evidence.
            raw_terms[name] = 1.
            continue
        path, limit = FIELDS[name]
        value = report
        for field in path.split('.'):
            value = value[field]
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid physical violation measurement: '+name)
        excess = max(0., value-limit)
        raw_terms[name] = excess/limit
        terms[name] = math.floor(excess/RESOLUTION+.5)*RESOLUTION/limit
    result['rank'] = [rank[0], rank[1], sum(terms.values()), rank[2]]
    result['hard_progress'] = {
        'revision': REVISION, 'normalized_failed_hard_deficits': terms,
        'unquantized_failed_hard_deficits': raw_terms,
        'ordering_resolution_native_units': RESOLUTION,
        'changes_success_criteria': False,
        'ordering': 'hard failure count, task failure count, hard deficit, task residual',
    }
    result['basis'] = 'unchanged hard/task failure counts, physical violation deficit, then task residual'
    return result
