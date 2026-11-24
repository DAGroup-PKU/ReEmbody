"""Rank partial source repairs by measured deficits, never alter task verdicts."""
import math

REVISION = 'source_failed_predicate_progress_v2'


def progress(evaluation, contract, legacy_residual):
    specifications = {p['id']: p for p in contract['predicates']}
    terms = {}
    for row in evaluation['predicates']:
        if row['passed']:
            continue
        spec = specifications[row['id']]
        kind, measured = row['kind'], row['measurements']
        value = 1.  # Boolean or unavailable measurement: no invented progress.
        if not kind.startswith('initial_'):
            if kind == 'stable':
                value = max(0., measured['maximum_motion_m']/spec['max_motion_m']-1.)
            elif kind == 'preserved':
                value = max(0., measured['maximum_displacement_m']/spec['max_displacement_m']-1.)
            elif kind == 'upright':
                value = max(0., measured['maximum_final_tilt_degrees']/spec['maximum_tilt_degrees']-1.)
            elif kind == 'particle_transfer':
                value = max(0., spec['minimum_fraction']-measured['fraction'])/spec['minimum_fraction']
            elif kind in ('table_contact', 'object_contact'):
                value = max(0., spec['minimum_fraction']-measured['contact_fraction'])/spec['minimum_fraction']
            elif kind == 'open_insert_close' and 'maximum_final_open_fraction' in measured:
                before=measured.get('maximum_open_fraction_before_insertion')
                during=measured.get('open_fraction_at_insertion')
                # Missing/reversed phase order stays a discrete failure. The
                # trace-derived closure deficit cannot imply a missing grasp.
                value=float(measured.get('first_insertion_s') is None or before is None)
                for fraction in (before,during):
                    value+=1. if fraction is None else max(0.,1.-fraction/spec['open_fraction'])
                value+=max(0.,measured['maximum_final_open_fraction']-spec['closed_fraction'])/(1.-spec['closed_fraction'])
            elif 'maximum_relation_violation_m' in measured:
                value = max(0., measured['maximum_relation_violation_m'])/.05
            elif kind == 'hold_source_interval' and measured.get('phase_complete', True):
                if 'maximum_contact_gap_s' in measured and 'bilateral_fraction' in measured:
                    value = (max(0., measured['maximum_contact_gap_s']/spec['max_contact_gap_s']-1.) +
                             max(0., 1.-measured['bilateral_fraction']/spec['minimum_bilateral_fraction']))
            # A sufficient lift without contact is still a discrete failure.
            elif kind == 'lift_with_contact' and measured['max_lift_m'] < spec['height_m']:
                value = max(0., 1.-measured['max_lift_m']/spec['height_m'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Invalid source progress metric')
        terms[row['id']] = float(value)
    if not math.isfinite(legacy_residual) or legacy_residual < 0:
        raise ValueError('Invalid source tie-break metric')
    residual = sum(terms.values()) + 1e-6*legacy_residual/(1.+legacy_residual)
    return {'revision': REVISION, 'normalized_failed_predicate_deficits': terms,
            'legacy_residual': legacy_residual, 'residual': residual,
            'changes_success_criteria': False}
