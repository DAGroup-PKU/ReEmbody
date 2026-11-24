"""Measured feedback for a rejected candidate, kept separate from the incumbent."""
import copy
from object_target_agent import diagnostic


def rejected_candidate(best, current):
    before=best['report'].get('criteria',{})
    after=current['report'].get('criteria',{})
    shared=sorted(before.keys() & after.keys())
    return copy.deepcopy({
        'trial':current['trial'],
        'compared_against_trial':best['trial'],
        'accepted':False,
        'program':current['program'],
        'objective':current['objective'],
        'report':diagnostic(current['report']),
        'regressed_predicates':[k for k in shared if before[k] is True and after[k] is False],
        'improved_predicates':[k for k in shared if before[k] is False and after[k] is True],
        'semantics':'Rejected evidence belongs to this candidate only. New patches still merge into best_program. Predicate changes do not establish causation.',
    })
