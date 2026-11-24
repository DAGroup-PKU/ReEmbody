"""Episode-local rule activation, exposure, feedback and append-only provenance."""
import json
from pathlib import Path

from rssr_augment.shared.family_autoresearch.store import digest, file_sha, pinned_json
from .adaptive_rule_protocol import OBS_ROOTS, RevisionMonitor, apply_patch, validate_patch
from .immutable_rule_guard import guard
from .relational_event_policy import decision_context
from .event_progress import filter_completed_reoffers
from .contact_bridge import contact_bridge_gaps


def plain(value):
    return json.loads(json.dumps(value, default=lambda x: x.tolist(), allow_nan=False))


def rule_observation(obs):
    """Use exactly the fields offered to RuleAgent throughout a rule's life."""
    return plain({key: value for key, value in obs.items() if key in OBS_ROOTS})


def native_goal_progress(before, after):
    """Describe native predicate movement after exposure without judging success."""
    initial = before['live_predicates']
    current = after['live_predicates']
    if set(initial) != set(current):
        raise ValueError('Native predicate identity changed within an episode')
    unsatisfied = [key for key in initial if not initial[key]['passed']]
    return {
        'unsatisfied_at_exposure': unsatisfied,
        'instantaneous_margin_delta': {
            key: (float(current[key]['instantaneous_margin']) -
                  float(initial[key]['instantaneous_margin']))
            for key in unsatisfied},
        'newly_passed': [key for key in unsatisfied if current[key]['passed']],
        'lost_previously_passed': [key for key in initial
                                  if initial[key]['passed'] and not current[key]['passed']],
        'local_effect_is_native_success': False,
    }


def compact_rule_feedback(record):
    """Keep measured rule outcomes in K3 context; retain full data in the journal.

    The full compact observation is already supplied at the current decision.
    Repeating up to eight prior observations here can exceed the explicit
    research request bound before the RuleAgent gets a chance to revise.
    """
    payload = record['payload']
    fields = ('rule_id', 'status', 'time_s', 'reason', 'detail',
              'native_success_certified', 'patch_sha256',
              'observation_sha256', 'rolled_back_patch_sha256',
              'restored_policy_sha256', 'goal_progress')
    return {'kind': record['kind'], 'index': record['index'],
            'journal_record_sha256': digest(record),
            'payload': {key: payload[key] for key in fields if key in payload}}


class RuleRuntime:
    def __init__(self, case, events, output, strategy_sha256, scene_profile=None):
        self.case, self.events = case, events
        self.output = Path(output)
        self.scene_profile = scene_profile
        self.seed_sha256 = digest({'strategy_sha256': strategy_sha256, 'rules': [],
                                   'scene_profile_sha256': (
                                       scene_profile['profile_sha256'] if scene_profile else None)})
        self.stack = []
        self.journal = []
        self.exposed_revisions = set()
        self.effect_revisions = set()
        self.last_status = {}
        self.log('initial_empty_rules', {'policy_sha256': self.seed_sha256,
                                        'scene_signature': scene_profile['scene_signature'] if scene_profile else None,
                                        'start_profile_sha256': scene_profile['profile_sha256'] if scene_profile else None,
                                        'prior_same_scene_cases': scene_profile['completed_same_scene_cases'] if scene_profile else [],
                                        'cross_case_rule_library_read': bool(scene_profile and scene_profile['completed_same_scene_cases'])})

    @property
    def policy_sha256(self):
        return self.stack[-1]['policy_sha256'] if self.stack else self.seed_sha256

    def log(self, kind, payload):
        record = {'schema': 'rssr.rule_runtime_record.v1', 'index': len(self.journal),
                  'previous_record_sha256': digest(self.journal[-1]) if self.journal else None,
                  'case_id': self.case['case_id'], 'kind': kind, 'payload': plain(payload)}
        pinned_json(self.output / f'event_{len(self.journal):04d}.json', record)
        self.journal.append(record)

    def _guard(self, event, obs):
        return guard(event, obs, self.case['contract'], self.events)

    def candidates(self, obs, history):
        base, context = decision_context(self.events, obs, self.case['contract'], history)
        context.setdefault('dead_end', False)
        application = {'offered_event_rule_owners': {}, 'deferred_event_rule_owners': {},
                       'immutable_rejections': [], 'dead_end': False}
        if self.stack:
            revision = self.stack[-1]
            available, application = apply_patch(revision['patch'], obs, base, self.events,
                self._guard, activated_time_s=revision['activated_time_s'])
        else:
            available = []
            for event in base:
                allowed, reasons = self._guard(event, obs)
                if allowed:
                    available.append(event)
                else:
                    application['immutable_rejections'].append({'event_id': event.event_id,
                                                                 'immutable_reasons': reasons})
        contact_gaps = contact_bridge_gaps(available, obs, self.events)
        available, completed_rejections = filter_completed_reoffers(
            available, obs, history, self.case['contract'])
        application['completed_reoffer_rejections'] = completed_rejections
        if not available:
            available = [e for e in self.events if e.action.event == 'hold']
            application['dead_end'] = True
        context.update(policy_sha256=self.policy_sha256,
                       active_patch_sha256=self.stack[-1]['patch_sha256'] if self.stack else None,
                       active_revision=str(self.stack[-1]['revision']) if self.stack else None,
                       rule_application=application)
        context['contact_bridge_gap_count'] = len(contact_gaps)
        context['contact_bridge_gaps'] = contact_gaps[:8]
        context['recent_rule_feedback'] = [compact_rule_feedback(r) for r in self.journal
            if r['kind'] in ('activation_rejected', 'effect_observation', 'rollback')][-8:]
        return available, context

    def activate(self, patch, revision, obs, history):
        revision = Path(revision)
        compact = rule_observation(obs)
        evidence_sha256 = file_sha(revision / 'packet.json')
        sha = validate_patch(patch, compact, self.events, self.policy_sha256, evidence_sha256)
        before, _ = self.candidates(obs, history)
        base, _ = decision_context(self.events, obs, self.case['contract'], history)
        after, application = apply_patch(patch, obs, base, self.events, self._guard,
                                         activated_time_s=obs['time_s'])
        after, completed_rejections = filter_completed_reoffers(
            after, obs, history, self.case['contract'])
        application['completed_reoffer_rejections'] = completed_rejections
        changed = ([e.as_packet() for e in before] != [e.as_packet() for e in after])
        progress_starved = (any(e.action.event != 'hold' for e in before) and
                            not any(e.action.event != 'hold' for e in after))
        if not after or not changed or progress_starved:
            rejection = {'activated': False, 'patch_sha256': sha,
                         'reason': ('all_actionable_events_removed' if progress_starved
                                    else 'no_guarded_candidate_change'),
                         'application': application,
                         'time_s': obs['time_s'], 'physical_state_reset': False}
            pinned_json(revision / 'activation.json', rejection)
            self.log('activation_rejected', {**rejection, 'revision': str(revision)})
            return False
        previous = self.policy_sha256
        policy_sha256 = digest({'seed_sha256': self.seed_sha256, 'patch_sha256': sha})
        record = {'activated': True, 'parent_policy_sha256': previous,
                  'policy_sha256': policy_sha256, 'patch_sha256': sha,
                  'time_s': obs['time_s'], 'physical_state_reset': False,
                  'before_candidates': [e.as_packet() for e in before],
                  'after_candidates': [e.as_packet() for e in after], 'application': application}
        pinned_json(revision / 'activation.json', record)
        self.stack.append({'patch': patch, 'patch_sha256': sha, 'policy_sha256': policy_sha256,
                           'revision': revision, 'activated_time_s': obs['time_s'],
                           'monitor': RevisionMonitor(patch, obs['time_s'])})
        self.log('activation', {**record, 'revision': str(revision)})
        return True

    def selected(self, event, context, obs):
        self.log('selection_context', {'time_s': obs['time_s'], 'event': event.as_packet(),
                                       'policy_sha256': self.policy_sha256,
                                       'active_revision': context['active_revision'],
                                       'application': context['rule_application']})
        if not self.stack:
            return
        revision = self.stack[-1]
        try:
            exposure = revision['monitor'].selected(event.event_id, context['rule_application'],
                                                     rule_observation(obs))
        except ValueError as exc:
            self.rollback(obs, 'expectation_vacuous_at_exposure', str(exc))
            return
        if exposure:
            self.exposed_revisions.add(str(revision['revision']))
            self.log('actual_exposure', {**exposure, 'revision': str(revision['revision']),
                                         'patch_sha256': revision['patch_sha256'],
                                         'before_observation': rule_observation(obs)})

    def rollback(self, obs, reason, detail=None):
        if not self.stack:
            return
        previous = self.stack.pop()
        self.log('rollback', {'time_s': obs['time_s'], 'reason': reason, 'detail': detail,
                              'revision': str(previous['revision']),
                              'rolled_back_patch_sha256': previous['patch_sha256'],
                              'restored_policy_sha256': self.policy_sha256,
                              'physical_state_reset': False})

    def observe(self, obs, reason):
        if not self.stack:
            return False
        revision = self.stack[-1]
        results = revision['monitor'].observe(rule_observation(obs),
            irreversible_hard_failure=not obs['immutable_hard']['all_hard_passed'],
            protected_relation_lost=reason == 'protected_predicate_lost')
        for result in results:
            key = (str(revision['revision']), result['rule_id'])
            if self.last_status.get(key) != result['status']:
                exposure = revision['monitor'].exposures.get(result['rule_id'])
                compact = rule_observation(obs)
                goal_progress = (native_goal_progress(exposure['before'], compact)
                                 if exposure is not None else None)
                self.log('effect_observation', {**result, 'revision': str(revision['revision']),
                                               'patch_sha256': revision['patch_sha256'],
                                               'goal_progress': goal_progress,
                                               'observation': compact,
                                               'observation_sha256': digest(compact)})
                self.last_status[key] = result['status']
            if result['status'] == 'observed_expected_change':
                self.effect_revisions.add(str(revision['revision']))
        failed = [r for r in results if r['status'].startswith('rollback_')]
        expired = all(obs['time_s'] - revision['activated_time_s'] >= r['ttl_s']
                      for r in revision['patch']['rules'])
        if failed or expired:
            self.rollback(obs, failed[0]['status'] if failed else 'simulated_ttl_expired')
            return True
        return False

    def summary(self):
        return {'schema': 'rssr.rule_runtime_summary.v1', 'journal': str(self.output),
                'journal_records': len(self.journal),
                'journal_head_content_sha256': digest(self.journal[-1]),
                'final_policy_sha256': self.policy_sha256,
                'exposed_revisions': sorted(self.exposed_revisions),
                'observed_effect_revisions': sorted(self.effect_revisions),
                'cross_case_rule_library_read': bool(self.scene_profile and self.scene_profile['completed_same_scene_cases']),
                'cross_case_rule_promotion': False}
