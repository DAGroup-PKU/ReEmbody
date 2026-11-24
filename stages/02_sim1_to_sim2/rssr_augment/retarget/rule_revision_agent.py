"""Charged runtime Luna RuleAgent; rules are compiled data, not source edits."""
import json
import math
import copy
from pathlib import Path

from rssr_augment.shared.family_autoresearch.agents import (
    DecisionUnavailable, InvalidCompleteResponse, MAX_REQUEST_CHARS,
    SCHEMA, _json, _profile, _request,
)
from rssr_augment.shared.family_autoresearch.store import digest, file_sha, pinned_json
from .adaptive_rule_protocol import OBS_ROOTS, apply_patch, expression, validate_patch
from .relational_event_policy import decision_context, episode_memory
from .task_context import build_task_context
from .scene_profile import trajectory_feedback
from .immutable_rule_guard import guard
from .event_progress import filter_completed_reoffers
from .rule_runtime import compact_rule_feedback
from .agent import MODEL_PROFILE
from .failure_stage_router import route_stage

RULE_MAX_OUTPUT_TOKENS = 8192


def template_packet(event, obs):
    """Expose current geometry for each contract-permitted action template."""
    packet = event.as_packet()
    hand, object_id = event.action.hand, event.action.object_id
    centers = obs.get('finger_center_m', {})
    positions = obs.get('object_position_m', {})
    if hand in centers and object_id in positions:
        distance = math.dist(centers[hand], positions[object_id])
        if not math.isfinite(distance):
            raise ValueError('Nonfinite current hand-to-object distance')
        packet['current_hand_object_distance_m'] = round(distance, 5)
    return packet


def request_evidence_projection(packet, *, compact_prior=False,
                                compact_all_prior=False, compact_blocked=False,
                                compact_templates=False):
    """Bound repeated prompt fields without changing the pinned source packet.

    Current task, live observation, event identities, and recent same-scene
    rules remain explicit. Older same-campaign rule hypotheses are summarized
    with exposure/outcome and content hashes; full records stay on disk.
    """
    projected = copy.deepcopy(packet)
    context = projected['trigger'].get('decision_context')
    if isinstance(context, dict):
        feedback = context.get('recent_rule_feedback')
        if isinstance(feedback, list):
            context['recent_rule_feedback'] = [
                compact_rule_feedback(row) if 'observation' in row.get('payload', {})
                else row for row in feedback]
        if compact_blocked and isinstance(context.get('blocked_events'), list):
            blocked = context['blocked_events']
            context['blocked_events'] = [{
                'event_id': row['event_id'],
                'unmet': [{key: fact[key] for key in
                           ('fact', 'object', 'hand', 'target_id') if key in fact}
                          for fact in row.get('unmet', [])]}
                for row in blocked]
            context['blocked_events_full_sha256'] = digest(blocked)
    if not (compact_prior or compact_all_prior or compact_blocked or compact_templates):
        return projected
    template_fields = ('event_id', 'physical_event', 'hand', 'object_id',
                       'target_id', 'target_offset_m', 'contact_mode',
                       'bystander_clearance_m',
                       'source_predicate', 'evidence', 'base_joint_target',
                       'base_axis_order',
                       'current_hand_object_distance_m')
    guarded = set(packet.get('current_guarded_candidate_ids', []))
    compact_template_fields = ('event_id', 'physical_event', 'hand',
                               'object_id', 'target_id')
    projected['primitive_templates'] = [
        {key: row[key] for key in
         (compact_template_fields if compact_templates and
          row.get('event_id') not in guarded else template_fields)
         if key in row}
        for row in packet['primitive_templates']]
    for projected_row, source_row in zip(projected['primitive_templates'],
                                         packet['primitive_templates']):
        if str(source_row.get('event_id', '')).endswith(':guarded_slow'):
            projected_row['max_speed_m_s'] = source_row['max_speed_m_s']
    profile = projected['trigger'].get('scene_profile')
    if compact_all_prior and isinstance(profile, dict):
        completed = profile.get('completed_same_scene_cases')
        if isinstance(completed, list):
            profile['completed_same_scene_case_count'] = len(completed)
            profile['completed_same_scene_cases_sha256'] = digest(completed)
            profile['completed_same_scene_cases'] = []
        same_campaign = projected.get('same_campaign_prior_scene_cases')
        if isinstance(same_campaign, list):
            projected['same_campaign_prior_scene_case_count'] = len(same_campaign)
            projected['same_campaign_prior_scene_cases_sha256'] = digest(same_campaign)
            projected['same_campaign_prior_scene_cases'] = []
    if isinstance(profile, dict) and isinstance(profile.get('recent_samples'), list):
        samples = profile['recent_samples']
        for sample in samples if compact_all_prior else samples[:-2]:
            feedback = sample.get('trajectory_feedback')
            if isinstance(feedback, dict):
                if compact_all_prior:
                    phases = {key: {field: value for field, value in row.items()
                                    if field in ('count', 'stop_reasons',
                                                 'margin_delta_sum', 'nonimproving')}
                              for key, row in feedback.get('phase_stats', {}).items()}
                    contacts = {key: {field: value for field, value in row.items()
                                      if field in ('any_contact_windows',
                                                   'bilateral_contact_windows')}
                                for key, row in feedback.get('contact_progress', {}).items()}
                else:
                    phases = feedback.get('phase_stats', {})
                    contacts = feedback.get('contact_progress', {})
                sample['trajectory_feedback'] = {
                    'schema': feedback.get('schema'),
                    'event_windows': feedback.get('event_windows'),
                    'phase_stats': phases,
                    'predicate_progress': {
                        key: {field: value for field, value in row.items()
                              if field in ('kind', 'first_passed_s',
                                           'max_instantaneous_margin',
                                           'final_instantaneous_margin', 'final_passed')}
                        for key, row in feedback.get('predicate_progress', {}).items()},
                    'contact_progress': contacts,
                    'non_target_contact_modes': feedback.get(
                        'non_target_contact_modes', [])[:4 if compact_all_prior else 8],
                    'concurrent_gripper_scene_contact': feedback.get(
                        'concurrent_gripper_scene_contact', {}),
                    'full_feedback_sha256': digest(feedback),
                    'full_event_history_retained_outside_prompt': True}
            summarized = []
            for proposal in sample.get('rule_proposals', []):
                summarized.append({
                    'proposal_sha256': proposal['proposal_sha256'],
                    'actually_exposed': proposal['actually_exposed'],
                    'effect_observations': proposal.get('effect_observations', []),
                    'rule_summary': [
                        {'id': rule['id'],
                         'offered_event_ids': [item['event_id'] for item in
                                               rule.get('offer', [])],
                         'expected': rule.get('expected')}
                        for rule in proposal['rules']]})
            sample['rule_proposals'] = summarized
            if not summarized:
                sample.pop('source_binding', None)
            elif compact_all_prior and isinstance(sample.get('source_binding'), dict):
                sample['source_binding'].pop('structure', None)
    projected['request_projection'] = {
        'schema': 'rssr.rule_request_projection.v1',
        'full_packet_sha256': None,
        'primitive_template_fields': list(template_fields),
        'blocked_event_details_compacted': compact_blocked,
        'noncandidate_template_details_compacted': compact_templates,
        'prior_sample_policy': ('all_prior_measured_progress_summarized'
                                if compact_all_prior else
                                'latest_two_detailed_older_measured_progress_summarized'),
        'full_evidence_retained_outside_prompt': True}
    return projected


def request_chars(evidence):
    """Use the exact charged request serializer and configured route before K3."""
    request = {'schema': SCHEMA, 'role': 'runtime_rule_revision',
               'profile': 'gpt6luna', 'model': _profile(MODEL_PROFILE)['model'],
               'prompt': PROMPT, 'evidence': evidence,
               'max_output_tokens': RULE_MAX_OUTPUT_TOKENS,
               'reasoning_effort': 'medium'}
    return len(_json(request))


def compact_recent_scene_feedback(samples):
    """Keep recent failure and strongest measured prerequisite evidence.

    This is an input projection only: the complete hash-bound scene samples
    remain in the pinned packet and cannot become executable robot commands.
    """
    if not samples:
        return []

    def progress(sample):
        feedback = sample['trajectory_feedback']
        return (bool(sample['native_task_success']),
                sum(row.get('first_passed_s') is not None for row in
                    feedback.get('predicate_progress', {}).values()),
                sum(row.get('bilateral_contact_windows', 0) for row in
                    feedback.get('contact_progress', {}).values()))

    selected = ([max(samples[:-1], key=progress)] if len(samples) > 1 else [])
    selected.append(samples[-1])
    result = []
    for sample in selected:
        feedback = sample['trajectory_feedback']
        contacts = sorted(feedback.get('contact_progress', {}).items(),
                          key=lambda pair: (-pair[1].get(
                              'bilateral_contact_windows', 0), pair[0]))[:3]
        phases = sorted(feedback.get('phase_stats', {}).items(),
                        key=lambda pair: (-pair[1].get('count', 0), pair[0]))[:6]
        result.append({
            'case_id': sample['case_id'],
            'source_binding': {key: value for key, value in
                               sample.get('source_binding', {}).items()
                               if key != 'structure'},
            'native_task_success': sample['native_task_success'],
            'strict_replay_accepted': sample['strict_replay_accepted'],
            'native_required_failures': sample['native_required_failures'],
            'native_hard_failures': sample['native_hard_failures'],
            'predicate_max_margin': {
                key: round(row['max_instantaneous_margin'], 4)
                for key, row in feedback.get('predicate_progress', {}).items()},
            'predicate_final_passed': {
                key: row['final_passed'] for key, row in
                feedback.get('predicate_progress', {}).items()},
            'top_bilateral_contacts': {
                key: {'any': row['any_contact_windows'],
                      'bilateral': row['bilateral_contact_windows']}
                for key, row in contacts},
            'phase_counts': {key: row['count'] for key, row in phases},
            'frequent_modes': feedback.get('frequent_modes', [])[:4],
            'collision_modes': feedback.get('non_target_contact_modes', [])[:2],
            'sample_sha256': sample['sample_sha256'],
        })
    return result


def compact_overlimit_evidence(evidence, *, reserve_chars=1024):
    """Keep live collision evidence while bounding redundant history fields."""
    target = MAX_REQUEST_CHARS - reserve_chars
    if request_chars(evidence) <= target:
        return evidence
    projection = evidence.setdefault('request_projection', {})
    projection.setdefault('full_packet_sha256', evidence.get('evidence_sha256'))
    feedback = evidence.get('current_episode_feedback')
    if isinstance(feedback, dict) and feedback.get('frequent_modes'):
        full = feedback.pop('frequent_modes')
        feedback['frequent_modes_count'] = len(full)
        feedback['frequent_modes_sha256'] = digest(full)
        projection['frequent_modes_compacted'] = True
    if request_chars(evidence) <= target:
        return evidence
    templates = evidence.get('primitive_templates')
    if isinstance(templates, list):
        slow = [row for row in templates if str(row.get('event_id', '')).endswith(
            ':guarded_slow')]
        if slow:
            keep = ('event_id', 'physical_event', 'hand', 'object_id',
                    'target_id', 'max_speed_m_s', 'contact_mode', 'source_predicate')
            evidence['primitive_templates'] = [
                {key: row[key] for key in keep if key in row}
                if str(row.get('event_id', '')).endswith(':guarded_slow') else row
                for row in templates]
            projection['slow_template_details_sha256'] = digest(slow)
            projection['slow_template_details_compacted'] = True
    if request_chars(evidence) <= target:
        return evidence
    if isinstance(feedback, dict) and isinstance(feedback.get('contact_progress'), dict):
        full = feedback['contact_progress']
        retained = sorted(full, key=lambda key: (
            -full[key].get('bilateral_contact_windows', 0),
            -full[key].get('any_contact_windows', 0), key))[:8]
        feedback['contact_progress'] = {key: full[key] for key in retained}
        feedback['contact_progress_full_count'] = len(full)
        feedback['contact_progress_full_sha256'] = digest(full)
        projection['contact_progress_compacted'] = True
    if request_chars(evidence) <= target:
        return evidence
    memory = evidence.get('episode_memory')
    if isinstance(memory, dict) and isinstance(memory.get('failed_attempts'), dict):
        full = memory['failed_attempts']
        retained = sorted(full, key=lambda key: (-full[key]['count'], key))[:12]
        memory['failed_attempts'] = {key: full[key] for key in retained}
        memory['failed_attempts_full_count'] = len(full)
        memory['failed_attempts_full_sha256'] = digest(full)
        projection['failed_attempts_compacted'] = True
    if request_chars(evidence) <= target:
        return evidence
    context = evidence.get('trigger', {}).get('decision_context')
    if isinstance(context, dict) and isinstance(context.get('blocked_events'), list):
        full = context['blocked_events']
        guarded = set(evidence.get('current_guarded_candidate_ids', []))
        priority = sorted(full, key=lambda row: (
            row.get('event_id') not in guarded, row.get('event_id', '')))
        context['blocked_events'] = priority[:12]
        context['blocked_event_count'] = len(full)
        context['blocked_event_ids'] = [row['event_id'] for row in full]
        context['blocked_events_full_sha256'] = digest(full)
        projection['blocked_event_details_sampled'] = True
    # The full blocked ID list repeats the catalog already supplied above.
    # Keep its count and hash on the pinned packet; a small current sample is
    # enough for this one revision. This is generic evidence projection only.
    if request_chars(evidence) > target and isinstance(context, dict):
        ids = context.pop('blocked_event_ids', None)
        if ids is not None:
            context['blocked_event_ids_count'] = len(ids)
            context['blocked_event_ids_sha256'] = digest(ids)
            projection['blocked_event_ids_compacted'] = True
    if request_chars(evidence) > target and isinstance(context, dict):
        recent = context.get('recent_rule_feedback')
        if isinstance(recent, list) and len(recent) > 2:
            context['recent_rule_feedback_full_count'] = len(recent)
            context['recent_rule_feedback_full_sha256'] = digest(recent)
            context['recent_rule_feedback'] = recent[-2:]
            projection['recent_rule_feedback_compacted'] = True
    if request_chars(evidence) > target and isinstance(context, dict):
        blocked = context.get('blocked_events')
        if isinstance(blocked, list) and len(blocked) > 6:
            context['blocked_events'] = blocked[:6]
            projection['blocked_event_sample_limit'] = 6
    if request_chars(evidence) > target:
        # The complete template catalog is pinned in packet.json.  Late in a
        # trajectory it can contain hundreds of variants, while the current
        # guarded choices are few.  Keep the full live choices and every other
        # event identity so the RuleAgent can still propose any allowed event.
        templates = evidence.get('primitive_templates')
        if isinstance(templates, list):
            guarded = set(evidence.get('current_guarded_candidate_ids', []))
            other = [row for row in templates if row.get('event_id') not in guarded]
            if other:
                evidence['primitive_templates'] = [
                    row if row.get('event_id') in guarded
                    else {'event_id': row['event_id']}
                    for row in templates]
                projection['noncandidate_template_details_sha256'] = digest(other)
                projection['noncandidate_template_details_compacted'] = len(other)
    if request_chars(evidence) > target:
        # Prior reusable rules may each repeat the same scene structure.  The
        # current task_context already supplies that structure; retain each
        # prior rule and its measured phase record while hashing its duplicate.
        profile = evidence.get('trigger', {}).get('scene_profile')
        if isinstance(profile, dict):
            prior = profile.get('eligible_reuse')
            portfolio = profile.get('eligible_reuse_portfolio', [])
            rows = ([prior] if isinstance(prior, dict) else []) + (
                portfolio if isinstance(portfolio, list) else [])
            compacted = 0
            for row in rows:
                binding = row.get('source_binding') if isinstance(row, dict) else None
                if isinstance(binding, dict) and 'structure' in binding:
                    structure = binding.pop('structure')
                    binding['structure_sha256'] = digest(structure)
                    compacted += 1
            if compacted:
                projection['prior_duplicate_structures_compacted'] = compacted
            if request_chars(evidence) > target:
                binding = profile.get('current_binding')
                if isinstance(binding, dict) and 'structure' in binding:
                    structure = binding.pop('structure')
                    binding['structure_sha256'] = digest(structure)
                    projection['current_duplicate_structure_compacted'] = True
    if request_chars(evidence) > target:
        # The executor, not the language model, consumes joint target vectors.
        # Preserve the guarded event's identity, geometry offset and contact
        # semantics while removing repeated low-level command vectors here.
        templates = evidence.get('primitive_templates')
        if isinstance(templates, list):
            guarded = set(evidence.get('current_guarded_candidate_ids', []))
            removed = {}
            for row in templates:
                if row.get('event_id') not in guarded:
                    continue
                details = {key: row.pop(key) for key in
                           ('base_joint_target', 'base_axis_order', 'evidence')
                           if key in row}
                if details:
                    removed[row['event_id']] = details
            if removed:
                projection['guarded_command_details_sha256'] = digest(removed)
                projection['guarded_command_details_compacted'] = len(removed)
    if request_chars(evidence) > target:
        profile = evidence.get('trigger', {}).get('scene_profile')
        portfolio = profile.get('eligible_reuse_portfolio') if isinstance(
            profile, dict) else None
        if isinstance(portfolio, list) and portfolio:
            profile['eligible_reuse_portfolio'] = [
                {'source_case_id': row.get('source_case_id'),
                 'source_proposal_sha256': row.get('source_proposal_sha256'),
                 'rule_ids': [rule.get('id') for rule in row.get('rules', [])],
                 'offered_event_ids': [offer.get('event_id') for rule in
                                       row.get('rules', []) for offer in
                                       rule.get('offer', [])],
                 'source_rule_phase_records_sha256': digest(
                     row.get('source_rule_phase_records', []))}
                for row in portfolio]
            projection['prior_portfolio_details_sha256'] = digest(portfolio)
            projection['prior_portfolio_summarized'] = len(portfolio)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        raise ValueError('RuleAgent request remains above frozen character limit')
    return evidence


def bounded_rule_request(packet, evidence_sha256, errors):
    """Project the pinned packet into the exact charged request evidence."""
    evidence = request_evidence_projection(packet)
    evidence.update(evidence_sha256=evidence_sha256,
                    previous_compile_errors=errors)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        evidence = request_evidence_projection(packet, compact_prior=True)
        evidence['request_projection']['full_packet_sha256'] = evidence_sha256
        evidence.update(evidence_sha256=evidence_sha256,
                        previous_compile_errors=errors)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        evidence = request_evidence_projection(packet, compact_all_prior=True)
        evidence['request_projection']['full_packet_sha256'] = evidence_sha256
        evidence.update(evidence_sha256=evidence_sha256,
                        previous_compile_errors=errors)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        evidence = request_evidence_projection(packet, compact_all_prior=True,
                                               compact_blocked=True)
        evidence['request_projection']['full_packet_sha256'] = evidence_sha256
        evidence.update(evidence_sha256=evidence_sha256,
                        previous_compile_errors=errors)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        evidence = request_evidence_projection(packet, compact_all_prior=True,
                                               compact_blocked=True,
                                               compact_templates=True)
        evidence['request_projection']['full_packet_sha256'] = evidence_sha256
        evidence.update(evidence_sha256=evidence_sha256,
                        previous_compile_errors=errors)
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        profile = evidence['trigger'].get('scene_profile', {})
        recent = profile.get('recent_samples', [])
        if recent:
            profile['recent_samples_full_sha256'] = digest(recent)
            profile['recent_samples'] = compact_recent_scene_feedback(recent)
            evidence['request_projection']['recent_sample_policy'] = (
                'latest_and_strongest_measured_prerequisite_summary')
        certified = profile.get('certified_rule_memory', [])
        if certified:
            profile['certified_rule_memory_full_sha256'] = digest(certified)
            for limit in (4, 2, 1, 0):
                profile['certified_rule_memory'] = certified[:limit]
                evidence['request_projection']['certified_memory_limit'] = limit
                if request_chars(evidence) <= MAX_REQUEST_CHARS - 1024:
                    break
        if (request_chars(evidence) > MAX_REQUEST_CHARS - 1024 and
                len(profile.get('recent_samples', [])) > 1):
            profile['recent_samples'] = profile['recent_samples'][:1]
            evidence['request_projection']['recent_sample_limit'] = 1
    if request_chars(evidence) > MAX_REQUEST_CHARS:
        feedback = evidence.get('current_episode_feedback')
        if isinstance(feedback, dict) and isinstance(
                feedback.get('recent_events'), list):
            recent = feedback.pop('recent_events')
            feedback['recent_events_count'] = len(recent)
            feedback['recent_events_sha256'] = digest(recent)
            evidence['request_projection']['recent_event_details_compacted'] = True
    compact_overlimit_evidence(evidence)
    return evidence


PROMPT = """You are the runtime rule-revision subagent of an RSSR retarget agent.
The failure_stage_route selects ONE physical dependency for this revision.
Act as that stage's specialist: use the complete native contract and current
measurements to invent one short, testable scene rule for the selected
dependency. Preserve already achieved predicates. If a measured prerequisite
is missing, state it and propose its immediate physical test rather than a
future-stage plan. The route is advisory; native guards and live evidence win.
If trigger.reason is repeated_event_cycle, inspect the supplied alternating
event IDs and native_margin_delta. A cycle can still make progress; use the
measured state to decide which immediate guarded event and bounded offset
could improve an unfinished physical dependency. Recheck the result after
one short segment rather than repeating the pair by name.
If trigger.newly_passed_native_predicates is nonempty, update the current rule
for the next unfinished physical dependency using the live observation. If
trigger.reason is rule_rollback, inspect the measured effect and why the prior
rule ended before proposing the next immediate rule. Do not preserve an old
offer merely because it produced a local scalar change.
If trigger.reason is ordered_prerequisite_at_risk, the current relation motion
paused because a native temporal prerequisite has too little measured reserve.
Inspect trigger.ordered_prerequisite_warning and the current native
relation and mechanism state. Choose a guarded short event that restores the
prerequisite or another measurable route; do not count a geometric relation
as ordered success unless the native sequence predicate actually passes.
When a required spatial relation is already measured true, identify any
remaining support, release, or ordered final-transition predicate. Propose
the next short event for that unfinished dependency. Repeatedly lifting or
transporting the placed object does not improve an already satisfied spatial
relation unless it was lost or the remaining dependency requires a changed
contact state. Use native predicate observations and available templates,
not a task-name recipe.
This revision controls the next physical event. The executor will request a new
revision after new measurements; rules expire within at most eight simulated
seconds. Propose a rule whose condition is TRUE now and which changes at least
one current_guarded_candidate_id, by adding a guarded event, changing a bounded
contact offset, or deferring an event. A future-stage plan alone is a no-op and
will be rejected. If the base candidates already include your intended event,
offer/defer must still alter the actual candidate set. Use a short physical
expected effect relevant to the unfinished native predicate or its necessary
prerequisite. Explain that link in reason.
First understand the current task from task_context: its frozen predicate
relations, object roles and geometry, source instruction, and available physical
primitives. The source instruction is context; native predicates alone define
success. Infer the next physical dependency from the live predicate state.
For a spatial relation, compare the measured component clearances and the
object center offset in the target frame. Select a guarded event that addresses
the currently limiting component, then check its measured change. A component
estimate is decision evidence, never a substitute for the native predicate.
The S1 chassis is fixed. Coupled waist motion is bounded by the executor and
may improve measured reach. Compare actual contact and task progress after
each event; revise or roll back rules when the effect does not improve.
Generate rules conditioned on the supplied predicate graph, object roles,
geometry and current physical measurements. Write conditions over supplied
observation fields and offers over supplied event IDs. Do not encode a task
name, family label, case ID or fixed scenario recipe in a rule. Native predicate
names may be used only when present in this case's contract. Do not copy a
threshold from another scene merely because its name or shape matches.
If a necessary physical primitive is absent, explain the exact
capability gap; a JSON rule cannot manufacture push, pour, object rotation,
hinge motion, or other missing robot skills.
If decision_context reports completed_reoffer_rejections, the named physical
events already reached their measured completion in this same trajectory and
remain complete now. Revise your soft rule toward the unfinished native
relation or a genuinely changed contact state; repeating those events cannot
advance the task. A lost grasp or fallen object can make them eligible again.
If decision_context.contact_bridge_gaps is nonempty, a measured hand pose is
within the immutable contact gate without object contact, and no contact event at that same relative
target is currently offered. If the gap names a contact template and its
offset_delta is within the rule envelope, consider offering that contact
template with the supplied offset delta, conditioned on current distance and
missing contact. Check actual pad contact over successive short segments.
Choose observe_s and ttl_s from the measured hand distance, gripper opening,
recent actuator progress and offered action speed. A finger that is still
closing can need more than one segment; do not demand physical contact before
the actuator could plausibly reach it. If that transit time is uncertain,
choose a measurable intermediate change such as decreasing gripper opening,
then revise toward contact using new observations. A reached approach is only
a local pose result; do not reoffer it as task completion.
If current_episode_feedback.gripper_closure_progress reports positive measured
closure with no target pad contact, treat the closure as actuator progress.
Compare its median positive rate with the current opening and the offered
gripper target before setting observe_s or ttl_s for expected hand contact.
Do not roll back a still-closing contact attempt solely because the first
short segment did not touch; retain hard-limit and collision stop conditions.
If the last event stopped because target_contact_started, one finger pad met the
target before the approach pose was reached. Treat this as a physical branch:
inspect the actual pad pattern, joint margin and contact normal. Consider a
short guarded retreat from partial contact, a different bounded approach, or
the available contact/grasp action only when its measured preconditions and
hard margins support it. Repeating the pressing approach without a changed
contact target risks a limited-joint violation.
Use current_episode_feedback to compare actual event margins, contact loss,
predicate progress and repeated modes in this same uninterrupted trajectory.
When a held object stalls, compare its bilateral grasp with contacts between
the same gripper and other scene objects. gripper_scene_contacts identifies
the contacted geometry and a measured direction away from contact. Decide
from task predicates whether another contact is useful support or an obstacle.
When an unsecured gripper touches a non-target fixture, a guarded retreat
may separate it before another approach; prefer measured contact clearance
over repeating an approach into the same fixture.
Use frequent_collision_modes and non_target_contact_modes as evidence of
which exact approach or contact modes repeatedly struck which objects. After
retreat, prefer an untried or measurably different guarded route. The generic
 overhead approach primitive may stage above a freely movable object before
 descending. Its overhead_high variant ends at a slightly higher measured
 contact region; consider it if a nearby support repeatedly blocks the center
 endpoint. Offer either only when observed geometry and collisions support
 that route. It is a physical hypothesis, not a guaranteed clear path.
 A supplied side_corridor route stages an open hand beyond a measured
 static fixture side before a separate approach to the object. Offer it
 when the direct approach loads intervening geometry, and require measured
 stage progress, retained prerequisite relations and hard checks.
 For a held object and articulated container, front_clearance and
 front_align are separate geometry-derived staging events before ordinary
 insertion. Offer them only when current loaded motion is blocked or the
 direct path threatens a measured prerequisite. A support_reserve variant
 requires another manipulator's bilateral feature contact and applies a
 bounded measured-axis target. Verify stage progress, contact, opening
 margin and hard gates before advancing; do not prewrite a full route.
If closing an articulated feature repeatedly loses measured contact during
motion, a guarded_slow variant may trade speed for contact persistence.
Offer it only with current contact evidence and compare actual joint-fraction
progress, contact loss and native predicates after exposure.
If a dependent transition requires an opening threshold and measured drift
leaves little reserve, a buffered opening event may build a small measured
margin before that transition. Offer it only when current contact and joint
state make it feasible; verify the actual fraction after one short segment.
If clearance is an available primitive, propose a condition and measurable
effect only when the current physical evidence supports it; check actual
contact loss and task-margin change after execution. Do not treat any contact
as automatically obstructive, and do not copy a scene-specific recovery rule.
If a copied-state forecast shows contact force between the inactive hand and
the moving target while articulation progress stalls, consider a supplied
bystander_clearance variant. Its inactive arm moves along the measured joint
axis; require bilateral active-hand contact, positive joint progress, and
every-step hard constraints after execution. A higher geometry-scaled contact
surface may be offered when measured low contact misses one pad or presses a
neighboring surface. Choose using pad contacts and geometry, not task names.
Change a rule only when its expected physical effect can be measured after
the rule actually influences an action; a proposed rule is not an improvement.
Read goal_progress in measured rule feedback. A local expected change can
occur while every unfinished native predicate remains stalled or regresses.
When this happens, identify the concrete physical prerequisite the rule
established and whether it remains necessary; otherwise revise the rule
toward an unfinished predicate. A newly passed predicate is evidence of
progress, not certification of whole-task success. Do not copy a locally
successful rule across cases solely because its expected scalar changed.
Diagnose the current physical dead end from observed evidence only. Write a
NEW executable declarative rule, not a repair-menu ID or a complete plan.
At the first observation, create evidence-conditioned initial rules from the current
contract, objects and approved prior scene samples if supplied. Diagnose
which prior rules were actually exposed, their measured effects and native
outcomes; revise conditions or offers accordingly. A prior proposal by itself
is not successful experience. Rebind prior object names using source_binding
and current_binding; never copy old names into another case blindly.
When certified_rule_memory is supplied, examine its rules rebound to the
current scene, their current applicability, and source_phase_evidence. The
source phase records which native predicates had already passed when that
rule actually affected a prior action. If source_prerequisites_met_now is
false, do not apply that rule as an immediate action merely because its
Boolean when is true. You may preserve it for a later phase or derive a new
rule from current measured evidence. A certified prior result does not
guarantee success here. If offered_event_ids_unavailable_now is nonempty,
adapt the physical objective to currently available manipulators and events;
never copy an unavailable event ID.
Use source_proposal_sha256 in your reason when a prior rule informs the patch.
If trigger.reason is source_phase_candidate_became_executable, inspect
newly_executable_certified_rule_keys and the corresponding certified memory.
Decide from the current physical observation whether to reuse, revise, or
reject a candidate; a phase match is evidence of timing, not success here.
If trigger.reason is reusable_rule_condition_now_true, an earlier same-campaign
rule has only become conditionally relevant now. Re-evaluate it against the
current task state and measured trajectory. Generate your own current rule;
do not repeat an old rule merely because its condition became true. The old
rule is not automatically activated and is not proof of native success here.
Use trajectory_feedback to identify the physical stage where progress stopped:
compare event margins, contact windows, predicate progress and final measured
state. If the prior episode reached a relation but lost contact or could not
close an articulation, write a rule for the measured remaining condition,
not a task-name shortcut. Do not treat a native failure label alone as a
complete diagnosis. Reusable rules were eligible through strict replay and
measured exposure, but still require fresh physical validation here.
latest_strict_success_anchor retains a bounded prior success witness even
when the recent episode window contains only failures. Its frequent event
modes are outcome context, never an executable trajectory or proof that a
mode caused success. Compare them with current reach, contact, collision and
predicate evidence before proposing a rule. If the current robot has fewer
manipulators or a contract binds a hand, obey those current capabilities.
The execution agent chooses an event from candidates produced by these rules.
Return precisely schema,parent_sha256,evidence_sha256,reason,rules.
schema is rssr.rule_patch.v1; copy the two supplied SHA256 values exactly.
Each of 1-3 rules has precisely id,when,offer,defer,expected,observe_s,ttl_s.
when is a boolean AST: {all:[AST,...]}, {any:[AST,...]}, {not:AST}, or
{path:[observation dictionary keys and optional nonnegative vector indices],
 op:eq|ne|lt|le|gt|ge|contains,value:typed scalar}. Use only supplied observations.
Contact lists change length during execution. For a contact object's presence,
prefer {path:["gripper_scene_contact_object_ids",hand],op:"contains",value:object_id}
over indexing gripper_scene_contacts by a fixed list position. A condition
whose referenced contact entry disappears becomes inactive as a whole.
Only supplied observation_roots may be addressed. Unknown or future evidence
cannot be assumed. Boolean ASTs are limited to depth5 and64nodes.
offer lists 0-6 parameterized events, each precisely event_id,priority,
offset_delta_m,hand. event_id must be a supplied primitive template.
priority is -10..10. offset_delta_m is a3vector; abs(component)<=.025m and
norm<=.04m. Nonzero approach/contact offsets may adjust a measured contact
region; transport/align may use ONLY [0,0,dz] with 0<=dz<=.025m to retain
measured carrying clearance during horizontal motion. hand is null, or an identifier
from available_manipulators when the contract-compiled primitive includes it for the same
physical event and object. A release may additionally use the hand that is
actually touching the object, subject to the independent immutable guard.
Use offset_delta_m:[0,0,0] for grasp, lift, open, close, push, retreat, release and
verify. Zero offset is also the default for approach/contact/transport/align.
Propose an upward carry offset only when live lift/contact measurements show
that transport or alignment loses clearance; after horizontal progress,
recheck the native target relation and return to the unshifted destination.
Add a nonzero contact offset only when current geometry or contact evidence
supports a bounded contact-region adjustment.
Primitive templates include current_hand_object_distance_m when measurable.
Use it as current reach evidence, not proof of a collision-free grasp or task
success. arm_torso_min_joint_margin_rad excludes scene articulations and
gripper fingers; manipulator_joint_margin_rad gives measured risk for each
available manipulator. If only one exists, consider feasible contact and
motion modes for it. If several exist, compare contract-permitted assignments
jointly against reach, occupancy and joint margin. A different assignment or
contact mode is a hypothesis to measure, not a guaranteed recovery. Preserve
contract bindings and do not invent a manipulator or unavailable primitive.
gripper_min_joint_margin_rad gives the measured finger-joint limit margin.
A warning prompts reassessment of the next event; it is not evidence of
contact, grasp or native task success.
If the available capability set cannot satisfy a dependency, report that gap
in the rule reason.
defer lists 0-6 existing event IDs.
Across each rule, defer contains at most 6 event IDs. Omit lower-priority
deferrals rather than exceeding this limit.
expected is precisely path,change,value: increase_by or decrease_by requires
a positive numeric delta with the native field units; becomes requires a
boolean/string value that does NOT already hold. Effects are measured against
the state at actual rule exposure, not merely proposal time. observe_s and
ttl_s use simulation time and satisfy .5<=observe_s<=ttl_s<=8.
expected.path MUST be a JSON array of concrete dictionary keys and, if
needed, a nonnegative vector index; never a dotted string or root-only name.
It must end at one currently measured scalar. For example, for an existing
object and hand use ["finger_contact", object_id, hand, 0], or for an existing
pad's measured normal load in newtons use
["finger_normal_force_n", object_id, hand, 0]. Geometric finger_contact
alone does not establish a load-bearing grasp; compare measured loads and
their changes with the live task state. Do not assume a universal force
threshold or invent a contact force where none was observed. For an existing
joint use ["articulation_fraction", joint_id]. The object_id, hand and joint_id
in the actual JSON must be literal keys present in the supplied observation.
expected.change MUST be exactly one string: "increase_by", "decrease_by", or
"becomes"; put its numeric or boolean target in the separate value field.
Expected change must be a physical contact, pose, velocity, articulation,
gripper, margin or live task measurement; elapsed time or metadata is not an effect.
For expected.path use only live_predicates, finger_contact,
finger_normal_force_n, hand_contact,
bilateral_contact_seconds, articulation_fraction, gripper_opening_m,
object_position_m, object_velocity_m_s, finger_center_m,
min_joint_margin_native or max_robot_table_penetration_m. Never use time_s,
immutable_hard, source text, scene structure, rule metadata or a static goal
parameter as expected.path. Inspect previous_compile_errors, if present, and
correct the specific rejected field before retrying.
The independent immutable guard may reject offered actions. You cannot alter
native success predicates/windows, hard limits, physics, actuator forces,
object state, time, safety envelopes or reference commands. New rules are
episode-local hypotheses; measured improvement is not native task success.
Explain why your generated condition addresses the observed failure. Do not
claim an untried mode if structured episode memory records it as tried.
Prior native outcomes are supervision for this frozen sequential campaign,
not an old external success oracle and not proof that a local rule caused success.
"""


def immediate_bridge_key(parent_sha256: str, trigger: dict):
    """Identify one new, guarded contact-gap review at a near-identical pose."""
    if not trigger.get('contact_bridge_review_due'):
        return None
    gaps = trigger.get('decision_context', {}).get('contact_bridge_gaps', [])
    rows = tuple(sorted(
        (gap['approach_event_id'], gap['contact_template_event_id'],
         tuple(gap['contact_offset_delta_m']))
        for gap in gaps if gap['rule_offset_within_envelope']))
    return (parent_sha256, rows) if rows else None


class RuleRevisionAgent:
    """New-role requests share the campaign's charged Store and provider route."""
    def __init__(self, store, output, max_revisions=8):
        self.store = store
        self.output = Path(output)
        self.max_revisions = max_revisions
        self.last_time_s = None
        self.last_immediate_bridge_key = None
        self.revisions = 0

    def propose(self, case_id, obs, history, contract, events, parent_sha256, trigger,
                scene=None, source_task_text=None):
        if self.revisions >= self.max_revisions:
            raise ValueError('Episode rule-revision budget exhausted')
        if self.last_time_s is not None and obs['time_s'] - self.last_time_s < .01 - 1e-8:
            # A just-activated rule can expose a contact gap at an otherwise
            # unchanged physical pose. This is new actionable context even
            # within 10 ms. The caller applies a four-second bridge cooldown;
            # this local key also rejects duplicate requests for the same gap.
            key = immediate_bridge_key(parent_sha256, trigger)
            if key is None or key == self.last_immediate_bridge_key:
                raise ValueError('Repeated rule request without a new observation')
            self.last_immediate_bridge_key = key
        revision = self.output / f'revision_{self.revisions:02d}'
        self.revisions += 1
        self.last_time_s = obs['time_s']
        obs = json.loads(json.dumps({k: v for k, v in obs.items() if k in OBS_ROOTS},
                                   default=lambda x: x.tolist(), allow_nan=False))
        base, _ = decision_context(events, obs, contract, history)
        immutable = lambda event, measured: guard(event, measured, contract, events)
        current = [event for event in base if immutable(event, obs)[0]]
        current, _ = filter_completed_reoffers(current, obs, history, contract)
        if not current:
            current = [event for event in events if event.action.event == 'hold']
        packet = {'schema': 'rssr.rule_revision_packet.v1', 'case_id': case_id,
                  'parent_sha256': parent_sha256, 'observation': obs,
                  'available_manipulators': sorted({e.action.hand for e in events
                                                     if e.action.hand is not None}),
                  'current_guarded_candidate_ids': [e.event_id for e in current],
                  'task_predicates': contract['predicates'],
                  'task_context': build_task_context(contract, scene, events,
                                                     source_task_text) if scene else None,
                  'primitive_templates': [template_packet(e, obs) for e in events
                                          if e.action.event != 'reference'],
                  'episode_memory': episode_memory(history), 'trigger': trigger,
                  'current_episode_feedback': trajectory_feedback(history),
                  'history_prefix_sha256': digest(history),
                  'observation_roots': sorted(OBS_ROOTS),
                  'historical_success_labels_supplied': bool(
                      trigger.get('scene_profile', {}).get('seed_source_sample_count')),
                  'external_historical_success_labels_supplied': bool(
                      trigger.get('scene_profile', {}).get('seed_source_sample_count')),
                  'same_campaign_prior_labels_supplied': any(
                      not case_id.startswith('seq189:') for case_id in
                      trigger.get('scene_profile', {}).get('completed_same_scene_cases', [])),
                  'same_campaign_prior_scene_cases': [case_id for case_id in
                      trigger.get('scene_profile', {}).get('completed_same_scene_cases', [])
                      if not case_id.startswith('seq189:')],
                  'prior_training_profile_sample_count': trigger.get(
                      'scene_profile', {}).get('seed_source_sample_count', 0)}
        packet['failure_stage_route'] = route_stage(contract, obs, trigger)
        pinned_json(revision / 'packet.json', packet)
        evidence_sha256 = file_sha(revision / 'packet.json')
        errors = []
        for attempt in range(2):
            request_dir = revision / f'request_{attempt:02d}'
            key = 'api:rule_revision:' + digest(str(request_dir.resolve()))[:20]
            charged = False

            def reserve(request_sha256):
                nonlocal charged
                self.store.reserve('api', key, {'case_id': case_id,
                    'role': 'runtime_rule_revision', 'request_sha256': request_sha256,
                    'output': str(request_dir)})
                charged = True

            evidence = bounded_rule_request(packet, evidence_sha256, errors)
            try:
                raw, request_sha256 = _request('runtime_rule_revision', PROMPT,
                    evidence, request_dir, MODEL_PROFILE, reserve,
                    max_output_tokens=RULE_MAX_OUTPUT_TOKENS)
            except (DecisionUnavailable, InvalidCompleteResponse) as error:
                if charged:
                    self.store.settle(key, 'error', str(request_dir))
                if isinstance(error, InvalidCompleteResponse) and attempt == 0:
                    errors.append('First provider response was not complete JSON')
                    pinned_json(request_dir / 'compile_result.json', {'accepted': False,
                        'error': 'invalid_complete_response',
                        'physical_state_changed': False})
                    continue
                raise
            if charged:
                self.store.settle(key, 'complete', str(request_dir / 'response.json'))
            pinned_json(request_dir / 'raw_model_proposal.json', raw)
            compiled = raw
            parent_copy_error = None
            copied_parent = raw.get('parent_sha256') if isinstance(raw, dict) else None
            if (isinstance(raw, dict) and raw.get('schema') == 'rssr.rule_patch.v1' and
                    raw.get('evidence_sha256') == evidence_sha256 and
                    isinstance(copied_parent, str) and len(copied_parent) == 64 and
                    copied_parent[:16] == parent_sha256[:16] and
                    all(char in '0123456789abcdef' for char in copied_parent)):
                mismatches = [i for i, (got, expected) in enumerate(
                    zip(copied_parent, parent_sha256)) if got != expected]
                if 1 <= len(mismatches) <= 2:
                    # Parent/evidence are executor-generated identifiers. A
                    # bounded copy error must not discard otherwise current,
                    # agent-authored rule content. Retain both versions so an
                    # independent audit can reconstruct the exact correction.
                    compiled = {**raw, 'parent_sha256': parent_sha256}
                    parent_copy_error = {'schema': 'rssr.rule_parent_copy_correction.v1',
                                         'raw_model_proposal_sha256': file_sha(
                                             request_dir / 'raw_model_proposal.json'),
                                         'evidence_sha256': evidence_sha256,
                                         'copied_parent_sha256': copied_parent,
                                         'bound_parent_sha256': parent_sha256,
                                         'mismatch_positions': mismatches,
                                         'rule_content_changed': False}
                    pinned_json(request_dir / 'normalization.json', parent_copy_error)
            pinned_json(request_dir / 'proposal.json', compiled)
            try:
                patch_sha256 = validate_patch(compiled, obs, events, parent_sha256, evidence_sha256)
                before = current
                after, application = apply_patch(compiled, obs, base, events, immutable,
                                                 activated_time_s=obs['time_s'])
                after, completed_rejections = filter_completed_reoffers(
                    after, obs, history, contract)
                application['completed_reoffer_rejections'] = completed_rejections
                rejected = application['immutable_rejections'][:6]
                rejected += completed_rejections[:6]
                if (any(event.action.event != 'hold' for event in before) and
                        not any(event.action.event != 'hold' for event in after)):
                    raise ValueError('Rule removes all currently actionable events; '
                                     'guard rejections: ' + str(rejected))
                if not after or [event.as_packet() for event in before] == [
                        event.as_packet() for event in after]:
                    raise ValueError('Rule changes no guarded candidate at this observation; '
                                     'current candidates: ' + str(packet[
                                         'current_guarded_candidate_ids'][:20]) +
                                     '; true rule IDs: ' + str([r['id'] for r in compiled['rules']
                                         if expression(r['when'], obs)]) +
                                     '; guard rejections: ' + str(rejected))
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                errors.append(str(exc))
                pinned_json(request_dir / 'compile_result.json', {'accepted': False,
                    'error': str(exc), 'physical_state_changed': False})
                continue
            pinned_json(request_dir / 'compile_result.json', {'accepted': True,
                'patch_sha256': patch_sha256, 'request_sha256': request_sha256,
                'response_sha256': file_sha(request_dir / 'response.json'),
                'parent_copy_correction': parent_copy_error is not None,
                'physical_state_changed': False, 'native_success_certified': False})
            return compiled, revision
        raise ValueError('RuleAgent produced no compilable patch; charged attempts retained')
