"""Append-only, same-campaign scene memory for concurrent rule evolution.

The signature is relational and independent of case IDs. Hash-pinned prior
campaign samples and completed cases in this run can contribute. This module
never certifies success or changes native physics.
"""
from __future__ import annotations

import json
import fcntl
import re
from collections import Counter
from pathlib import Path
from statistics import median

from rssr_augment.shared.family_autoresearch.store import digest, file_sha, pinned_json


def append_sample(root: Path, sample: dict) -> Path:
    """Serialize completion-order sample allocation across worker processes."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.append.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            prior_paths = sorted(root.glob('sample_*.json'))
            if any(json.loads(p.read_text())['case_id'] == sample['case_id']
                   for p in prior_paths):
                raise ValueError('A scene sample cannot be appended twice')
            indices = [int(p.stem.split('_')[1]) for p in prior_paths]
            index = max(indices, default=-1) + 1
            path = root / f'sample_{index:03d}.json'
            pinned_json(path, sample)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
    return path


def trajectory_feedback(history: list[dict]) -> dict:
    """Bounded causal evidence from the *completed same-campaign* episode."""
    phase = {}
    mode = Counter()
    predicates = {}
    contact = {}
    concurrent = {}
    blocked_modes = Counter()
    blocked_objects = {}
    previous_opening = {}
    previous_time = None
    closure_progress = {}
    for row in history:
        event_id = row['event_id']
        key = event_id.split(':', 1)[0]
        openings = row.get('gripper_opening_m', {})
        hand = row.get('selected_hand')
        if (key in ('contact', 'grasp') and hand in openings and
                hand in previous_opening and previous_time is not None):
            elapsed = float(row['time_s']) - previous_time
            if elapsed > 0:
                before = float(previous_opening[hand])
                after = float(openings[hand])
                rate = (before - after) / elapsed
                entry = closure_progress.setdefault(hand, {'windows': [],
                                                            'positive_rates_m_s': []})
                entry['windows'].append({
                    'event_id': event_id, 'elapsed_s': elapsed,
                    'opening_before_m': before, 'opening_after_m': after,
                    'closure_rate_m_s': rate,
                    'target_contact_observed': bool(row.get('hand_contact', {}).get(
                        row.get('selected_object_id'), {}).get(hand, False)),
                })
                if rate > 0:
                    entry['positive_rates_m_s'].append(rate)
        for observed_hand, opening in openings.items():
            previous_opening[observed_hand] = float(opening)
        previous_time = float(row['time_s'])
        mode[event_id] += 1
        bucket = phase.setdefault(key, {'count': 0, 'stop_reasons': Counter(),
                                        'margin_delta_sum': 0., 'nonimproving': 0,
                                        'last_time_s': None, 'last_margin': None})
        bucket['count'] += 1
        bucket['stop_reasons'][row.get('stop_reason', 'unknown')] += 1
        delta = float(row.get('delta_margin', 0.))
        bucket['margin_delta_sum'] += delta
        bucket['nonimproving'] += delta <= 0.
        bucket['last_time_s'] = row['time_s']
        bucket['last_margin'] = row.get('signed_margin')
        if row.get('stop_reason') == 'non_target_contact':
            hand = row.get('selected_hand')
            target = row.get('selected_object_id')
            blockers = sorted({c['object_id'] for c in
                               row.get('gripper_scene_contacts', {}).get(hand, [])
                               if c.get('object_id') != target})
            blocked_modes[event_id] += 1
            blocked_objects.setdefault(event_id, Counter()).update(blockers)
        for predicate_id, measured in row.get('live_predicates', {}).items():
            margin = float(measured['instantaneous_margin'])
            entry = predicates.setdefault(predicate_id, {
                'kind': measured['kind'], 'first_passed_s': None,
                'max_instantaneous_margin': margin, 'final_instantaneous_margin': margin,
                'final_passed': False, 'final_measurements': {}})
            if measured['passed'] and entry['first_passed_s'] is None:
                entry['first_passed_s'] = row['time_s']
            entry['max_instantaneous_margin'] = max(entry['max_instantaneous_margin'], margin)
            entry['final_instantaneous_margin'] = margin
            entry['final_passed'] = bool(measured['passed'])
            entry['final_measurements'] = measured.get('measurements', {})
        for obj, hands in row.get('finger_contact', {}).items():
            for hand, pads in hands.items():
                key = obj + ':' + hand
                entry = contact.setdefault(key, {'any_contact_windows': 0,
                                                 'bilateral_contact_windows': 0,
                                                 'last_bilateral_contact_s': None})
                entry['any_contact_windows'] += any(pads)
                entry['bilateral_contact_windows'] += all(pads)
                if all(pads):
                    entry['last_bilateral_contact_s'] = row['time_s']
        carried, hand = row.get('selected_object_id'), row.get('selected_hand')
        if (carried in row.get('finger_contact', {}) and
                hand in row['finger_contact'][carried] and
                all(row['finger_contact'][carried][hand])):
            for other_id in sorted({c['object_id'] for c in
                                    row.get('gripper_scene_contacts', {}).get(hand, [])}):
                if other_id == carried:
                    continue
                key = carried + ':' + hand + '|' + other_id
                entry = concurrent.setdefault(key, {'windows': 0,
                                                      'first_s': row['time_s'],
                                                      'last_s': row['time_s']})
                entry['windows'] += 1
                entry['last_s'] = row['time_s']
    modes = [{'event_id': name, 'count': count}
             for name, count in mode.most_common(12)]
    collisions = [{'event_id': name, 'count': count,
                   'contacted_object_ids': dict(blocked_objects[name].most_common(4))}
                  for name, count in blocked_modes.most_common(6)]
    recent = [{'event_id': r['event_id'], 'time_s': r['time_s'],
               'stop_reason': r.get('stop_reason'),
               'signed_margin': r.get('signed_margin'),
               'delta_margin': r.get('delta_margin')} for r in history[-8:]]
    return {'schema': 'rssr.scene_trajectory_feedback.v1',
            'event_windows': len(history),
            'phase_stats': {k: {**v, 'stop_reasons': dict(v['stop_reasons'])}
                            for k, v in phase.items()},
            'frequent_modes': modes, 'predicate_progress': predicates,
            'non_target_contact_modes': collisions,
            'contact_progress': contact, 'recent_events': recent,
            'gripper_closure_progress': {
                hand: {'recent_windows': entry['windows'][-6:],
                       'positive_window_count': len(entry['positive_rates_m_s']),
                       'median_positive_rate_m_s': (median(entry['positive_rates_m_s'][-8:])
                                                    if entry['positive_rates_m_s'] else None),
                       'latest_opening_m': previous_opening[hand]}
                for hand, entry in closure_progress.items()},
            'concurrent_gripper_scene_contact': concurrent,
            'full_event_history_retained_outside_prompt': True}


def scene_signature(contract: dict, scene: dict) -> tuple[str, dict]:
    objects = {o['id']: o for o in scene['objects']}
    roles: dict[str, str] = {}
    joints: dict[str, str] = {}

    def bind(name: str) -> str:
        if name not in roles:
            roles[name] = f'object_{len(roles):02d}'
        return roles[name]

    for pred in contract['predicates']:
        for key in ('object', 'container', 'region', 'support', 'base', 'upper'):
            if isinstance(pred.get(key), str) and pred[key] in objects:
                bind(pred[key])
        for name in pred.get('objects', []):
            if name in objects:
                bind(name)
        if isinstance(pred.get('joint'), str):
            joints.setdefault(pred['joint'], f'joint_{len(joints):02d}')
    for name in objects:
        bind(name)
    signature = {
        'schema': 'rssr.scene_structure.v1',
        'predicates': [
            {'kind': p['kind'],
             'relations': {k: roles[p[k]] for k in
                           ('object', 'container', 'region', 'support', 'base', 'upper')
                           if isinstance(p.get(k), str) and p[k] in roles},
             'objects': [roles[x] for x in p.get('objects', []) if x in roles],
             'hand': p.get('hand'),
             'joint': joints.get(p.get('joint'))}
            for p in contract['predicates']],
        'objects': [{'binding': roles[o['id']], 'shape': o.get('shape'),
                     'role': o.get('role')} for o in scene['objects']],
    }
    return digest(signature), {'schema': 'rssr.scene_binding.v1',
                               'object_bindings': roles, 'joint_bindings': joints,
                               'structure': signature}


class SceneProfileBank:
    """Per-scene append-only samples with a pinned prior training snapshot."""
    def __init__(self, run: Path, signature: str, binding: dict):
        self.run = Path(run)
        self.root = self.run / 'scene_profiles' / signature
        self.seed_root = self.run / 'seed_scene_profiles' / signature
        self.signature = signature
        self.binding = binding

    def context(self) -> dict:
        rows = []
        seed_manifest = json.loads((self.run / 'seed_profile_manifest.json').read_text())
        source_seq = int(seed_manifest['source_strategy_seq'])
        # Imported samples are immutable training evidence. Namespace their
        # IDs so rerunning the same official case can append a new, distinct
        # sample without modifying the source campaign or losing provenance.
        imported = [item for item in seed_manifest['samples']
                    if Path(item['relative_path']).parent.name == self.signature]
        for item in imported:
            path = self.run / item['relative_path']
            if file_sha(path) != item['sha256']:
                raise ValueError('Imported scene profile sample hash differs')
            row = json.loads(path.read_text())
            if row['scene_signature'] != self.signature:
                raise ValueError('Cross-scene training profile contamination')
            item_seq = int(item.get('source_strategy_seq', source_seq))
            source_id = row['case_id']
            # New samples already carry their source campaign and attempt.
            # Legacy samples have only the official case ID and need a source
            # campaign prefix on import.
            row['case_id'] = (source_id if re.match(r'^seq\d+:', source_id)
                              else f'seq{item_seq}:{source_id}')
            row['seed_source_case_id'] = source_id
            row['seed_source_strategy_seq'] = item_seq
            for proposal in row.get('rule_proposals', []):
                name = proposal['source_case_id']
                proposal['source_case_id'] = (name if re.match(r'^seq\d+:', name)
                                              else f'seq{item_seq}:{name}')
            for proposal in row.get('eligible_reuse', []):
                name = proposal['source_case_id']
                proposal['source_case_id'] = (name if re.match(r'^seq\d+:', name)
                                              else f'seq{item_seq}:{name}')
            rows.append({**row, 'sample_sha256': file_sha(path)})
        pilot_manifest_path = self.run / 'prior_pilot_profile_manifest.json'
        if pilot_manifest_path.exists():
            pilot_manifest = json.loads(pilot_manifest_path.read_text())
            for item in pilot_manifest['samples']:
                if Path(item['relative_path']).parent.name != self.signature:
                    continue
                path = self.run / item['relative_path']
                if file_sha(path) != item['sha256']:
                    raise ValueError('Prior pilot profile sample hash differs')
                row = json.loads(path.read_text())
                if row['scene_signature'] != self.signature:
                    raise ValueError('Prior pilot scene signature differs')
                item_seq = int(item['source_strategy_seq'])
                source_id = row['case_id']
                row['case_id'] = (source_id if re.match(r'^seq\d+:', source_id)
                                  else f'seq{item_seq}:{source_id}')
                row['seed_source_case_id'] = source_id
                row['seed_source_strategy_seq'] = item_seq
                row['prior_pilot_provenance_sha256'] = item['source_sha256']
                for proposal in row.get('rule_proposals', []):
                    source = proposal['source_case_id']
                    proposal['source_case_id'] = (
                        source if re.match(r'^seq\d+:', source)
                        else f'seq{item_seq}:{source}')
                for proposal in row.get('eligible_reuse', []):
                    source = proposal['source_case_id']
                    proposal['source_case_id'] = (
                        source if re.match(r'^seq\d+:', source)
                        else f'seq{item_seq}:{source}')
                rows.append({**row, 'sample_sha256': file_sha(path)})
        for path in sorted(self.root.glob('sample_*.json')):
            row = json.loads(path.read_text())
            if row['scene_signature'] != self.signature:
                raise ValueError('Cross-scene profile contamination')
            rows.append({**row, 'sample_sha256': file_sha(path)})
        # Full provenance remains on disk; K3 sees bounded recent evidence.
        tail = []
        for row in rows[-5:]:
            tail.append({'case_id': row['case_id'],
                         'source_binding': row['source_binding'],
                         'native_task_success': row['native_task_success'],
                         'strict_replay_accepted': row['strict_replay_accepted'],
                         'native_required_failures': row['native_required_failures'],
                         'native_hard_failures': row['native_hard_failures'],
                         'trajectory_feedback': row['trajectory_feedback'],
                         'event_history_sha256': row['event_history_sha256'],
                         'rule_proposals': [{
                             'rules': p['proposal']['rules'],
                             'reason': p['proposal']['reason'],
                             'actually_exposed': p['actually_exposed'],
                             'effect_observations': [{
                                 'rule_id': e.get('rule_id'), 'status': e.get('status'),
                                 'goal_progress': e.get('goal_progress')}
                                 for e in p['effect_observations'][-3:]],
                             'proposal_sha256': p['proposal_sha256']}
                             for p in row['rule_proposals'][-2:]],
                         'sample_sha256': row['sample_sha256']})
        # A later failed episode does not erase a previously certified rule.
        # Reuse is still restricted to actual exposure, measured change,
        # strict native success and exact replay by append()'s admission gate.
        certified_portfolio = [proposal for row in rows
                               if row['strict_replay_accepted']
                               for proposal in row['eligible_reuse']]
        # The last few episodes can all be failures, hiding a certified
        # successful event sequence from the RuleAgent. Preserve a bounded
        # success witness across recency churn. It contains event categories
        # and counts only: no old robot commands, object poses, timings or
        # automatic choice of a hand or event in the new episode.
        last_success = next((row for row in reversed(rows)
                             if row['strict_replay_accepted'] and
                             row['native_task_success']), None)
        success_anchor = None
        if last_success is not None:
            source_binding = last_success['source_binding']
            name_map = {}
            for kind in ('object_bindings', 'joint_bindings'):
                current_by_role = {role: name for name, role in
                                   self.binding[kind].items()}
                for name, role in source_binding[kind].items():
                    if role not in current_by_role:
                        raise ValueError('Successful source event has no current scene role')
                    name_map[name] = current_by_role[role]

            def current_event_id(source_id: str) -> str:
                # Replace complete colon-delimited identifiers only. Rule
                # names and event kinds are not object bindings.
                return ':'.join(name_map.get(part, part)
                                for part in source_id.split(':'))

            useful = {'approach', 'contact', 'grasp', 'lift', 'transport',
                      'align', 'open', 'insert', 'close', 'push_close', 'release'}
            modes = [
                {'event_id': current_event_id(mode['event_id']), 'count': mode['count']}
                for mode in last_success['trajectory_feedback']['frequent_modes']
                if mode['event_id'].split(':', 1)[0] in useful
            ][:8]
            success_anchor = {
                'source_case_id': last_success['case_id'],
                'source_sample_sha256': last_success['sample_sha256'],
                'strict_replay_accepted': True,
                'frequent_physical_event_modes': modes,
                'scope': 'prior same-structure outcome evidence; no robot commands or causal rule credit',
            }
        latest_eligible = next((row for row in reversed(rows)
                                if row['strict_replay_accepted'] and
                                row['eligible_reuse']), None)
        reusable = latest_eligible['eligible_reuse'][-1] if latest_eligible else None
        return {'schema': 'rssr.scene_profile_context.v1',
                'scene_signature': self.signature,
                'profile_sha256': digest(rows),
                'current_binding': self.binding,
                'completed_same_scene_cases': [r['case_id'] for r in rows],
                'recent_samples': tail,
                'latest_strict_success_anchor': success_anchor,
                'eligible_reuse': reusable,
                'eligible_reuse_portfolio': certified_portfolio,
                'eligible_reuse_portfolio_sha256': digest(certified_portfolio),
                'seed_source_strategy_seq': source_seq,
                'seed_source_strategy_seqs': sorted({r['seed_source_strategy_seq']
                                                      for r in rows
                                                      if 'seed_source_strategy_seq' in r}),
                'seed_source_sample_count': sum('seed_source_case_id' in r for r in rows),
                'cross_scene_samples_read': False}

    def initial_patch(self, context: dict, parent_sha256: str,
                      evidence_sha256: str) -> dict | None:
        """Rebind only a previously exposed, measured, strict-success rule."""
        source = context.get('eligible_reuse')
        if source is None:
            return None
        from_binding = source['source_binding']
        to_binding = context['current_binding']
        remap = {}
        for kind in ('object_bindings', 'joint_bindings'):
            target = {role: name for name, role in to_binding[kind].items()}
            for name, role in from_binding[kind].items():
                if role not in target:
                    raise ValueError('Promoted rule has no current scene role')
                remap[name] = target[role]
        ordered = sorted(remap, key=len, reverse=True)

        def rebind(value):
            if isinstance(value, dict):
                return {k: rebind(v) for k, v in value.items()}
            if isinstance(value, list):
                return [rebind(v) for v in value]
            if isinstance(value, str):
                result = value
                for i, name in enumerate(ordered):
                    result = result.replace(name, f'__SCENE_ROLE_{i}__')
                for i, name in enumerate(ordered):
                    result = result.replace(f'__SCENE_ROLE_{i}__', remap[name])
                return result
            return value

        return {'schema': 'rssr.rule_patch.v1',
                'parent_sha256': parent_sha256,
                'evidence_sha256': evidence_sha256,
                'reason': ('Pinned same-scene reusable rule from ' +
                           source['source_case_id'] + '; native success does not '
                           'prove this rule caused success'),
                'rules': rebind(source['rules'])}

    def append(self, receipt: dict, rule_journal: Path, revision_root: Path,
               start_context: dict, contract: dict) -> Path:
        attempt_id = Path(rule_journal).parent.name
        if not re.fullmatch(r'attempt_\d+', attempt_id):
            raise ValueError('Scene profile requires an explicit attempt directory')
        sample_id = (f"seq{receipt['strategy_seq']}:"
                     f"{receipt['case_id']}:{attempt_id}")
        if sample_id in start_context['completed_same_scene_cases']:
            raise ValueError('A scene sample cannot be appended twice')
        if start_context['scene_signature'] != self.signature:
            raise ValueError('Scene signature changed during episode')
        journal = [json.loads(p.read_text()) for p in sorted(rule_journal.glob('event_*.json'))]
        exposed = {r['payload'].get('revision') for r in journal if r['kind'] == 'actual_exposure'}
        exposure_rows = [r['payload'] for r in journal if r['kind'] == 'actual_exposure']
        actual_exposures = {(r['revision'], r['rule_id']): r for r in exposure_rows}
        if len(actual_exposures) != len(exposure_rows):
            raise ValueError('Duplicate certified source rule exposure')
        predicate_kinds = [row['kind'] for row in contract['predicates']]
        predicate_ids = [row['id'] for row in contract['predicates']]

        def measured_source_phase(revision: str, rule_ids: set[str]) -> list[dict]:
            records = []
            for rule_id in sorted(rule_ids):
                exposure = actual_exposures.get((revision, rule_id))
                if exposure is None:
                    raise ValueError('Certified rule lacks actual source exposure')
                if digest(exposure['before_observation']) != exposure['before_sha256']:
                    raise ValueError('Certified source exposure observation hash differs')
                before = exposure['before_observation']['live_predicates']
                if set(before) != set(predicate_ids):
                    raise ValueError('Certified source predicate identity differs')
                records.append({
                    'rule_id': rule_id,
                    'native_predicate_positions_passed_at_source_exposure': [
                        index for index, predicate_id in enumerate(predicate_ids)
                        if before[predicate_id]['passed']],
                    'source_predicate_kinds': predicate_kinds,
                    'source_exposure_time_s': exposure['time_s'],
                    'source_exposed_event_id': exposure['event_id'],
                    'source_exposure_observation_sha256': exposure['before_sha256'],
                })
            return records
        effects = [r['payload'] for r in journal if r['kind'] == 'effect_observation']
        proposals = []
        reuse = []
        for revision in sorted(revision_root.glob('revision_*')):
            for request in sorted(revision.glob('request_*')):
                proposal = request / 'proposal.json'
                compile_result = request / 'compile_result.json'
                if not proposal.exists() or not compile_result.exists():
                    continue
                status = json.loads(compile_result.read_text())
                if status.get('accepted'):
                    proposal_data = json.loads(proposal.read_text())
                    effect_rows = [e for e in effects if e.get('revision') == str(revision)]
                    proposals.append({'proposal': proposal_data,
                                      'proposal_sha256': file_sha(proposal),
                                      'source_case_id': sample_id,
                                      'actually_exposed': str(revision) in exposed,
                                      'effect_observations': effect_rows})
                    if receipt['accepted'] and not receipt['native']['hard_failures'] and str(revision) in exposed:
                        measured = {e['rule_id'] for e in effect_rows
                                    if e['status'] == 'observed_expected_change'}
                        rules = [r for r in proposal_data['rules'] if r['id'] in measured]
                        if rules:
                            reuse.append({'source_case_id': sample_id,
                                          'source_binding': self.binding,
                                          'source_proposal_sha256': file_sha(proposal),
                                          'rules': rules,
                                          'source_rule_phase_records': measured_source_phase(
                                              str(revision), {r['id'] for r in rules}),
                                          'eligibility': 'strict native+replay; actual rule exposure; measured expected change'})
        seed_dir = Path(rule_journal).parent / 'profile_seed'
        seed_proposal = seed_dir / 'proposal.json'
        if (seed_proposal.exists() and receipt['accepted'] and
                not receipt['native']['hard_failures'] and str(seed_dir) in exposed):
            seed = json.loads(seed_proposal.read_text())
            measured = {e['rule_id'] for e in effects
                        if e.get('revision') == str(seed_dir) and
                        e['status'] == 'observed_expected_change'}
            rules = [r for r in seed['rules'] if r['id'] in measured]
            if rules:
                reuse.append({'source_case_id': sample_id,
                              'source_binding': self.binding,
                              'source_proposal_sha256': file_sha(seed_proposal),
                              'rules': rules,
                              'source_rule_phase_records': measured_source_phase(
                                  str(seed_dir), {r['id'] for r in rules}),
                              'eligibility': 'reused same-scene rule; strict native+replay; actual exposure and measured change'})
        sample = {
            'schema': 'rssr.scene_profile_sample.v1',
            'scene_signature': self.signature,
            'source_binding': self.binding,
            'case_id': sample_id,
            'official_case_id': receipt['case_id'],
            'attempt_id': attempt_id,
            'start_profile_sha256': start_context['profile_sha256'],
            'start_prior_case_ids': start_context['completed_same_scene_cases'],
            'native_task_success': bool(receipt['native']['task_text_physical_success']),
            'strict_replay_accepted': bool(receipt['accepted']),
            'native_required_failures': receipt['native']['required_failures'],
            'native_hard_failures': receipt['native']['hard_failures'],
            'event_history_sha256': digest(receipt['event_history']),
            'trajectory_feedback': trajectory_feedback(receipt['event_history']),
            'rule_proposals': proposals[-4:],
            'eligible_reuse': reuse[-2:],
            'rule_journal_head_sha256': receipt['rule_runtime']['journal_head_content_sha256'],
            'receipt_sha256': file_sha(Path(rule_journal).parent / 'receipt.json'),
            'historical_external_success_oracle_used': False,
        }
        # Cases in one structural family may now run concurrently. Their
        # start snapshots can have the same sample count, so allocate the next
        # completion-order index under a per-profile filesystem lock.
        return append_sample(self.root, sample)
