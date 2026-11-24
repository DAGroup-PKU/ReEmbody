"""Scene-neutral phase checks for transferring measured historical rules."""
from __future__ import annotations

import json
from pathlib import Path

from rssr_augment.shared.family_autoresearch.store import file_sha


class CertifiedRulePhaseIndex:
    def __init__(self, path: Path, expected_sha256: str,
                 expected_seed_manifest_sha256: str):
        path = Path(path)
        if file_sha(path) != expected_sha256:
            raise ValueError('Certified rule phase index hash differs')
        data = json.loads(path.read_text())
        source_seqs = data.get('source_strategy_seqs',
                               [data.get('source_strategy_seq')])
        if (data.get('schema') != 'rssr.certified_rule_source_phase_index.v1' or
                not source_seqs or not all(isinstance(seq, int) for seq in source_seqs) or
                data.get('errors') or
                data.get('seed_manifest_sha256') != expected_seed_manifest_sha256):
            raise ValueError('Certified rule phase index provenance differs')
        self.records = {}
        for record in data['records']:
            key = (record['source_case_id'], record['source_proposal_sha256'],
                   record['source_rule_id'])
            if key in self.records:
                raise ValueError('Certified rule phase index duplicate')
            self.records[key] = record
        if len(self.records) != data['record_count']:
            raise ValueError('Certified rule phase index count differs')

    def describe(self, source: dict, observation: dict, contract: dict) -> list[dict]:
        predicates = contract['predicates']
        current = observation['live_predicates']
        if {row['id'] for row in predicates} != set(current):
            raise ValueError('Current native predicate identity differs')
        passed = {index for index, row in enumerate(predicates)
                  if current[row['id']]['passed']}
        result = []
        for rule in source['rules']:
            key = (source['source_case_id'], source['source_proposal_sha256'], rule['id'])
            record = self.records.get(key)
            origin = 'frozen_import'
            if record is None:
                # Same-run strict successes can add certified rules after the
                # frozen import index was built. Their source-phase witness is
                # stored with the append-only scene sample at actual exposure.
                same_run = [row for row in source.get('source_rule_phase_records', [])
                            if row['rule_id'] == rule['id']]
                if not same_run:
                    result.append({'rule_id': rule['id'], 'source_phase_known': False,
                                   'source_prerequisites_met_now': None,
                                   'missing_source_passed_predicate_positions_now': []})
                    continue
                if len(same_run) != 1:
                    raise ValueError('Same-run source rule phase witness is not unique')
                record = same_run[0]
                origin = 'same_run_witness'
                if record['source_predicate_kinds'] != [row['kind'] for row in predicates]:
                    raise ValueError('Same-run source predicate structure differs')
                if (not isinstance(record['source_exposure_time_s'], (int, float)) or
                        not isinstance(record['source_exposed_event_id'], str) or
                        len(record['source_exposure_observation_sha256']) != 64):
                    raise ValueError('Same-run source phase witness invalid')
            source_passed = set(record['native_predicate_positions_passed_at_source_exposure'])
            if (any(not isinstance(position, int) or position < 0
                    for position in source_passed)):
                raise ValueError('Source predicate position invalid')
            if max(source_passed, default=-1) >= len(predicates):
                raise ValueError('Source predicate position exceeds current contract')
            missing = sorted(source_passed - passed)
            result.append({'rule_id': rule['id'], 'source_phase_known': True,
                           'source_phase_origin': origin,
                           'source_exposure_time_s': record['source_exposure_time_s'],
                           'source_exposed_event_id': record['source_exposed_event_id'],
                           'source_prerequisites_met_now': not missing,
                           'missing_source_passed_predicate_positions_now': missing,
                           'source_passed_predicate_positions': sorted(source_passed)})
        return result


def executable_certified_rule_keys(memory: list[dict],
                                   available_event_ids: set[str]) -> set[tuple[str, str, str]]:
    """Identify prior rules whose measured phase, condition and offer align now."""
    keys = set()
    for source in memory:
        if source['offered_event_ids_unavailable_now']:
            continue
        ready = {row['rule_id'] for row in source['source_phase_evidence']
                 if row['source_prerequisites_met_now'] is True}
        applicable = set(source['applicable_rule_ids_now'])
        for rule in source['rules_rebound_to_current_scene']:
            if (rule['id'] in ready and rule['id'] in applicable and
                    any(offer['event_id'] in available_event_ids
                        for offer in rule['offer'])):
                keys.add((source['source_case_id'],
                          source['source_proposal_sha256'], rule['id']))
    return keys
