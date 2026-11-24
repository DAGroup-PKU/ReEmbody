"""Prepare isolated runs; initialization never calls APIs or steps physics."""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil

from rssr_augment.shared.family_autoresearch.store import Store, file_sha, atomic_json
from .common import BASE_RUN, BRANCH, MODULE, ROOT, RESULT, read, pin, relocate, resolve_path, within_branch


def selected_cases(config, available):
    cohort = config.get('cohort')
    if cohort is None:
        return list(available)
    families = cohort['families']
    count = cohort['cases_per_family']
    if not isinstance(count, int) or not 1 <= count <= 50 or len(set(families)) != len(families):
        raise ValueError('Invalid source cohort')
    groups = {family: sorted(cid for cid in available if cid.startswith(family + '_ep'))[:count]
              for family in families}
    if any(len(group) != count for group in groups.values()):
        raise ValueError('Insufficient reconstructed source cases')
    return [groups[family][index] for index in range(count) for family in families]


def seed_knowledge(run, mode, include_prior_pilot):
    seed = read(BASE_RUN / 'seed_profile_manifest.json')
    pilot = read(BASE_RUN / 'prior_pilot_profile_manifest.json')
    selected = seed['samples'] if mode == 'warm_start' else []
    selected_pilot = pilot['samples'] if mode == 'warm_start' and include_prior_pilot else []
    for item in selected + selected_pilot:
        relative = Path(item['relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid historical sample path')
        source = BASE_RUN / relative
        if file_sha(source) != item['sha256']:
            raise ValueError('Historical sample hash changed: ' + str(source))
        target = run / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        if file_sha(target) != item['sha256']:
            raise ValueError('Initialization copy changed: ' + str(target))
    new_seed = {**seed, 'samples': selected, 'source_training_samples': len(selected),
                'lyc_knowledge_mode': mode, 'historical_origin': str(BASE_RUN)}
    if mode == 'cold_start':
        new_seed.update(source_certified_reusable_rule_sets=0,
                        source_certified_reusable_rule_entries=0)
    pin(run / 'seed_profile_manifest.json', new_seed)
    pin(run / 'prior_pilot_profile_manifest.json',
        {**pilot, 'samples': selected_pilot, 'source_training_samples': len(selected_pilot),
         'source_strategies': pilot['source_strategies'] if selected_pilot else [],
         'lyc_knowledge_mode': mode})
    phase = read(BASE_RUN / 'source_rule_phase_index.json')
    if mode == 'cold_start':
        phase.update(records=[], record_count=0)
    phase['seed_manifest_sha256'] = file_sha(run / 'seed_profile_manifest.json')
    pin(run / 'source_rule_phase_index.json', phase)
    # Explicit inventories do not pretend the old empty operators.json was the
    # actual knowledge channel. Executable rules are embedded in scene samples.
    inventory = {'schema': 'rssr.lyc_dev_v2.knowledge_inventory.v1', 'mode': mode,
                 'historical_feedback_samples': len(selected),
                 'prior_pilot_samples': len(selected_pilot),
                 'initial_total_samples': len(selected) + len(selected_pilot),
                 'certified_phase_records': phase['record_count'],
                 'source': str(BASE_RUN), 'source_run_modified': False,
                 'learned_rules_embedded_in_samples': True,
                 'current_sample_directory': str(run / 'scene_profiles'),
                 'initial_samples_are_local_copies_not_symlinks': True,
                 'knowledge_is_writable_and_continues_learning': True}
    pin(run / 'knowledge_inventory.json', inventory)
    pin(run / 'initial_knowledge.json', {'operators': [], 'experiences': [],
                                       'actual_learning_channel': 'scene_profiles'})
    return inventory


def initialize(name, mode, include_prior_pilot=True, physical=None, config=None):
    if not name or Path(name).name != name or name in ('.', '..'):
        raise ValueError('Run name must be one directory component')
    cfg = deepcopy(config or read(BRANCH / 'config/pipeline.json'))
    if cfg.get('direct_run_layout')=='physical_groups_at_batch_root':
        if name not in ('physical_00','physical_01','physical_02'):raise ValueError('Invalid formal physical group name')
        run=within_branch(RESULT/name)
    else:
        run = within_branch(RESULT / 'runs' / mode / name)
    strategy_file = run / 'strategy.json'
    if strategy_file.exists():
        saved = read(strategy_file)
        if (saved['lyc_dev_v2']['knowledge_mode'] != mode or
                saved['lyc_dev_v2']['include_prior_pilot'] != (include_prior_pilot and mode == 'warm_start') or
                saved.get('physical_augmentation') != physical or
                saved['lyc_dev_v2']['config'] != cfg):
            raise ValueError('Existing run differs; do not reset progress or knowledge')
        return run
    if (run / 'manifest.json').exists():
        raise ValueError('Partial initialization already has a manifest; inspect it')
    run.mkdir(parents=True, exist_ok=True)
    inventory = seed_knowledge(run, mode, include_prior_pilot)
    original = BRANCH / 'config/base_strategy.json'
    strategy = relocate(read(original))
    cases = selected_cases(cfg, strategy['cases'])
    strategy['cases'] = cases
    strategy['campaign']['pilot_cases'] = cases
    strategy['campaign']['minimum_goal_successes_per_family'] = {
        family: int(cfg['cohort']['cases_per_family'] * .7) + 1
        for family in cfg['cohort']['families']} if cfg.get('cohort') else strategy['campaign']['minimum_goal_successes_per_family']
    strategy['campaign']['minimum_goal_successes'] = sum(strategy['campaign']['minimum_goal_successes_per_family'].values())
    strategy['full_run_policy']['all_unique_official_cases'] = len(cases)
    from .retarget.case import RUNTIME  # Loads the vendored schema dependencies first.
    from .retarget.agent import VERSION
    strategy.update(seq=343000, at=datetime.now(timezone.utc).isoformat(),
                    author='lyc_dev_v2', status='launch_authorized', new_run=str(run),
                    executor_module=MODULE, agent_version=VERSION,
                    needs_feedback_before_launch=False,
                    physical_augmentation=physical,
                    aligned_initialization=cfg.get('aligned_initialization'),
                    initial_knowledge=inventory,
                    lyc_dev_v2={'knowledge_mode': mode,
                                'include_prior_pilot': include_prior_pilot and mode == 'warm_start',
                                'config': cfg, 'baseline_strategy_sha256': file_sha(original),
                                'not_the_historical_seq343_run': True})
    strategy['campaign']['max_case_concurrency'] = cfg['concurrency']
    strategy['s1_model_manifest'] = cfg.get('s1_model_manifest')
    strategy['robot'] = 'S1'
    strategy['campaign']['max_api_concurrency'] = cfg['concurrency']
    strategy['campaign']['initial_knowledge'] = inventory
    strategy['evidence']['empty_knowledge_file'] = str(run / 'initial_knowledge.json')
    strategy['evidence']['empty_knowledge_sha256'] = file_sha(run / 'initial_knowledge.json')
    strategy['training_profile_seed'].update(
        manifest_file=str(run / 'seed_profile_manifest.json'),
        manifest_sha256=file_sha(run / 'seed_profile_manifest.json'),
        source_rule_phase_index_file=str(run / 'source_rule_phase_index.json'),
        source_rule_phase_index_sha256=file_sha(run / 'source_rule_phase_index.json'))
    strategy['prior_pilot_profile_seed'].update(
        manifest_file=str(run / 'prior_pilot_profile_manifest.json'),
        manifest_sha256=file_sha(run / 'prior_pilot_profile_manifest.json'))
    # Snapshot every branch source plus shared budget/transport dependencies.
    paths = list((BRANCH / 'retarget').glob('*.py')) + list((BRANCH / 'vendor/runtime').rglob('*.py'))
    paths += list(BRANCH.glob('*.py')) + list((BRANCH / 'shared/family_autoresearch').glob('*.py'))
    paths += [BRANCH / 'shared/responses_transport.py']
    paths += [BRANCH / 'shared/joint_robot_object_v2_20260923/task_subagent.py']
    paths += list(BRANCH.rglob('*.py')) + list(BRANCH.parent.glob('*.py'))
    paths += [BRANCH/'config/pipeline.json',BRANCH/'config/measured_alignment.json']
    strategy['implementation_sha256'] = {str(p): file_sha(p) for p in sorted(set(paths))}
    pin(strategy_file, strategy)
    original_manifest = read(BASE_RUN / 'manifest.json')
    original_manifest.update(cases=cases, pilot_cases=cases,
                             limits={**original_manifest['limits'],
                                     'full': len(cases) * original_manifest['case_limits']['full'],
                                     'api': len(cases) * original_manifest['case_limits']['api']})
    manifest = {**original_manifest, 'strategy_sha256': file_sha(strategy_file),
                'strategy_seq': strategy['seq'], 'executor_module': MODULE,
                'agent_version': VERSION,
                'training_profile_seed_sha256': strategy['training_profile_seed']['manifest_sha256'],
                'prior_pilot_profile_manifest_sha256': strategy['prior_pilot_profile_seed']['manifest_sha256'],
                'lyc_knowledge_mode': mode, 'historical_scores_not_imported': True}
    Store(run).initialize(manifest, strategy['cases'])
    return run


def validate_startup(strategy_path, case_id, *, allow_legacy_snapshot=False):
    strategy = read(strategy_path)
    run = within_branch(strategy['new_run'])
    manifest = read(run / 'manifest.json')
    expected_module = strategy['executor_module'] if allow_legacy_snapshot and strategy['executor_module'] == 'lyc_dev_v2.retarget' else MODULE
    if (strategy['executor_module'] != expected_module or strategy['allow_s1_virtual_base_experiment'] is not False or
            manifest['strategy_sha256'] != file_sha(strategy_path) or
            manifest['executor_module'] != expected_module or manifest['cases'] != strategy['cases'] or
            case_id not in strategy['cases'] or
            strategy['cases'] != selected_cases(strategy['lyc_dev_v2']['config'],
                                               read(BRANCH / 'config/base_strategy.json')['cases'])):
        raise ValueError('Branch launch binding differs')
    for path, expected in strategy['implementation_sha256'].items():
        if file_sha(path) != expected:
            raise ValueError('Pinned implementation changed: ' + path)
    for entry in (strategy['training_profile_seed'], strategy['prior_pilot_profile_seed']):
        if file_sha(entry['manifest_file']) != entry['manifest_sha256']:
            raise ValueError('Initial knowledge manifest changed; create a versioned new run')
        for item in read(entry['manifest_file'])['samples']:
            if file_sha(run / item['relative_path']) != item['sha256']:
                raise ValueError('Initialization evidence changed: ' + item['relative_path'])
    if strategy['lyc_dev_v2']['knowledge_mode'] == 'cold_start':
        if (read(run / 'seed_profile_manifest.json')['samples'] or
                read(run / 'prior_pilot_profile_manifest.json')['samples'] or
                read(run / 'source_rule_phase_index.json')['records']):
            raise ValueError('Cold start cannot import historical learned knowledge')
    from .retarget.transfer_phase import CertifiedRulePhaseIndex
    seed = strategy['training_profile_seed']
    CertifiedRulePhaseIndex(Path(seed['source_rule_phase_index_file']),
                           seed['source_rule_phase_index_sha256'], seed['manifest_sha256'])
    Store(run).validate_events()
    return strategy


def refresh_unstarted_code(run):
    """Finalize a development setup only while all costs and attempts are zero."""
    import fcntl
    from rssr_augment.shared.family_autoresearch.store import digest, _event
    run = within_branch(run)
    with (run/'launcher.lock').open('a') as launch_lock:
        fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Store(run).transaction() as box:
            state = box[0]
            if (any(state['spent'].values()) or state['reservations'] or
                    state['trials'] or any(c['status'] != 'pending' for c in state['cases'].values()) or
                    list((run/'cases').glob('*/attempt_*'))):
                raise ValueError('Cannot reseal any started/charged run; create a new version')
            strategy = read(run/'strategy.json')
            previous = file_sha(run/'strategy.json')
            strategy['implementation_sha256'] = {p: file_sha(p) for p in strategy['implementation_sha256']}
            atomic_json(run/'strategy.json', strategy)
            manifest = read(run/'manifest.json')
            manifest['strategy_sha256'] = file_sha(run/'strategy.json')
            atomic_json(run/'manifest.json', manifest)
            state['manifest_sha256'] = digest(manifest)
            _event(state,'unstarted_development_code_finalized',{'previous_strategy_sha256':previous,
                    'strategy_sha256':manifest['strategy_sha256'],'charged_api_and_physics':0})
