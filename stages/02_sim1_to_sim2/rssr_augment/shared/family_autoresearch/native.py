"""Process-isolated native v80 compiler and physical executor.

Preflight performs IK/FK only. Evaluation consumes the *same* compiled controls
and frozen inputs; it cannot silently replan. Run one job in each subprocess::

    python -m pipeline.family_autoresearch.native --job job.json --phase preflight
    python -m pipeline.family_autoresearch.native --job job.json --phase evaluate

Jobs contain case_id, run_root, index, candidate (or proposal_path), and parent
evidence for repairs. An omitted candidate replays the v74 selection. An exact
historical replay additionally supplies baseline_trajectory_path. Partial
rollouts are deliberately unsupported and can never be counted as successes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from contextlib import contextmanager

ROOT = Path(__file__).resolve().parents[2]
import os as _os
V74 = Path(_os.environ.get('REEMBODY_SOURCE_CASES_DIR', str(Path(__file__).resolve().parents[5] / 'examples')))
RUNTIME = ROOT / 'pipeline/cobot150_runtime_v80_task_completion_20260923'
FIELDS = ('program', 'scene', 'windows', 'model_profile', 'cost', 'decision')
HANDS = ('left', 'right')
INPUTS = ('source_action_state.npz', 'case.json', 'frame_action_mapping.npz',
          'sim2_robot_only_scene.xml')


class PreflightReject(ValueError):
    """An inadmissible proposal, never evidence against a physical mechanism."""


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf8')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def verify_artifacts(expected):
    for name, expected_hash in expected.items():
        if not Path(name).is_file() or sha(name) != expected_hash:
            raise PreflightReject('Frozen artifact changed or missing: ' + str(name))


def candidate_fields(value):
    if not isinstance(value, dict) or not all(k in value for k in ('program', 'scene')):
        raise PreflightReject('Candidate requires program and scene')
    defaults = {'windows': {}, 'model_profile': {}, 'cost': 0.,
                'decision': {'lever': 'baseline', 'target_predicate': 'baseline'}}
    result = {k: copy.deepcopy(value.get(k, defaults.get(k))) for k in FIELDS}
    digest(result)  # Reject NaN/Infinity, including in decision metadata.
    if not isinstance(result['decision'], dict):
        raise PreflightReject('Decision must be an object')
    if type(result['cost']) not in (int, float) or result['cost'] < 0:
        raise PreflightReject('Candidate cost must be finite and nonnegative')
    return result


def physical_identity(candidate):
    return {k: candidate[k] for k in ('program', 'scene', 'windows', 'model_profile')}


def validate_frozen_candidate(candidate, selected):
    if candidate['scene'] != selected['scene']:
        raise PreflightReject('Scene differs from frozen v74 selection')
    if candidate['model_profile'] != selected.get('model_profile', {}):
        raise PreflightReject('Physical model_profile differs from frozen v74 selection')


def intended_scope(candidate, parent, replay=False):
    """Derive scope from actual changes; metadata cannot disguise grip edits."""
    if replay:
        return {'kind': 'replay', 'hands': list(HANDS)}
    before, after = copy.deepcopy(parent['program']), copy.deepcopy(candidate['program'])
    before_release = before.pop('release_opening', {})
    after_release = after.pop('release_opening', {})
    grip_hands = [h for h in HANDS if
                  parent['windows'].get(h, []) != candidate['windows'].get(h, [])
                  or before_release.get(h, []) != after_release.get(h, [])]
    if before == after:
        if not grip_hands:
            raise PreflightReject('Candidate has no control intervention')
        return {'kind': 'gripper_only', 'hands': grip_hands}
    decision = candidate['decision']
    robot = decision.get('robot') or {}
    hands = robot.get('hands') or ([robot['hand']] if robot.get('hand') else [])
    if not hands and decision.get('hand') in HANDS:
        hands = [decision['hand']]
    if not hands and decision.get('lever') in ('ik_profile', 'ik_health', 'robot_ik'):
        hands = list(HANDS)
    if (not isinstance(hands, list) or not hands or len(set(hands)) != len(hands)
            or any(h not in HANDS for h in hands)):
        raise PreflightReject('Spatial intervention must explicitly declare robot hand(s)')
    if any(h not in hands for h in grip_hands):
        raise PreflightReject('Gripper edit lies outside declared intervention hand(s)')
    return {'kind': 'spatial', 'hands': [h for h in HANDS if h in hands]}


def freeze_trajectory(trajectory, parent_trajectory, scope):
    """Freeze collateral IK motion for either hand, including shared upstream."""
    import numpy as np
    trajectory, parent_trajectory = Path(trajectory), Path(parent_trajectory)
    with np.load(trajectory, allow_pickle=False) as current, \
            np.load(parent_trajectory, allow_pickle=False) as parent:
        if (set(current.files) != set(parent.files)
                or not np.array_equal(current['source_time_s'], parent['source_time_s'])):
            raise PreflightReject('Parent and candidate target fields/clocks differ')
        values = {key: current[key].copy() for key in current.files}
        spatial_hands = () if scope['kind'] == 'gripper_only' else scope['hands']
        frozen = []
        for hand in HANDS:
            if hand not in spatial_hands:
                frozen.extend((hand + '_q', hand + '_target_position',
                               hand + '_target_rotation'))
            if hand not in scope['hands']:
                frozen.append('source_' + hand + '_gripper')
        if len(spatial_hands) != 2:
            frozen.append('upstream_q')
        for key in frozen:
            values[key] = parent[key].copy()
    np.savez_compressed(trajectory, **values)
    return {'exact_parent_keys': frozen, 'scope': scope,
            'parent_trajectory_path': str(parent_trajectory),
            'parent_trajectory_sha256': sha(parent_trajectory),
            'trajectory_sha256': sha(trajectory)}


def check_control_scope(model, candidate_stream, parent_stream, scope, safety):
    """Check compiled desired joints, not merely IK metadata or source grip."""
    import numpy as np
    times, candidate = candidate_stream[:2]
    parent_times, parent = parent_stream[:2]
    if (not np.array_equal(times, parent_times) or candidate.shape != parent.shape
            or not np.isfinite(candidate).all() or not np.isfinite(parent).all()):
        raise PreflightReject('Nonfinite or mismatched candidate/parent command stream')
    mask = np.ones(model.nq, dtype=bool)
    if scope['kind'] == 'replay':
        mask[:] = False
    else:
        for hand in scope['hands']:
            indices = safety.hand_qpos_indices(model, hand)
            mask[indices[7:] if scope['kind'] == 'gripper_only' else indices] = False
        if scope['kind'] == 'spatial' and scope['hands'] == list(HANDS):
            # Shared robot upstream is allowed for an explicitly bilateral edit.
            for j in range(model.njnt):
                if (model.joint(j).name or '').startswith('astribot_torso_joint_'):
                    mask[int(model.jnt_qposadr[j])] = False
    if not np.array_equal(candidate[:, mask], parent[:, mask]):
        raise PreflightReject('Compiled control changes coordinates outside declared scope')
    delta = float(np.max(np.abs(candidate - parent)))
    if scope['kind'] != 'replay' and delta <= 1e-12:
        raise PreflightReject('Intervention did not change the actual desired command stream')
    return {'scope': scope, 'unmodified_qpos_exact': True,
            'protected_qpos_coordinates': np.flatnonzero(mask).tolist(),
            'max_command_delta': delta}


def _runtime():
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(key, '1')
    os.environ.setdefault('MUJOCO_GL', 'egl')
    from pipeline.joint_robot_object_v2_20260923 import runner_core as core
    from pipeline.crossfamily_stage_v58_seq104 import safety
    import pilot_ik
    import pilot_physics
    return core, safety, pilot_ik, pilot_physics


def _environment(config):
    import mujoco
    import numpy as np
    expected = config.get('mujoco_version_expected')
    if not expected or mujoco.__version__ != expected:
        raise PreflightReject('MuJoCo version differs from frozen v74 config: '
                              + str(mujoco.__version__) + ' != ' + str(expected))
    module_dir = Path(mujoco.__file__).resolve().parent
    binaries = list(module_dir.glob('*.so*'))
    if not binaries:
        raise PreflightReject('Cannot fingerprint the native MuJoCo binary')
    return {'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
            'python_version': sys.version,
            'python_executable': str(Path(sys.executable).resolve()),
            'binary_sha256': {str(path): sha(path) for path in sorted(binaries)}}


def _paths(job):
    case = job.get('case_id', '')
    if not re.fullmatch(r'(stack_cups|stack_blocks|drawer)_ep\d{6}', case):
        raise PreflightReject('Invalid or unsupported case_id')
    index = job.get('index', 0)
    if type(index) is not int or index < 0:
        raise PreflightReject('index must be a nonnegative integer')
    run = Path(job['run_root']).resolve()
    if run == V74.parent or V74 in run.parents:
        raise PreflightReject('Cannot write into frozen v74 history')
    output = run / 'cases' / case
    trial = output / f'trial_{index:02d}'
    return run, output, trial, trial / 'native'


def _inputs(folder, source, config, candidate, run):
    inputs = Path(folder) / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    for name in INPUTS:
        target = inputs / name
        original = (Path(source) / 'inputs' / name).resolve()
        if target.exists() or target.is_symlink():
            if target.resolve() != original:
                raise PreflightReject('Existing input has incorrect source: ' + str(target))
        else:
            target.symlink_to(original)
    result = copy.deepcopy(config)
    result.update(import_case=str(inputs), run_root=str(folder), batch_root=str(run),
                  api_budget_root=str(run), mode='infer',
                  model_profile=candidate['model_profile'],
                  _release_opening=candidate['program'].get('release_opening', {}))
    return result


def _load_job(job):
    if job.get('partial') or job.get('probe') or job.get('mode') in ('probe', 'partial'):
        raise PreflightReject('Partial probes are not supported by the frozen native adapter')
    run, output, trial, folder = _paths(job)
    source = V74 / job['case_id']
    selected = read(source / 'sim2/selection.json')
    config = read(source / 'config.json')
    if job.get('source_dir') and Path(job['source_dir']).resolve() != source.resolve():
        raise PreflightReject('source_dir must be the canonical v74 case')
    value = job.get('candidate')
    if value is None:
        value = read(job['proposal_path']) if job.get('proposal_path') else selected
    candidate = candidate_fields(value)
    validate_frozen_candidate(candidate, selected)
    parent_evidence = job.get('parent') or {}
    if parent_evidence:
        value = parent_evidence.get('candidate')
        if value is None:
            value = read(parent_evidence['proposal_path'])
        parent = candidate_fields(value)
        validate_frozen_candidate(parent, selected)
        parent_trajectory = parent_evidence.get('trajectory_path')
        if not parent_trajectory:
            parent_trajectory = Path(parent_evidence['proposal_path']).parent / 'inputs/agent_trajectory_full.npz'
        replay = False
    else:
        parent = copy.deepcopy(candidate)
        parent_trajectory = job.get('baseline_trajectory_path') or job.get('trajectory_path')
        if not parent_trajectory:
            if physical_identity(candidate) != physical_identity(candidate_fields(selected)):
                raise PreflightReject('A non-v74 baseline requires its exact compiled trajectory')
            parent_trajectory = Path(selected['trial']).parent / 'inputs/agent_trajectory_full.npz'
        replay = True
    if not Path(parent_trajectory).is_file():
        raise PreflightReject('Missing parent compiled trajectory: ' + str(parent_trajectory))
    scope = intended_scope(candidate, parent, replay=replay)
    spec = {'case_id': job['case_id'], 'source': str(source), 'config': config}
    return run, output, trial, folder, spec, selected, candidate, parent, Path(parent_trajectory), scope


def _tracking_inputs(model, stream, folder, parent_trajectory, scope, safety):
    """Recompute FK diagnostics after freezes; inherited planned FK may be stale."""
    import mujoco
    import numpy as np
    times, qpos = stream[:2]
    paths = [Path(parent_trajectory).resolve().parent / 'goal_tracking.npz',
             Path(parent_trajectory).parent.parent / 'ik/goal_tracking.npz']
    parent_goals = next((path for path in paths if path.is_file()), None)
    fresh = folder / 'ik/goal_tracking.npz'
    target_source = fresh if fresh.is_file() else parent_goals
    with np.load(folder / 'inputs/agent_trajectory_full.npz', allow_pickle=False) as z:
        values = {'time_s': times.copy(), **{
            hand + '_target_position': z[hand + '_target_position'].copy() for hand in HANDS}}
    origins = {hand: 'immutable_source_reference' for hand in HANDS}
    for hand in HANDS:
        use_parent = scope['kind'] == 'gripper_only' or hand not in scope['hands']
        target = parent_goals if use_parent else target_source
        if target is not None:
            with np.load(target, allow_pickle=False) as goals:
                if not np.array_equal(goals['time_s'], times):
                    raise PreflightReject('Tracking goal clock differs from compiled commands')
                values[hand + '_target_position'] = goals[hand + '_target_position'].copy()
            origins[hand] = str(target)
    scratch = mujoco.MjData(model)
    positions = np.asarray([safety.point_positions(model, scratch, row)[:, 0, :]
                            for row in qpos])
    for h, hand in enumerate(HANDS):
        values[hand + '_planned_site_position'] = positions[:, h, :]
    destination = folder / 'goal_tracking.npz'
    np.savez_compressed(destination, **values)
    return {'path': str(destination), 'sha256': sha(destination),
            'target_origins': origins, 'planned_fk_recomputed_after_freeze': True}


def _xml_assets(path, seen=None):
    seen = set() if seen is None else seen
    path = Path(path).resolve()
    if path in seen:
        return seen
    seen.add(path)
    root = ET.parse(path).getroot()
    compiler = root.find('compiler')
    for node in root.iter():
        file = node.get('file')
        if not file:
            continue
        asset = Path(file)
        if not asset.is_absolute():
            directory = ''
            if compiler is not None and node.tag in ('mesh', 'texture'):
                directory = compiler.get('meshdir' if node.tag == 'mesh' else 'texturedir', '')
            asset = path.parent / directory / asset
        if not asset.is_file():
            raise PreflightReject('Missing frozen XML dependency: ' + str(asset))
        if node.tag == 'include':
            _xml_assets(asset, seen)
        else:
            seen.add(asset.resolve())
    return seen


def source_dependency_hashes(case_id, *, source_root=None, _hash_cache=None):
    """Fingerprint canonical case inputs without importing IK or physics.

Includes the original bound reference, selected compiled control, static XML
dependencies, source commands/rollout, engineered assets and task-frame/IK
model inputs. ``source_root`` exists for isolated fixtures; production callers
omit it so the canonical v74 root is authoritative. Shared files have identical
keys and can be deduplicated by the campaign manifest.
    """
    if not re.fullmatch(r'(stack_cups|stack_blocks|drawer)_ep\d{6}', case_id):
        raise PreflightReject('Invalid or unsupported case_id')
    source = Path(source_root if source_root is not None else V74) / case_id
    config = read(source / 'config.json')
    selected = read(source / 'sim2/selection.json')
    files = {source / name for name in ('config.json', 'sim2/selection.json',
                                       'sim2/task_contract.json')}
    files.update(source / 'inputs' / name for name in (*INPUTS, 'agent_trajectory_full.npz'))
    files.update(_xml_assets(source / 'inputs/sim2_robot_only_scene.xml'))
    selected_trajectory = Path(selected['trial']).parent / 'inputs/agent_trajectory_full.npz'
    files.add(selected_trajectory)
    selected_goals = selected_trajectory.resolve().parent / 'goal_tracking.npz'
    if selected_goals.is_file():
        files.add(selected_goals)
    if config.get('model_root'):
        model_source = Path(config['model_root']) / 'source'
        files.add(model_source / 'scene/task_frame_contract.json')
        files.update(_xml_assets(model_source / 'scene/s1_tabletop.xml'))
    if config.get('source_reference_artifact'):
        reference = Path(config['source_reference_artifact'])
        files.update(reference / name for name in ('scene.xml', 'command_targets.npz'))
        files.update(_xml_assets(reference / 'scene.xml'))
        files.update(reference / name for name in ('report.json', 'rollout.npz', 'task_contract.json')
                     if (reference / name).is_file())
        if (reference.parent / 'scene.json').is_file():
            files.add(reference.parent / 'scene.json')
    for key in ('engineered_task_asset', 'task_asset', 'task_scene_asset'):
        if config.get(key):
            asset = Path(config[key])
            if not asset.is_dir():
                raise PreflightReject('Missing frozen asset directory: ' + str(asset))
            files.update(path for path in asset.rglob('*') if path.is_file())
    for key in ('source_success_receipt', 'frozen_scene_path', '_source_eef_receipt'):
        if config.get(key):
            files.add(Path(config[key]))
    result = {}
    cache = {} if _hash_cache is None else _hash_cache
    for path in sorted(files):
        resolved = path.resolve()
        metadata = resolved.stat()
        stamp = (metadata.st_mtime_ns, metadata.st_size)
        previous = cache.get(str(resolved))
        if previous is None or previous[0] != stamp:
            value = sha(resolved)
            after = resolved.stat()
            if (after.st_mtime_ns, after.st_size) != stamp:
                raise PreflightReject('Source artifact changed while being frozen: ' + str(resolved))
            cache[str(resolved)] = (stamp, value)
        result[str(resolved)] = cache[str(resolved)][1]
    return result


def source_artifact_manifest(cases):
    """Merge canonical dependencies for campaign initialization, without FK."""
    result, cache = {}, {}
    for item in cases:
        case_id = item if isinstance(item, str) else item['case_id']
        if isinstance(item, dict) and item.get('source_dir'):
            if Path(item['source_dir']).resolve() != (V74 / case_id).resolve():
                raise PreflightReject('Campaign source_dir differs from canonical v74 case')
        result.update(source_dependency_hashes(case_id, _hash_cache=cache))
    return result


def _artifact_manifest(job, spec, folder, parent_trajectory):
    canonical = source_dependency_hashes(spec['case_id'])
    files = {Path(name) for name in canonical}
    files.update(RUNTIME.glob('*.py'))
    files.update({Path(__file__), ROOT / 'run_semantic_gpt150_task_completion_core_seq34.py',
                  ROOT / 'pipeline/crossfamily_stage_v58_seq104/safety.py'})
    files.update(_xml_assets(folder / 'physics/scene.xml'))
    files.add(parent_trajectory)
    files.add(folder / 'inputs/agent_trajectory_full.npz')
    files.add(folder / 'world_fk_reference.npz')
    files.add(folder / 'goal_tracking.npz')
    files.add(folder / 'candidate.json')
    for obj in (job, job.get('parent') or {}):
        for name in ('proposal_path', 'report_path', 'receipt_path', 'trajectory_path', 'baseline_trajectory_path'):
            if obj.get(name):
                files.add(Path(obj[name]))
    for module in tuple(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value and value.endswith('.py'):
            path = Path(value).resolve()
            if path.is_relative_to(ROOT) or path.is_relative_to(Path(spec['config']['model_root']) / 'source'):
                files.add(path)
    return {str(path.resolve()): sha(path) for path in sorted(files)}


def _job_identity(job):
    ignored = {'mode', 'phase', 'result_path', 'timeout_s', 'recover_only'}
    return digest({k: v for k, v in job.items() if k not in ignored})


def _load_preflight(job, folder):
    receipt = read(folder / 'receipt.json')
    if receipt['job_sha256'] != _job_identity(job):
        raise PreflightReject('Job differs from the approved preflight')
    verify_artifacts(receipt['artifact_sha256'])
    verify_artifacts(job.get('expected_artifact_sha256', {}))
    return receipt


def preflight(job):
    run, output, trial, folder, spec, selected, candidate, parent, parent_trajectory, scope = _load_job(job)
    stage_mode = job.get('stage_mode')
    if stage_mode is not None:
        if (stage_mode != 'bounded_stage_v1'
                or candidate['decision'].get('stage_plan') != stage_mode
                or scope['kind'] != 'spatial' or len(scope['hands']) != 1
                or job.get('grasp_clearance_probe') is None):
            raise PreflightReject('Bounded stage plan requires one declared spatial hand and clearance probe')
    verify_artifacts(job.get('expected_artifact_sha256', {}))
    if (folder / 'receipt.json').exists():
        receipt = _load_preflight(job, folder)
        if receipt['environment'] != _environment(spec['config']):
            raise PreflightReject('Execution environment differs from approved preflight')
        return _validated_result(job, folder, receipt)
    environment = _environment(spec['config'])
    core, safety, pilot_ik, physics = _runtime()
    from semantic_control import validate_windows
    # Validation is observational; do not normalize a pinned historical program.
    core.validate_program(candidate['program'], {})
    core.validate_combined_ik_bounds(candidate['program'], spec['config'])
    validate_windows(candidate['windows'])
    core.validate_scene(candidate['scene'])
    config = _inputs(folder, spec['source'], spec['config'], candidate, run)
    trajectory = folder / 'inputs/agent_trajectory_full.npz'
    if scope['kind'] == 'replay' or scope['kind'] == 'gripper_only':
        shutil.copy2(parent_trajectory, trajectory)
        freeze = {'scope': scope, 'parent_trajectory_path': str(parent_trajectory),
                  'parent_trajectory_sha256': sha(parent_trajectory),
                  'all_spatial_targets_exact': True}
    else:
        planned = pilot_ik.plan(config, folder / 'ik', candidate['program'])
        shutil.copy2(planned['trajectory'], trajectory)
        freeze = freeze_trajectory(trajectory, parent_trajectory, scope)
    model = physics.build(config, 'sim2', folder / 'physics', candidate['scene'])
    with core.physical.installed_windows(candidate['windows']):
        current = physics.commands(config, 'sim2', model)
    parent_config = _inputs(folder / 'parent', spec['source'], spec['config'], parent, run)
    shutil.copy2(parent_trajectory, folder / 'parent/inputs/agent_trajectory_full.npz')
    with core.physical.installed_windows(parent['windows']):
        previous = physics.commands(parent_config, 'sim2', model)
    source_candidate = candidate_fields(selected)
    source_config = copy.deepcopy(spec['config'])
    source_config.update(import_case=str(Path(spec['source']) / 'inputs'),
                         model_profile=selected.get('model_profile', {}),
                         _release_opening=selected['program'].get('release_opening', {}))
    with core.physical.installed_windows({}):
        original = physics.commands(source_config, 'sim2', model)
    exact = check_control_scope(model, current, previous, scope, safety)
    grasp_clearance = None
    if job.get('grasp_clearance_probe') is not None:
        from .clearance import compare_grasp_window
        try:
            grasp_clearance = compare_grasp_window(
                model, current, previous, job['grasp_clearance_probe'])
        except ValueError as exc:
            raise PreflightReject('Grasp clearance hypothesis rejected: ' + str(exc)) from exc
    grasp_alignment = None
    if job.get('grasp_alignment_probe') is not None:
        from .grasp_alignment import compare_alignment
        probe = job['grasp_alignment_probe']
        if not isinstance(probe, dict) or set(probe) != {'hand', 'object', 'close_s'}:
            raise PreflightReject('Grasp alignment probe fields are invalid')
        try:
            grasp_alignment = compare_alignment(model, current, previous,
                hand=probe['hand'], object_id=probe['object'],
                close_s=float(probe['close_s']), require_bilateral=False)
        except (ValueError, KeyError) as exc:
            raise PreflightReject('Grasp alignment hypothesis rejected: ' + str(exc)) from exc
    grasp_table_diagnostic = None
    if job.get('grasp_table_diagnostic_probe') is not None:
        from .clearance import sample_window
        probe = job['grasp_table_diagnostic_probe']
        if not isinstance(probe, dict) or set(probe) != {'hand', 'start_s', 'end_s'}:
            raise PreflightReject('Grasp table diagnostic probe fields are invalid')
        try:
            before = sample_window(model, previous, probe['hand'],
                                   float(probe['start_s']), float(probe['end_s']))
            after = sample_window(model, current, probe['hand'],
                                  float(probe['start_s']), float(probe['end_s']))
        except (ValueError, KeyError) as exc:
            raise PreflightReject('Grasp table diagnostic rejected: ' + str(exc)) from exc
        grasp_table_diagnostic = {'schema': 'rssr.family.grasp_table_diagnostic.v1',
                                  'parent': before, 'candidate': after,
                                  'static_only': True, 'hard_constraint_requires_physics': True}
    tracking = _tracking_inputs(model, current, folder, parent_trajectory, scope, safety)
    if stage_mode:
        from . import stage_safety
        try:
            receipt = stage_safety.precompute(model, {'candidate': current, 'parent': previous,
                'source': original}, folder / 'world_fk_reference.npz')
        except stage_safety.StageEnvelopeReject as exc:
            raise PreflightReject('Bounded stage safety rejected: ' + str(exc)) from exc
        single_limit = total_limit = stage_safety.WORLD_AXIS_BOUND_M
    else:
        try:
            receipt = safety.precompute(model, {'candidate': current, 'parent': previous,
                                               'source': original}, folder / 'world_fk_reference.npz')
        except safety.EnvelopeReject as exc:
            raise PreflightReject('Whole-stream target envelope rejected: ' + str(exc)) from exc
        single_limit, total_limit = .040, .160
    import numpy as np
    # All hands must satisfy cumulative bounds, even a replay with no new edits.
    for key, limit in (('max_single_action_world_axis_m', single_limit),
                       ('max_total_source_world_axis_m', total_limit)):
        if np.max(receipt[key]) > limit + safety.TOLERANCE_M:
            raise PreflightReject(key + ' exceeds the frozen bilateral bound')
    write(folder / 'candidate.json', candidate)
    receipt.update(schema='rssr.family_autoresearch.native_receipt.v1',
                   job_sha256=_job_identity(job), case_id=job['case_id'],
                   environment=environment,
                   candidate_sha256=digest(physical_identity(candidate)),
                   parent_candidate_sha256=digest(physical_identity(parent)),
                   source_candidate_sha256=digest(physical_identity(source_candidate)),
                   candidate_trajectory_path=str(trajectory),
                   candidate_ik_trajectory_sha256=sha(trajectory),
                   frozen_scene_sha256=digest(selected['scene']),
                   frozen_model_profile_sha256=digest(selected.get('model_profile', {})),
                   guarded_hands=list(HANDS), exact_control_scope=exact,
                   grasp_clearance=grasp_clearance,
                   grasp_alignment=grasp_alignment,
                   grasp_table_diagnostic=grasp_table_diagnostic,
                   safety_mode=stage_mode or 'legacy_dsl',
                   goal_tracking=tracking,
                   trajectory_freeze=freeze, physical_started=False,
                   safety_code_sha256=sha(Path(safety.__file__)),
                   stage_safety_code_sha256=(sha(Path(stage_safety.__file__))
                                             if stage_mode else None))
    receipt['artifact_sha256'] = _artifact_manifest(job, spec, folder, parent_trajectory)
    write(folder / 'receipt.json', receipt)
    return _validated_result(job, folder, receipt)


def _validated_result(job, folder, receipt):
    return {'status': 'validated', 'phase_status': 'preflight_passed',
            'case_id': job['case_id'], 'physical_started': False,
            'complete': False, 'complete_native_rollout': False, 'native_pass': False,
            'receipt_path': str(folder / 'receipt.json'),
            'candidate_path': str(folder / 'candidate.json'),
            'trajectory_path': receipt['candidate_trajectory_path'],
            'candidate_sha256': receipt['candidate_sha256'],
            'modified_hands': receipt['modified_hands'],
            'exact_control_scope': receipt['exact_control_scope']}


@contextmanager
def _approved_runtime(pilot_ik, physics, receipt, folder, marker):
    import mujoco
    old_plan, old_build, old_step = pilot_ik.plan, physics.build, mujoco.mj_step

    def pinned_plan(config, destination, program):
        if sha(receipt['candidate_trajectory_path']) != receipt['candidate_ik_trajectory_sha256']:
            raise PreflightReject('Approved trajectory changed before execution')
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        trajectory = destination / 'trajectory.npz'
        shutil.copy2(receipt['candidate_trajectory_path'], trajectory)
        shutil.copy2(receipt['goal_tracking']['path'], destination / 'goal_tracking.npz')
        planned = {'program': program, 'trajectory': str(trajectory),
                   'sha256': sha(trajectory), 'exact_preflight_replay': True,
                   'preflight_receipt': str(folder / 'receipt.json')}
        write(destination / 'plan.json', planned)
        return planned

    def checked_build(config, kind, destination, scene=None):
        model = old_build(config, kind, destination, scene)
        if kind != 'sim2' or sha(Path(destination) / 'scene.xml') != sha(folder / 'physics/scene.xml'):
            raise PreflightReject('Runtime physical scene differs from frozen preflight')
        return model

    def marked_step(*args, **kwargs):
        if not marker.exists():
            write(marker, {'physical_started': True, 'unix_time': time.time(),
                           'receipt_path': str(folder / 'receipt.json')})
        return old_step(*args, **kwargs)

    pilot_ik.plan, physics.build, mujoco.mj_step = pinned_plan, checked_build, marked_step
    try:
        yield
    finally:
        pilot_ik.plan, physics.build, mujoco.mj_step = old_plan, old_build, old_step


def evaluate(job):
    run, output, trial, folder, spec, selected, candidate, parent, trajectory, scope = _load_job(job)
    if not (folder / 'receipt.json').exists():
        raise PreflightReject('Evaluation requires a successful saved preflight')
    receipt = _load_preflight(job, folder)
    if receipt['environment'] != _environment(spec['config']):
        raise PreflightReject('Execution environment differs from approved preflight')
    audit_path = trial / 'safety/runtime_guard.json'
    marker = folder / 'physical_started.json'
    report_path = trial / 'physics/report.json'
    evaluation_path = trial / 'evaluation.json'
    completed_path = folder / 'completed.json'
    if completed_path.exists():
        completed = read(completed_path)
        if (completed.get('status') != 'complete' or not completed.get('complete')
                or not completed.get('complete_native_rollout')
                or completed.get('candidate_sha256') != receipt['candidate_sha256']):
            raise PreflightReject('Saved completion does not match approved candidate')
        verify_artifacts(completed['output_sha256'])
        return completed
    if job.get('recover_only'):
        raise PreflightReject('Recovery requires an existing verified native/completed.json; no physics attempted')
    if marker.exists() or report_path.exists():
        raise PreflightReject('Interrupted physical trial cannot be silently rerun; allocate a new index')
    core, safety, pilot_ik, physics = _runtime()
    targets = core.object_targets(spec['config'], selected['scene'])
    specifications = core.task_specs(spec)
    if receipt.get('safety_mode') == 'bounded_stage_v1':
        from . import stage_safety
        guard = stage_safety.RuntimeGuard(receipt, trial / 'safety')
    else:
        guard = safety.RuntimeGuard(receipt, trial / 'safety')
    # Check *both* hands at every physical step. Exact scope was checked against
    # parent qpos; RuntimeGuard also matches the entire audited candidate qpos.
    guard.modified = HANDS
    with _approved_runtime(pilot_ik, physics, receipt, folder, marker):
        with safety.installed(guard):
            row = core.evaluate(run, spec, output, job.get('index', 0), candidate,
                                targets, specifications)
    audit = read(audit_path)
    if (not audit.get('complete_native_rollout') or audit.get('first_trigger')
            or audit.get('physical_control_frames') != receipt['physical_frames']
            or not marker.exists()):
        raise RuntimeError('Incomplete or overbound native physical rollout')
    report = read(report_path)  # Raw frozen v80 output; no post-hoc judge migration.
    if report.get('judgment_version') != 'task_completion_v1':
        raise RuntimeError('Report is not native frozen v80 task_completion_v1')
    forbidden = ('objects_forced', 'object_weld_used', 'robot_qpos_forced_during_execution')
    if any(report.get(key) for key in forbidden):
        raise RuntimeError('Native physical execution invariant violated')
    verify_artifacts(receipt['artifact_sha256'])
    row.update(bilateral_safety_audit_path=str(audit_path),
               bilateral_safety_reference_sha256=receipt['reference_sha256'])
    write(evaluation_path, row)
    result = {'status': 'complete', 'phase_status': 'complete', 'case_id': job['case_id'],
              'physical_started': True, 'complete': True, 'complete_native_rollout': True,
              'native_pass': bool(core.physical.passed(report)),
              'report_path': str(report_path), 'proposal_path': row['proposal_path'],
              'candidate_path': str(folder / 'candidate.json'),
              'trajectory_path': str(trial / 'inputs/agent_trajectory_full.npz'),
              'guard_path': str(audit_path), 'receipt_path': str(folder / 'receipt.json'),
              'evaluation_path': str(evaluation_path), 'evaluation': row,
              'candidate_sha256': receipt['candidate_sha256'],
              'output_sha256': {str(path): sha(path) for path in
                                [audit_path, evaluation_path, folder / 'receipt.json',
                                 Path(row['proposal_path']), trial / 'inputs/agent_trajectory_full.npz',
                                 *sorted((trial / 'physics').glob('*'))] if path.is_file()}}
    write(completed_path, result)
    return result


def run_job(job, phase=None):
    phase = phase or job.get('mode', job.get('phase', 'preflight'))
    try:
        if phase in ('preflight', 'validate'):
            return preflight(job)
        if phase in ('evaluate', 'replay'):
            return evaluate(job)
        raise PreflightReject('Unsupported native job phase: ' + str(phase))
    except Exception as exc:
        try:
            folder = _paths(job)[3]
            started = (folder / 'physical_started.json').exists()
        except Exception:
            started = False
        is_rejection = isinstance(exc, (ValueError, PreflightReject)) and not started
        return {'status': 'rejected' if is_rejection else 'tool_error',
                'phase_status': 'preflight_rejected' if is_rejection else 'worker_error',
                'case_id': job.get('case_id'), 'physical_started': started,
                'complete': False, 'complete_native_rollout': False, 'native_pass': False,
                'error_type': type(exc).__name__, 'error': str(exc),
                'traceback': traceback.format_exc()[-10000:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--phase', choices=('preflight', 'evaluate'))
    args = parser.parse_args()
    job = read(args.job)
    result = run_job(job, args.phase)
    destination = args.result or Path(job.get('result_path', str(args.job.with_name('result.json'))))
    write(destination, result)
    print(json.dumps({key: result.get(key) for key in
                      ('status', 'case_id', 'physical_started', 'complete', 'native_pass', 'error')}))
    return 0 if result['status'] in ('validated', 'complete') else 2


if __name__ == '__main__':
    raise SystemExit(main())
