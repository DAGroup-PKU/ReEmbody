#!/usr/bin/env python3
"""Verify and reconstruct a registered initial state without advancing physics."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

REQUIRED_MUJOCO = '3.3.7'
REQUIRED = ('qpos', 'qvel', 'ctrl', 'time_s', 'robot_scalar_names', 'scalar_state32',
            'cup_names', 'cup_position', 'cup_quaternion',
            'camera_names', 'camera_position', 'camera_rotation')
OPTIONAL_STATE = ('act', 'mocap_pos', 'mocap_quat', 'userdata')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def declared_path(root, value):
    p = Path(value)
    return (p if p.is_absolute() else root / p).resolve(strict=True)


def pin(path, expected, pins):
    require(isinstance(expected, str) and len(expected) == 64, 'Missing/invalid SHA-256: ' + str(path))
    actual = sha(path)
    require(actual == expected, 'SHA-256 mismatch: ' + str(path))
    pins[str(path)] = actual


def names(array, label):
    require(array.ndim == 1 and array.dtype.kind in 'US', label + ' must be a1D string array')
    values = [str(x) for x in array.tolist()]
    require(len(values) == len(set(values)) and all(values), label + ' must contain unique nonempty names')
    return values


def finite_array(value, shape, label):
    require(value.shape == shape, f'{label} shape {value.shape} != {shape}')
    require(value.dtype.kind == 'f' and np.isfinite(value).all(), label + ' must contain finite floating-point values')


def max_error(actual, expected):
    return float(np.max(np.abs(actual - expected), initial=0.))


def atomic_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('x') as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    require(not path.exists(), 'Refusing to overwrite: ' + str(path))
    temp.replace(path)


def reconstruct(args, report):
    require(mujoco.__version__ == REQUIRED_MUJOCO,
            f'需要MuJoCo / Required MuJoCo {REQUIRED_MUJOCO}; found {mujoco.__version__}')
    registry = args.registry_dir.resolve(strict=True)
    manifest_path = registry / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['mujoco_version'] == REQUIRED_MUJOCO, 'Registry MuJoCo version differs')
    cases_path = declared_path(registry, manifest['cases_file'])
    require(cases_path.is_relative_to(registry), 'cases_file must be within registry-dir')
    pins = {str(manifest_path): sha(manifest_path), str(Path(__file__).resolve()): sha(__file__)}
    pin(cases_path, manifest['cases_sha256'], pins)
    rows = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    require(len(rows) == manifest['total_cases'] == 111, 'This registry must contain exactly111 cases')
    require(len({r['source_id'] for r in rows}) == len(rows), 'Duplicate source_id in cases')
    require(sorted(r['episode_index'] for r in rows) == list(range(111)), 'episode_index must cover0..110 once')
    selected = [r for r in rows if r['episode_index'] == args.episode_index] if args.episode_index is not None else [r for r in rows if r['source_id'] == args.source_id]
    require(len(selected) == 1, 'Requested episode/source_id not uniquely present')
    row = selected[0]
    scene = declared_path(registry, row['scene_xml'])
    snapshot = declared_path(registry, row['snapshot'])
    pin(scene, row['scene_xml_sha256'], pins)
    pin(snapshot, row['snapshot_sha256'], pins)

    assets_verified = 0
    assets_declared = 'assets_file' in manifest or 'assets_sha256' in manifest
    if assets_declared:
        require('assets_file' in manifest and 'assets_sha256' in manifest, 'assets_file/assets_sha256 must be provided together')
        asset_file = declared_path(registry, manifest['assets_file'])
        pin(asset_file, manifest['assets_sha256'], pins)
        assets = json.loads(asset_file.read_text())
        if 'files' in assets:
            records = assets['files']
            if isinstance(records, dict):
                records = [{'path': p, 'sha256': h} for p, h in records.items()]
        else:
            records = [{'path': p, 'sha256': h} for p, h in assets.items()]
        require(isinstance(records, list), 'assets.json must contain files list or path:sha256 mapping')
        for entry in records:
            asset = declared_path(registry, entry['path'])
            pin(asset, entry['sha256'], pins)
            assets_verified += 1

    with np.load(snapshot, allow_pickle=False) as z:
        stored = {k: z[k].copy() for k in z.files}
    require(set(REQUIRED) <= set(stored), 'Snapshot is missing required fields: ' + str(set(REQUIRED) - set(stored)))
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    finite_array(stored['qpos'], (model.nq,), 'qpos')
    finite_array(stored['qvel'], (model.nv,), 'qvel')
    finite_array(stored['ctrl'], (model.nu,), 'ctrl')
    require(stored['time_s'].size == 1 and np.isfinite(stored['time_s']).all(), 'time_s must be a finite scalar')
    saved_time = float(stored['time_s'].reshape(()))
    require(saved_time == 0.0, 'Initial-state registry requires time_s==0; smoke/mid-episode snapshots are not accepted')
    robot_names = names(stored['robot_scalar_names'], 'robot_scalar_names')
    require(len(robot_names) == 32, 'Exactly32 named scalar robot states required')
    expected_names = [model.joint(j).name for j in range(model.njnt)
                      if (model.joint(j).name or '').startswith('astribot_')
                      and model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)]
    require(set(robot_names) == set(expected_names), 'Named scalar32 does not match compiled robot joints')
    addresses = np.asarray([int(model.joint(n).qposadr[0]) for n in robot_names])
    finite_array(stored['scalar_state32'], (32,), 'scalar_state32')
    cup_names = names(stored['cup_names'], 'cup_names')
    camera_names = names(stored['camera_names'], 'camera_names')
    require(cup_names and camera_names, 'Cup/camera FK checks require nonempty name lists')
    cup_ids = [model.body(n).id for n in cup_names]
    camera_ids = [model.camera(n).id for n in camera_names]
    finite_array(stored['cup_position'], (len(cup_names), 3), 'cup_position')
    finite_array(stored['cup_quaternion'], (len(cup_names), 4), 'cup_quaternion(wxyz)')
    finite_array(stored['camera_position'], (len(camera_names), 3), 'camera_position')
    require(stored['camera_rotation'].shape in [(len(camera_names), 9), (len(camera_names), 3, 3)], 'camera_rotation must be Nx9 orNx3x3')
    expected_camera_rotation = stored['camera_rotation'].reshape(len(camera_names), 3, 3)
    require(np.isfinite(expected_camera_rotation).all(), 'camera_rotation is nonfinite')
    require(np.allclose(np.linalg.norm(stored['cup_quaternion'], axis=1), 1., rtol=0, atol=1e-6), 'Cup quaternion is not unit length')

    # Restore recorded state once. These calls never advance the clock.
    mujoco.mj_resetData(model, data)
    data.qpos[:] = stored['qpos']
    data.qvel[:] = stored['qvel']
    data.ctrl[:] = stored['ctrl']
    data.time = saved_time
    omitted, restored_optional = [], []
    for key in OPTIONAL_STATE:
        dest = getattr(data, key)
        if key in stored:
            finite_array(stored[key], dest.shape, key)
            dest[:] = stored[key]
            restored_optional.append(key)
        else:
            omitted.append({'field': key, 'size': int(dest.size), 'handling': 'mj_resetData defaults; not recorded in source'})
    mujoco.mj_forward(model, data)

    actual_scalar = data.qpos[addresses]
    if stored['scalar_state32'].dtype == np.float32:
        scalar_matches = np.array_equal(actual_scalar.astype(np.float32), stored['scalar_state32'])
    else:
        scalar_matches = np.allclose(actual_scalar, stored['scalar_state32'], rtol=0, atol=1e-10)
    actual_cup_quat = data.xquat[cup_ids].copy()
    quat_error = np.minimum(np.max(np.abs(actual_cup_quat - stored['cup_quaternion']), axis=1),
                            np.max(np.abs(actual_cup_quat + stored['cup_quaternion']), axis=1))
    errors = {'scalar_state32_float64_max_abs': max_error(actual_scalar, stored['scalar_state32']),
              'cup_position_m_max_abs': max_error(data.xpos[cup_ids], stored['cup_position']),
              'cup_quaternion_sign_invariant_max_abs': float(np.max(quat_error, initial=0.)),
              'camera_position_m_max_abs': max_error(data.cam_xpos[camera_ids], stored['camera_position']),
              'camera_rotation_max_abs': max_error(data.cam_xmat[camera_ids].reshape(-1, 3, 3), expected_camera_rotation)}
    checks = {'qpos_exactly_restored_after_forward': np.array_equal(data.qpos, stored['qpos']),
              'qvel_exactly_restored_after_forward': np.array_equal(data.qvel, stored['qvel']),
              'ctrl_exactly_restored_after_forward': np.array_equal(data.ctrl, stored['ctrl']),
              'time_exactly_restored_and_not_advanced': data.time == saved_time,
              'scalar_state32_matches': bool(scalar_matches),
              'cup_position_matches': errors['cup_position_m_max_abs'] <= args.position_tolerance_m,
              'cup_quaternion_matches': errors['cup_quaternion_sign_invariant_max_abs'] <= args.rotation_tolerance,
              'camera_position_matches': errors['camera_position_m_max_abs'] <= args.position_tolerance_m,
              'camera_rotation_matches': errors['camera_rotation_max_abs'] <= args.rotation_tolerance}
    checks['recorded_optional_states_exactly_restored'] = all(np.array_equal(getattr(data, k), stored[k]) for k in restored_optional)
    checks['input_pins_unchanged'] = all(sha(p) == digest for p, digest in pins.items())
    packet = {'schema': np.asarray('rssr.v3_human_align.restored_initial_state.v1'),
              'source_id': np.asarray(row['source_id']), 'episode_index': np.asarray(row['episode_index']),
              'qpos': data.qpos.copy(), 'qvel': data.qvel.copy(), 'ctrl': data.ctrl.copy(),
              'time_s': np.asarray(data.time), 'robot_scalar_names': np.asarray(robot_names),
              'scalar_state32': actual_scalar.astype(np.float32), 'cup_names': np.asarray(cup_names),
              'cup_position': data.xpos[cup_ids].copy(), 'cup_quaternion': actual_cup_quat,
              'camera_names': np.asarray(camera_names), 'camera_position': data.cam_xpos[camera_ids].copy(),
              'camera_rotation': data.cam_xmat[camera_ids].copy(),
              'scene_xml_sha256': np.asarray(row['scene_xml_sha256']),
              'source_snapshot_sha256': np.asarray(row['snapshot_sha256']),
              'zero_time_reconstruction': np.asarray(True)}
    packet.update({key: getattr(data, key).copy() for key in OPTIONAL_STATE})
    provenance = {k: v for k, v in stored.items() if k.startswith('source_')}
    packet.update(provenance)  # Documentation only; never applied as a command.
    for key, value in provenance.items():
        if value.dtype.kind in 'fc':
            require(np.isfinite(value).all(), 'Nonfinite source provenance array: ' + key)
    report.update(source_id=row['source_id'], episode_index=row['episode_index'],
                  scene_xml=str(scene), snapshot=str(snapshot), time_s=saved_time,
                  model_shape={k: int(getattr(model, k)) for k in ['nq', 'nv', 'nu', 'na', 'nmocap', 'nuserdata', 'njnt', 'nbody', 'ncam']},
                  checks=checks, maximum_errors=errors,
                  tolerances={'position_m': args.position_tolerance_m, 'rotation_components': args.rotation_tolerance},
                  input_sha256=pins, external_assets_manifest_present=assets_declared,
                  declared_external_assets_hash_verified=assets_verified,
                  external_asset_closure_completeness_independently_proven=False,
                  optional_recorded_state_fields_restored=restored_optional,
                  optional_state_fields_omitted_from_source=omitted,
                  source_demonstration_arrays_not_executed={k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in provenance.items()},
                  ctrl_semantics='Recorded native actuator ctrl0 after the original first servo.apply; restored exactly. This is not action22 or a newly issued model action.',
                  reconstruction_basis='deterministic_reconstruction_from_actual_qpos',
                  registry_self_contained=manifest.get('self_contained'),
                  registry_global_robot_kinematics_fingerprint=manifest.get('global_robot_kinematics_fingerprint', manifest.get('robot_kinematics_sha256')),
                  future_control_defaults=manifest.get('future_control_defaults'),
                  checkpoint_action_mapping_assumed=False,
                  full_integrator_snapshot=False,
                  unrecorded_integrator_state_examples=['qacc_warmstart', 'solver history', 'applied forces', 'plugin state', 'original optional act/mocap/userdata if absent'],
                  limitations=['Only zero-time state reconstruction and FK consistency are checked. No mj_step or model inference is performed.',
                               'Original source did not record every integration/solver field; this packet is not a bitwise continuous-dynamics replay snapshot.',
                               'Source initial reference/first issued demonstration targets are copied only as provenance, never automatically executed.',
                               'Assets declared in assets.json are hash-checked; completeness of that declaration is the registry producer responsibility.'],
                  passed=bool(all(checks.values())))
    require(report['passed'], 'Initial-state/FK verification failed: ' + str([k for k, v in checks.items() if not v]))
    return packet


def main():
    parser = argparse.ArgumentParser(
        description='零时间恢复并核验v3初始状态 / Verify and reconstruct v3 initial state without stepping physics.',
        epilog='仅MuJoCo3.3.7，reset+forward，不推进物理、不执行来源首指令或action model。'
               ' Only MuJoCo3.3.7; reset+forward, no mj_step, model inference, or automatic source-target execution. '
               '这不是完整积分器快照 / This is not a complete integrator snapshot. '
               'horizon50/hz30仅未来默认配置；不假定checkpoint32→22映射 / No checkpoint32-to22 mapping is assumed.')
    parser.add_argument('--registry-dir', type=Path, required=True, help='含manifest.json与cases.jsonl的目录 / Registry directory')
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument('--episode-index', type=int, help='已登记episode索引 / Registered episode index')
    select.add_argument('--source-id', help='已登记source_id / Registered source ID')
    parser.add_argument('--output', type=Path, required=True, help='新NPZ路径；报告为<output>.report.json / Fresh NPZ output; JSON report is appended .report.json')
    parser.add_argument('--check-only', action='store_true', help='只重建核验并写JSON报告，不写NPZ / Verify and write JSON only; do not write packet')
    parser.add_argument('--position-tolerance-m', type=float, default=1e-8)
    parser.add_argument('--rotation-tolerance', type=float, default=1e-7)
    args = parser.parse_args()
    require(args.position_tolerance_m > 0 and args.rotation_tolerance > 0, 'Tolerances must be positive')
    output = args.output.resolve()
    report_path = output.with_name(output.name + '.report.json')
    require(not output.exists() and not report_path.exists(), 'Refusing to overwrite output or report')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'rssr.v3_human_align.initial_state_restore_check.v1',
              'at_utc': datetime.now(timezone.utc).isoformat(), 'mujoco_version': mujoco.__version__,
              'registry_dir': str(args.registry_dir.resolve()), 'check_only': args.check_only,
              'mj_step_calls': 0, 'policy_inference_calls': 0, 'gpu_operations': 0,
              'passed': False, 'packet_written': False}
    try:
        packet = reconstruct(args, report)
        if not args.check_only:
            temp = output.with_name(output.name + '.tmp')
            with temp.open('xb') as f:
                np.savez_compressed(f, **packet)
            require(not output.exists(), 'Refusing to overwrite packet')
            temp.replace(output)
            report.update(packet_written=True, packet=str(output), packet_sha256=sha(output))
        atomic_json(report_path, report)
        print(json.dumps({'passed': True, 'source_id': report['source_id'], 'episode_index': report['episode_index'],
                          'packet_written': report['packet_written'], 'report': str(report_path)}, ensure_ascii=False))
        return 0
    except Exception as exc:
        report.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        atomic_json(report_path, report)
        print(json.dumps({'passed': False, 'error': str(exc), 'report': str(report_path)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
