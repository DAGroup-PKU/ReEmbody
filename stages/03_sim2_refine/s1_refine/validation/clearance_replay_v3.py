"""Fresh exact native replay with independent every-step convex-clearance proof.

Never reconstruct force/contact claims from saved qpos. Geometry alone is read
on a private FK MjData at the actual current solver qpos, so stale integration
geometry cannot silently shift the time of the clearance check.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
if __package__:
    from .clearance_gate import REQUIRED_CATEGORIES, VERSION, DEPS
    from .clearance_cross_side import CrossSideNativeGate as ClearanceGate
else:
    from clearance_gate import REQUIRED_CATEGORIES, VERSION, DEPS
    from clearance_cross_side import CrossSideNativeGate as ClearanceGate


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--strategy', type=Path, required=True)
    p.add_argument('--case-id', required=True)
    p.add_argument('--implementation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--min-clearance', type=float, default=.020)
    p.add_argument('--initial-pose', type=Path)
    p.add_argument('--table-geometry-pose', type=Path)
    p.add_argument('--four-axis-template', type=Path)
    a = p.parse_args()
    if a.min_clearance < .020:
        raise ValueError('User requires >=20mm: threshold cannot be reduced')
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    sys.path.insert(0, str(a.implementation))
    from rssr_augment.common import read
    from rssr_augment.retarget.case import load_case
    from rssr_augment.retarget import run_case

    pin_paths = [a.strategy, a.native/'receipt.json', a.native/'execute/scene.xml',
        a.native/'execute/rollout.npz', a.native/'execute/immutable_hard_trace.npz',
        a.native/'issued_robot_commands.npz', Path(__file__), Path(__file__).with_name('clearance_gate.py'),
        Path(__file__).with_name('clearance_fast.py'), Path(__file__).with_name('clearance_cross_side.py')]
    pin_paths += sorted(DEPS.glob('fcl/*.so')) + sorted(DEPS.glob('python_fcl.libs/*.so*'))
    pin_paths += [path for path in (a.initial_pose, a.table_geometry_pose,
                                   a.four_axis_template) if path is not None]
    pins_before = {str(f.resolve()):sha(f) for f in pin_paths}
    case = load_case(read(a.strategy), a.case_id, a.output)
    if a.initial_pose is not None:
        task_dir = Path(__file__).resolve().parents[1] / 'tasks/stack_blocks'
        sys.path.insert(0, str(task_dir))
        if a.four_axis_template is not None:
            if a.table_geometry_pose is None:
                raise ValueError('Four-axis replay needs pinned table geometry')
            from full4_adapter import apply_full4
            apply_full4(case, a.four_axis_template, a.initial_pose, a.table_geometry_pose)
        else:
            from gui_initial_pose import apply_gui_initial_pose
            apply_gui_initial_pose(case, a.initial_pose, a.table_geometry_pose)
    (a.output/'execute').symlink_to(a.native/'execute', target_is_directory=True)
    commands = np.load(a.native/'issued_robot_commands.npz')
    records = list(zip(commands['time_s'], commands['qpos'], commands['qvel'], commands['event']))
    native_receipt = read(a.native/'receipt.json')
    stop = float(native_receipt.get('horizon_s', 14.965 if
                 native_receipt.get('schema') == 'rssr.stack_blocks.short_refine_candidate.v1'
                 else native_receipt['horizon_s']))
    original = run_case.HardSnapshotter
    rows = []; sampled_minima = {}; interval_minima = {}; stages = {}
    started = time.monotonic()

    class ClearanceSnapshotter:
        def __init__(self, model):
            self.original = original(model)
            self.gate = ClearanceGate(model, a.min_clearance)
            self.model = model; self.last_time = None
            self.last_qpos = None; self.last_result = None
            self.manifest = self.gate.geometry_manifest()
        def __call__(self, data):
            original_result = self.original(data)
            t = float(data.time)
            if self.last_time is not None and t == self.last_time:
                return original_result
            if self.last_time is not None and t <= self.last_time:
                raise RuntimeError('Nonmonotone native clearance sample clock')
            qpos = data.qpos.copy()
            result = self.gate.at_qpos(qpos)
            ci = min(len(records)-1, max(0, int(np.searchsorted(commands['time_s'], t+1e-9)-1)))
            stage = str(commands['event'][ci])
            stage_row = stages.setdefault(stage, dict(samples=0, start_s=t, end_s=t, minima={}))
            stage_row['samples'] += 1; stage_row['end_s'] = t
            for category, value in result['minima'].items():
                detail = dict(value, time_s=t, sample_index=len(rows), stage=stage)
                if category not in sampled_minima or detail['distance_m'] < sampled_minima[category]['distance_m']:
                    sampled_minima[category] = detail
                if category not in stage_row['minima'] or detail['distance_m'] < stage_row['minima'][category]['distance_m']:
                    stage_row['minima'][category] = detail
            margins = None; continuous = None
            if self.last_qpos is not None:
                margins = self.gate.interval_motion_margin(self.last_qpos, qpos)
                continuous = {}
                for category in REQUIRED_CATEGORIES:
                    start_d = self.last_result['minima'][category]['distance_m']
                    end_d = result['minima'][category]['distance_m']
                    lower = min(start_d, end_d)-margins[category]
                    detail = dict(lower_bound_m=lower, motion_margin_m=margins[category],
                        start_time_s=self.last_time, end_time_s=t, stage=stage,
                        min_endpoint_clearance_m=min(start_d, end_d),
                        endpoint_minimizer=(self.last_result if start_d <= end_d else result)['minima'][category])
                    continuous[category] = lower
                    if category not in interval_minima or lower < interval_minima[category]['lower_bound_m']:
                        interval_minima[category] = detail
            rows.append((t, qpos, [result['minima'][c]['distance_m'] for c in self.gate.categories],
                [continuous[c] if continuous else np.nan for c in REQUIRED_CATEGORIES]))
            self.last_time = t; self.last_qpos = qpos; self.last_result = result
            if len(rows)%5000 == 0:
                print(json.dumps(dict(samples=len(rows), time_s=t, elapsed_s=time.monotonic()-started,
                    continuous_min_m={c:v['lower_bound_m'] for c,v in interval_minima.items()})), flush=True)
            return original_result

    run_case.HardSnapshotter = ClearanceSnapshotter
    try:
        replay = run_case._replay(case, a.output, records, stop_s=stop)
    finally:
        run_case.HardSnapshotter = original
    if not rows:
        raise RuntimeError('Replay emitted no clearance samples')
    # _replay guarantees native hard trace and 100Hz full state arrays exact.
    exact = bool(replay['exact'] and replay['every_step_hard_trace_exact'])
    native_hard = np.load(a.native/'execute/immutable_hard_trace.npz')
    t = np.array([r[0] for r in rows])
    clock_exact = bool(np.array_equal(t, native_hard['time_s']))
    pins_after = {str(f.resolve()):sha(f) for f in pin_paths}
    pinned = pins_before == pins_after
    sampled_pass = all(sampled_minima[c]['distance_m'] >= a.min_clearance for c in REQUIRED_CATEGORIES)
    continuous_pass = all(interval_minima[c]['lower_bound_m'] >= a.min_clearance for c in REQUIRED_CATEGORIES)
    gap_ok = bool(len(t) == len(records)+1 and np.allclose(np.diff(t), case['config']['sim_dt'] if 'sim_dt' in case['config'] else .0005, rtol=0, atol=1e-9))
    np.savez_compressed(a.output/'clearance_trace.npz', time_s=t,
        qpos=np.stack([r[1] for r in rows]),
        categories=np.asarray(sorted(sampled_minima)),
        distance_lower_bound_m=np.stack([r[2] for r in rows]),
        required_categories=np.asarray(REQUIRED_CATEGORIES),
        interval_lower_bound_m=np.stack([r[3] for r in rows]))
    # Gate instance metadata is deterministic from pinned native scene.
    manifest = ClearanceGate(mujoco.MjModel.from_xml_path(str(a.native/'execute/scene.xml')), a.min_clearance).geometry_manifest()
    checks = dict(exact_native_replay=exact, every_step_clock_exact=clock_exact,
        every_native_step_present=gap_ok, input_pins_unchanged=pinned,
        original_native_hard_passed=replay['every_step_hard_summary']['all_hard_passed'],
        sampled_cross_side_clearance_ge_20mm=sampled_pass,
        scalar_interpolated_cross_side_clearance_ge_20mm=continuous_pass)
    report = dict(schema='rssr.v3_human_align.every_step_clearance_replay.v3',
        accepted=all(checks.values()), checks=checks, native_root=str(a.native.resolve()),
        case_id=a.case_id, required_clearance_m=a.min_clearance,
        samples=len(t), intervals=len(t)-1, max_sample_gap_s=float(np.max(np.diff(t))),
        horizon_end_s=float(t[-1]), pins=pins_before, manifest=manifest,
        sampled_minima=sampled_minima, interval_minima=interval_minima,
        stage_minima=stages, replay=replay,
        trace='clearance_trace.npz', trace_sha256=sha(a.output/'clearance_trace.npz'),
        continuous_path_definition='Piecewise scalar-joint-linear interpolation between every 0.5ms actual native qpos sample. Conservative geometric motion bounds cover that defined path.',
        limitations=[
            'Camera envelope is assumed 90x40x40mm at fixed fitted extrinsics; hardware dimensions and calibration have not been measured.',
            'This certifies the simulated model and defined native interpolation. It does not bound real robot calibration error, servo tracking error, or unmodelled attachments.',
            'Convex hull rejection can be conservative for concave visual mesh gaps; no gate threshold is relaxed.',
            'Table is not subject to 20mm user threshold; original native penetration checks remain enforced.',
        ], wall_seconds=time.monotonic()-started)
    (a.output/'clearance_receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(accepted=report['accepted'], checks=checks, samples=len(t),
        sampled_min_m={c:sampled_minima[c]['distance_m'] for c in REQUIRED_CATEGORIES},
        continuous_min_m={c:interval_minima[c]['lower_bound_m'] for c in REQUIRED_CATEGORIES},
        output=str(a.output))), flush=True)


if __name__ == '__main__':
    main()
