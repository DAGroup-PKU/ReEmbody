"""Drawer-specific admission: unchanged ordered task, complete short physics.

This gate never substitutes cup lift/first-grasp checks for opening, inserting
and closing the drawer. Native predicates and final windows are all retained.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np

VERSION = 'rssr.sim2_refine.drawer.quality.v1'

def audit(native, output):
    native, output = Path(native), Path(output)
    if output.exists():
        raise FileExistsError(output)
    report_path = native/'execute/report.json'
    contract_path = native/'execute/task_contract.json'
    report = json.loads(report_path.read_text())
    contract = json.loads(contract_path.read_text())
    commands = np.load(native/'issued_robot_commands.npz',allow_pickle=False)
    rollout = np.load(native/'execute/rollout.npz',allow_pickle=False)
    hard = np.load(native/'execute/immutable_hard_trace.npz',allow_pickle=False)
    history = json.loads((native/'drawer_refine_progress.json').read_text())
    strict = report['judgments']['strict_v1']['task_evaluation']
    operation = [p for p in contract['predicates'] if p['kind']=='open_insert_close']
    if len(operation) != 1:
        raise ValueError('Expected exactly one original drawer open_insert_close predicate')
    operation = operation[0]
    ordered = next(p for p in strict['predicates'] if p['id']==operation['id'])
    measured = ordered['measurements']
    ct, ht, nt = commands['time_s'], hard['time_s'], rollout['time_s']
    dt = float(report['joint_limit_audit']['timestep_s'])
    expected = int(np.ceil(14.965/dt))+1
    finite = lambda archive: all(np.isfinite(archive[k]).all() for k in archive.files if np.issubdtype(archive[k].dtype,np.number))
    phases = history['history']
    phase_names = [r['phase'] for r in phases]
    required = ['approach_handle','contact_handle','grasp_handle','open_drawer',
        'approach_payload','contact_payload','grasp_payload','lift_payload',
        'insert_payload','release_payload','close_drawer','release_handle']
    checks = {
        'native_timestep_unchanged_0_5ms': dt==.0005,
        'complete_native_command_clock': len(ct)==expected and bool(np.allclose(ct,np.arange(expected)*dt,rtol=0,atol=1e-8)),
        'issued_targets_and_velocities_complete': commands['qpos'].ndim==2 and commands['qvel'].shape==commands['qpos'].shape and len(commands['qpos'])==len(ct)==len(commands['event']),
        'complete_every_step_hard_clock': len(ht)==len(ct)+1 and bool(np.allclose(ht[:-1],ct,rtol=0,atol=1e-8)) and abs(float(ht[-1])-expected*dt)<1e-8,
        'recorded_native_clock_matches_actual_commands': bool(np.all(np.diff(nt)>0)) and abs(float(nt[0]))<1e-8 and abs(float(nt[-1])-float(ct[-1]))<1e-8,
        'physical_duration_le_15s': 0<float(ht[-1])<=15. and abs(float(report['simulation_seconds'])-float(ht[-1]))<1e-8,
        'all_numeric_trace_arrays_finite': bool(all(finite(z) for z in (commands,rollout,hard))),
        'every_step_original_native_hard_constraints_pass': bool(hard['measurements_valid'].all() and hard['hard_constraints'].all()),
        'all_original_strict_predicates_identical_and_pass': {p['id'] for p in strict['predicates']}=={p['id'] for p in contract['predicates']} and all(p['passed'] is True for p in strict['predicates']),
        'all_original_strict_hard_constraints_pass': bool(strict['hard_constraints']) and all(v is True for v in strict['hard_constraints'].values()),
        'native_drawer_task_success': strict['task_success'] is True,
        'drawer_opened_before_payload_insertion': measured.get('opened_before_insertion') is True,
        'drawer_open_at_first_payload_insertion': measured.get('open_during_insertion') is True and measured.get('first_insertion_s') is not None,
        'drawer_closed_during_complete_original_final_window': measured.get('closed_at_end') is True and ordered['passed'] is True,
        'original_final_stability_and_release_windows_preserved': all(float(p['window_s'])==1. for p in contract['predicates'] if p['kind'] in ('stable','released')),
        'one_forward_sequence_with_no_retries': phase_names==required and all(r['stop_reason']=='event_reached' for r in phases),
        'no_controller_rejection': history['failure_reason'] is None,
        'no_objects_forced_welded_or_robot_teleported': all(report.get(k) is False for k in ('objects_forced','object_weld_used','robot_qpos_forced_during_execution')),
        'native_head_view_quality_pass': report.get('head_view_audit',{}).get('passed') is True,
    }
    inputs=[report_path,contract_path,native/'issued_robot_commands.npz',native/'execute/rollout.npz',native/'execute/immutable_hard_trace.npz',native/'drawer_refine_progress.json',Path(__file__)]
    result={'schema':VERSION,'accepted':all(checks.values()),'checks':checks,
            'native_root':str(native.resolve()),'physical_duration_s':float(ht[-1]),
            'operation_native_thresholds':operation,'ordered_measurements':measured,
            'strict_native_evaluation':strict,'phase_history':phases,
            'input_pins':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
            'limitations':['Single simulated candidate; acceptance also requires independent full-state/action alias and every-step 20mm geometry gates.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=audit(args.native,args.output)
    print(json.dumps({'accepted':result['accepted'],'failed_checks':[k for k,v in result['checks'].items() if not v],'output':str(args.output)}))

if __name__=='__main__':
    main()
