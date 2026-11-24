"""Execute one measured-S1 drawer candidate; every failed gate stays visible.

No cup controllers, prerecorded commands, object pose/force corrections,
physics threshold edits, trajectory crops or video time scaling are used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT.parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strategy',type=Path,required=True)
    parser.add_argument('--case-id',default='drawer_ep000014')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--implementation',type=Path,default=PROJECT/'sim1-sim2/s1_short_refine_v2')
    parser.add_argument('--drawer-implementation',type=Path,default=PROJECT/'sim1-sim2/s1_drawer_v1')
    parser.add_argument('--result-root',type=Path,default=PROJECT/'result/sim2-refine/s1_drawer_v1_20261019')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (ROOT,args.implementation,args.drawer_implementation):
        sys.path.insert(0,str(path.resolve()))
    from rssr_drawer.runtime import configure
    configure(args.implementation.resolve(),args.result_root.resolve())
    from rssr_drawer.case import load_case,pilot_physics,rssr_servo
    from rssr_augment.retarget import session as session_module
    from rssr_augment.retarget.live_predicates import LivePredicates
    from rssr_augment.shared.family_autoresearch.store import atomic_json
    from tasks.drawer.controller import DrawerRefineAgent,RefineRejected,STOP_S,VERSION,PHASES
    from tasks.drawer.clearance import ClearanceRecorder
    from tasks.drawer import servo
    from tasks.drawer.quality_gate import audit as quality_audit
    from validation.statealias_gate_v2 import audit as alias_audit
    strategy=json.loads(args.strategy.read_text())
    from validation._shared.paths import SOURCE_CLOCK_ROOT,SHARED_RUNTIME_ROOT,STATEALIAS_REFERENCE_PACKAGE,FCL_DEPS
    dependency_paths=[args.strategy,*sorted(Path(__file__).parent.glob('*.py')),
        *sorted((ROOT/'validation').rglob('*.py')),
        *sorted((args.drawer_implementation/'rssr_drawer').glob('*.py')),
        args.drawer_implementation/'config/measured_drawer.json',
        STATEALIAS_REFERENCE_PACKAGE/'meta/info.json',
        SOURCE_CLOCK_ROOT/'rssr_render/source_clock.py',SOURCE_CLOCK_ROOT/'rssr_render/common.py',
        SHARED_RUNTIME_ROOT/'rssr_augment/layout.py',
        *sorted(FCL_DEPS.glob('fcl/*.py')),*sorted(FCL_DEPS.glob('fcl/*.so')),
        *sorted(FCL_DEPS.glob('python_fcl.libs/*.so*'))]
    pins=dict(strategy.get('implementation_sha256',{}))
    for path in dependency_paths:
        resolved=str(path.resolve());actual=sha(path)
        if resolved in pins and pins[resolved]!=actual:
            raise ValueError('Strategy/new dependency pin conflict: '+resolved)
        pins[resolved]=actual
    for path,expected in strategy.get('implementation_sha256',{}).items():
        if sha(path)!=expected:
            raise ValueError('Pinned implementation changed: '+path)
    args.output.mkdir(parents=True)
    started=time.monotonic()
    atomic_json(args.output/'drawer_run_started.json',{'schema':VERSION,'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'case_id':args.case_id,'strategy':str(args.strategy.resolve()),'input_pins':pins,
        'native_requested_stop_s':STOP_S,'duration_limit_s':15.,
        'drawer_loaded_robot_target_speed_limit_m_s':servo.DRAWER_LOADED_SPEED_M_S,
        'native_hard_action_speed_limit_m_s':.1,'phase_budget':PHASES,
        'task':'open drawer, put red block inside, close drawer',
        'robot_commands_from_live_fk':True,'native_physics_and_goal_unchanged':True})
    case=load_case(strategy,args.case_id,args.output)
    for path in [args.output/'mobile_model_preflight/scene.xml',
                 args.output/'physical_augmentation.json',args.output/'s1_command_initialization.json']:
        pins[str(path.resolve())]=sha(path)
    atomic_json(args.output/'drawer_input_pins.json',pins)
    atomic_json(args.output/'drawer_expected_native_contract.json',case['contract'])
    pins[str((args.output/'drawer_expected_native_contract.json').resolve())]=sha(args.output/'drawer_expected_native_contract.json')
    monitor=LivePredicates(case['contract'],case['scene'])
    recorder=ClearanceRecorder(.020)
    original_snapshotter=session_module.HardSnapshotter
    session_module.HardSnapshotter=recorder.snapshotter(original_snapshotter)
    class DrawerSession(session_module.ContinuousSession):
        @staticmethod
        def _control(action,snapshot,q,dq,model):
            return servo.control_step(action,snapshot,q,dq,model)
    session=DrawerSession(pilot_physics,rssr_servo,case['config'],case['scene'],
        args.output/'execute',case['target_qpos'],stop_s=STOP_S,observer=monitor)
    agent=DrawerRefineAgent(session,case,args.output,atomic_json)
    finished=False
    failure=None
    try:
        obs=session.start()
        for path in [args.output/'execute/scene.xml',args.output/'execute/task_contract.json']:
            if path.exists():
                pins[str(path.resolve())]=sha(path)
        atomic_json(args.output/'drawer_input_pins.json',pins)
        try:
            obs=agent.run(obs)
        except RefineRejected as error:
            failure=str(error)
            obs=session.observe()
            agent.hold_to_end(obs)
        session.join()
        session.finish()
        obs=session.observe()
        finished=True
    except BaseException as error:
        # Preserve the actual issued prefix as incomplete diagnostic evidence.
        # An interface/interrupt failure can never become a task success label.
        partial=session.command_records
        if partial:
            np.savez_compressed(args.output/'issued_robot_commands_partial.npz',
                time_s=np.asarray([r[0] for r in partial]),qpos=np.stack([r[1] for r in partial]),
                qvel=np.stack([r[2] for r in partial]),event=np.asarray([r[3] for r in partial],dtype='U40'))
        atomic_json(args.output/'receipt.json',{'schema':'rssr.sim2_refine.drawer.receipt.v1',
            'case_id':args.case_id,'accepted':False,'execution_status':'incomplete_interface_or_execution_error',
            'exception_type':type(error).__name__,'failure_reason':str(error),
            'issued_prefix_rows':len(partial),'not_a_training_release':True,
            'native':{'task_goal_met':False},'replay':None,'complete_physical_horizon_verified':False,
            'original_outputs_retained':True})
        raise
    finally:
        if not finished:
            session.abort()
        session_module.HardSnapshotter=original_snapshotter
    rows=session.command_records
    np.savez_compressed(args.output/'issued_robot_commands.npz',
        time_s=np.asarray([r[0] for r in rows]),qpos=np.stack([r[1] for r in rows]),
        qvel=np.stack([r[2] for r in rows]),event=np.asarray([r[3] for r in rows],dtype='U40'))
    # Persist final actual time without changing any phase completion outcome.
    agent._persist(obs)
    hard=np.load(args.output/'execute/immutable_hard_trace.npz',allow_pickle=False)
    dt=float(case['config'].get('sim_dt',.0005))
    clearance=recorder.finish(args.output,hard['time_s'],dt)
    quality=quality_audit(args.output,args.output/'drawer_quality_receipt.json')
    alias=alias_audit(args.output,args.output/'alias_receipt.json')
    replay=None
    commands_before_replay=sha(args.output/'issued_robot_commands.npz')
    if failure is None and quality['accepted'] and clearance['accepted'] and alias['accepted']:
        from rssr_augment.retarget.run_case import _replay
        replay=_replay(case,args.output,session.command_records,stop_s=STOP_S)
        if sha(args.output/'issued_robot_commands.npz')!=commands_before_replay:
            raise ValueError('Independent replay changed canonical U40 issued command archive bytes')
        atomic_json(args.output/'drawer_replay_receipt.json',replay)
    unchanged=all(sha(path)==expected for path,expected in pins.items())
    accepted=bool(failure is None and quality['accepted'] and clearance['accepted'] and alias['accepted'] and unchanged and replay and replay['exact'] and replay['every_step_hard_trace_exact'] and replay['every_step_hard_summary']['all_hard_passed'])
    receipt={'schema':'rssr.sim2_refine.drawer.receipt.v1','case_id':args.case_id,
        'accepted':accepted,'task':'open drawer, insert red block, close drawer',
        'controller':VERSION,'strategy_sha256':sha(args.strategy),'failure_reason':failure,
        'horizon_s':STOP_S,'actual_physical_end_s':float(hard['time_s'][-1]),
        'wall_seconds':time.monotonic()-started,'same_simulation_state':True,
        'object_state_written_by_agent':False,'source_action_commands_replayed':False,
        'native_physics_actuators_forces_and_task_thresholds_unchanged':True,
        'robot_action_policy_changes':{'drawer_loaded_speed_m_s':servo.DRAWER_LOADED_SPEED_M_S,
            'former_loaded_speed_m_s':.025,'immutable_action_speed_upper_bound_m_s':.1,
            'one_forward_sequence_without_regrasp_or_retry':True},
        'bindings':agent.bindings.packet(),'hand_assignment':agent.assignment,
        'event_history':agent.history,'implementation_pins_unchanged':unchanged,
        'implementation_pins':pins,'issued_commands_sha256':sha(args.output/'issued_robot_commands.npz'),
        'gates':{'drawer_task_quality':{'accepted':quality['accepted'],'receipt':str(args.output/'drawer_quality_receipt.json')},
                 'every_step_20mm_clearance':{'accepted':clearance['accepted'],'receipt':str(args.output/'clearance_receipt.json')},
                 'full_32state_22action_alias':{'accepted':alias['accepted'],'receipt':str(args.output/'alias_receipt.json')}},
        'native':{'task_goal_met':quality['checks']['native_drawer_task_success'],
                  'task_text_physical_success':quality['accepted'],
                  'original_strict_evaluation':quality['strict_native_evaluation']},
        'replay':replay,
        'not_a_training_release':True,'independent_actual_command_replay_completed':bool(replay),
        'formal_publication_admission_completed':False,
        'formal_publication_requires':['independent actual-issued-command replay exactness',
            'all unchanged native hard and drawer predicates','every-step 20mm geometry',
            'state/action alias gate','verified camera/render clock and source provenance'],
        'rendering_requires_all_gates_accepted':True}
    atomic_json(args.output/'receipt.json',receipt)
    print(json.dumps({'accepted':accepted,'failure_reason':failure,'actual_physical_end_s':receipt['actual_physical_end_s'],
        'wall_seconds':receipt['wall_seconds'],'output':str(args.output)},allow_nan=False),flush=True)

if __name__=='__main__':
    main()
