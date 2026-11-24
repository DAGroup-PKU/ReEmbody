from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]
SOURCE = PROJECT / 'result/franka_fr3_duo_v2_20261106_attempt_01'
PACKAGE = PROJECT / 'result/lerobot/150case_data_enhance_blender_franka_v2_20261106_attempt_02'

def artifact(path):
    p = Path(path)
    data = p.read_bytes()
    return {'path': str(p), 'sha256': hashlib.sha256(data).hexdigest(), 'text': data.decode()}

schema = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'task': {'type': 'string'},
        'controller_source': {'type': 'string', 'minLength': 100, 'maxLength': 180000},
        'reasoning': {'type': 'string', 'maxLength': 12000},
        'gaze_target_definition': {'type': 'string', 'maxLength': 6000},
        'expected_motion_duration_s': {'type': 'number', 'minimum': 0, 'maximum': 18},
        'uncertain_or_missing_capabilities': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 20},
        'software_test_source': {'type': 'string', 'maxLength': 30000},
    },
    'required': ['task', 'controller_source', 'reasoning', 'gaze_target_definition',
                 'expected_motion_duration_s', 'uncertain_or_missing_capabilities', 'software_test_source'],
}

PROMPT = '''You are the requested GPT-6 Luna trajectory-refinement API agent. Author a complete, executable
Python controller module for the supplied task on the actual Franka FR3 Duo + Robotiq 2F85 + fixed-mounted D405
wrist cameras. Use the provided current source controller as a baseline, but optimize it to meet the user's
requirements on ALL 100 exact source placements. Do not replace data, sample easier layouts, change the robot,
move virtual cameras independently, change camera mounts, loosen validators, edit source data, retime a video,
or claim predicted success as measured success. Your program will be reviewed and executed in fresh output only.

Required behavior: before a grasping hand translates toward its object, first rotate that hand's wrist camera
to face the object while holding its TCP position as closely as IK/servo permit; then move above the actual grasp
region, then descend vertically in a genuine top-down grasp. Maintain object-directed wrist gaze throughout
approach and transport. The active camera is the wrist camera of the grasping hand, not the fixed arch D455.
The complete motion and required 2-second terminal settling must be at most20 seconds: budget motion <=18s.
Preserve the original task, state initializations, independent replay, joint position/velocity/tracking limits,
arm-to-arch clearance >=6cm, stacking inter-arm clearance >=10cm, grasp/idle/ambiguity gates, and for the latest
drawer the no-hand-entry/no-drawer-contact insertion of the original admitted source. Do not raise runtime
STEP_MAX or any hard bound to gain speed. Optimize path length, redundant waits and independent-arm overlap.

Output a full module exporting run(S) and grasps(S), optionally OPTIONS. See the exact trusted Session interface.
Reuse the supplied baseline via read-only imports or complete copied functions if helpful; never modify imported
global runtime/validator constants. It is acceptable to install instance-level planning helpers that preserve
all original collision/path checks. A plain endpoint look_at plus orientation Slerp does not guarantee gaze in
between. Compute gaze-aware orientations from the ACTUAL fixed camera-to-TCP transform at intermediate samples,
solve bounded IK continuously, and validate the resulting path before execution. Initial wrist alignment must be
a rotation before translation, not a relabeled ordinary joint-space approach.

Optical geometry matters: TCP local+Z is the approach axis. Optical forward is -camera_xmat[:,2]. The fixed
camera is offset from the TCP, so an exact centroid ray and exact vertical cup-wall grasp can conflict. A gaze
target may be a physically bound grasp surface/patch on the actual object, such as the selected cup wall, with
its local coordinates declared and transformed with the object. It must stay on the object; do not aim at an
arbitrary empty-space surrogate. Report centroid and patch errors separately; do not invent a new acceptance
threshold or conceal geometric impossibility. The supplied10deg/15deg audit settings are diagnostics, not user
specified gates. Above-before-down should be relative to the legal object/grasp footprint, not force the TCP
above a cup centroid when its wall must be pinched. Preserve the source task contact mechanism where required.

The baseline audit's earliest decreasing gripper command can be a start/pre-shape transient, not the actual
grasp closure. Do not assume that field proves wrong closure timing; use actual object contacts and meaningful
opening-to-close events. Explain unresolved feasibility limitations honestly. Include a self-contained Python
software_test_source exposing test_controller(module), testing geometry/math/interface without claiming physics
success. The actual provided source files and rigid geometry are authoritative, not prior Panda/S1 assumptions.
Do not access credentials/network/files except read-only source imports. All trajectory choices/code must be
your own response, not a request for the host to silently design a replacement controller. Return through the
required function schema, with complete code rather than a plan or an ellipsis.
'''

contract = json.loads((HERE.parent/'trajectory_v3/franka_controller_contract.json').read_text())
geometry = json.loads((HERE/'case0_rigid_geometry.json').read_text())

for folder, task, source_folder, base_file, audit_file in [
    ('cup', 'stack_cups', 'stack_cups', 'duo_stack_cups.py', 'audit_cup_ep0.json'),
    ('block', 'stack_blocks', 'stack_blocks', 'duo_stack_blocks_v2.py', 'audit_block_ep0.json'),
    ('v2_draw_new', 'drawer', 'drawer_new', 'duo_drawer_v2d.py', 'audit_v2_draw_new_ep0.json'),
]:
    rows = [json.loads(s) for s in (PACKAGE/folder/'meta/source_cases.jsonl').read_text().splitlines()]
    first = Path(rows[0]['run'])
    evidence = {
        'effective_user_goal': json.loads((HERE/'effective_goal.json').read_text()),
        'task': task, 'source_folder': folder, 'expected_cases': 100,
        'runtime_contract': contract, 'actual_rigid_geometry_and_initial_conditions': geometry,
        'baseline_controller': artifact(SOURCE/'controllers'/base_file),
        'baseline_wrapper': artifact(SOURCE/'refine'/source_folder/'controller.py'),
        'arch_clearance_helper': artifact(SOURCE/'code/arch_away.py'),
        'representative_native_record': json.loads((first/'record.json').read_text()),
        'representative_camera_topdown_audit': json.loads((HERE/audit_file).read_text()),
        'source_cases': [{'episode_index': r['episode_index'], 'variation': r['variation'],
                          'run': r['run'], 'options': r.get('options'), 'duration_s': r['duration_s']}
                         for r in rows],
        'post_source_change_policy': 'Fresh output, exact source variation+initial model; keep failures visible.',
    }
    output = HERE / f'context_{task}_round00.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps({'prompt': PROMPT, 'evidence': evidence, 'strategy_schema': schema},
                                 ensure_ascii=False, indent=2))
    print(task, output, output.stat().st_size)
