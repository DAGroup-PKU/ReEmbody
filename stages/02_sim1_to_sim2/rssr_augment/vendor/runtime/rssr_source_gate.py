"""A target run requires evidence from this exact successful source instance."""
import pathlib
from common import read,sha
from rssr_protocol import digest

def require(config,scene):
    path=config.get('source_success_receipt')
    if not path:raise ValueError('Sim2 blocked: missing successful Sim1 physical replay receipt')
    proof=read(path)
    if proof.get('schema')!='rssr.source_success.v2' or proof.get('passed') is not True:
        raise ValueError('Sim2 blocked: Sim1 did not pass this source contract')
    if proof['scene_sha256']!=digest(scene):raise ValueError('Sim2 scene differs from successful Sim1')
    if proof['source_registration']!=config.get('source_base_world_shift_m'):
        raise ValueError('Source coordinate registration changed after Sim1 verification')
    source=pathlib.Path(config['import_case'])/'source_action_state.npz'
    if proof['action_sha256']!=sha(source):raise ValueError('Source action checksum changed')
    if proof['asset_manifest_sha256']!=config.get('engineered_task_asset_sha256'):
        raise ValueError('Task asset differs from successful Sim1')
    report=pathlib.Path(proof['report']);physical=read(report)
    if sha(report)!=proof['report_sha256'] or physical.get('task_success') is not True or not all(physical['criteria'].values()):
        raise ValueError('Sim1 physical proof changed or has failed criteria')
    if physical.get('objects_forced') or physical.get('object_weld_used') or physical.get('robot_qpos_forced_during_execution'):
        raise ValueError('Source proof uses forced state instead of physical replay')
    for filename,key in [('command_targets.npz','commanded_joint_sha256'),('rollout.npz','rollout_sha256'),('scene.xml','model_xml_sha256'),('task_contract.json','task_contract_sha256')]:
        if sha(report.parent/filename)!=proof[key]:raise ValueError('Source evidence changed: '+filename)
    code=pathlib.Path(__file__).parent
    for filename,key in [('pilot_physics.py','physics_code_sha256'),('rssr_source_eef.py','source_eef_code_sha256'),('rssr_task_contracts.py','task_evaluator_sha256')]:
        if sha(code/filename)!=proof[key]:raise ValueError('Source runtime differs from verified physics: '+filename)
    return proof
