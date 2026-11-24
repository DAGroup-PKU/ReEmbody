"""Compile measured Agent success into one reusable parameter program.

No action trajectory, scene pose or human-selected repair is stored. A program
is merely a candidate on the next demonstration and must pass normal physics.
Keep the first successful program per compatibility class to bound growth.
"""
import copy
from pathlib import Path
import numpy as np
from common import sha
from pilot_ik import edit, initial_program
from pilot_learning import validate_operator
from rssr_physical_feedback import grasp_events
from rssr_protocol import digest

POLICY = 'first_physically_successful_composed_program_v1'


def compatibility(config, scene):
    root = Path(config['model_root'])/'source/scene'
    with np.load(Path(config['import_case'])/'source_action_state.npz') as source:
        times = source['timestamp']-source['timestamp'][0]
        counts = {side:len(grasp_events(times,np.clip(source['action'][:,index]/1.05,0,1)))
                  for side,index in (('left',13),('right',6))}
    return {'task_family': config['task_family'],
            'interaction_revision':config.get('interaction_revision','legacy'),
            'model_sha256':sha(root/'s1_tabletop.xml'),
            'task_frame_contract_sha256':sha(root/'task_frame_contract.json'),
            'source_grasp_cycle_counts':counts,
            'scene_object_ids':sorted(o['id'] for o in scene['objects']),
            'task_bindings':copy.deepcopy(scene.get('bindings',{}))}


def record_success(config, knowledge, best, baseline):
    """Called only after the ordinary measured evaluation, with no API call."""
    if config.get('mode','train') != 'train' or not best['objective']['passed']:
        return
    if baseline['passed'] or tuple(best['objective']['rank']) >= tuple(baseline['rank']):
        return
    program = edit(initial_program(), best['program'])
    signature = compatibility(config, best['scene'])
    snapshot = knowledge.snapshot('retarget',include_referenced=True)
    context = {'task_family':config['task_family'],'phase':'object_task',
               'compiled_program_policy':POLICY,'compatibility':signature}
    operator = validate_operator({
        'name':'Replay a physically successful composed retarget program',
        'mechanism':'Apply the full cumulative parameter program produced by prior Agent decisions; recompute IK on the current source action and validate it with unchanged physics.',
        'preconditions':'Same task family, robot assets, task frame contract, object roles and grasp-cycle counts. Previous success does not imply transfer.',
        'tool':'retarget_patch','parameters':{},'template':program},'retarget')
    trace = {'operator_id':operator['id'],'operator_definition':operator,'arguments':{},
             'diagnosis':'The current executed candidate passed all fixed physics and task predicates.',
             'hypothesis':'The complete accumulated parameter program may avoid repeated repair decisions on a compatible demonstration.',
             'knowledge_version':snapshot['version'],'mode':'train','case':config['case_id'],
             'decision_source':'fixed_agent_compiles_measured_success',
             'experience_context':context,
             'source_program_sha256':digest(program),
             'source_physics_report_sha256':sha(Path(best['trial'])/'report.json'),
             'source_trajectory_sha256':sha(best['trajectory']),
             'contains_saved_trajectory':False,'transfer_verified_here':False}
    proof=None
    if config.get('interaction_revision')=='v16':
        proof=dict(hard_before=baseline['hard_failures'],hard_after=best['objective']['hard_failures'],
                     evidence_policy='interaction_v16',evidence_level='task_success',
                     task_before=baseline['task_failures'],task_after=[],scene_unverified=False)
    knowledge.record('retarget',config['case_id'],trace,baseline['rank'],
                     best['objective']['rank'],context,True,admission_evidence=proof)
