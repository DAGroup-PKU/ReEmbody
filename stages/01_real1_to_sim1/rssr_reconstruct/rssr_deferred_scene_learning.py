"""Admit a future Agent success only after its changed scene passes review.

Pending observations are case-local, not learned knowledge. Historical bank
records are never replayed or rewritten, and no new model/physics call is made.
"""
import copy
from pathlib import Path
from common import read,write,sha
from pilot_learning import digest as evidence_digest
from rssr_protocol import digest
from rssr_knowledge import normalize_context

POLICY='selected_success_after_same_scene_review_v1'


def enabled(config,bank):
    return (config.get('mode','train')=='train' and getattr(bank,'mode','train')=='train'
            and config.get('learning_revision')=='v17')


def defer_success(config,bank,trial,current,trace,before,context,proof):
    if (not enabled(config,bank) or not trace or not trace.get('operator_id')
        or current['objective'].get('passed') is not True
        or proof.get('scene_unverified') is not True
        or proof.get('hypothesis_verified') is not True
        or proof.get('experiment',{}).get('diagnostic_only')):
        return False
    trial=Path(trial)
    item={'stage':'retarget','case':config['case_id'],'trace':copy.deepcopy(trace),
        'before':list(before),'after':list(current['objective']['rank']),
        'context':normalize_context(context),'passed':True,
        'improved':tuple(current['objective']['rank'])<tuple(before),
        'admission_evidence':copy.deepcopy(proof)}
    pending={'policy':POLICY,'observation':item,'scene_sha256':digest(current['scene']),
        'model_profile_sha256':digest(current.get('model_profile',{})),
        'actual_program_sha256':digest(current['report']['ik']['program']),
        'physics_report_sha256':sha(trial/'physics/report.json'),
        'trajectory_sha256':sha(current['trajectory'])}
    path=trial/'pending_scene_learning.json'
    if path.exists() and read(path)!=pending:raise ValueError('Pending scene-learning observation changed')
    if not path.exists():write(path,pending)
    return True


def release_reviewed_success(config,bank,target,review_receipt):
    if (not enabled(config,bank) or target.get('passed') is not True
        or not target.get('scene_modified_by_retarget')):return False
    case_root=Path(config['run_root']).resolve();physics=Path(target['trial']).resolve()
    if case_root/'sim2' not in physics.parents:raise ValueError('Selected physics outside current case')
    trial=physics.parent;path=trial/'pending_scene_learning.json'
    # No backfill for a cached/historical outcome without this explicit spool.
    if not path.exists():return False
    receipt_path=Path(review_receipt).resolve()
    if case_root not in receipt_path.parents:raise ValueError('Review outside current case')
    pending=read(path);receipt=read(receipt_path);body=dict(receipt);identity=body.pop('result_id')
    if digest(body)!=identity or receipt.get('stage')!='scene.final_review':
        raise ValueError('Final scene review receipt identity mismatch')
    review=receipt['data']
    if review.get('objective',{}).get('passed') is not True:return False
    from pilot_scene import scene_score
    if (review.get('immutable_final_scene_review') is not True or
        not scene_score(review.get('verdict',{}),review.get('numeric',{}))['passed']):
        raise ValueError('Final scene review does not support its pass')
    item=pending['observation'];proof=copy.deepcopy(item['admission_evidence'])
    execution=proof.get('execution_context',{})
    checks=[pending.get('policy')==POLICY,item['case']==config['case_id'],
        item['stage']=='retarget',item['passed'] is True,
        proof.get('evidence_policy')=='interaction_v17',proof.get('scene_unverified') is True,
        proof.get('hypothesis_verified') is True,not proof.get('experiment',{}).get('diagnostic_only'),
        target['objective']['passed'] is True,item['after']==target['objective']['rank'],
        proof.get('hard_after')==target['objective']['hard_failures']==[],
        proof.get('task_after')==target['objective']['task_failures']==[],
        digest(review['scene'])==digest(target['scene'])==pending['scene_sha256'],
        execution.get('observed_result_scene_sha256')==pending['scene_sha256'],
        digest(target.get('model_profile',{}))==pending['model_profile_sha256'],
        execution.get('observed_result_model_profile')==target.get('model_profile',{}),
        digest(target['report']['ik']['program'])==pending['actual_program_sha256'],
        execution.get('observed_result_action_program')==target['report']['ik']['program'],
        sha(physics/'report.json')==pending['physics_report_sha256'],
        read(physics/'report.json').get('task_success') is True,
        sha(target['trajectory'])==pending['trajectory_sha256']]
    if not all(checks):raise ValueError('Reviewed success differs from pending executed observation')
    proof['scene_unverified']=False
    proof['scene_review_verification']={'policy':POLICY,'result_id':identity,
        'review_receipt_sha256':sha(receipt_path),'pending_observation_sha256':sha(path),
        'scene_sha256':pending['scene_sha256'],'physics_report_sha256':pending['physics_report_sha256']}
    released=copy.deepcopy(item);released['admission_evidence']=proof
    evidence_id=evidence_digest(released)
    bank.record('retarget',item['case'],item['trace'],item['before'],item['after'],
                item['context'],True,proof)
    evidence_path=bank.root/'evidence/retarget'/(evidence_id+'.json')
    if not evidence_path.exists() or read(evidence_path)!=released:
        raise ValueError('Reviewed observation was not recorded with its bound evidence')
    publication={'policy':POLICY,'case':item['case'],'evidence_id':evidence_id,
        'review_result_id':identity,'pending_observation_sha256':sha(path),
        'scope':'scene-validated execution; original prediction and admission rules still apply'}
    dest=trial/'reviewed_scene_learning.json'
    if dest.exists() and read(dest)!=publication:raise ValueError('Scene-learning publication changed')
    if not dest.exists():write(dest,publication)
    return True
