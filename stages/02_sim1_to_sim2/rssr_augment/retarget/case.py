"""Bind frozen Cobot cup evidence to a new measured four-torso S1 execution."""
from copy import deepcopy
from pathlib import Path
import sys
import mujoco
import numpy as np
from rssr_augment.common import BRANCH,read,resolve_path,relocate
from rssr_augment.shared.family_autoresearch.store import file_sha,atomic_json
from rssr_augment import measured_s1
from rssr_augment.physical import augment_scene
RUNTIME=BRANCH/'vendor/runtime'
sys.path.insert(0,str(RUNTIME))
import pilot_physics
import rssr_servo
from rssr_task_contracts import compile_contract
measured_s1.install_backend(pilot_physics)

def load_case(strategy,case_id,output):
 if case_id not in strategy['cases'] or not case_id.startswith('stack_cups_ep'):raise ValueError('Case outside cup cohort')
 if strategy.get('s1_model_manifest'):
  binding=strategy['s1_model_manifest']
  if file_sha(binding['path'])!=binding['sha256']:raise ValueError('Frozen S1 model manifest changed')
  for path,expected in read(binding['path'])['files'].items():
   if file_sha(path)!=expected:raise ValueError('Frozen S1 model input changed: '+path)
 cards=read(strategy['evidence']['mechanism_cards'])['mechanism_cards'];card=next(r for r in cards if r['case_id']==case_id)
 for path,expected in card['fixed_evidence_sha256'].items():
  if file_sha(resolve_path(path))!=expected:raise ValueError('Frozen source changed: '+path)
 selections=[resolve_path(p) for p in card['fixed_evidence_sha256'] if p.endswith('/sim2/selection.json')]
 if len(selections)!=1:raise ValueError('Ambiguous frozen source')
 selection_path=selections[0];selection=read(selection_path);trial=resolve_path(selection['trial']);config=relocate(read(selection_path.parent.parent/'config.json'))
 official=resolve_path(strategy['official_dataset_root'])/'150case/cases'/case_id;merged=output/'inputs';merged.mkdir()
 for source in (official/'inputs').iterdir():(merged/source.name).symlink_to(source)
 artifact=strategy['derived_ik_artifacts'][case_id]
 if file_sha(artifact['path'])!=artifact['sha256']:raise ValueError('Frozen source IK changed')
 (merged/'agent_trajectory_full.npz').symlink_to(artifact['path'])
 if file_sha(official/'inputs/source_action_state.npz')!=file_sha(selection_path.parent.parent/'inputs/source_action_state.npz'):raise ValueError('Official source action differs')
 for key in ('engineered_task_asset','engineered_task_asset_sha256','task_asset','task_asset_sha256'):config.pop(key,None)
 raw=measured_s1.pose();config.update(import_case=str(merged),run_root=str(output),batch_root=str(output),api_budget_root=str(output),mode='infer',
  python=sys.executable,mujoco_version_expected='3.3.7',_release_opening={},robot_type='s1',
  collision_integrity_revision='s1_cross_arm_visual_gripper_v2',_s1_measured_torso_rad=raw['torso_rad'],_lyc_head_target_rad=raw['head_rad'],
  model_profile={'solver':{'timestep_s':.0005},'coupling':{f'eq_{i}':{'time_constant_s':.001} for i in range(10)}})
 normalized=measured_s1.normalize_scene(selection['scene']);profile=strategy.get('physical_augmentation',{}).get('profiles',{}).get(case_id)
 scene,receipt=augment_scene(normalized,config,profile);config['_lyc_head_target_rad']=raw['head_rad']
 aligned=(strategy.get('aligned_initialization') or {}).get('profiles',{}).get(case_id)
 if aligned is not None:
  if aligned['case_id']!=case_id:raise ValueError('Aligned initialization case mismatch')
  cup_objects={o['id']:o for o in scene['objects'] if o['shape']=='cup'}
  if set(aligned['cups_world_m'])!=set(cup_objects):raise ValueError('Aligned cup bindings differ')
  for name,xyz in aligned['cups_world_m'].items():
   xyz=np.asarray(xyz,dtype=float)
   if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('Invalid aligned cup center')
   cup_objects[name]['position_m']=xyz.tolist()
  config['_s1_initial_robot_qpos_by_name']=deepcopy(aligned['qpos_by_name'])
  atomic_json(output/'aligned_initialization.json',aligned)
  if receipt:
   receipt.update(aligned_initialization=aligned,augmented_scene=deepcopy(scene),
    free_object_offsets_xy_m={name:(np.asarray(cup_objects[name]['position_m'][:2])-np.asarray(next(o for o in normalized['objects'] if o['id']==name)['position_m'][:2])).tolist() for name in cup_objects},
    initialization_scope='Declared arm perturbations and directly facing cup distances; fixed tabletop/head/torso/fingers')
 preflight=output/'mobile_model_preflight';model=pilot_physics.build(config,'sim2',preflight,deepcopy(scene));times,full,grip=pilot_physics.commands(config,'sim2',model)
 robot=[model.joint(i) for i in range(model.njnt) if (model.joint(i).name or '').startswith('astribot_')];robot.sort(key=lambda j:int(j.qposadr[0]));names=[j.name for j in robot];qa=[int(j.qposadr[0]) for j in robot]
 config['_reference_robot_joint_names']=names;reference=full[:,qa].copy()
 d=mujoco.MjData(model);d.qpos[:]=full[0];mujoco.mj_forward(model,d)
 from .immutable_hard_observation import HardSnapshotter
 hard=HardSnapshotter(model)(d)
 if not hard['all_hard_passed']:raise ValueError('Measured S1 startup hard gate failed: '+repr(hard))
 atomic_json(output/'s1_command_initialization.json',dict(robot='Astribot S1',joint_names=names,initial_joint_values_rad=reference[0].tolist(),
  measured_non_gripper_joints_preserved=aligned is None,measured_head_torso_and_finger_joints_preserved=True,
  aligned_initialization_applied=aligned is not None,source_robot_joint_values_reused=False,sdk_gripper_raw=raw['gripper_sdk_0_100'],
  sdk_gripper_angle_mapping_verified=False,gripper_mapping_assumption='SDK remaining nominal opening fraction geometrically inverted through original S1 linkage',
  initial_gripper_linkage_rad={side:float(reference[0,names.index('astribot_gripper_'+side+'_joint_L1')]) for side in ('left','right')},
  startup_hard=hard,fresh_closed_loop_execution_required=True))
 if receipt:
  receipt.update(robot='S1',cup_model=measured_s1.CUP,source_scene_before_cup_correction=selection['scene'],
   measured_pose_path=measured_s1.reference()['pose_path'],sdk_gripper_angle_mapping_verified=False,
   object_shapes_masses_friction_unchanged=False,task_object_ids_and_goal_order_preserved=True)
  atomic_json(output/'physical_augmentation.json',receipt)
 contract=compile_contract(scene['task_id'],scene,scene['bindings'],model_health_required=True);scene_sha=file_sha(preflight/'scene.xml')
 return dict(case_id=case_id,official_source=str(official),selection_path=selection_path,source_task_text=read(official/'inputs/case.json').get('task'),
  trial=trial,scene=scene,contract=contract,config=config,target_qpos=reference,baseline_report=read(trial/'report.json'),
  baseline_scene_sha256=scene_sha,mobile_scene_sha256=scene_sha,original_baseline_scene_sha256=file_sha(trial/'scene.xml'),baseline_command_targets_sha256=file_sha(trial/'command_targets.npz'))
