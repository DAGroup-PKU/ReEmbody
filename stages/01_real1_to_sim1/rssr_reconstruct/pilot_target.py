"""Object-state repair under frozen tool, uncertainty and physics contracts."""
import copy,pathlib,os,json
import numpy as np
from common import read,write,sha
from pilot_api import propose
from protocol_decision import DecisionOutputExhausted
from pilot_ik import initial_program,edit,plan,NUMERIC,CATEGORICAL
from pilot_physics import rollout
from object_target_agent import objective,diagnostic,state_feedback,HARD,TASK

CONTRACT={'tool':'retarget_patch','fields':{'ik':dict(NUMERIC,**CATEGORICAL),'arms':{'left':'ik_pos_weight .1..4, ik_rot_weight 0..4','right':'same'},'seed_index':'integer 0..65535','correction':{'hand_offsets_m':'left/right each [x,y,z], absolute per-axis <= .16m','phase_hand_offsets_m':'left/right -> grasp/release -> [x,y,z], extra <=.08m, sum <=.16m','vertical_grasp_mix':'0..1','vertical_grasp_mix_by_hand':'left/right 0..1'}},'semantics':'partial patch merged into best program; offsets absolute relative to immutable source reference. Scene, mass, geometry, camera, controller, timing and criteria cannot be changed.'}
CONTRACT['fields']['event_corrections']={'left/right':'list of {cycle,grasp_offset_m,release_offset_m,grasp_rotation_degrees,release_rotation_degrees}',
    'cycle':'zero-based immutable source grasp event, as listed in source_grasp_events',
    'patch_semantics':'cycle is required; the four vector fields are optional. Omitted fields retain the current value, or zero for a new cycle. Other cycles and the other hand are preserved. Explicit zero vectors reset a field.',
    'translations':'each 3-vector per-axis <=.08m; total including all correction fields <=.16m',
    'rotations':'world xyz Euler offsets, each 3-vector per-axis <=35 degrees; smooth source-event interpolation',
    'no_retiming':True}
CONTRACT['fields']['initialization']=['model_default','joint_midpoint','random']
CONTRACT['fields']['time_knots']={'left/right':'list of {time_s,offset_m?,rotation_degrees?}; time and at least one vector required; at most 12 strictly increasing source-time knots per hand; [] clears that hand',
    'bounds':'translation each axis <=.12m, world xyz Euler correction <=35deg; combined global/phase/event/local/time displacement still <=.16m per world axis, combined event/time rotation <=35deg',
    'semantics':'provided hand list replaces that hand only; omitted vector means zero; implicit zero corrections at source episode endpoints; explicit endpoint knots must be zero; timestamps/gripper commands/scene unchanged; smoothstep between knots',
    'localization':'use zero knots before and after the diagnosed interval to preserve other goal segments; knot times and values are Agent choices, not a predefined repair'}
CONTRACT['fields']['vertical_alignment_frame']=['tool_site_legacy','declared_grasp']
CONTRACT['fields']['grasp_frame_offsets_m']={'left/right':'optional partial map of 3-vectors in metres, each axis within .16; all combined displacements must remain within .16 per world axis'}
CONTRACT['grasp_frame_offset_semantics']='Agent-chosen displacement of the IK site in contract-declared grasp axes, rotated by the final goal orientation after vertical and event rotations at every source frame. It adds to world/phase/event shifts under their shared world bound; it does not change the frame contract, object state, timing or criteria. Omitted/zero offsets preserve previous goals. Explicit zero vectors reset one hand; other hand fields are preserved. Grasp frame definitions and measured component geometry are observations, not prescribed offset values.'
CONTRACT['vertical_alignment_semantics']='Optional frame for correction.vertical_grasp_mix and its per-hand overrides. Omitted means tool_site_legacy, preserving existing recipes: mix=1 aligns tool-site +Z downward. declared_grasp instead aligns the contract-declared grasp +Z approach axis downward and maps back to the IK site frame. These axes need not coincide. Mix=0 preserves orientation; event world rotations still apply afterwards. The frame selection and mix are Agent decisions, not a contact-success guarantee.'
CONTRACT['initialization_semantics']='Model default and joint midpoint use declared robot geometry and limits, never prior learned trajectories. seed_index only changes the random policy. From-zero refers to empty knowledge, not a random robot pose. Waist constraints apply to all policies.'
CONTRACT['tracking_semantics']='At source events, ik_tracking compares the corrected IK tool-site goal with FK of the solved joints; servo_tracking compares FK of commanded joints with physically executed FK at the same time. Finger grasp centres and object distances are separate. A hand-object miss alone does not identify which layer failed.'
CONTRACT['upstream_weight_semantics']='Zero freezes the upstream joints at the initialization; a positive value enables the allowed coupled torso 1/2 motion. Torso 3/4 stay fixed and torso 1+2 remains -0.28403 rad for every policy.'
CONTRACT['inherited_operator_semantics']='Imported v22 operators may use solver_operations, a bounded expression bridge acting only on existing IK fields. Select their supplied operator IDs and arguments. Their evidence concerns kinematic trajectories; all new candidates require the current controller, object and contact predicates.'

def candidate_config(config,folder,trajectory):
    dest=pathlib.Path(folder)/'inputs';dest.mkdir(parents=True,exist_ok=True);source=pathlib.Path(config['import_case'])
    for name in ['source_action_state.npz','case.json','frame_action_mapping.npz','sim2_robot_only_scene.xml']:
        target=dest/name
        if not target.exists():target.symlink_to(source/name)
    target=dest/'agent_trajectory_full.npz'
    if not target.exists():target.symlink_to(trajectory)
    if target.resolve()!=pathlib.Path(trajectory).resolve():raise ValueError('Candidate trajectory changed')
    return dict(config,import_case=str(dest))

def run(config,knowledge,root,scene):
    from rssr_errors import CandidateInvalid,StageUnresolved
    from rssr_protocol import ValidationError
    from rssr_interaction import capabilities,feasibility,diagnosis,admission,scene_patch
    from rssr_task_contracts import compile_contract
    from rssr_knowledge import normalize_context
    root=pathlib.Path(root);dest=root/'selection.json'
    if dest.exists():return read(dest)
    initial_scene=copy.deepcopy(scene);generic=scene.get('schema')=='rssr.scene.v2'
    fixed_asset=bool(config.get('task_scene_asset') or config.get('source_success_receipt'))
    if config.get('task_scene_asset'):
        from rssr_task_assets import load as load_task_asset
        load_task_asset(config,scene)
    if config.get('source_success_receipt'):
        from rssr_engineered_assets import validate_source_receipt
        validate_source_receipt(config,scene)
    v16=config.get('interaction_revision')=='v16'
    from rssr_model_tools import enabled,catalog,edit_profile,PROBES
    import rssr_learning_rules as learning
    import rssr_search_branch as search
    v17=enabled(config);ledger=[];probes=set()
    tool=copy.deepcopy(CONTRACT)
    if config.get('source_success_receipt'):
        tool['require_explicit_search_base']=True
        tool['search_base_semantics']='If exploration_candidate is present, base_candidate must explicitly name incumbent or exploration in the decision envelope; it is not an operator parameter. Your diagnosis, expected metrics and edits must all refer to that same executed candidate.'
    caps=capabilities(config) if v16 else None
    if v16:
        tool['fields']['scene_patch']={'object_xyz_shift_m':'free non-excluded object -> vector, step <=.04m; total <=.10m',
            'object_yaw_delta_rad':'free non-excluded object -> radians, step <=.3',
            'object_size_scale':'free non-excluded object -> isotropic [s,s,s], step .9..1.1; total .9..1.1'}
        tool['semantics']='Action fields merge into best_program. scene_patch edits best_scene inside frozen reconstruction uncertainty. No role, object-count, mass, friction, controller, task threshold, region, fixture, particle or excluded-object edits. Every trial restarts full physics; no teleport during execution. Changed scenes need final visual revalidation.'
        tool['target_capabilities']=caps
    if v17:
        if not generic or not v16:raise ValueError('v17 requires the v16 generic interaction backend')
        available=catalog(config,scene)
        model_contract_sha=sha(pathlib.Path(config['model_root'])/'source/scene/s1_tabletop.xml')
        if isinstance(getattr(knowledge,'query_context',None),dict):knowledge.query_context['model_contract_sha256']=model_contract_sha
        tool['fields'].update(model_patch=available['profile_schema'],diagnostic_probe={'type':'string','enum':list(PROBES),'example':PROBES[0],'not_an_array':True},experiment=learning.contract()['experiment'])
        tool['learning_rules']=learning.contract();tool['model_tools']=available
        tool['semantics']+=' model_patch changes only the bounded numerical representation described in model_tools. Each repair carries experiment metadata and edits exactly one layer.'
    if fixed_asset:
        tool['fields'].pop('scene_patch',None)
        tool['semantics']+=' Frozen source-validated task asset: scene_patch is disabled. Object identity, shape, initial pose, mass, contact friction and task criteria cannot be changed. Repair robot actions using measured object/contact state; equivalent numerical model experiments remain allowed.'
        tool['task_asset']={'manifest_sha256':config.get('task_scene_asset_sha256') or config.get('engineered_task_asset_sha256'),
                            'source_replay_validated':True,'scene_editing_allowed':False}
    contract=compile_contract(scene['task_id'],scene,scene['bindings'],model_health_required=v17) if generic else {'hard':list(HARD),'task':list(TASK)}
    write(root/'task_contract.json',contract)
    best=None;history=[];seen=set();program=initial_program();latest_rejected=None;baseline=None;invalid_anchor=None
    exploration=None;branch_evaluations=0
    for i in range(config['max_target_agent_rounds']):
        trial=root/f'trial_{i:02d}';trial.mkdir(parents=True,exist_ok=True);trace=None
        working_scene=copy.deepcopy(best['scene'] if best else initial_scene)
        base_program=best['program'] if best else initial_program()
        model_profile=copy.deepcopy(best.get('model_profile',{}) if best else {})
        experiment=None;probe=None;spec=None;experiment_before=best['report'] if best else invalid_anchor
        control=best;from_branch=False
        if i:
            empty={'passed':False,'rank':[100,100,1e9],'hard_failures':['no_valid_rollout'],'task_failures':['unmeasured']}
            payload={'task':config['task'],'best_program':base_program,'scene':working_scene,
                     'best_report':diagnostic(best['report']) if best else copy.deepcopy(invalid_anchor or {'candidate_invalid':True,'state_trace':[]}),
                     'best_objective':best['objective'] if best else empty,'history':history[-10:],
                     'tool_contract':tool,'remaining_evaluations':config['max_target_agent_rounds']-i}
            if v16:
                payload['best_report']['repair_diagnosis']=diagnosis(best['report']) if best else {'interaction_phase':'initialization'}
                payload['best_report']['grasp_feasibility']=feasibility(working_scene,caps)
                if hasattr(knowledge,'query_context'):
                    knowledge.query_context['repair_phase']=payload['best_report']['repair_diagnosis']['interaction_phase']
            if latest_rejected is not None:payload['latest_rejected_candidate']=latest_rejected
            if v17:
                if hasattr(knowledge,'query_context'):
                    knowledge.query_context['model_profile']=copy.deepcopy(model_profile)
                    knowledge.query_context['model_reuse_catalog']={
                        'couplings':dict.fromkeys(available.get('couplings',{}),{}),
                        'box_objects':copy.deepcopy(available.get('box_objects',[]))}
                    # The editable input may omit empty optional fields. Compare
                    # the executed plan used by proof(), not its sparse spelling.
                    actual_program=(experiment_before or {}).get('ik',{}).get('program')
                    if isinstance(actual_program,dict):knowledge.query_context['action_program']=copy.deepcopy(actual_program)
                    else:knowledge.query_context.pop('action_program',None)
                    from rssr_protocol import digest as context_digest
                    knowledge.query_context['scene_sha256']=context_digest(working_scene)
                    failures=(experiment_before or {}).get('objective',{}).get('hard_failures',[])
                    knowledge.query_context['unresolved_hard_predicates']=list(failures)
                    knowledge.query_context['unresolved_task_predicates']=list(
                        (experiment_before or {}).get('objective',{}).get('task_failures',[]))
                    knowledge.query_context['model_health_unresolved']=any(
                        name in failures for name in ('gripper_coupling_within_2mrad','limited_joints_within_2mrad_or_2mm','invalid_physics'))
                payload['best_report']['learning_state']={'model_profile':model_profile,'measured_metrics':learning.metrics(experiment_before) if experiment_before else {},
                    'hypothesis_ledger':ledger[-8:],'diagnostic_probes_used':sorted(probes),'diagnostic_probes_remaining':2-len(probes)}
                payload['tool_contract']=copy.deepcopy(tool)
                unused=[name for name in PROBES if name not in probes]
                field=payload['tool_contract']['fields']['diagnostic_probe'];field['enum']=unused
                if unused:field['example']=unused[0]
                else:field.pop('example',None)
                payload['tool_contract']['model_tools'].setdefault('diagnostic_probe',{})['currently_available']=unused
                optional=search.observation(best,exploration,branch_evaluations)
                if optional is not None:
                    payload['exploration_candidate']=optional
                    if (payload.get('latest_rejected_candidate') or {}).get('trial')==optional['trial']:
                        payload.pop('latest_rejected_candidate')
            try:
                replay=None
                if i==1 and best and config.get('retarget_experience_probe',False) and not v17:
                    from rssr_experience_probe import first_probe
                    replay=first_probe(config,knowledge,trial/'decision',payload)
                patch,trace=replay if replay is not None else propose(config,knowledge,'retarget',trial/'decision',payload)
            except DecisionOutputExhausted as e:
                history.append({'trial':i,'stopped':True,'method_budget_exhausted':str(e),'unexecuted_proposal':True});write(trial/'outcome.json',history[-1]);break
            except (ValueError,TypeError,KeyError,ValidationError) as e:
                history.append({'trial':i,'invalid_operator':str(e),'decision_path':str(trial/'decision')});write(trial/'outcome.json',history[-1]);continue
            if patch is None:break
        else:patch={}
        try:
            action_patch=dict(patch);geometry=action_patch.pop('scene_patch',None)
            if v17 and i:
                control,from_branch=search.select(trace,best,exploration,branch_evaluations,patch)
                if from_branch:
                    working_scene=copy.deepcopy(control['scene']);base_program=copy.deepcopy(control['program'])
                    model_profile=copy.deepcopy(control.get('model_profile',{}));experiment_before=control['report']
                if experiment_before is None:raise ValueError('No measured initialization evidence')
                spec=learning.validate_experiment(patch,experiment_before)
                comparable=[x for x in ledger if x.get('search_base','incumbent')==('exploration' if from_branch else 'incumbent')
                            and (not from_branch or x.get('control_trial')==control['trial'])]
                learning.enforce_no_repeat(patch,comparable)
                action_patch.pop('experiment')
                probe=action_patch.pop('diagnostic_probe',None)
                profile_patch=action_patch.pop('model_patch',None)
                if profile_patch is not None:model_profile=edit_profile(model_profile,profile_patch,available)
                if probe:
                    if best is None:raise ValueError('Isolation probe requires a complete control trajectory')
                    learning.validate_probe(probe,probes)
                    probes.add(probe)
            if geometry is not None:
                if fixed_asset:raise ValueError('Frozen source-validated task asset forbids scene_patch')
                if not v16:raise ValueError('Scene repair is not enabled')
                working_scene=scene_patch(working_scene,initial_scene,geometry)
            program=copy.deepcopy(base_program) if v17 and not action_patch else edit(base_program,action_patch) if i else base_program
            identity=json.dumps({'program':program,'scene':working_scene,'model_profile':model_profile,'diagnostic_probe':probe},sort_keys=True)
            if identity in seen:
                history.append({'trial':i,'duplicate_proposal':True,'trace':trace});write(trial/'outcome.json',history[-1]);continue
            seen.add(identity)
            if from_branch:branch_evaluations+=1
            planned=read(pathlib.Path(config['import_case'])/'default_ik/plan.json') if i==0 else control['report']['ik'] if v17 and control and program==base_program else plan(config,trial/'ik',program)
            runtime=candidate_config(config,trial,planned['trajectory']);physics=trial/'physics'
            if v17:runtime=dict(runtime,model_profile=model_profile,_diagnostic_probe=probe)
            if generic:
                contract=compile_contract(working_scene['task_id'],working_scene,working_scene['bindings'],model_health_required=v17)
                write(trial/'task_contract.json',contract);runtime=dict(runtime,task_contract=str(trial/'task_contract.json'))
            report=read(physics/'report.json') if (physics/'report.json').exists() else rollout(runtime,'sim2',physics,working_scene)
            from rssr_hold_feedback import event_queries
            feedback=state_feedback(physics,working_scene,ik_plan=planned,
                contact_queries=event_queries(report,contract) if generic else ());write(trial/'state_feedback.json',feedback)
            report=dict(report,state_trace=feedback['state_trace'],ik=planned)
            score=report['objective'] if generic else objective(report,working_scene)
            if generic:
                from rssr_contact_feedback import with_approach_objective
                score=with_approach_objective(score,contract,feedback);report['objective']=score
                from rssr_hard_progress import with_hard_progress
                score=with_hard_progress(score,report);report['objective']=score
            if v17 and i:
                experiment=learning.evidence(experiment_before,report,spec,patch)
                if best is None:experiment['invalid_control_not_admissible']=True
                experiment.update(trial=i,structure=learning.structure(patch),hypothesis=trace.get('hypothesis') if trace else None)
                experiment.update(control_trial=control['trial'] if control else None,search_base='exploration' if from_branch else 'incumbent')
                if probe:
                    experiment.update(probe=probe,control_trial=best['trial'],
                        measured_comparison={'control':learning.metrics(experiment_before),'probe':learning.metrics(report)},
                        probe_hard_failures=score['hard_failures'],probe_task_failures=score['task_failures'])
                ledger.append(experiment);write(trial/'experiment.json',experiment)
                if probe:
                    note={'trial':i,'diagnostic_only':True,'accepted':False,'probe':probe,'experiment':experiment,
                          'physics_seconds':report['wall_seconds'],'not_task_evidence':True}
                    history.append(note);write(trial/'outcome.json',note);write(root/'hypothesis_ledger.json',ledger)
                    continue
        except (ValueError,KeyError,TypeError,ValidationError) as e:
            note={'trial':i,'invalid_proposal':str(e),'trace':trace,'candidate_rejected':True}
            if isinstance(e,CandidateInvalid):note['physical_failure_evidence']=e.evidence
            if v17 and i==0 and isinstance(e,CandidateInvalid):
                invalid_anchor={'candidate_invalid':str(e),'state_trace':[],'physical_failure_evidence':e.evidence,
                    'objective':{'hard_failures':['invalid_physics'],'task_failures':[p['id'] for p in contract['predicates']]},
                    'model_health':e.evidence.get('model_health',{}),'not_a_complete_control':True}
            history.append(note);write(trial/'outcome.json',note)
            if v17 and spec:
                ledger.append({'trial':i,'structure':learning.structure(patch),'prediction_matched':False,'diagnostic_only':bool(probe),
                    'invalid':str(e),'hypothesis':trace.get('hypothesis') if trace else None,
                    'control_trial':control['trial'] if control else None,'search_base':'exploration' if from_branch else 'incumbent'})
                write(root/'hypothesis_ledger.json',ledger)
            if isinstance(e,CandidateInvalid) or best is None:
                latest_rejected={'program':program,'scene':working_scene,'report':{'candidate_invalid':str(e),'state_trace':[]},'accepted':False}
            continue
        changed=working_scene!=initial_scene
        current={'trial':str(physics),'scene':copy.deepcopy(working_scene),'program':program,'report':report,'objective':score,'trajectory':planned['trajectory']}
        if v17:current['model_profile']=model_profile
        if baseline is None:baseline=copy.deepcopy(score)
        if best and trace:
            proof=admission(best['report'],report,changed) if v16 else None
            if v17:
                proof=learning.proof(control['report'],report,changed,experiment,
                                     control_scene=control['scene'],observed_scene=working_scene)
                if from_branch:proof=search.bind_proof(proof,control,best,current)
            if not v16 and 'hard_failures' in best['objective'] and 'hard_failures' in score:
                proof={'hard_before':best['objective']['hard_failures'],'hard_after':score['hard_failures']}
            context=normalize_context(trace.get('experience_context',{'task_family':config.get('task_family','di1'),'phase':'object_task'}))
            if v16 and not context.get('compiled_program_policy'):
                context=dict(context,repair_phase=diagnosis(control['report'])['interaction_phase'])
            if v17:
                context=dict(context,learning_revision='v17',model_contract_sha256=model_contract_sha,
                    evidence_scope=learning.evidence_scope(experiment))
            evidence={'admission_evidence':proof} if proof is not None else {}
            deferred=False
            if v17:
                from rssr_deferred_scene_learning import defer_success
                deferred=defer_success(config,knowledge,trial,current,trace,control['objective']['rank'],context,proof)
            if not deferred:
                knowledge.record('retarget',config['case_id'],trace,control['objective']['rank'],score['rank'],context,score['passed'],**evidence)
        accepted=best is None or tuple(score['rank'])<tuple(best['objective']['rank']) or score['passed']
        relation_gain=False;holding_gain=False
        if v17 and best:
            hard_safe=set(score['hard_failures'])<=set(best['objective']['hard_failures'])
            task_safe=set(score['task_failures'])<=set(best['objective']['task_failures'])
            # A verified health improvement can retain an otherwise tied model.
            health_gain=(learning.model_health_metric(experiment['metric']) and experiment['edited_layer']==['model'] and experiment['prediction_matched'])
            if health_gain:
                before_metrics,after_metrics=learning.metrics(best['report']),learning.metrics(report)
                health_gain=all(after_metrics.get(k,0.)<=before_metrics.get(k,0.)+.0001 for k in ('coupling_residual','joint_excess'))
            restores_hard=set(score['hard_failures'])<set(best['objective']['hard_failures'])
            relation_gain=learning.retain_relation_progress(best['report'],report,experiment)
            holding_gain=learning.retain_holding_progress(best['report'],report,experiment)
            # Repair may remove an artifact-dependent contact. Keep a branch
            # that restores hard feasibility, but admission still rejects its
            # task regression. Exploration and learned claims are separate.
            accepted=bool(hard_safe and ((accepted and (task_safe or restores_hard)) or (health_gain and task_safe) or relation_gain or holding_gain))
        retained_exploration=False
        if accepted:best=copy.deepcopy(current);latest_rejected=None;exploration=None
        else:
            from rssr_repair_feedback import rejected_candidate
            latest_rejected=rejected_candidate(best,current)
            if v17 and branch_evaluations<search.MAX_BRANCH_EVALUATIONS and search.eligible(best,current,experiment):
                exploration=copy.deepcopy(current);retained_exploration=True
        note={'trial':i,'accepted':bool(accepted),'objective':score,'trace':trace,'ik_seconds':planned['wall_seconds'],'physics_seconds':report['wall_seconds']}
        if from_branch:note['search_control_trial']=control['trial']
        if retained_exploration:note['retained_for_exploration_only']=True
        if relation_gain:note['retained_measured_relation_progress_not_task_success']=True
        if holding_gain:note['retained_measured_holding_progress_not_task_success']=True
        history.append(note);write(trial/'outcome.json',note)
        if v17:write(root/'hypothesis_ledger.json',ledger)
        write(root/'progress.json',{'history':history,'best_objective':best['objective'],'best_trial':best['trial']})
        if score['passed']:break
    if best is None:raise StageUnresolved('retarget','No numerically valid candidate within fixed evaluation budget; see per-trial outcomes')
    best.update(history=history,passed=best['objective']['passed'],source_task_success_required=False,
                scene_modified_by_retarget=best['scene']!=initial_scene,old_v22_trajectory_imported=False)
    if config.get('retarget_experience_probe',False) and not best['scene_modified_by_retarget'] and not v17:
        from rssr_success_recipe import record_success
        record_success(config,knowledge,best,baseline)
    if v17:best.update(learning_revision='v17',hypothesis_ledger=ledger,diagnostic_probes_used=sorted(probes),search_branch_evaluations=branch_evaluations)
    write(dest,best);return best
