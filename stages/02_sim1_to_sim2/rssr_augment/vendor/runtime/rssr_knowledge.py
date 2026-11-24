"""Typed stage libraries with selective imports and physically read-only inference.

Index JSON is small; full evidence is one immutable JSON per observation. Inference
keeps newly proposed operators in episode memory, never in an exported library.
"""
import contextlib
import copy
import fcntl
import math
import pathlib
import time

from common import read, write, sha
from pilot_learning import STAGES, validate_operator, instantiate, digest


def empty_stage(stage):
    return {'schema': 'rssr.knowledge.stage.v2', 'stage': stage, 'revision': 0,
            'operators': [], 'experiences': [], 'negative_examples': [], 'evidence_count': 0,
            'scene_priors': [],
            'trajectory_priors': [],
            'training_cases': [], 'inherited_training_cases': [], 'events': []}


def clean_operator(operator):
    return {k: copy.deepcopy(operator[k]) for k in
            ('name', 'mechanism', 'preconditions', 'tool', 'parameters', 'template')}


def normalize_context(context):
    if isinstance(context, str):
        family, _, phase = context.partition(':')
        return {'task_family': family, 'phase': phase}
    if not isinstance(context, dict) or not isinstance(context.get('task_family'), str):
        raise ValueError('Knowledge context must name the task family')
    return copy.deepcopy(context)


def check_stage(state, stage):
    if state.get('schema') != 'rssr.knowledge.stage.v2' or state.get('stage') != stage:
        raise ValueError('Wrong library schema or stage: ' + stage)
    ids = set()
    for operator in state['operators']+state.get('candidate_operators',[])+state.get('archived_operators',[]):
        checked = validate_operator(clean_operator(operator), stage)
        if operator['id'] != checked['id'] or operator['id'] in ids:
            raise ValueError('Invalid/duplicate operator identity')
        ids.add(operator['id'])
    if state.get('operator_policy'):
        from rssr_operator_policy import POLICY, ACTIVE_LIMIT
        if state['operator_policy']!=POLICY or len(state['operators'])>ACTIVE_LIMIT:
            raise ValueError('Operator admission policy/active limit violated')
        pinned=set(state.get('pinned_operator_ids',[]))
        for op in state['operators']:
            if op['id'] not in pinned and len(state['operator_registry'][op['family_id']]['improved_cases'])<2:
                raise ValueError('Active operator lacks two distinct demonstrations')
    for experience in state['experiences']:
        if experience['operator_id'] not in ids or not experience['improved_cases']:
            raise ValueError('Unverified or unresolved experience')
    prior_ids = set()
    inherited_ids=set()
    for prior in state.get('trajectory_priors',[]):
        from rssr_v22_bridge import validate_call
        if (stage!='retarget' or prior['id'] in inherited_ids or prior.get('current_object_task_verified') is not False
                or prior.get('evidence_level')!='constrained_trajectory_success' or prior['trajectory_success_count']<1):
            raise ValueError('Invalid inherited trajectory evidence')
        for call in prior['patch']['solver_operations']:validate_call(call)
        inherited_ids.add(prior['id'])
    if not isinstance(state.get('scene_priors', []), list):
        raise ValueError('Scene priors must be a list')
    for prior in state.get('scene_priors', []):
        family = prior['task_family']
        if (stage != 'reconstruction' or prior['id'] != digest({'scene_prior_task_family': family}) or
                prior['id'] in prior_ids or not prior['accepted_scene_cases'] or
                prior['prototype']['task_id'] != family):
            raise ValueError('Invalid or duplicated scene prior')
        if any(set(o) & {'position_m','yaw_rad','roll_pitch_rad','head_center','evidence'}
               for o in prior['prototype']['objects']):
            raise ValueError('Scene prior must not include prior-episode pose or image evidence')
        prior_ids.add(prior['id'])
    return state


class KnowledgeBank:
    def __init__(self, root, mode='train', query_context=None, maintenance_policy=None):
        if mode not in ('train', 'infer'):
            raise ValueError('Library mode must be train or infer')
        self.root = pathlib.Path(root)
        self.mode = mode
        self.query_context = normalize_context(query_context) if query_context else None
        self.transient = {stage: {} for stage in STAGES}
        self._evidence_cache = {}
        if maintenance_policy:
            from rssr_library_lifecycle import POLICY
            if maintenance_policy!=POLICY:raise ValueError('Unknown library maintenance policy')
        self.maintenance_policy=maintenance_policy

    @contextlib.contextmanager
    def locked(self):
        with (self.root/'knowledge.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    @staticmethod
    def initialize(root, mode='train', imports=None, required_stages=STAGES, admission_policy=None):
        root = pathlib.Path(root)
        root.mkdir(parents=True, exist_ok=True)
        imports = imports or {}
        if set(imports) - set(STAGES):
            raise ValueError('Unknown imported stage')
        if mode == 'infer' and set(imports) != set(required_stages):
            raise ValueError('Inference requires all configured stage libraries, including empty frozen ones')
        sources = {}
        imported = {}
        for stage, reference in imports.items():
            reference = pathlib.Path(reference)
            checkpoint = read(reference)
            if checkpoint.get('schema') != 'rssr.knowledge.checkpoint.v2':
                raise ValueError('Import requires a sealed checkpoint, not a mutable bank path')
            body = dict(checkpoint)
            identity = body.pop('checkpoint_id')
            if digest(body) != identity:
                raise ValueError('Checkpoint digest mismatch')
            entry = checkpoint['stages'][stage]
            file = (reference.parent/entry['path']).resolve()
            if sha(file) != entry['sha256']:
                raise ValueError('Checkpoint stage content changed')
            imported[stage] = check_stage(read(file), stage)
            sources[stage] = {'checkpoint': str(reference.resolve()), 'checkpoint_id': identity,
                              'stage_sha256': entry['sha256']}
        init = {'schema': 'rssr.knowledge.initialization.v2', 'mode': mode,
                'imports': sources, 'empty_stages': [s for s in STAGES if s not in imports]}
        if admission_policy:
            from rssr_operator_policy import POLICY
            if admission_policy!=POLICY:raise ValueError('Unknown operator admission policy')
            init['operator_admission_policy']=admission_policy
        bank = KnowledgeBank(root, mode)
        with bank.locked():
            path = root/'initialization.json'
            if path.exists():
                if read(path) != init:
                    raise ValueError('Knowledge initialization changed; use a new run directory')
                for stage in STAGES:
                    check_stage(read(root/(stage+'.json')), stage)
                return bank
            if any((root/(s+'.json')).exists() for s in STAGES):
                raise ValueError('Partial initialization requires explicit recovery')
            for stage in STAGES:
                state = copy.deepcopy(imported.get(stage, empty_stage(stage)))
                # Old evidence stays in its sealed source artifact; the run stores
                # only new event bodies while retaining source attribution.
                state['inherited_training_cases'] = sorted(set(state['inherited_training_cases'] + state['training_cases']))
                state['training_cases'] = []
                state['events'] = []
                if admission_policy:
                    from rssr_operator_policy import configure
                    configure(state)
                write(root/(stage+'.json'), state)
            write(path, init)
        return bank

    def snapshot(self, stage, include_referenced=False):
        state = check_stage(read(self.root/(stage+'.json')), stage)
        if state.get('operator_policy'):
            from rssr_operator_policy import snapshot
            def load_evidence(identity):
                if identity not in self._evidence_cache:
                    # Evidence bodies are immutable and content-addressed. A
                    # missing inherited body is unknown, never a positive fact.
                    if len(identity)!=64 or any(c not in '0123456789abcdef' for c in identity):
                        raise ValueError('Invalid evidence identity')
                    path=self.root/'evidence'/stage/(identity+'.json')
                    if not path.exists():return None
                    self._evidence_cache[identity]=read(path)
                return self._evidence_cache[identity]
            result=snapshot(state,self.mode,self.query_context,self.transient[stage],
                            load_evidence=load_evidence if stage=='retarget' else None)
            if stage=='retarget':result['trajectory_priors']=copy.deepcopy(state.get('trajectory_priors',[]))
            if stage=='reconstruction':
                family=(self.query_context or {}).get('task_family')
                result['scene_priors']=[{k:copy.deepcopy(p[k]) for k in ('id','task_family','prototype','basis')} |
                    {'accepted_scene_case_count':len(p['accepted_scene_cases']),'same_prototype_transfer_proven':False}
                    for p in state.get('scene_priors',[]) if family is None or p['task_family']==family]
            return result
        family = self.query_context.get('task_family') if self.query_context else None
        experiences = [e for e in state['experiences'] if family is None or e['context']['task_family'] == family]
        experiences.sort(key=lambda e: (len(e['successful_cases']), len(e['improved_cases']), e['uses']), reverse=True)
        experiences = experiences[:8]
        needed = {e['operator_id'] for e in experiences}
        operators = [o for o in state['operators'] if o['id'] in needed]
        # Generic mechanisms may be visible across families; context-bound
        # parameter recipes are not automatically transferred across tasks.
        remaining = max(0, 24-len(operators))
        if remaining:
            operators += [o for o in state['operators'] if o['id'] not in needed and (o.get('effect_validated') or o.get('inherited_solver_candidate'))][-remaining:]
        known = {o['id'] for o in operators}
        operators += [o for i, o in self.transient[stage].items() if i not in known]
        negatives = [e for e in state['negative_examples'] if family is None or e['context']['task_family'] == family][-8:]
        result = {'version': digest(state), 'operators': operators, 'experiences': experiences,
                  'negative_examples': negatives, 'mode': self.mode}
        if stage == 'reconstruction':
            result['scene_priors'] = [{k: copy.deepcopy(p[k]) for k in
                                      ('id','task_family','prototype','basis')} |
                                     {'accepted_scene_case_count': len(p['accepted_scene_cases']),
                                      'same_prototype_transfer_proven': False}
                for p in state.get('scene_priors', []) if family is None or p['task_family'] == family]
        if stage == 'retarget' and state.get('trajectory_priors'):
            result['trajectory_priors']=copy.deepcopy(state['trajectory_priors'])
        return result

    def record_scene_prior(self, case, reconstruction, context):
        """Store an accepted visual structure, including first-attempt success.

        This is a geometry prior, not an improved operator or a successful
        physical retarget. One first accepted prototype per task family; later
        accepted cases contribute evidence counts, not duplicate skills.
        """
        if self.mode == 'infer' or not reconstruction['objective']['passed']:
            return
        from pilot_scene import scene_score
        from rssr_scene import validate
        if not scene_score(reconstruction['verdict'], reconstruction['numeric'])['passed']:
            raise ValueError('Scene-prior acceptance contradicts actual scene evidence')
        scene = reconstruction['scene']
        if scene.get('schema') != 'rssr.scene.v2': return
        validate(scene)
        context = normalize_context(context)
        family = context['task_family']
        if scene['task_id'] != family:
            raise ValueError('Scene prior belongs to another task family')
        omitted = {'position_m','yaw_rad','roll_pitch_rad','head_center','evidence'}
        prototype = {'schema': 'rssr.scene.prior.v1', 'task_id': family,
            'objects': [{k:copy.deepcopy(v) for k,v in o.items() if k not in omitted} for o in scene['objects']],
            'bindings_example':copy.deepcopy(scene['bindings']),
            'reestimate_from_current_images_and_action':sorted(omitted),
            'not_an_executable_scene':True,'metric_calibration':False}
        item = {'stage':'reconstruction','kind':'accepted_scene_prior','case':case,
                'context':context,'accepted_scene_sha256':digest(scene),
                'verdict':reconstruction['verdict'],'numeric':reconstruction['numeric'],
                'prototype':prototype,'sim1_task_success_used':False,'physical_transfer_verified':False}
        identity = digest(item)
        with self.locked():
            path = self.root/'reconstruction.json'
            state = check_stage(read(path), 'reconstruction')
            if identity in state['events']: return
            evidence_path = self.root/'evidence/reconstruction'/(identity+'.json')
            if evidence_path.exists() and read(evidence_path) != item:
                raise ValueError('Scene-prior evidence changed')
            write(evidence_path,item)
            priors = state.setdefault('scene_priors', [])
            prior_id = digest({'scene_prior_task_family':family})
            prior = next((p for p in priors if p['id']==prior_id),None)
            if prior is None:
                prior = {'id':prior_id,'task_family':family,'prototype':prototype,
                         'basis':'first accepted approximate visual structure; poses reinferred per episode; not physical success',
                         'prototype_source_case':case,'accepted_scene_cases':[],'evidence_ids':[]}
                priors.append(prior)
            prior['accepted_scene_cases'] = sorted(set(prior['accepted_scene_cases']+[case]))
            prior['evidence_ids'].append(identity)
            state['events'].append(identity);state['evidence_count']+=1;state['revision']+=1
            state['training_cases']=sorted(set(state['training_cases']+[case]))
            write(path,state)

    def resolve(self, stage, decision, case):
        if not decision.get('diagnosis') or not decision.get('hypothesis'):
            raise ValueError('Diagnosis and hypothesis are required')
        if decision.get('stop') is True:
            return None, {'stop': True, 'diagnosis': decision['diagnosis']}
        if bool(decision.get('operator_id')) == bool(decision.get('operator_proposal')):
            raise ValueError('Choose one existing operator or propose one new operator')
        state = check_stage(read(self.root/(stage+'.json')), stage)
        if decision.get('operator_proposal'):
            operator = validate_operator(decision['operator_proposal'], stage)
        else:
            from rssr_operator_policy import all_definitions
            available=list(all_definitions(state).values()) if state.get('operator_policy') else state['operators']
            operator = next((o for o in [*available, *self.transient[stage].values()]
                             if o['id'] == decision['operator_id']), None)
            if operator is None:
                raise ValueError('Unknown operator')
        arguments = decision.get('arguments', {})
        patch = instantiate(operator, arguments)
        # Validation happens before even transient insertion. A proposal is not
        # a learned operator until the simulator/visual reviewer measures gain.
        self.transient[stage][operator['id']] = copy.deepcopy(operator)
        return patch, {'operator_id': operator['id'], 'operator_definition': copy.deepcopy(operator),
                       'arguments': arguments, 'diagnosis': decision['diagnosis'],
                       'hypothesis': decision['hypothesis'], 'knowledge_version': digest(state),
                       'mode': self.mode, 'case': case}

    def record(self, stage, case, trace, before, after, context, passed=False, admission_evidence=None):
        if self.mode == 'infer' or not trace or not trace.get('operator_id'):
            return
        if len(before) != len(after) or not before or any(type(x) not in (int, float) or not math.isfinite(x) for x in [*before, *after]):
            raise ValueError('Finite comparable observed scores required')
        context = normalize_context(context)
        item = {'stage': stage, 'case': case, 'trace': trace, 'before': list(before), 'after': list(after),
                'context': context, 'passed': bool(passed), 'improved': tuple(after) < tuple(before)}
        observation_id=digest(item)
        if admission_evidence is not None:item['admission_evidence']=copy.deepcopy(admission_evidence)
        identity = digest(item)
        with self.locked():
            path = self.root/(stage+'.json')
            state = read(path)
            if identity in state['events']:
                return
            if state.get('operator_policy'):
                from rssr_operator_policy import observe
                if observation_id in state.get('observation_ids',[]):return
                event_path=self.root/'evidence'/stage/(identity+'.json')
                if event_path.exists() and read(event_path)!=item:raise ValueError('Evidence identity collision')
                write(event_path,item)
                observe(state,item,identity)
                if self.maintenance_policy:
                    from rssr_library_lifecycle import update
                    from rssr_operator_policy import refresh
                    before=digest(state);changes=update(state,item,identity);refresh(state)
                    write(self.root/'maintenance'/stage/(identity+'.json'),{
                        'policy':self.maintenance_policy,'evidence_id':identity,'case':case,
                        'state_before_sha256':before,'state_after_sha256':digest(state),'changes':changes,
                        'agent_written':True,'human_entries_added':False})
                state.setdefault('observation_ids',[]).append(observation_id)
                state['events'].append(identity);state['evidence_count']+=1;state['revision']+=1
                state['training_cases']=sorted(set(state['training_cases']+[case]))
                check_stage(state,stage)
                write(path,state)
                return
            if context.get('compiled_program_policy'):
                from rssr_success_recipe import POLICY
                if stage != 'retarget' or context['compiled_program_policy'] != POLICY:
                    raise ValueError('Unknown composed-program experience policy')
                # Admission and deduplication share the same writer lock.
                # Concurrent successes cannot create two prototypes for one
                # compatibility class; repeated success of the same program
                # contributes transfer evidence instead of a new capability.
                existing = next((e for e in state['experiences']
                                 if e['context'] == context and e['successful_cases']), None)
                if existing is not None and existing['operator_id'] != trace['operator_id']:
                    return
                if existing is not None and passed and case in existing['successful_cases']:
                    return
            # Write event before index: a crash can replay this exact event safely.
            event_path = self.root/'evidence'/stage/(identity+'.json')
            if event_path.exists() and read(event_path) != item:
                raise ValueError('Evidence identity collision')
            write(event_path, item)
            state['events'].append(identity)
            state['evidence_count'] += 1
            state['training_cases'] = sorted(set(state['training_cases'] + [case]))
            operator = copy.deepcopy(trace['operator_definition'])
            if item['improved']:
                if not any(o['id'] == operator['id'] for o in state['operators']):
                    operator['effect_validated'] = True
                    state['operators'].append(operator)
                key = digest({'operator': operator['id'], 'context': context})
                experience = next((e for e in state['experiences'] if e['id'] == key), None)
                if experience is None:
                    experience = {'id': key, 'stage': stage, 'operator_id': operator['id'],
                                  'mechanism': operator['mechanism'], 'preconditions': operator['preconditions'],
                                  'context': context, 'arguments': trace['arguments'], 'uses': 0,
                                  'improved_cases': [], 'successful_cases': [], 'evidence_ids': []}
                    state['experiences'].append(experience)
                experience['uses'] += 1
                experience['improved_cases'] = sorted(set(experience['improved_cases'] + [case]))
                if passed:
                    experience['successful_cases'] = sorted(set(experience['successful_cases'] + [case]))
                experience['transfer_verified'] = len(experience['improved_cases']) >= 2
                experience['evidence_ids'].append(identity)
            else:
                negative = {'id': identity, 'context': context, 'operator_id': operator['id'],
                            'operator_definition': operator, 'arguments': trace['arguments'],
                            'before': before, 'after': after, 'case': case}
                state['negative_examples'] = (state['negative_examples'] + [negative])[-120:]
            state['revision'] += 1
            write(path, state)

    def export(self, destination, coverage):
        if self.mode != 'train':
            raise ValueError('Inference cannot publish a trained checkpoint')
        expected = coverage.get('expected_case_ids', [])
        processed = coverage.get('processed_case_ids', [])
        if not expected or len(expected) != len(set(expected)) or len(processed) != len(set(processed)) or set(expected) != set(processed):
            raise ValueError('All declared training cases must be accounted for before freezing')
        if coverage.get('execution_error_case_ids') or coverage.get('unsupported_case_ids'):
            raise ValueError('Execution errors or unsupported tasks are incomplete training, not completed failures')
        stages = {}
        destination = pathlib.Path(destination)
        with self.locked():
            for stage in STAGES:
                state = check_stage(read(self.root/(stage+'.json')), stage)
                path = destination/(stage+'.json')
                if path.exists() and read(path) != state:
                    raise ValueError('Checkpoint destination is immutable')
                write(path, state)
                stages[stage] = {'path': path.name, 'sha256': sha(path)}
            body = {'schema': 'rssr.knowledge.checkpoint.v2', 'stages': stages,
                    'coverage': coverage, 'initialization': read(self.root/'initialization.json')}
            body['checkpoint_id'] = digest(body)
            path = destination/'checkpoint.json'
            if path.exists() and read(path) != body:
                raise ValueError('Checkpoint metadata changed')
            write(path, body)
        return path
