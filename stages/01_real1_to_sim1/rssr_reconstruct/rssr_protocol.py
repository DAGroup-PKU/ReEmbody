"""Versioned stage messages: compact evidence, strict outputs, legacy tool adapter.

This module never evaluates task success, edits candidates or writes knowledge.
Full input remains an audit artifact. Only presentation to the LLM is compacted.
"""
import copy
import hashlib
import json
import math
import pathlib
import sys

try:
    from jsonschema import Draft202012Validator
except ModuleNotFoundError:
    sys.path.insert(0, str(pathlib.Path(__file__).parent / 'vendor'))
    from jsonschema import Draft202012Validator

from jsonschema.exceptions import ValidationError

VERSION = 'rssr.stage.v1'
STAGES = ('scene.initialize', 'scene.review', 'reconstruction.repair',
          'camera.observe', 'camera.review', 'camera.repair', 'retarget.repair')
ROLES = ('head', 'left_wrist', 'right_wrist')
SCENE_CRITERIA = ('object_identity', 'relative_layout', 'geometry_scale', 'table_support')
TOOLS = {'reconstruction.repair': 'scene_patch', 'camera.repair': 'camera_patch',
         'retarget.repair': 'retarget_patch'}


class ProtocolError(ValueError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def parse(text):
    def invalid(value):
        raise ProtocolError('Nonfinite JSON value: ' + value)
    # Markdown, multiple JSON messages and duplicate keys are rejected.
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=invalid)


def obj(properties, required=None):
    return {'type': 'object', 'properties': properties, 'required': list(properties) if required is None else required,
            'additionalProperties': False}


def string(n=300):
    return {'type': 'string', 'maxLength': n}


def array(items, n=100, minimum=0):
    return {'type': 'array', 'items': items, 'minItems': minimum, 'maxItems': n}


def vector(n, low=None, high=None):
    item = {'type': 'number'}
    if low is not None: item['minimum'] = low
    if high is not None: item['maximum'] = high
    return array(item, n, n)


BOOL = {'type': 'boolean'}
NULL = {'type': 'null'}


def data_schema(stage):
    if stage in TOOLS:
        bound = {'oneOf': [obj({'min': {'type': 'number'}, 'max': {'type': 'number'}}),
                           obj({'enum': array({'type': ['string', 'number']}, 20, 1)})]}
        operator = obj({'name': string(120), 'mechanism': string(300), 'preconditions': string(300),
                        'tool': {'const': TOOLS[stage]},
                        'parameters': {'type': 'object', 'maxProperties': 32, 'additionalProperties': bound},
                        'template': {'type': 'object', 'minProperties': 1}})
        result=obj({'diagnosis': dict(string(400), minLength=1),
                    'hypothesis': dict(string(300), minLength=1), 'stop': BOOL,
                    'operator_id': {'type': ['string', 'null'], 'maxLength': 100},
                    'operator_proposal': {'oneOf': [NULL, operator]},
                    'arguments': {'type': 'object', 'maxProperties': 32}})
        if stage=='retarget.repair':
            result['properties']['base_candidate']={'enum':['incumbent','exploration']}
        return result
    if stage == 'scene.initialize':
        shape = obj({'id': {'enum': ['basket', 'blue_cube', 'red_cube', 'green_cube']},
                     'shape': {'enum': ['box', 'open_box']}, 'position_m': vector(3),
                     'size_m': vector(3, .015, .45), 'yaw_rad': {'type': 'number'},
                     'rgba': vector(4, 0, 1), 'mass_kg': {'type': 'number'}, 'wall_m': {'type': 'number'}},
                    ['id', 'shape', 'position_m', 'size_m', 'yaw_rad', 'rgba', 'mass_kg'])
        scene = obj({'container_id': {'const': 'basket'}, 'movable_id': {'const': 'blue_cube'},
                     'table_shift_z_m': {'const': 0}, 'objects': array(shape, 4, 2)})
        from rssr_scene import schema as scene_schema
        return obj({'supported': BOOL, 'reason': string(400), 'scene': {'oneOf': [NULL, scene, scene_schema()]}})
    if stage == 'scene.review':
        criterion = obj({'pass': BOOL, 'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                         'evidence': dict(string(300), minLength=1)})
        return obj({'criteria': obj({k: criterion for k in SCENE_CRITERIA}),
                    'camera_issues': array(string(200), 8), 'scene_issues': array(string(200), 8),
                    'suggested_diagnosis': string(300)})
    if stage == 'camera.observe':
        landmark = obj({'object_id': string(80), 'visible': BOOL,
                        'center_xy_normalized': {'oneOf': [NULL, vector(2, 0, 1)]},
                        'reason': dict(string(240), minLength=1)})
        landmark['properties']['bbox_xyxy_normalized'] = {'oneOf': [NULL, vector(4, 0, 1)]}
        return obj({'views': array(obj({'view': {'enum': list(ROLES)},
                                       'landmarks': array(landmark, 40)}), 3, 3)})
    if stage == 'camera.review':
        view = obj({'kind': {'enum': ['sim1', 'sim2']}, 'view': {'enum': list(ROLES)},
                    'orientation_consistent': BOOL, 'fov_adequate': BOOL, 'role_correct': BOOL,
                    'key_objects_visible': BOOL, 'reason': dict(string(300), minLength=1)})
        return obj({'images_visible': BOOL, 'views': array(view, 6, 3), 'unresolved': array(string(240), 8)})
    raise ProtocolError('Unknown stage: ' + stage)


def response_schema(stage):
    return dict(obj({'protocol': {'const': VERSION}, 'request_id': string(64),
                     'stage': {'const': stage}, 'data': data_schema(stage)}),
                **{'$schema': 'https://json-schema.org/draft/2020-12/schema'})


INPUT_FIELDS = {
    'scene.initialize': ('task action_geometry_hint table', 'knowledge scene_contract task_id'),
    'scene.review': ('task scene numeric image_order', ''),
    'reconstruction.repair': ('best_scene best_verdict best_numeric history tool_contract knowledge', ''),
    'camera.observe': ('task image_order objects', ''),
    'camera.review': ('task available_stages source_frame_indices source_task_success_required diagnostics not_measured_calibration', 'frame_mapping camera_phase'),
    'camera.repair': ('task tool_contract camera_profiles diagnostics history remaining_candidates first_frame_landmarks instructions knowledge', 'camera_origin frame_mapping camera_phase latest_rejected_camera image_order'),
    'retarget.repair': ('task best_program scene best_report best_objective history tool_contract remaining_evaluations knowledge', 'latest_rejected_candidate exploration_candidate'),
}


def request_schema(stage=None):
    if stage is None:
        return {'$schema':'https://json-schema.org/draft/2020-12/schema',
                'oneOf':[request_schema(s) for s in STAGES]}
    required, optional = (x.split() for x in INPUT_FIELDS[stage])
    types = {}
    for key in required + optional:
        if key in ('task','instructions','camera_phase','task_id','scene_contract'): value = {'type':'string'}
        elif key in ('history','image_order','objects','available_stages','source_frame_indices'): value = {'type':'array'}
        elif key in ('source_task_success_required','not_measured_calibration'): value = BOOL
        elif key in ('remaining_evaluations','remaining_candidates'): value = {'type':'integer','minimum':0}
        else: value = {'type':'object'}
        types[key] = value
    context = obj({k:string(500) for k in ('case_id','full_evidence_sha256','candidate_sha256','frame',
                   'length_unit','angle_unit','camera_axes','calibration','sim2_success','state_trace_presentation')} |
                   {'memory_version': {'type':['string','null']},'sim1_success_required':{'const':False}})
    return dict(obj({'protocol': {'const': VERSION}, 'request_id': string(64),
                     'stage': {'const': stage}, 'context': context,
                     'input': obj(types,required),
                     'assets': array(obj({'id': string(30), 'label': string(180), 'sha256': string(64)}), 16)}),
                **{'$schema': 'https://json-schema.org/draft/2020-12/schema'})


def pack_rows(rows, rounding=None):
    """Lossless column encoding; optional rounding only for large state traces.

    Explicit null, false, zero and a missing field remain distinct via present.
    Coordinates/orientations/constraint diagnostics are never inferred.
    """
    columns, flat_rows = [], []
    def flatten(value, prefix=()):
        if isinstance(value, dict) and value:
            result = {}
            for key, child in value.items(): result.update(flatten(child, prefix + (key,)))
            return result
        return {prefix: value}
    def rounded(value):
        if isinstance(value, float): return round(value, rounding) if rounding is not None else value
        if isinstance(value, list): return [rounded(x) for x in value]
        if isinstance(value, dict): return {k: rounded(v) for k, v in value.items()}
        return value
    for row in rows:
        flat = flatten(row);flat_rows.append(flat)
        for key in flat:
            if key not in columns: columns.append(key)
    result = {'encoding': 'columns.v1', 'columns': [list(k) for k in columns],
              'rows': [[rounded(row.get(k)) for k in columns] for row in flat_rows]}
    if any(len(row) != len(columns) for row in flat_rows):
        result['present'] = [[i for i, k in enumerate(columns) if k in row] for row in flat_rows]
    if rounding is not None: result['decimal_places'] = rounding
    return result


def unpack_rows(packed):
    rows = []
    for index, values in enumerate(packed['rows']):
        out = {};present = packed.get('present', [list(range(len(values)))] * len(packed['rows']))[index]
        for i in present:
            keys = packed['columns'][i];parent = out
            if not keys:
                out = copy.deepcopy(values[i]);continue
            for key in keys[:-1]: parent = parent.setdefault(key, {})
            parent[keys[-1]] = values[i]
        rows.append(out)
    return rows


def _history(rows):
    out = []
    for row in rows:
        keep = {k: copy.deepcopy(v) for k, v in row.items() if k in
                ('trial', 'round', 'candidate', 'score', 'before_score', 'accepted', 'passed', 'stopped',
                 'invalid_operator', 'invalid_proposal', 'duplicate_proposal', 'tool_error', 'objective',
                 'diagnostic_only', 'probe', 'experiment', 'not_task_evidence', 'physical_failure_evidence',
                 'camera_feedback','retained_measured_relation_progress_not_task_success','retained_measured_holding_progress_not_task_success',
                 'search_control_trial','retained_for_exploration_only')}
        trace = row.get('trace') or {}
        keep.update({k: copy.deepcopy(trace[k]) for k in ('operator_id', 'arguments', 'diagnosis', 'hypothesis','base_candidate') if k in trace})
        out.append(keep)
    return out


def _execution_example(example):
    keys=('control_model_profile','observed_result_model_profile','scope','standalone_patch_from_default_verified',
          'policy','default_model_control_observed','control_action_program','observed_result_action_program',
          'control_scene_sha256','observed_result_scene_sha256','current_control_comparison')
    out={k:copy.deepcopy(example[k]) for k in keys if k in example}
    # Store full executed conditions in evidence, but do not send identical
    # before/after programs and profiles twice. Missing stays distinct from equal.
    aliases={}
    if out.get('policy')=='observed_execution_conditions_v2':
        for control,result in (('control_model_profile','observed_result_model_profile'),
                               ('control_action_program','observed_result_action_program'),
                               ('control_scene_sha256','observed_result_scene_sha256')):
            if control in out and result in out and out[control]==out[result]:
                out.pop(result);aliases[result]=control
    if aliases:out['identical_to_control_field']=aliases
    out['supporting_demonstrations']=len(example.get('cases',[]))
    return out


def _knowledge(knowledge, history):
    # Referenced historical operator definitions must remain available once,
    # even when absent from the library's recent-operator snapshot.
    catalog = {o['id']: o for o in knowledge.get('operators', [])}
    for row in history:
        op = (row.get('trace') or {}).get('operator_definition')
        if op: catalog.setdefault(op['id'], op)
    for row in knowledge.get('negative_examples', []):
        op = row.get('operator_definition')
        if op: catalog.setdefault(op['id'], op)
    op_keys = ('id', 'name', 'mechanism', 'preconditions', 'tool', 'parameters', 'template', 'effect_validated',
               'family_id','admission_state','validated_demonstrations')
    result = {'version': knowledge.get('version'),
            'operators': [{k: o[k] for k in op_keys if k in o} for o in catalog.values()],
            'experiences': [{k: e[k] for k in ('id', 'operator_id', 'arguments', 'context', 'transfer_verified',
                            'evidence_levels','attempt_count','interaction_gain_count','task_success_count','numeric_only_count',
                            'model_health_gain_count','prediction_matched_count','admission_state','retrieval_outcomes',
                            'selected_recipe_improved_case_count') if k in e} |
                            {'improved_case_count': len(e.get('improved_cases', [])),
                             'successful_case_count': len(e.get('successful_cases', [])),
                             'selected_recipe_successful_case_count':len(e.get('selected_recipe_successful_cases',[])),
                             'execution_examples': [_execution_example(x) for x in e.get('execution_examples',[])]}
                            for e in knowledge.get('experiences', [])],
            'negative_examples': [{k:e[k] for k in ('id','operator_id','context','arguments','before','after','experiment','hypothesis','admission_rejection') if k in e}
                                  for e in knowledge.get('negative_examples', [])]}
    if 'scene_priors' in knowledge:
        result['scene_priors'] = copy.deepcopy(knowledge['scene_priors'])
    if 'admission_policy' in knowledge:result['admission_policy']=copy.deepcopy(knowledge['admission_policy'])
    if 'library_lifecycle' in knowledge:result['library_lifecycle']=copy.deepcopy(knowledge['library_lifecycle'])
    if 'observed_model_reuse_options' in knowledge:
        result['observed_model_reuse_options']=copy.deepcopy(knowledge['observed_model_reuse_options'])
    if 'trajectory_priors' in knowledge:
        result['trajectory_priors']=[{k:p[k] for k in ('id','title','trajectory_success_count','evidence_level','current_object_task_verified')}|
            {'calls':[{'operator_id':c['definition']['id'],'scope':c['scope'],'parameters':c['parameters']}
                      for c in p['patch']['solver_operations']]}
            for p in sorted(knowledge['trajectory_priors'],key=lambda p:-p['trajectory_success_count'])[:8]]
    return result


def _camera_numeric(numeric):
    result = copy.deepcopy(numeric)
    if isinstance(result.get('observations'), list): result['observations'] = pack_rows(result['observations'])
    return result


def compact(stage, payload):
    result = copy.deepcopy(payload)
    history = payload.get('history', [])
    if 'knowledge' in payload: result['knowledge'] = _knowledge(payload['knowledge'], history)
    if 'history' in payload: result['history'] = _history(history)
    if stage == 'scene.initialize':
        hint=result['action_geometry_hint']
        hint.pop('task',None);hint.pop('render_calls',None)
        # Same source evidence, presented to a coarse visual initializer. Full
        # source geometry remains archived and is used unchanged by physics.
        def rounded(value):
            if isinstance(value,float):return round(value,6)
            if isinstance(value,list):return [rounded(v) for v in value]
            if isinstance(value,dict):return {k:rounded(v) for k,v in value.items()}
            return value
        result['action_geometry_hint']=rounded(hint)
    if stage == 'retarget.repair':
        reports=[result['best_report']]
        if 'latest_rejected_candidate' in result:
            reports.append(result['latest_rejected_candidate']['report'])
        if 'exploration_candidate' in result:
            reports.append(result['exploration_candidate']['report'])
        for report in reports:
            if isinstance(report.get('state_trace'), list): report['state_trace'] = pack_rows(report['state_trace'], 6)
            for key in ('scene_sha256', 'wall_seconds', 'simulation_seconds', 'render_calls'):
                report.pop(key, None)
            ik = report.get('ik')
            if ik is not None:
                report['ik'] = {k: ik[k] for k in ('defaults', 'mean_errors_left_mm_deg_right_mm_deg',
                                                 'max_adjacent_joint_step_rad', 'source_reference','grasp_frame_definitions') if k in ik}
    if stage == 'camera.repair':
        pairs = [(result['camera_profiles'],result['diagnostics'])]
        if 'latest_rejected_camera' in result:
            rejected = result['latest_rejected_camera']
            pairs.append((rejected['profiles'],rejected['report']))
        for profiles,report in pairs:
            for cameras in profiles.values():
                for name, spec in cameras.items():
                    cameras[name] = {k: spec[k] for k in ('parent_body', 'position_in_parent_m', 'rotation_in_parent', 'fovy_degrees')}
            report.pop('images', None)
            if 'numerical' in report: report['numerical'] = _camera_numeric(report['numerical'])
    if stage == 'camera.review': result['diagnostics'] = _camera_numeric(result['diagnostics'])
    if stage == 'reconstruction.repair':
        result.get('best_numeric', {}).pop('objects', None)  # same objects in best_scene
        result.get('best_numeric', {}).pop('model_sha256', None)
    if stage == 'scene.review':
        result.get('numeric', {}).pop('objects', None)  # same objects in scene
        result.get('numeric', {}).pop('model_sha256', None)
    return result


def _shape(schema):
    """Small wire contract; complete JSON Schemas are stored with the code."""
    if 'const' in schema: return schema['const']
    if 'enum' in schema: return {'enum': schema['enum']}
    if 'oneOf' in schema: return {'oneOf': [_shape(s) for s in schema['oneOf']]}
    kind = schema.get('type')
    if kind == 'object':
        if 'properties' in schema:
            result = {k: _shape(v) for k, v in schema['properties'].items()}
            optional = set(result) - set(schema['required'])
            if optional: result['optional_fields'] = sorted(optional)
            return result
        return {'type': 'object', **({'values': _shape(schema['additionalProperties'])} if isinstance(schema.get('additionalProperties'), dict) else {})}
    if kind == 'array': return {'array_of': _shape(schema['items']), 'min': schema['minItems'], 'max': schema['maxItems']}
    return {k: v for k, v in schema.items() if k in ('type', 'maxLength', 'minLength', 'minimum', 'maximum')}


def build_request(config, stage, payload, image_assets):
    if stage not in STAGES: raise ProtocolError('Unknown stage')
    evidence_hash = digest(payload)
    current = payload.get('best_program', payload.get('camera_profiles', payload.get('best_scene', payload.get('scene'))))
    context = {'case_id': config['case_id'], 'full_evidence_sha256': evidence_hash,
               'candidate_sha256': digest(current), 'memory_version': payload.get('knowledge', {}).get('version'),
               'frame': 'simulation_world_with_shared_table', 'length_unit': 'm', 'angle_unit': 'rad_unless_field_says_degrees',
               'camera_axes': '+X right,+Y up,-Z forward; rotation local-to-parent',
               'calibration': 'estimated_not_hardware_measured', 'sim1_success_required': False,
               'sim2_success': 'all_original_physics_predicates_and_robot_constraints',
               'state_trace_presentation': '6 decimals; hard-constraint metrics and thresholds retain full precision'}
    request = {'protocol': VERSION, 'stage': stage, 'context': context,
               'input': compact(stage, payload), 'assets': image_assets}
    request['request_id'] = 'rssr_' + digest(request)[:24]
    Draft202012Validator(request_schema(stage)).validate(request)
    return request


def system_prompt(original, stage):
    if stage=='camera.observe':
        from camera_landmarks import LANDMARK_PROMPT
        original = LANDMARK_PROMPT
    text=original+'\n只返回规定的JSON数据，不返回Markdown、工具调用或封装元信息；理由短写，reason/evidence最多240字符。'
    if stage in ('camera.repair','camera.review','retarget.repair'):
        text+=(' columns.v1的columns为字段路径，rows为数值；present若存在则给出该行实际存在的列。'
               'null、缺失、false和0不同。相机局部轴+X右/+Y上/-Z前，安装旋转为local-to-parent；米/弧度，标degrees字段用度。')
    if stage in TOOLS:
        text+=(' 必填字段：'+','.join(data_schema(stage)['required'])+'，禁止额外字段。diagnosis≤400字符，'
               'hypothesis/mechanism/preconditions≤240字符，name≤120字符。operator_id和operator_proposal不用的设null；'
               'stop=true时两者为null且arguments={}；否则二选一。')
    if stage=='retarget.repair':
        text+=(' base_candidate也是允许的顶层字段，不属于arguments或operator_proposal.template。'
               '取值只能是incumbent或exploration。若输入tool_contract.require_explicit_search_base=true、'
               '存在exploration_candidate且stop=false，必须显式填写base_candidate；'
               '诊断与修改均须针对所选候选，不能省略后让执行器猜测。没有探索候选时使用incumbent或省略。')
    if stage=='scene.initialize':text+=' supported=false时scene=null。'
    return text


def validate_response(response, request):
    dumps(response)  # also reject NaN/Infinity supplied as a Python object
    validator = Draft202012Validator(response_schema(request['stage']))
    errors = list(validator.iter_errors(response))
    if errors:
        from rssr_schema_feedback import explain_errors
        raise ProtocolError(explain_errors(errors))
    if response['request_id'] != request['request_id']: raise ProtocolError('Response belongs to another request')
    data, stage, payload = response['data'], request['stage'], request['input']
    if (stage=='retarget.repair' and not data['stop'] and payload.get('exploration_candidate') is not None
            and payload.get('tool_contract',{}).get('require_explicit_search_base')
            and 'base_candidate' not in data):
        raise ProtocolError('Explicit base_candidate is required when both incumbent and exploration candidates are offered; choose the state your diagnosis and edits refer to.')
    if stage in TOOLS:
        if data['stop']:
            if data['operator_id'] is not None or data['operator_proposal'] is not None or data['arguments']:
                raise ProtocolError('Stop decision cannot contain an edit')
        else:
            if bool(data['operator_id']) == bool(data['operator_proposal']):
                raise ProtocolError('Exactly one existing operator or new operator is required')
            from pilot_learning import validate_operator, instantiate
            if data['operator_proposal']:
                operator = validate_operator(data['operator_proposal'], stage.split('.')[0])
            else:
                operator = next((o for o in payload.get('knowledge', {}).get('operators', []) if o['id'] == data['operator_id']), None)
                if operator is None: raise ProtocolError('Unknown operator ID')
            instantiate(operator, data['arguments'])
    elif stage == 'scene.initialize':
        if data['supported']:
            if data['scene'].get('schema')=='rssr.scene.v2':
                from rssr_scene import validate
                validate(data['scene'])
                if data['scene']['task_id']!=payload.get('task_id'):raise ProtocolError('Scene belongs to a different task')
            else:
                if payload.get('scene_contract')=='rssr.scene.v2':raise ProtocolError('Generic task requires generic scene schema')
                from pilot_scene import validate_scene
                validate_scene(data['scene'])
        elif data['scene'] is not None: raise ProtocolError('Unsupported scene must be null')
    elif stage == 'camera.observe':
        views = [v['view'] for v in data['views']]
        if set(views) != set(ROLES) or len(views) != 3: raise ProtocolError('Need three unique camera roles')
        ids = {o['id'] for o in payload['objects']}
        for view in data['views']:
            seen = set()
            for row in view['landmarks']:
                if row['object_id'] not in ids or row['object_id'] in seen: raise ProtocolError('Unknown/duplicate object observation')
                seen.add(row['object_id'])
                from camera_landmarks import validate_landmark
                try: validate_landmark(row)
                except ValueError as error: raise ProtocolError(str(error)) from error
    elif stage == 'camera.review':
        expected = {(kind, view) for kind in payload['available_stages'] for view in ROLES}
        actual = [(v['kind'], v['view']) for v in data['views']]
        if len(actual) != len(expected) or set(actual) != expected: raise ProtocolError('Missing/duplicate or cross-stage camera view')
    return data


def export_schemas(folder):
    folder = pathlib.Path(folder);folder.mkdir(parents=True, exist_ok=True)
    (folder / 'request.schema.json').write_text(json.dumps(request_schema(), ensure_ascii=False, indent=2), encoding='utf8')
    for stage in STAGES:
        schema = response_schema(stage);Draft202012Validator.check_schema(schema)
        (folder / (stage + '.response.schema.json')).write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding='utf8')
        (folder / (stage + '.request.schema.json')).write_text(json.dumps(request_schema(stage), ensure_ascii=False, indent=2), encoding='utf8')


if __name__ == '__main__':
    export_schemas(pathlib.Path(__file__).parent / 'schemas/rssr-stage-v1')
