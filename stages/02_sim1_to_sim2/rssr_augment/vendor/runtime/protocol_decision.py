"""Bounded JSON stage transport; local schema validation, no tool execution."""
import base64
import copy
import json
import os
import pathlib
import time
from openai import OpenAI
from common import read, write, sha, event
from api_scheduler import api_slot
from rssr_protocol import (VERSION, build_request, response_schema, system_prompt,
                           validate_response, parse, dumps, digest, ProtocolError, data_schema, STAGES)


class DecisionUnavailable(RuntimeError):
    """A bounded visit failed; the immutable request may have attempts remaining."""
    pass


class ProviderRequestRejected(RuntimeError):
    """The provider rejected the request; this is not a model JSON mistake."""


class DecisionOutputExhausted(RuntimeError):
    """Two bounded outputs were truncated despite explicit concise feedback."""


def gateway_rejection(raw):
    content = raw.get('content')
    return (isinstance(content,str) and content.startswith('[req_') and
            '**Bad request from AI provider**' in content[:200] and
            'Your request was rejected by the AI provider' in content[:400])


def provider_format(config, stage):
    """Native constrained output where enabled; local validation stays mandatory.

    Live probes cover camera observations and generated operator templates.
    Other stages use the same JSON-schema path, with local validation retained.
    A rejected provider request may use the declared JSON-object fallback;
    every transition is receipted and local semantic validation is unchanged.
    """
    stages = config.get('k3_native_schema_stages', list(STAGES))
    if stage not in stages:
        return {'type': 'json_object'}
    def adapt(value):
        if isinstance(value, dict):
            return {('anyOf' if k == 'oneOf' else k): adapt(v) for k, v in value.items()}
        if isinstance(value, list):
            return [adapt(v) for v in value]
        return value
    return {'type': 'json_schema', 'json_schema': {
        'name': 'rssr_'+stage.replace('.', '_'), 'strict': True,
        'schema': adapt(data_schema(stage))}}


def normalize_explanations(data, stage):
    """Clip explanatory prose only; never repair or change an executable choice.

    The original API response is retained separately. A few excess characters
    in a diagnosis should not require another expensive model decision.
    """
    result, edits = copy.deepcopy(data), []
    def visit(value, schema, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                if key in ('arguments', 'operator_proposal', 'operator_id', 'scene'):
                    continue
                rule=schema.get('properties',{}).get(key,{})
                limit=rule.get('maxLength')
                if key in ('diagnosis','hypothesis','reason','evidence') and isinstance(child,str) and limit and len(child)>limit:
                    edits.append({'path':list(path+(key,)),'original_chars':len(child),'retained_chars':limit})
                    value[key]=child[:limit]
                else:visit(child,rule,path+(key,))
        elif isinstance(value,list):
            for index, child in enumerate(value):visit(child,schema.get('items',{}),path+(index,))
    visit(result,data_schema(stage))
    return result, edits


def normalize_wire_shape(data, stage):
    """Canonicalize only lossless, unambiguous equivalents of a camera map.

    Every role and landmark is retained. Extra or missing roles remain invalid;
    no object coordinates, decisions, executable fields or unknowns are edited.
    """
    # The provider sometimes nests decision-envelope fields inside a new
    # operator definition. Move only known fields with no competing top-level
    # value; never invent values, merge conflicts, or relax schema validation.
    if stage == 'retarget.repair' and isinstance(data, dict):
        proposal = data.get('operator_proposal')
        if isinstance(proposal, dict):
            fields = [k for k in ('arguments', 'base_candidate') if k in proposal]
            if fields and all(k not in data for k in fields):
                result = copy.deepcopy(data)
                for key in fields:
                    result[key] = result['operator_proposal'].pop(key)
                return result, 'operator_nested_envelope_to_top_level:' + ','.join(fields)
    if stage!='camera.observe' or not isinstance(data,dict):
        return data, None
    roles=('head','left_wrist','right_wrist')
    source = data.get('views') if set(data)=={'views'} else data
    if not isinstance(source,dict) or set(source)!=set(roles) or not all(isinstance(source[r],list) for r in roles):
        return data,None
    return {'views':[{'view':r,'landmarks':copy.deepcopy(source[r])} for r in roles]}, 'named_role_map_to_view_array'


def prepare(config, folder, stage, original_prompt, payload, images):
    if config.get('api_identity'):
        from rssr_api_identity import assert_api_identity
        assert_api_identity(config['api_identity'])
    labels = payload.get('image_order', [])
    assets = [{'id': 'image_' + str(i), 'label': str(labels[i]) if i < len(labels) else pathlib.Path(p).stem,
               'sha256': sha(p)} for i, p in enumerate(images)]
    wire = build_request(config, stage, payload, assets)
    prompt = system_prompt(original_prompt, stage)
    request = {'system': prompt, 'payload': wire['input'], 'protocol_envelope':wire,
               'images': {str(p): sha(p) for p in images},
               'model': os.environ.get('KIMI_MODEL', 'Kimi-K3'),
               'provider_response_format': provider_format(config, stage),
               'schema_retry_policy': 'preserve_assistant_message_v1',
               'transport_policy': 'native_schema_then_json_object_on_rejection_v1',
               'truncation_policy': 'one_concise_retry_then_stop_v1',
               'max_tokens': config['object_decision_max_tokens'],
               'reasoning_effort': config.get('k3_reasoning_effort')}
    if config.get('llm_api', 'chat_completions') == 'responses':
        request['llm_api'] = 'responses'
    folder = pathlib.Path(folder);folder.mkdir(parents=True, exist_ok=True)
    if (folder/'input.json').exists() and digest(read(folder/'input.json')) != digest(request):
        raise RuntimeError('Decision identity changed; cannot reuse another input')
    write(folder/'input.json', request)
    write(folder/'full_input.json', {'system': original_prompt, 'payload': payload, 'assets': assets})
    write(folder/'wire_request.json', wire)
    write(folder/'response.schema.json', response_schema(stage))
    write(folder/'protocol_manifest.json', {
        'protocol': VERSION, 'request_id': wire['request_id'], 'stage': stage,
        'schema_sha256': digest(response_schema(stage)),
        'original_text_chars': len(original_prompt) + len(json.dumps(payload,ensure_ascii=False)),
        'wire_text_chars': len(prompt) + len(dumps(wire['input'])),
        'full_input_payload_chars': len(dumps(payload)), 'compact_input_payload_chars': len(dumps(wire['input'])),
        'provider_response_format': request['provider_response_format']['type'],
        'provider_schema_requested': request['provider_response_format']['type']=='json_schema',
        'provider_schema_enforcement_independently_verified': False,
        'local_schema_validation': True, 'acceptance_and_tool_validation_unchanged': True})
    return request


def decide_stage(config, budget, folder, stage, original_prompt, payload, images=()):
    folder = pathlib.Path(folder)
    request = prepare(config, folder, stage, original_prompt, payload, images)
    wire = request['protocol_envelope']
    if (folder/'result.json').exists():
        data = validate_response(read(folder/'wire_response.json'), wire)
        if data != read(folder/'result.json'): raise ProtocolError('Decoded result differs from checked response')
        return data
    # The executor fills protocol/identity metadata. The model only writes
    # stage data; it need not spend tokens copying hashes or request IDs.
    content = [{'type': 'text', 'text': dumps(wire['input'])}]
    for path in images:
        binary = pathlib.Path(path).read_bytes();mime = 'image/png' if binary.startswith(b'\x89PNG') else 'image/jpeg'
        content.append({'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(binary).decode()}})
    extra = {'reasoning_effort': config['k3_reasoning_effort']} if config.get('k3_reasoning_effort') else {}
    messages = [{'role': 'system', 'content': request['system']}, {'role': 'user', 'content': content}]
    client = OpenAI(api_key=os.environ['KIMI_API_KEY'], base_url=os.environ['KIMI_BASE_URL'],
                    timeout=config['k3_timeout_s'], max_retries=0)
    schema_issue = None
    response_format = copy.deepcopy(request['provider_response_format'])
    truncations = 0
    created=0
    for attempt in range(config.get('k3_max_attempts_per_decision_total',config['k3_attempts_per_decision'])):
        raw_path = folder/f'raw_{attempt:02d}.json';receipt = folder/f'request_{attempt:02d}.json'
        error_path = folder/f'error_{attempt:02d}.json'
        # Replay transport adaptation, never replay a provider error as an
        # assistant answer. No billed attempt is silently sent a second time.
        if error_path.exists() and not raw_path.exists():
            failure = read(error_path)
            if failure.get('status') in (400,401,403,404,422):
                if failure.get('status') in (400,422) and response_format['type']=='json_schema':
                    response_format={'type':'json_object'}
                else:
                    raise ProviderRequestRejected('Provider rejected the bounded request; see '+str(error_path))
            continue
        if raw_path.exists():
            raw = read(raw_path)
        else:
            if receipt.exists(): continue  # uncertain request remains charged; never silently reissue it
            if created>=config['k3_attempts_per_decision']:break
            # Retry, recovery and cache lookup do not monopolize a shared
            # slot. Preserve the existing slot paths and cumulative ledger.
            with api_slot(config) as scheduling:
                sequence = budget.reserve('k3', config['case_id'], str(folder))
                created+=1
                write(receipt, {'sequence': sequence, 'at': time.time(), 'request_id': wire['request_id'],
                                'messages_sha256': digest(messages), 'message_count':len(messages),
                                'response_format_type':response_format['type'],
                                'response_format_sha256':digest(response_format),
                                'scheduling':dict(scheduling,scope='one_network_request')})
                event(pathlib.Path(config['run_root']), 'stage_json_started', stage=stage, sequence=sequence,
                      request_id=wire['request_id'], images=len(images), queue_wait_s=scheduling['queue_wait_s'])
                started = time.monotonic()
                try:
                    if request.get('llm_api') == 'responses':
                        from responses_transport import create
                        raw = create(client, request, messages, response_format)
                    else:
                        response = client.chat.completions.create(model=request['model'],
                            messages=copy.deepcopy(messages),
                            response_format=response_format, temperature=1., stream=False,
                            max_tokens=request['max_tokens'], extra_body=extra)
                        choice = response.choices[0]
                        raw = {'content': choice.message.content, 'finish_reason': choice.finish_reason,
                               'assistant_message': choice.message.model_dump(exclude_none=True),
                               'provider_model':getattr(response,'model',None),
                               'provider_response_id':getattr(response,'id',None),
                               'usage': response.usage.model_dump() if response.usage else None,
                               'tool_calls': [t.model_dump() for t in choice.message.tool_calls] if choice.message.tool_calls else None}
                    raw.update(latency_s=time.monotonic()-started, schema_retry_feedback=schema_issue)
                    write(raw_path, raw)
                except Exception as error:
                    status = getattr(error, 'status_code', None)
                    body = getattr(error,'body',None)
                    detail=body.get('error',body) if isinstance(body,dict) else {}
                    detail=detail if isinstance(detail,dict) else {}
                    write(error_path, {'type': type(error).__name__, 'status':status,
                          'provider_code':str(detail.get('code',''))[:160],
                          'provider_type':str(detail.get('type',''))[:160],
                          'response_format_type':response_format['type'],
                          'latency_s': time.monotonic()-started})
                    if status in (400,401,403,404,422):
                        if status in (400,422) and response_format['type']=='json_schema':
                            write(folder/f'transport_adaptation_{attempt:02d}.json',{
                                'reason':'provider_request_rejected','from':'json_schema','to':'json_object',
                                'local_validation_unchanged':True,'charged_attempt':attempt})
                            response_format={'type':'json_object'}
                        else:
                            raise ProviderRequestRejected('Provider rejected the bounded request; see '+str(error_path)) from error
                    continue
        if gateway_rejection(raw):
            write(folder/f'transport_rejection_{attempt:02d}.json',{
                'kind':'gateway_embedded_provider_rejection','raw_sha256':sha(raw_path),
                'model_answer':False,'response_format_type':response_format['type']})
            if response_format['type']=='json_schema':
                response_format={'type':'json_object'}
                write(folder/f'transport_adaptation_{attempt:02d}.json',{
                    'reason':'gateway_embedded_provider_rejection','from':'json_schema','to':'json_object',
                    'local_validation_unchanged':True,'charged_attempt':attempt})
                continue
            raise ProviderRequestRejected('Provider also rejected JSON-object transport; see '+str(raw_path))
        if raw.get('finish_reason')=='length':
            truncations+=1
            write(folder/f'validation_{attempt:02d}.json',{'passed':False,
                  'error':'Output token budget exhausted before a complete decision',
                  'truncation_count':truncations,'partial_answer_used':False})
            if truncations>=2:
                raise DecisionOutputExhausted('Two outputs exhausted the token limit; original evidence retained')
            messages.append({'role':'user','content':
                'The preceding request exhausted its output token budget before returning complete JSON. '
                'No partial answer was executed. Make one bounded next decision from the supplied evidence; '
                'do not plan the entire remaining experiment or repeatedly compare speculative alternatives. '
                'Keep the diagnosis and hypothesis concise, and return the complete required JSON now. '
                'All original evidence, tool bounds and acceptance criteria remain unchanged.'})
            continue
        try:
            if raw['finish_reason'] != 'stop' or raw.get('tool_calls'):
                raise ProtocolError('Expected one final JSON message; finish_reason=' + str(raw['finish_reason']))
            canonical, shape_edit = normalize_wire_shape(parse(raw['content']),stage)
            normalized, edits = normalize_explanations(canonical,stage)
            if shape_edit:
                write(folder/f'wire_shape_normalization_{attempt:02d}.json',
                      {'raw_sha256':sha(raw_path),'normalization':shape_edit,'values_changed':False})
            response = {'protocol':VERSION,'request_id':wire['request_id'],'stage':stage,
                        'data':normalized}
            if edits:
                write(folder/f'explanation_normalization_{attempt:02d}.json',
                      {'raw_sha256':sha(raw_path),'edits':edits,'executable_fields_changed':False})
            data = validate_response(response, wire)
            write(folder/'wire_response.json', response);write(folder/'result.json', data)
            write(folder/'validation.json', {'passed': True, 'request_id': wire['request_id'],
                                             'stage': stage, 'attempt': attempt})
            return data  # legacy tools still enforce numerical bounds and task predicates
        except (ValueError, TypeError, KeyError) as error:
            schema_issue = str(error)[:1200]
            write(folder/f'validation_{attempt:02d}.json', {'passed': False, 'error': schema_issue})
            previous = raw.get('assistant_message')
            # K3 requires its complete assistant message, including provider
            # reasoning fields, when continuing a conversation. Reconstruct
            # this same history from saved raw replies on a worker restart.
            # Interrupted answers/tool calls do not establish a complete turn.
            if (raw['finish_reason']=='stop' and not raw.get('tool_calls') and
                    isinstance(previous,dict) and previous.get('role')=='assistant'):
                messages.append(copy.deepcopy(previous))
                messages.append({'role':'user','content':
                    'Your preceding answer failed the fixed interface validation: '+schema_issue+
                    '\nReturn the corrected complete JSON object for the same input. '
                    'Keep valid executable choices unchanged unless the error requires a correction. '
                    'Do not change observations, tool bounds or success criteria.'})
    raise DecisionUnavailable('K3 bounded stage decision failed; preserved request and validation receipts')
