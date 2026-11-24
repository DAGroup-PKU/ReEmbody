"""Responses wire adapter; keep RSSR evidence, validation and budgets unchanged."""
import copy
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import time


@contextmanager
def request_slot():
    """Share an optional batch-wide request limit across case processes."""
    directory = os.environ.get('RSSR_API_SLOT_DIRECTORY')
    if not directory:
        yield
        return
    count = int(os.environ.get('RSSR_API_SLOT_COUNT', '14'))
    if not 1 <= count <= 100:
        raise ValueError('Invalid shared request concurrency')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    selected = None
    while selected is None:
        for index in range(count):
            stream = (root / f'slot_{index:02d}.lock').open('a')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                continue
            selected = stream
            break
        if selected is None:
            time.sleep(.2)
    try:
        yield
    finally:
        selected.close()


def input_messages(messages):
    result = []
    for message in messages:
        role = message['role']
        if role not in ('system', 'developer', 'user', 'assistant'):
            raise ValueError('Unsupported Responses message role')
        content = message['content']
        if isinstance(content, str):
            result.append({'role': role, 'content': content})
            continue
        parts = []
        for part in content:
            if part['type'] == 'text':
                parts.append({'type': 'input_text', 'text': part['text']})
            elif part['type'] == 'image_url' and role == 'user':
                image = part['image_url']
                converted = {'type': 'input_image', 'image_url': image['url']}
                if 'detail' in image: converted['detail'] = image['detail']
                parts.append(converted)
            else:
                raise ValueError('Unsupported modality; Responses adapter accepts text and images only')
        result.append({'role': role, 'content': parts})
    return result


def create(client, request, messages, response_format):
    fmt = copy.deepcopy(response_format)
    if fmt['type'] == 'json_schema':
        fmt = dict(type='json_schema', **fmt['json_schema'])
    elif fmt != {'type': 'json_object'}:
        raise ValueError('Unsupported Responses output format')
    args = {'model': request['model'], 'input': input_messages(messages),
            'text': {'format': fmt}, 'max_output_tokens': request['max_tokens'],
            'store': False, 'stream': False}
    if request.get('reasoning_effort'):
        args['reasoning'] = {'effort': request['reasoning_effort']}
    with request_slot():
        response = client.responses.create(**args)
    body = response.model_dump(exclude_none=True)
    status = body.get('status')
    reason = body.get('incomplete_details', {}).get('reason')
    if status == 'incomplete' and reason == 'max_output_tokens':
        finish = 'length'
    elif status == 'completed':
        finish = 'stop'
    else:
        raise RuntimeError('Responses did not complete: ' + str(status) + '/' + str(reason))
    texts, refusals, unexpected = [], [], []
    for item in body.get('output', []):
        if item['type'] == 'reasoning':
            continue
        if item['type'] != 'message' or item.get('role') != 'assistant':
            unexpected.append(item['type'])
            continue
        for part in item.get('content', []):
            if part['type'] == 'output_text': texts.append(part['text'])
            elif part['type'] == 'refusal': refusals.append(part.get('refusal', ''))
            else: unexpected.append(part['type'])
    text = ''.join(texts)
    if refusals: finish = 'content_filter'
    if unexpected: raise RuntimeError('Unexpected Responses output types: ' + ','.join(unexpected))
    # A complete text-only assistant message is sufficient for a stateless
    # schema-correction turn. No previous_response_id or remote storage is used.
    return {'content': text, 'finish_reason': finish,
            'assistant_message': {'role': 'assistant', 'content': text},
            'provider_model': body.get('model'), 'provider_response_id': body.get('id'),
            'provider_status': status, 'provider_incomplete_reason': reason,
            'provider_api': 'responses', 'usage': body.get('usage'),
            'tool_calls': None, 'refusals': refusals}
