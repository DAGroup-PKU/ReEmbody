"""Resume a paid visual decision using its original, hash-checked evidence.

Rendering the same MuJoCo state twice need not produce identical pixels. Once
images have been submitted, recovery must read those images, not regenerate
them and silently attach an old verdict to different evidence.
"""
from pathlib import Path
from common import read, write, sha
from rssr_protocol import digest


def load(path, identity):
    path = Path(path)
    if not path.exists():
        return None
    try:
        record = read(path)
    except (ValueError,TypeError) as error:
        raise RuntimeError('Saved visual evidence receipt is unreadable: ' + str(path)) from error
    body = {k: v for k, v in record.items() if k != 'sha256'}
    if record.get('sha256') != digest(body):
        raise RuntimeError('Saved visual evidence receipt changed: ' + str(path))
    if digest(record['identity']) != digest(identity):
        raise RuntimeError('Saved visual evidence inputs changed: ' + str(path))
    for filename, expected in record['files'].items():
        if not Path(filename).is_file() or sha(filename) != expected:
            raise RuntimeError('Saved visual evidence file changed: ' + filename)
    return record['data']


def save(path, identity, data, files=()):
    record = {'schema': 'rssr.visual_evidence.v1', 'identity': identity,
              'data': data, 'files': {str(p): sha(p) for p in files}}
    record['sha256'] = digest(record)
    if Path(path).exists():
        previous = load(path, identity)
        if digest(previous) != digest(data) or read(path) != record:
            raise RuntimeError('Cannot replace committed visual evidence: ' + str(path))
    else:
        write(path, record)
    return data
