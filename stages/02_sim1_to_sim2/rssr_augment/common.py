"""Branch-local paths, atomic artifacts and explicit historical path relocation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from rssr_augment.shared.family_autoresearch.store import atomic_json, file_sha
from .layout import resolve_path as resolve_layout_path

BRANCH = Path(__file__).resolve().parent
ROOT = next(p for p in BRANCH.parents if (p/'.git').exists())
RESULT = ROOT/'result/sim1-sim2/s1_short_refine_v2_20261018'
SOURCE_ROOT = Path(os.environ.get('REEMBODY_SOURCE_ROOT', os.environ.get('RSSR_SOURCE_ROOT', str(Path(__file__).resolve().parents[3] / 'examples')))).resolve()
BASE_RUN = SOURCE_ROOT / 'runs/rssr_fixed_chassis_exact_preview_full150_seq343_20261004'
PHOTO_ROOT = SOURCE_ROOT / 'sim2-real2/user_optimization/outputs_20261008/cups48_seq343_single295_wrist_v6'
import sys as _sys
PYTHON = Path(os.environ.get('REEMBODY_PHYSICS_PYTHON', _sys.executable))
FFMPEG = PYTHON.parent / 'ffmpeg'
FFPROBE = PYTHON.parent / 'ffprobe'
MODULE = 'rssr_augment.retarget'

def read(path):
    return relocate(json.loads(resolve_path(path).read_text()))

def pin(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Frozen artifact changed; use another run: ' + str(path))
    else:
        atomic_json(path, value)

def resolve_path(path):
    """Resolve migrated files without modifying the historical evidence bytes."""
    path = resolve_layout_path(path)
    if path.exists():
        return path
    text = str(path)
    mappings = [
        ('${REEMBODY_SOURCE_ROOT}/', str(SOURCE_ROOT) + '/'),
    ]
    if text == '${REEMBODY_SOURCE_ROOT}':
        return SOURCE_ROOT
    for before, after in mappings:
        if text.startswith(before):
            candidate = Path(after + text[len(before):])
            if candidate.exists():
                return candidate
    return path

def relocate(value):
    if isinstance(value, str) and value.startswith('/'):
        return str(resolve_path(value))
    if isinstance(value, list):
        return [relocate(x) for x in value]
    if isinstance(value, dict):
        return {str(resolve_path(k)) if k.startswith('/') else k: relocate(v)
                for k, v in value.items()}
    return value

def stable_seed(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)

def within_branch(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RESULT.resolve()) and not path.is_relative_to(BRANCH):
        raise ValueError('Outputs must remain in the Sim1-Sim2 result directory: ' + str(path))
    return path
