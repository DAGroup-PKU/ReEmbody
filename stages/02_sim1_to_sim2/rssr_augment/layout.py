"""Resolve historical artifact paths without changing frozen evidence files."""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = next((p for p in Path(__file__).resolve().parents if (p / '.git').exists()),
            Path(__file__).resolve().parents[3])


def archive_root():
    configured = os.environ.get('RSSR_ARCHIVE_ROOT')
    if configured:
        return Path(configured)
    manifest = ROOT / 'result/organization_manifest.json'
    if manifest.is_file():
        return Path(json.loads(manifest.read_text())['archive'])
    raise FileNotFoundError('Historical inputs require result/organization_manifest.json or RSSR_ARCHIVE_ROOT')


def resolve_path(path):
    """Map exact former directory prefixes; leave current paths unchanged."""
    path = Path(path)
    try:
        relative = path.relative_to(ROOT)
    except ValueError:
        return path
    if not relative.parts:
        return path
    name, *rest = relative.parts
    if name == 'experiments':
        return (ROOT / 'result/experiments').joinpath(*rest).resolve()
    if name == 'provenance':
        return (ROOT / 'result/provenance').joinpath(*rest).resolve()
    if name in {'lyc_dev_v2', 'lyc_dev_blender', 'pipeline', 'config', 'knowledge', 'research_20260929'}:
        return archive_root().joinpath(name, *rest).resolve()
    return path


def relocate(value):
    """Relocate JSON values and pin keys in memory; byte hashes remain original."""
    if isinstance(value, str) and value.startswith('/'):
        return str(resolve_path(value))
    if isinstance(value, list):
        return [relocate(item) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            mapped = str(resolve_path(key)) if key.startswith('/') else key
            relocated = relocate(item)
            if mapped in result and result[mapped] != relocated:
                raise ValueError('Historical paths have conflicting bindings: ' + mapped)
            result[mapped] = relocated
        return result
    return value
