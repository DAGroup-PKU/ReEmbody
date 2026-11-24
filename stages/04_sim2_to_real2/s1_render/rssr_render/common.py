"""Durable shared-filesystem artifacts for the independent CPU render route."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rssr_augment.layout import resolve_path, relocate

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'.git').exists()), Path(__file__).resolve().parents[1])
MODULE = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'result/sim2-real2/_inputs/blender_sim2_real2_20261012'
PHYSICAL = ROOT / 'result/sim1-sim2/latest'
CAMERAS = ('head_rgbd', 'left_wrist_rgbd', 'right_wrist_rgbd')
FAMILIES = ('stack_cups', 'stack_blocks', 'drawer')
FPS = 24
RASTER = 224  # Paired official Real1 native square raster, not an upscaled preview.
import shutil as _shutil
FFMPEG = Path(os.environ.get('REEMBODY_FFMPEG') or os.environ.get('RSSR_FFMPEG') or _shutil.which('ffmpeg') or 'ffmpeg')
FFPROBE = Path(os.environ.get('RSSR_FFPROBE', str(FFMPEG.with_name('ffprobe'))))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return relocate(json.loads(resolve_path(path).read_text()))


def parse_scene(path):
    """Resolve asset locations in memory while preserving the original XML."""
    import xml.etree.ElementTree as ET
    path = resolve_path(path)
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find('compiler')
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key.startswith('file'):
                asset = Path(value)
                if not asset.is_absolute():
                    directory = {'mesh': 'meshdir', 'texture': 'texturedir'}.get(element.tag)
                    base = compiler.get(directory, '') if compiler is not None and directory else ''
                    asset = path.parent / base / asset
                element.set(key, str(resolve_path(asset)))
    if compiler is not None:
        for directory in ('meshdir', 'texturedir'):
            if directory in compiler.attrib:
                compiler.set(directory, '')
    return tree


def load_xml_model(path):
    import mujoco
    import xml.etree.ElementTree as ET
    return mujoco.MjModel.from_xml_string(ET.tostring(parse_scene(path).getroot(), encoding='unicode'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def save_arrays(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(tmp, path)


def code_pins():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(MODULE.rglob('*.py'))}


def verify_pins(pins):
    for path, digest in pins.items():
        if sha(ROOT / path) != digest:
            raise ValueError('Pinned input changed: ' + path)


def sample_clock(state_times, command_times, duration, fps=FPS):
    """Original speed, causal zero-order sampling; no interpolation or time warping."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError('Video frame rate must be positive')
    video = np.arange(int(np.ceil(duration * fps - 1e-8)) + 1, dtype=np.float64) / fps
    physical_queries = np.minimum(video, duration)
    indices = []
    queries = physical_queries
    for times in (state_times, command_times):
        times = np.asarray(times, dtype=np.float64)
        if times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0):
            raise ValueError('Physical and action clocks must be strictly increasing')
        selected = np.searchsorted(times, queries + 1e-10, side='right') - 1
        if np.any(selected < 0) or np.any(selected >= len(times)):
            raise ValueError('Full video clock lies outside physical/action source')
        gap = queries - times[selected]
        if np.any(gap < -1e-10) or np.any(gap > np.max(np.diff(times)) + 1e-8):
            raise ValueError('Video/action sampling lost causal time alignment')
        indices.append(selected.astype(np.int64))
        # Bind desired actions to the physical state actually rendered. An
        # action issued between that saved state and the video clock must not
        # be assigned to the preceding picture.
        queries = times[selected]
    return video, *indices, physical_queries
