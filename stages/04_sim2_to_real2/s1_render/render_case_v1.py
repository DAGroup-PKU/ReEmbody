"""Export + Blender-render one admitted centred-layout native drawer case.

Export: export_source_center_v1.py (MuJoCo 3.3.7, read-only FK, requires the
native receipt with exact replay and the bound quality-v7 receipt).
Render: renderer_center_v1.py (bpy 4.5, pinned desk blend, v9b appearance,
SDK nominal camera mounts, 224x126 content + 49 px black padding, 12 samples,
as the v9b production runs).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
import os as _os, sys as _sys
EXPORT_PY = _os.environ.get('REEMBODY_PHYSICS_PYTHON', _sys.executable)
RENDER_PY = _os.environ.get('REEMBODY_BLENDER_PYTHON', _sys.executable)
CONFIG = HERE / 'render_config_center_sdk_v1.json'
LOCAL_GPU = dict(index=0, uuid='GPU-254e779d-602d-8448-b69e-06c9722de0ef', pci='00000000:61:00.0')


def run(argv, log, env):
    with Path(log).open('x') as stream:
        rc = subprocess.run(argv, cwd=HERE, env=env, stdin=subprocess.DEVNULL, stdout=stream,
                            stderr=subprocess.STDOUT).returncode
    if rc:
        raise RuntimeError(f'command failed ({rc}); see {log}')


def render_case(native: Path, output: Path, device='OPTIX', samples=12, share_group='drawercenter1',
                preview=False):
    native = Path(native).resolve(strict=True)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((native / 'receipt.json').read_text())
    if not (receipt.get('accepted') is True and (receipt.get('replay') or {}).get('exact') is True):
        raise ValueError('native case is not admitted')
    base = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
                PYTHONDONTWRITEBYTECODE='1')
    run([EXPORT_PY, '-u', str(HERE / 'export_source_center_v1.py'), '--native', str(native),
         '--qa-receipt', str(native / 'drawer_quality_receipt.json'), '--config', str(CONFIG),
         '--output', str(output / 'source')], output / 'export.log', dict(base, CUDA_VISIBLE_DEVICES=''))
    argv = [RENDER_PY, '-u', str(HERE / 'renderer_center_v1.py'), '--source', str(output / 'source'),
            '--config', str(CONFIG), '--output', str(output / 'video'), '--device', device,
            '--samples', str(samples), '--width', '224', '--height', '126']
    if preview:
        argv.append('--preview')
    if device == 'OPTIX':
        argv += ['--gpu-index', '0', '--gpu-uuid', LOCAL_GPU['uuid'], '--gpu-pci', LOCAL_GPU['pci'],
                 '--allow-shared-gpu', '--share-group', share_group, '--max-renderers-per-gpu', '4']
        env = dict(base, CUDA_VISIBLE_DEVICES=str(LOCAL_GPU['index']))
    else:
        env = dict(base, CUDA_VISIBLE_DEVICES='')
    run(argv, output / 'render.log', env)
    rr = json.loads((output / 'video/render_receipt.json').read_text())
    result = dict(native=str(native), output=str(output), frames=len(rr.get('frames', [])),
                  videos={c: str(output / 'video' / f'{c}.mp4') for c in rr['camera_names']},
                  elapsed_s=rr.get('elapsed_s'), renderer_revision=rr.get('renderer_revision'))
    (output / 'render_case_result.json').write_text(json.dumps(result, indent=1))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--native', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--device', choices=('OPTIX', 'CPU'), default='OPTIX')
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--preview', action='store_true')
    a = ap.parse_args()
    print(json.dumps(render_case(a.native, a.output, a.device, a.samples, preview=a.preview)))


if __name__ == '__main__':
    main()
