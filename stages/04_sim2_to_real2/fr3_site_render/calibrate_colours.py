"""Calibrate the S1 objects' paint in the site light (2026-11-06; user: "方块任务按S1的绿和红重新渲染",
"抽屉任务需要完全复用图中S1的抽屉").

The site scene's area lights are about twice as strong as S1's (540 W vs 289 W, both AgX, exposure 0), so S1's paint
renders paler and more neutral here (S1 cabinet top sRGB 157 151 143, site 190 190 187; S1 red cube 118 47 57, site
167 116 126). This finds the linear base colour that renders, in the head camera at one frame of one episode, the mean
sRGB S1's own render shows for that object. Per iteration and channel: colour *= lin(target) / lin(rendered) (lin: sRGB
-> linear), so the update is exact where the render is linear and converges in a few steps through AgX.

    t1_retarget/bin/python calibrate_colours.py blocks|cabinet --source <export> --config <local config> --out <dir>
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
import os as _os, sys as _sys
BLENDER_PY = _os.environ.get('REEMBODY_BLENDER_PYTHON', _sys.executable)
FRAME = 15
# S1's rendered mean sRGB (head camera, episode 0 at 0.5 s): v8 block (cam_high) and v9 lean30 drawer (cam_high)
TARGET = {'green': [81, 126, 73], 'red': [118, 47, 57], 'cabinet': [159.5, 153.5, 145.5]}
CABINET_BOXES = ((100, 142, 63, 76), (100, 135, 90, 112))  # x0 x1 y0 y1 of the cabinet top and fronts, padded frame


def lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def render(source, cfg, out):
    subprocess.run(['nice', '-n', '10', BLENDER_PY, str(HERE / 'render_site_v2.py'), '--source', str(source), '--config', str(cfg),
                    '--output', str(out), '--device', 'CPU', '--start', str(FRAME), '--stop', str(FRAME + 1)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.asarray(Image.open(out / f'frame_{FRAME:06d}_head.png').convert('RGB')).astype(float)


def measure(kind, a):
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    if kind == 'blocks':
        masks = {'green': (g > r + 12) & (g > b + 12) & (a.sum(-1) > 120), 'red': (r > g + 20) & (r > b + 15)}
        return {k: (a[m].mean(0), int(m.sum())) for k, m in masks.items()}
    px = np.concatenate([a[y0:y1, x0:x1].reshape(-1, 3) for x0, x1, y0, y1 in CABINET_BOXES])
    return {'cabinet': (px.mean(0), len(px)),
            **{f'cabinet_{n}': (a[y0:y1, x0:x1].reshape(-1, 3).mean(0), 0) for n, (x0, x1, y0, y1) in zip(('top', 'fronts'), CABINET_BOXES)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=('blocks', 'cabinet'))
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(a.config.read_text())
    if a.kind == 'blocks':
        keys = {'green': 'blue_cube_geom', 'red': 'purple_cube_geom'}
        colour = {k: np.asarray(cfg['s1_cubes'][g]['linear_rgb'], float) for k, g in keys.items()}
    else:
        gain = np.ones(3)
    log = []
    for it in range(a.iterations + 1):
        if a.kind == 'blocks':
            for k, g in keys.items():
                cfg['s1_cubes'][g]['linear_rgb'] = colour[k].round(4).tolist()
        else:
            cfg['s1_drawer']['site_gain'] = dict(cabinet=gain.round(4).tolist(), handle=gain.round(4).tolist())
        c = a.out / f'config_{it}.json'
        c.write_text(json.dumps(cfg, indent=1) + '\n')
        m = measure(a.kind, render(a.source, c, a.out / f'render_{it}'))
        row = dict(iteration=it, rendered={k: [v[0].round(1).tolist(), v[1]] for k, v in m.items()},
                   colour={k: v.round(4).tolist() for k, v in colour.items()} if a.kind == 'blocks' else None,
                   gain=gain.round(4).tolist() if a.kind == 'cabinet' else None)
        log.append(row)
        print(json.dumps(row), flush=True)
        if it == a.iterations:
            break
        if a.kind == 'blocks':
            for k in keys:
                colour[k] = np.clip(colour[k] * lin(TARGET[k]) / np.maximum(lin(m[k][0]), 1e-4), 0.0, 1.0)
        else:
            body = np.asarray(cfg['s1_drawer']['appearance']['cabinet_body_linear_rgb'], float)
            gain = np.minimum(gain * lin(TARGET['cabinet']) / np.maximum(lin(m['cabinet'][0]), 1e-4), 1.0 / body)
    (a.out / 'calibration.json').write_text(json.dumps(dict(kind=a.kind, frame=FRAME, target=TARGET, log=log), indent=1) + '\n')


if __name__ == '__main__':
    main()
