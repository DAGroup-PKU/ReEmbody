"""Compare the site scene with the user's D455 frame (S1 analogue: the style fit of s1_heightfit_v2 / s1_bgfit_v3).

Renders the head camera of a preview source with the robot arms hidden (their pose in the user's frame is unknown),
then reports the mean sRGB of image regions the real arms do not cover: cloth (desk top centre), carpet (between the
desk feet and the mount), pillar (above the desk), desk feet; and the render/real ratio per region. The fitted values
(cloth colour, carpet multiplier, exposure) go into site_parameters.json / look.json; nothing here edits the scene.

    native_photo/bin/python style_fit.py --source <preview source> --config <config> --real <d455 png> --out <dir>
"""
import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_site import SiteScene  # noqa: E402

# regions in the 640x480 stream: (x0, y0, x1, y1)
REGIONS = {'cloth': (230, 140, 420, 220), 'carpet': (240, 300, 400, 370), 'pillar': (230, 8, 420, 70),
           'feet': (178, 300, 196, 330)}


def stats(img):
    a = np.asarray(img.convert('RGB').resize((640, 480), Image.BILINEAR), float)
    return {k: a[y0:y1, x0:x1].reshape(-1, 3).mean(0).round(1).tolist() for k, (x0, y0, x1, y1) in REGIONS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--real', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=32)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = SiteScene(a.source, a.config, a.samples, 'OPTIX')
    for j, ob in S.geoms:
        if ob['mj_body'].startswith(('left_', 'right_')):
            ob.hide_render = True
    S.scene.render.resolution_percentage = 400
    img = S.render(0, 0, a.out / 'head_noarm.png')
    real = Image.open(a.real)
    rs, ss = stats(real), stats(img)
    ratio = {k: (np.array(ss[k]) / np.maximum(np.array(rs[k]), 1)).round(3).tolist() for k in REGIONS}
    both = Image.new('RGB', (1280, 480))
    both.paste(real.convert('RGB').resize((640, 480)), (0, 0))
    both.paste(img.resize((640, 480)), (640, 0))
    both.save(a.out / 'real_vs_render.png')
    rep = dict(regions_640x480=REGIONS, real_srgb=rs, render_srgb=ss, render_over_real=ratio)
    (a.out / 'style_fit.json').write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep))


if __name__ == '__main__':
    main()
