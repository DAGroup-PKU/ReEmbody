"""Persistent Blender render worker for closed-loop policy evaluation.

Loads the same pinned desk blend, v9b appearance, lights, materials, SDK camera
mounts and 224x126 raster as renderer_center_v1.py (its AssetScene class is
used unchanged) from a time-zero diagnostic export, then renders requested
live poses.  Protocol: one JSON line per request on stdin
  {"id": n, "pose": "<npz with geom_position (G,3), geom_rotation (G,9),
   camera_position (3,3), camera_rotation (3,9)>", "out": "<npy path>"}
writes (3,224,224,3) uint8 (head, left wrist, right wrist; 49 px black bars)
and prints "RSSR_DONE <id>" on stdout.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import renderer_center_v1 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--samples', type=int, default=12)
    ap.add_argument('--scratch', type=Path, required=True)
    a = ap.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else None)
    scene = R.AssetScene(a.source, a.config, a.samples, 'CPU', 224, 126, reference=True)
    n_geoms = scene.pose['geom_position'].shape[1]
    a.scratch.mkdir(parents=True, exist_ok=True)
    print('RSSR_READY', flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        with np.load(req['pose']) as z:
            pose = {k: z[k][None].astype(np.float64) for k in
                    ('geom_position', 'geom_rotation', 'camera_position', 'camera_rotation')}
        assert pose['geom_position'].shape[1] == n_geoms
        pose['geom_rotation'] = pose['geom_rotation'].reshape(1, n_geoms, 9)
        pose['camera_rotation'] = pose['camera_rotation'].reshape(1, 3, 9)
        scene.pose.update(pose)
        frames = []
        for ci in range(3):
            raw = scene.render(0, ci, a.scratch / 'current.png')
            assert raw.size == (224, 126)
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            image.paste(raw, (0, 49))
            frames.append(np.asarray(image, np.uint8))
        out = np.stack(frames)
        tmp = Path(req['out'] + '.tmp.npy')
        np.save(tmp, out)
        tmp.rename(req['out'])
        print(f"RSSR_DONE {req['id']}", flush=True)


if __name__ == '__main__':
    main()
