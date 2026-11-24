"""Per-episode appearance config for render_site.py, with S1's v8/v9 appearance randomisation
(s1_stack_cups_random300_r15_extra200_v1/run_v2.py appearance_config + s1_bgfit_v3/campaign.py lighting_recipe):
  * every baseline light: position + U(-0.2, 0.2) m per axis, aim point + U(-0.1, 0.1) m per axis, power x U(0.8, 1.2),
    colour x U(0.9, 1.1) per channel (clipped to [0, 1]);
  * one or two added disk area lights, U(8, 30) W, size U(0.8, 1.8) m, 1.8-2.8 m high within +-1.8 m (across) and
    +-1.2 m (along) of the desk centre, aimed at the desk top near its centre;
  * every camera's exposure: its baseline offset + U(-0.2, 0.2) EV.
The seed is sha256('fr3site:<base_seed>:<episode index>:<variation>'), as S1 seeds per proposal.

    python make_config.py --site-dir <env dir> --cup-blend <measured_cups.blend> --episode-index 3 \
        --variation stack_cups_site_aug_0007 --out config.json [--asset-root <remote assets dir>]
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

BASE_SEED = 20261103


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recipe(seed, baseline, desk_xy, desk_top):
    rng = np.random.default_rng(seed)
    lights = []
    for v in baseline:
        color = np.clip(np.asarray(v['color'], float) * rng.uniform(0.9, 1.1, 3), 0, 1)
        lights.append(dict(name=v['name'], position_m=(np.asarray(v['position_m']) + rng.uniform(-0.2, 0.2, 3)).tolist(),
                           target_m=(np.asarray(v['target_m']) + rng.uniform(-0.1, 0.1, 3)).tolist(),
                           power_scale=float(rng.uniform(0.8, 1.2)), color=color.tolist(), size_scale=1.0))
    added = []
    for i in range(int(rng.integers(1, 3))):
        added.append(dict(name=f'site_added_{i}',
                          position_m=[float(desk_xy[0] + rng.uniform(-1.8, 1.8)), float(desk_xy[1] + rng.uniform(-1.2, 1.2)),
                                      float(rng.uniform(1.8, 2.8))],
                          target_m=[float(desk_xy[0] + rng.uniform(-0.25, 0.25)), float(desk_xy[1] + rng.uniform(-0.2, 0.2)),
                                    float(desk_top)],
                          power_w=float(rng.uniform(8, 30)), color=[1.0, 1.0, 1.0], size_m=float(rng.uniform(0.8, 1.8))))
    return rng, dict(seed=int(seed), lights=lights, added_lights=added)


def make(site_dir, cup_blend, index, variation, asset_root=None, base_seed=BASE_SEED):
    site_dir, cup_blend = Path(site_dir), Path(cup_blend)
    scene = json.loads((site_dir / 'site_scene.json').read_text())
    look = json.loads((site_dir / 'look.json').read_text())  # fitted baseline exposure and camera offsets
    params = json.loads((site_dir / 'site_parameters.json').read_text())  # frozen copy next to the scene
    assert hashlib.sha256((site_dir / 'site_parameters.json').read_bytes()).hexdigest() == scene['parameters_sha256']
    text = f'fr3site:{base_seed}:{int(index)}:{variation}'
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little')
    rng, rec = recipe(seed, scene['lights'], params['desk_centre_xy_m'], params['desk_top_m'])
    offsets = {}
    for cam, base in look['camera_exposure_offset_ev'].items():
        offsets[cam] = float(base + rng.uniform(-0.2, 0.2))
    root = Path(asset_root) if asset_root else None
    return dict(schema='rssr.fr3_duo.site_render_config.v1', variation=variation, episode_index=int(index),
                seed_text=text,
                site_blend=str((root or site_dir) / 'site_scene.blend'), site_blend_sha256=scene['derived_sha256'],
                site_dependencies=scene['asset_dependencies'],
                cup_blend=str((root / cup_blend.name) if root else cup_blend), cup_blend_sha256=sha(cup_blend),
                exposure_ev=float(look['exposure_ev']), camera_exposure_offset_ev=offsets,
                camera_exposure_baseline_ev=look['camera_exposure_offset_ev'], lighting_recipe=rec,
                randomization_ranges=dict(light_position_delta_each_axis_m=[-0.2, 0.2], light_target_delta_each_axis_m=[-0.1, 0.1],
                                          light_energy_factor=[0.8, 1.2], light_rgb_multiplier_each_channel=[0.9, 1.1],
                                          added_area_light_count=[1, 2], added_area_light_energy_w=[8, 30],
                                          added_area_light_size_m=[0.8, 1.8], each_camera_exposure_delta_ev=[-0.2, 0.2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-dir', type=Path, required=True)
    ap.add_argument('--cup-blend', type=Path, required=True)
    ap.add_argument('--episode-index', type=int, required=True)
    ap.add_argument('--variation', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--asset-root')
    a = ap.parse_args()
    cfg = make(a.site_dir, a.cup_blend, a.episode_index, a.variation, a.asset_root)
    a.out.write_text(json.dumps(cfg, indent=1) + '\n')


if __name__ == '__main__':
    main()
