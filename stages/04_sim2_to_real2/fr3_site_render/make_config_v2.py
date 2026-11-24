"""Per-episode render config of the v2 data: sim2-real2/franka_fr3_duo_site_v1/make_config.py (S1's seeded appearance
randomisation, unchanged) plus what render_site_v2.py needs for layout fr3_site_20261105d: the sim -> Blender shift of
site_v8 and the S1 drawer appearance (s1_drawer_v1/drawer_appearance_v9b_uniform_cabinet_upright.json and the material
recipe of s1_drawer_center_v1/render_config_center_sdk_v1.json, copied in with their sha256).

    python make_config_v2.py --site-dir ../env/site_v8 --cup-blend <measured_cups.blend> --episode-index 3 \
        --variation <id> --out config.json [--asset-root <remote assets dir>]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE4_CONFIGS = HERE.parent / 'configs'
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import make_config as MC  # noqa: E402

SIM_TO_BLENDER = [0.0316, 0.15, 0.04]  # site_v8: sim desk top 0.75 m vs 0.79 m, mount on the midline, desk 15 cm closer
S1_APPEARANCE = STAGE4_CONFIGS / 'drawer_appearance_v9b_uniform_cabinet_upright.json'
S1_RECIPE = STAGE4_CONFIGS / 'render_config_center_sdk_v1.json'
# S1 v8 block render palette (sim2-refine/tasks/stack_blocks/random200_near/build_manifest200_cupstyle_v5.py
# color_derivative: green [.32 .60 .30], red [.52 .16 .22] linear; render_real2_cupstyle_flatcube_v2.py matte surface):
# the task string's green base and red top. linear_rgb is that palette calibrated to the site light
# (calibrate_colours.py, ../dev/colour_calibration_20261106/blocks): rendered head-camera means sRGB 82 127 74 and
# 118 47 57 against S1's 81 126 73 and 118 47 57 (S1's palette itself renders 136 157 132 and 164 113 124 here).
S1_CUBES = {'blue_cube_geom': dict(linear_rgb=[0.0751, 0.3381, 0.0669], s1_palette_linear_rgb=[.32, .60, .30], roughness=.82,
                                   specular_ior_level=.02, coat=0.0, s1_colour='green'),
            'purple_cube_geom': dict(linear_rgb=[0.1993, 0.025, 0.0422], s1_palette_linear_rgb=[.52, .16, .22], roughness=.82,
                                     specular_ior_level=.02, coat=0.0, s1_colour='red')}
# Cabinet / handle paint gain in the site light (render_site_v2.s1_rgb), calibrated the same way
# (../dev/colour_calibration_20261106/cabinet): cabinet top + fronts render sRGB 160.5 154.4 146.0 against S1's
# 159.5 153.5 145.5 (ungained: 188.5 184.7 178.0).
DRAWER_SITE_GAIN = dict(cabinet=[0.5041, 0.4818, 0.4807], handle=[0.5041, 0.4818, 0.4807])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make(site_dir, cup_blend, index, variation, asset_root=None):
    cfg = MC.make(site_dir, cup_blend, index, variation, asset_root)
    cfg['schema'] = 'rssr.fr3_duo.site_render_config.v2'
    cfg['sim_to_blender_m'] = SIM_TO_BLENDER
    cfg['s1_drawer'] = dict(appearance=json.loads(S1_APPEARANCE.read_text()),
                            recipe=json.loads(S1_RECIPE.read_text())['native_render_material_overrides'],
                            appearance_source=str(S1_APPEARANCE), appearance_sha256=sha(S1_APPEARANCE),
                            recipe_source=str(S1_RECIPE), recipe_sha256=sha(S1_RECIPE))
    if variation.startswith('stack_blocks'):
        cfg['s1_cubes'] = S1_CUBES
    if variation.startswith('drawer') and DRAWER_SITE_GAIN:
        cfg['s1_drawer']['site_gain'] = DRAWER_SITE_GAIN
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--site-dir', type=Path, required=True)
    ap.add_argument('--cup-blend', type=Path, required=True)
    ap.add_argument('--episode-index', type=int, required=True)
    ap.add_argument('--variation', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--asset-root')
    a = ap.parse_args()
    a.out.write_text(json.dumps(make(a.site_dir, a.cup_blend, a.episode_index, a.variation, a.asset_root), indent=1) + '\n')


if __name__ == '__main__':
    main()
