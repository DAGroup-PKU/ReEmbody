#!/usr/bin/env python3
"""Per-episode Blender appearance config for the SDK wrist-camera cup renders.

Starts from the exact config the published v8 cup episode used
(v4 extra200 evidence reference[episode % 200]/config.json). Only the two
wrist entries of camera_alignment.cameras change: mount pos/quat = Astribot
SDK S1v2 kinematic.xml, reference_fovy_deg = fovy = 57 (vertical, 16:9 raster),
identity in-camera delta. Head camera, exposure offsets, lighting, style and
assets are unchanged. Optional --asset-root rewrites the two .blend paths for
a staged remote copy (dependencies are resolved relative to asset_blend).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_source_sdkcam_v1 import SDK_SOURCE, SDK_WRIST  # noqa: E402

import os as _os
ROOT = Path(__file__).resolve().parents[3]
OLD = Path(_os.environ.get('REEMBODY_S1_BASE_RENDER_CONFIG_DIR', str(ROOT / 'stages/04_sim2_to_real2/configs')))


def base_config(episode: int) -> tuple[Path, dict]:
    reference = [json.loads(line) for line in (OLD / 'meta/source_cases.jsonl').read_text().splitlines()]
    path = OLD / reference[episode % 200]['evidence'] / 'config.json'
    return path, json.loads(path.read_text())


def sdk_config(episode: int, asset_root: str | None = None) -> dict:
    path, cfg = base_config(episode)
    out = copy.deepcopy(cfg)
    cams = out['camera_alignment']['cameras']
    for name, (pos, quat, fovy) in SDK_WRIST.items():
        spec = cams[name]
        spec.update(pos=list(pos), quat=list(quat), fovy=fovy, reference_fovy_deg=fovy, reference_aspect=16 / 9,
                    fovy_deg=math.degrees(2 * math.atan(16 / 9 * math.tan(math.radians(fovy) / 2))),
                    position_in_camera_m=[0.0, 0.0, 0.0], quaternion_in_camera_wxyz=[1.0, 0.0, 0.0, 0.0],
                    mount_source=SDK_SOURCE, mount_kind='astribot_sdk_s1v2_kinematic_xml_nominal',
                    image_roll_180_about_optical_axis=True,
                    verified_against_real_wrist_frame=False)
        for k in ('baseline_world_position_m', 'baseline_world_rotation'):
            spec.pop(k, None)
    out['camera_alignment']['wrist_override'] = dict(
        reason='cup wrist cameras switched to the SDK nominal mount (~10 cm from pads, fovy 57) per user decision 2026-10-31',
        base_config=str(path), head_camera_unchanged=True, exposure_offsets_unchanged=True)
    if asset_root:
        out['asset_blend'] = str(Path(asset_root) / Path(cfg['asset_blend']).name)
        out['cup_asset_blend'] = str(Path(asset_root) / Path(cfg['cup_asset_blend']).name)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episode', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--asset-root')
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(sdk_config(a.episode, a.asset_root), indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(config=str(a.output))))


if __name__ == '__main__':
    main()
