"""Explicit, uncalibrated rolling-resistance prior for round scene assets.

MuJoCo's rolling coefficient is a length. Scale it with radius rather than
sharing a millimetre-scale constant between balls and millimetre-scale grains.
This is an asset assumption, not an inference-time action or success override.
"""
import math
import numpy as np

REVISION = 'radius_scaled_round_rolling_v1'


def apply(xml, scene, policy):
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {'revision', 'radius_ratio', 'calibrated'}:
        raise ValueError('Unknown round-contact asset policy')
    ratio = policy['radius_ratio']
    if (policy['revision'] != REVISION or policy['calibrated'] is not False or
            type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= .05):
        raise ValueError('Invalid or falsely calibrated round-contact prior')
    objects = {o['id']: o for o in scene['objects'] if o['shape'] in ('sphere', 'grain')}
    if not objects:
        return
    import mujoco
    import xml.etree.ElementTree as ET
    original = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    # Inactive rolling defaults on the other geom otherwise dominate MuJoCo's
    # componentwise-max contact mixing. Sliding/torsional friction are retained.
    for geom in xml.findall('.//worldbody//geom'):
        name = geom.get('name')
        if not name:
            if geom.get('contype') == '0' and geom.get('conaffinity') == '0':
                continue  # Unnamed display-only robot meshes cannot mix friction.
            raise ValueError('Rolling prior requires named physical geoms')
        friction = original.geom_friction[original.geom(name).id].copy()
        friction[2] = 0.
        geom.set('friction', ' '.join(map(str, friction)))
    for name, obj in objects.items():
        body = xml.find(".//body[@name='"+name+"']")
        if body is None:
            raise ValueError('Missing round body')
        count = 0
        for geom in body.findall('geom'):
            if geom.get('type') != 'sphere':
                continue
            radius = float(obj['size_m'][0])/2
            if radius <= 0 or not math.isfinite(radius):
                raise ValueError('Invalid round-object radius')
            friction = np.fromstring(geom.get('friction'), sep=' ')
            friction[2] = ratio*radius
            geom.set('friction', ' '.join(map(str, friction)))
            geom.set('condim', '6')
            count += 1
        if count != 1:
            raise ValueError('Round prior expects exactly one sphere per round object')
