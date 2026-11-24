"""Equivalent table collision surfaces dispatched by shape, opt-in only.

Round scene objects use analytic boxes; other pairs retain convex tables.
The original collision-pair graph is preserved and checked after compilation.
"""
import copy
import itertools
import xml.etree.ElementTree as ET
import numpy as np

REVISION = 'round_analytic_table_dispatch_v1'


def compatible(model, a, b):
    return bool((int(model.geom_contype[a]) & int(model.geom_conaffinity[b])) or
                (int(model.geom_contype[b]) & int(model.geom_conaffinity[a])))


def apply(xml, scene):
    import mujoco
    names = []
    for obj in scene['objects']:
        if obj['shape'] in ('sphere', 'grain'):
            body = xml.find(".//body[@name='"+obj['id']+"']")
            names.extend(g.get('name') for g in body.findall('geom') if g.get('type') == 'sphere')
    if not names:
        return {'revision': REVISION, 'round_geoms': [], 'duplicates': {}}
    if xml.findall('.//contact/pair'):
        raise ValueError('Explicit collision pairs require separate dispatch validation')
    old = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    round_ids = [old.geom(n).id for n in names]
    if len({(int(old.geom_contype[i]), int(old.geom_conaffinity[i])) for i in round_ids}) != 1:
        raise ValueError('Round-object masks must share one collision policy')
    elements = {g.get('name'): g for g in xml.findall('.//worldbody//geom') if g.get('name')}
    body = xml.find(".//body[@name='worktable']")
    if body is None:
        raise ValueError('Worktable required')
    if body.find('inertial') is None:
        b = old.body('worktable').id
        ET.SubElement(body, 'inertial', pos=' '.join(map(str, old.body_ipos[b])),
                      quat=' '.join(map(str, old.body_iquat[b])), mass=str(old.body_mass[b]),
                      diaginertia=' '.join(map(str, old.body_inertia[b])))
    used = int(np.bitwise_or.reduce(old.geom_contype | old.geom_conaffinity))
    bit = next((1 << i for i in range(30) if not used & (1 << i)), None)
    if bit is None:
        raise ValueError('No collision mask bit available')
    tables = {g.get('name'): g for g in body.findall('geom') if g.get('type') == 'mesh'}
    if not tables:
        raise ValueError('Equivalent convex worktable required')
    for i in range(old.ngeom):
        name = old.geom(i).name
        if i in round_ids:
            elements[name].set('contype', str(bit))
            elements[name].set('conaffinity', str(bit if compatible(old, i, round_ids[0]) else 0))
        elif name not in tables and compatible(old, i, round_ids[0]):
            if name not in elements:
                raise ValueError('Unnamed physical geom cannot be dispatched safely')
            elements[name].set('conaffinity', str(int(old.geom_conaffinity[i]) | bit))
    duplicates = {}
    for name, geom in tables.items():
        if not compatible(old, old.geom(name).id, round_ids[0]):
            continue
        mesh = xml.find(".//mesh[@name='"+geom.get('mesh')+"']")
        if mesh is None or mesh.get('scale') or mesh.get('file'):
            raise ValueError('Expected unscaled inline equivalent box vertices')
        vertices = np.fromstring(mesh.get('vertex', ''), sep=' ').reshape(-1, 3)
        half = np.max(np.abs(vertices), axis=0)
        expected = np.array(list(itertools.product((-1, 1), repeat=3))) * half
        if {tuple(v) for v in vertices} != {tuple(v) for v in expected}:
            raise ValueError('Worktable mesh is not an axis-aligned equivalent box')
        twin = copy.deepcopy(geom)
        twin_name = name+'_round_analytic'
        twin.set('name', twin_name)
        twin.set('type', 'box')
        twin.set('size', ' '.join(map(str, half)))
        del twin.attrib['mesh']
        twin.set('contype', '0')
        twin.set('conaffinity', str(bit))
        twin.set('rgba', '0 0 0 0')
        twin.set('group', '5')
        body.append(twin)
        duplicates[name] = twin_name
    new = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    def mapped(i):
        name = old.geom(i).name
        if name:
            return new.geom(name).id
        body_id = old.geom_bodyid[i]
        local_index = list(np.flatnonzero(old.geom_bodyid == body_id)).index(i)
        return int(np.flatnonzero(new.geom_bodyid == body_id)[local_index])
    # Validate every original geom pair, counting the equivalent table twin.
    for a in range(old.ngeom):
        na = mapped(a)
        for b in range(a):
            nb = mapped(b)
            alternatives_a = [na] + ([new.geom(duplicates[old.geom(a).name]).id] if old.geom(a).name in duplicates else [])
            alternatives_b = [nb] + ([new.geom(duplicates[old.geom(b).name]).id] if old.geom(b).name in duplicates else [])
            count = sum(compatible(new, x, y) for x in alternatives_a for y in alternatives_b)
            if count != int(compatible(old, a, b)):
                raise ValueError('Collision pair coverage changed or doubled')
    for field in ('body_mass', 'body_inertia', 'body_ipos', 'body_iquat', 'jnt_range'):
        if not np.array_equal(getattr(old, field), getattr(new, field)):
            raise ValueError('Physical property changed: '+field)
    return {'revision': REVISION, 'round_geoms': names, 'duplicates': duplicates,
            'collision_pair_graph_preserved': True, 'reserved_bit': bit}
