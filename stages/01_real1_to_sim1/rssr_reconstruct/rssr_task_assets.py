"""Load a versioned task asset before bounded numerical model experiments.

v1 certifies one source episode. Rebinding a different episode must create and
validate a new placement receipt; a successful source scene is not universal.
"""
import copy,pathlib,xml.etree.ElementTree as ET
import numpy as np
from common import read,sha


def load(config,scene):
    root=pathlib.Path(config['task_scene_asset']);manifest=read(root/'manifest.json')
    if manifest.get('schema')!='rssr.task_scene_asset.v1':raise ValueError('Unknown task asset')
    expected=config.get('task_scene_asset_sha256')
    if expected and sha(root/'manifest.json')!=expected:raise ValueError('Task asset version changed')
    for name,identity in manifest['files'].items():
        path=(root/name).resolve()
        if path.parent!=root.resolve() or sha(path)!=identity:raise ValueError('Task asset file changed: '+name)
    if read(root/'scene.json')!=scene:raise ValueError('Frozen task asset layout/geometry cannot be edited in retarget')
    receipt=read(root/'validation.json')
    if not receipt.get('sim1_task_success') or not receipt.get('repeated_arrays_identical'):
        raise ValueError('Source task asset has not passed validation')
    if sha(pathlib.Path(config['import_case'])/'source_action_state.npz')!=receipt['source_action_sha256']:
        raise ValueError('This v1 placement is validated only for its recorded source action')
    return root


def apply(xml,config,kind,scene):
    if not config.get('task_scene_asset'):return
    root=load(config,scene);world=xml.find('worldbody')
    objects=ET.parse(root/'object_bodies.xml').getroot().find('objects')
    expected={o['id'] for o in scene['objects']}
    if {b.get('name') for b in objects}!=expected:raise ValueError('Asset object set changed')
    for body in objects:
        old=world.find("body[@name='"+body.get('name')+"']")
        if old is None:raise ValueError('Asset v1 requires direct world object bodies')
        world.remove(old);world.append(copy.deepcopy(body))
    if kind!='sim1':return
    # Reproduce the explicitly recorded source collision/numerical adapter.
    from scipy.spatial.transform import Rotation
    params=read(root/'parameters.json');xml.find('option').set('timestep','.001')
    for side in ('left','right'):
        base=world.find("body[@name='"+side+"_base']")
        position=np.fromstring(base.get('pos'),sep=' ');position[2]+=params.get('source_base_z_offset',0)
        base.set('pos',' '.join(map(str,position)))
        if params.get('flat_finger_pads'):
            for suffix,sign in [('7',-1),('8',1)]:
                body=xml.find(".//body[@name='"+side+'_link'+suffix+"']")
                q=np.fromstring(body.get('quat'),sep=' ');r=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
                p=r.T@(np.array([0.,sign*.003,.113])-np.fromstring(body.get('pos'),sep=' '))
                q=Rotation.from_matrix(r.T).as_quat()
                ET.SubElement(body,'geom',name=side+'_finger_pad_'+suffix,type='box',pos=' '.join(map(str,p)),
                    quat=' '.join(map(str,q[[3,0,1,2]])),size='.02 .003 .021',mass='.001',
                    contype='1',conaffinity='6',condim='4',friction='1 .01 .002',solref='.006 1',rgba='.04 .04 .04 1')
    rolling=params.get('finite_patch_rolling_radius',0)
    if rolling:
        for g in xml.iter('geom'):
            if g.get('name','').startswith(('basket_','left_finger_pad_','right_finger_pad_')):
                g.set('condim','6');g.set('friction',f'1 .01 {rolling}')
    for joint in xml.iter('joint'):
        if joint.get('name','').startswith(('left_joint','right_joint')):
            joint.set('solreflimit','.003 1');joint.set('solimplimit','.99 .999 .001')
