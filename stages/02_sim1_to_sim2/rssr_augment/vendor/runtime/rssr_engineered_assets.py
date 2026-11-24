"""Human-engineered task geometry, episode-local placement, ordinary dynamics.

Asset templates are inputs, not learned memory or assertions of success. Only
an executed Sim1 receipt can freeze an instance for target retargeting.
"""
import copy
import pathlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from common import read, write, sha
from rssr_scene import validate, lowest_geometry_offset

VERSION = 'rssr.engineered_task_asset.v1'


def load(config):
    root = pathlib.Path(config['engineered_task_asset'])
    manifest = read(root/'manifest.json')
    if manifest.get('schema') != VERSION:
        raise ValueError('Unknown engineered task asset')
    if config.get('engineered_task_asset_sha256') != sha(root/'manifest.json'):
        raise ValueError('Engineered asset version changed')
    for name, identity in manifest['files'].items():
        p = (root/name).resolve()
        if p.parent != root.resolve() or sha(p) != identity:
            raise ValueError('Engineered asset content changed: '+name)
    return root, read(root/'definition.json')


def supported(obj):
    if obj['support_id'] == 'table':
        obj['position_m'][2] = .7503-lowest_geometry_offset(obj)


def _move_group(scene, name, shift):
    """Translate a fixture and all physically attached/supported descendants."""
    members = {name}
    while True:
        more = {o['id'] for o in scene['objects']
                if o['parent_id'] in members or o['support_id'] in members}
        if more <= members: break
        members |= more
    for obj in scene['objects']:
        if obj['id'] in members:
            obj['position_m'] = (np.asarray(obj['position_m'])+shift).tolist()


def fit_release_goals(scene, goals, evidence, tracks, table_bounds_xy=None):
    """Initialize receivers from source FK, without executing or forcing objects.

    Rigid holding is a placement hypothesis only. The full physical rollout
    must still verify grasping, insertion, release and all task predicates.
    """
    from rssr_scene import object_rotation
    objects={o['id']:o for o in scene['objects']};receipts=[]
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for goal in goals:
        receiver=objects[goal['receiver']]
        if receiver['parent_id'] is not None or receiver.get('joint'):
            raise ValueError('Release fit cannot move an articulated/attached receiver')
        rotation=object_rotation(receiver);points=[];predictions=[]
        for assignment in goal['releases']:
            obj=objects[assignment['object']];hand=assignment['hand'];cycle=assignment.get('cycle',0)
            events=evidence['events'][hand].get('cycles',[])
            if cycle>=len(events):continue
            event=events[cycle];closed=event['closed_index'];released=event['release_index']
            orientations=tracks[hand+'_rotation'];centers=tracks[hand+'_center']
            rel=orientations[closed].T@(np.asarray(obj['position_m'])-centers[closed])
            center=centers[released]+orientations[released]@rel
            orientation=orientations[released]@orientations[closed].T@object_rotation(obj)
            if obj['shape'] in ('sphere','grain'):
                local=center@rotation+signs*obj['size_m'][0]/2
            else:
                vertices=(signs*np.asarray(obj['size_m'])/2)@orientation.T+center
                local=vertices@rotation
            points.extend(local);predictions.append({'object':obj['id'],'hand':hand,'cycle':cycle,
                'source_release_index':int(released),'predicted_center_m':center.tolist()})
        if len(predictions)!=len(goal['releases']):
            receipts.append({'receiver':receiver['id'],'applied':False,'reason':'missing_source_release'});continue
        # Center the union of predicted footprints in the unchanged receiver.
        points=np.asarray(points);lo=points[:,:2].min(axis=0);hi=points[:,:2].max(axis=0)
        local=np.asarray(receiver['position_m'])@rotation
        target=local.copy();target[:2]=(lo+hi)/2
        layout_fit=None
        if goal.get('avoid_initial_overlap'):
            from rssr_action_layout import collision_free_receiver_center
            if table_bounds_xy is None:raise ValueError('Overlap-aware initialization requires actual table bounds')
            wall=receiver.get('wall_m',0.) if receiver['shape']=='open_box' else 0.
            clearance=(np.asarray(receiver['size_m'][:2])-2*wall-(hi-lo))/2
            layout_fit=collision_free_receiver_center(scene,receiver['id'],target[:2],clearance,table_bounds_xy=table_bounds_xy)
            if layout_fit['applied']:target[:2]=layout_fit['selected_local_xy_m']
            else:
                receipts.append({'receiver':receiver['id'],'applied':False,'reason':layout_fit['reason'],
                                 'initial_overlap_fit':layout_fit,'geometry_resized':False})
                continue
        shift=rotation@(target-local);shift[2]=0.
        before=list(receiver['position_m']);_move_group(scene,receiver['id'],shift)
        wall=receiver.get('wall_m',0.) if receiver['shape']=='open_box' else 0.
        clearance=(np.asarray(receiver['size_m'][:2])-2*wall-(hi-lo))/2
        receipts.append({'receiver':receiver['id'],'applied':True,'initial_position_before_m':before,
            'initial_position_after_m':list(receiver['position_m']),'predictions':predictions,
            'predicted_xy_clearance_m':clearance.tolist(),'geometry_resized':False,
            'basis':'source FK rigid-hold hypothesis; not measured object motion or success evidence'})
        if layout_fit is not None:receipts[-1]['initial_overlap_fit']=layout_fit
    return receipts


def bind(config, folder):
    """Rebind source-action anchors; preserve source joint state and timestamps."""
    from pilot_physics import action_evidence
    root, definition = load(config)
    scene = copy.deepcopy(definition['scene'])
    folder = pathlib.Path(folder)
    ev = action_evidence(config, 'sim1', folder/'action_geometry')
    z = np.load(folder/'action_geometry/kinematic_reference.npz')
    objects = {o['id']: o for o in scene['objects']}
    receipts = []
    for anchor in definition['anchors']:
        side = anchor['hand']; event = ev['events'][side]
        cycles = event.get('cycles', [])
        cycle = anchor.get('cycle', 0)
        if cycle >= len(cycles):
            receipts.append(dict(anchor=anchor, applied=False, reason='no_matching_source_grasp'))
            continue
        k = cycles[cycle]['closed_index']
        rot = z[side+'_rotation'][k]
        p = z[side+'_center'][k]-rot[:, 2]*definition['grasp_depth_m']
        obj = objects[anchor['object']]
        old = np.asarray(obj['position_m']).copy()
        if anchor['mode'] == 'basket_wall':
            normal = rot[:2, 1].copy(); normal /= np.linalg.norm(normal)
            obj['yaw_rad'] = float(np.arctan2(normal[1], normal[0]))
            slope = definition.get('tapered_baskets', {}).get(obj['id'], 0.)
            h = obj['size_m'][2]; width = obj['size_m'][0]
            radius = (width-obj['wall_m'])/2+(p[2]-obj['position_m'][2]-h/2)*np.tan(slope)
            obj['position_m'][:2] = (p[:2]-normal*radius).tolist()
        elif anchor['mode'] == 'round_rim':
            normal = rot[:2, 1].copy(); normal /= np.linalg.norm(normal)
            obj['position_m'][:2] = (p[:2]-normal*(obj['size_m'][0]-obj['wall_m'])/2).tolist()
        elif anchor['mode'] == 'handle':
            handle = objects[anchor['handle']]
            shift = p-np.asarray(handle['position_m']); shift[2] = 0.
            _move_group(scene, obj['id'], shift)
        else:
            shift = p-old; shift[2] = 0.
            _move_group(scene, obj['id'], shift)
            if obj['shape'] in ('box', 'triangular_prism'):
                obj['yaw_rad'] = float(np.arctan2(-rot[0, 1], rot[1, 1]))
        supported(obj)
        receipts.append(dict(anchor=anchor, applied=True, source_index=int(k),
                             source_grasp_m=p.tolist(), initial_position_m=obj['position_m']))
    orientation_receipts=[]
    for goal in definition.get('orientation_goals',[]):
        if goal.get('mode')!='horizontal_to_upright_release':raise ValueError('Unknown orientation initialization')
        obj=objects[goal['object']]
        if obj['shape']!='bottle' or obj['parent_id'] is not None:raise ValueError('Expected an independent bottle')
        hand=goal['hand'];cycle=goal.get('cycle',0);events=ev['events'][hand].get('cycles',[])
        if cycle>=len(events):
            orientation_receipts.append({'object':obj['id'],'applied':False,'reason':'missing_source_cycle'});continue
        from rssr_action_layout import horizontal_axis_for_upright_release
        event=events[cycle]
        hypothesis=horizontal_axis_for_upright_release(z[hand+'_rotation'][event['closed_index']],
                                                      z[hand+'_rotation'][event['release_index']])
        obj['yaw_rad']=hypothesis['yaw_rad'];obj['roll_pitch_rad']=hypothesis['roll_pitch_rad'];supported(obj)
        orientation_receipts.append(dict(hypothesis,object=obj['id'],hand=hand,cycle=cycle,applied=True))
    table_bounds=None
    if any(goal.get('avoid_initial_overlap') for goal in definition.get('release_goals',[])):
        import mujoco
        from rssr_action_layout import table_bounds_from_model
        table_bounds=table_bounds_from_model(mujoco.MjModel.from_xml_path(str(folder/'action_geometry/scene.xml')))
    release_receipts=fit_release_goals(scene,definition.get('release_goals',[]),ev,z,table_bounds)
    scene = validate(scene)
    write(folder/'initial_scene.json', scene)
    write(folder/'placement.json', {'asset_sha256':sha(root/'manifest.json'),
        'source_action_sha256':sha(pathlib.Path(config['import_case'])/'source_action_state.npz'),
        'anchors':receipts,'release_goals':release_receipts,'orientation_goals':orientation_receipts,
        'success_asserted':False, 'geometry_is_approximate':True})
    return scene


def apply(xml, config, kind, scene):
    if not config.get('engineered_task_asset'): return
    _, definition = load(config)
    original = {o['id']:o for o in definition['scene']['objects']}
    if scene['task_id'] != definition['scene']['task_id'] or scene['bindings'] != definition['scene']['bindings']:
        raise ValueError('Task identity/bindings cannot be changed')
    if set(original) != {o['id'] for o in scene['objects']}:
        raise ValueError('Task object set cannot be changed')
    for obj in scene['objects']:
        if any(obj.get(k) != original[obj['id']].get(k) for k in
               ('shape','role','color','size_m','mass_kg','mobility','parent_id','support_id','joint','wall_m')):
            raise ValueError('Asset geometry or semantics changed: '+obj['id'])
    if definition.get('cabinet_open_face') == '+y':
        for obj in scene['objects']:
            if obj['shape'] != 'cabinet': continue
            back = xml.find(".//geom[@name='"+obj['id']+"_back']")
            pos = np.fromstring(back.get('pos'), sep=' '); pos[1] *= -1
            back.set('pos',' '.join(map(str,pos)))
    limit_settings=definition.get('articulation_limit_solver')
    if limit_settings:
        for obj in scene['objects']:
            if not obj.get('joint'):continue
            j=xml.find(".//joint[@name='"+obj['joint']['name']+"']")
            for field,value in limit_settings.items():j.set(field,' '.join(map(str,value)))
    # Match the engineered flared basket used in source and target dynamics.
    for name, slope in definition.get('tapered_baskets', {}).items():
        body = xml.find(".//body[@name='"+name+"']")
        obj = original[name]; sx, sy, h = obj['size_m']; t = obj['wall_m']
        dx = h/2*np.tan(slope)
        if min(sx/2, sy/2)-2*dx <= t: raise ValueError('Invalid basket taper')
        for geom in body.findall('geom'):
            if geom.get('name','').endswith('bottom'):
                geom.set('size', f'{sx/2-2*dx} {sy/2-2*dx} {t/2}')
                continue
            pos = np.fromstring(geom.get('pos'), sep=' ')
            size = np.fromstring(geom.get('size'), sep=' ')
            axis = 0 if abs(pos[0]) > 0 else 1; sign = np.sign(pos[axis])
            pos[axis] -= sign*dx; size[2] = h/2/np.cos(slope)
            rot = Rotation.from_euler('y' if axis == 0 else 'x', sign*slope*(1 if axis == 0 else -1))
            q = rot.as_quat()
            geom.set('pos',' '.join(map(str,pos))); geom.set('size',' '.join(map(str,size)))
            geom.set('quat',' '.join(map(str,q[[3,0,1,2]])))
    if definition.get('box_collision'):
        from rssr_model_tools import apply_box_collisions
        apply_box_collisions(xml,definition['box_collision'],mesh_prefix='engineered_equivalent_')
    if kind != 'sim1':
        if definition.get('round_contact_model'):
            from rssr_round_contact import apply as apply_round_contact
            apply_round_contact(xml, scene, definition['round_contact_model'])
        return
    xml.find('option').set('timestep','.001')
    for side in ('left','right'):
        for suffix, sign in [('7',-1),('8',1)]:
            body = xml.find(".//body[@name='"+side+'_link'+suffix+"']")
            q = np.fromstring(body.get('quat'), sep=' ')
            r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            p = r.T@(np.array([0.,sign*.003,.113])-np.fromstring(body.get('pos'),sep=' '))
            q = Rotation.from_matrix(r.T).as_quat()
            ET.SubElement(body,'geom',name=side+'_finger_pad_'+suffix,type='box',
                pos=' '.join(map(str,p)),quat=' '.join(map(str,q[[3,0,1,2]])),size='.02 .003 .021',
                mass='.001',contype='1',conaffinity='6',condim='4',friction='1 .01 .002',
                solref='.006 1',rgba='.04 .04 .04 1')
    for j in xml.iter('joint'):
        if j.get('name','').startswith(('left_joint','right_joint')):
            j.set('solreflimit','.003 1'); j.set('solimplimit','.99 .999 .001')
    if definition.get('round_contact_model'):
        from rssr_round_contact import apply as apply_round_contact
        apply_round_contact(xml, scene, definition['round_contact_model'])
    precision=config.get('source_precision')
    if precision:
        if set(precision)!={'timestep_s','joint_limit_time_constant_s'}:
            raise ValueError('Unknown source precision parameter')
        dt=float(precision['timestep_s']);tau=float(precision['joint_limit_time_constant_s'])
        if dt not in (.00025,.0005,.001) or not 2*dt<=tau<=.003:
            raise ValueError('Invalid source numerical precision')
        xml.find('option').set('timestep',str(dt))
        for j in xml.iter('joint'):
            if j.get('name','').startswith(('left_joint','right_joint')):
                j.set('solreflimit',f'{tau} 1')


def validate_source_receipt(config, scene):
    """Required before Sim2: same scene and corrected source command checksum."""
    from rssr_protocol import digest
    receipt = read(config['source_success_receipt'])
    if not receipt.get('passed') or receipt['scene_sha256'] != digest(scene):
        raise ValueError('Sim2 requires a successful frozen Sim1 scene')
    if receipt['action_sha256'] != sha(pathlib.Path(config['import_case'])/'source_action_state.npz'):
        raise ValueError('Retarget source commands differ from successful Sim1')
    if receipt['asset_manifest_sha256'] != config['engineered_task_asset_sha256']:
        raise ValueError('Sim1 and Sim2 asset versions differ')
    report = pathlib.Path(receipt['report'])
    if sha(report) != receipt['report_sha256'] or read(report).get('task_success') is not True:
        raise ValueError('Source physical proof changed or did not pass')
