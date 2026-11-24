"""Task-independent rigid, articulated and granular scene construction.

Geometry is an explicit coarse simulation approximation. The model supplies
visual layout and identities; masses, friction and solver settings are tools,
not parameters the repair model can tune to manufacture success.
"""
import copy
import math
import pathlib
import re
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

VERSION='rssr.scene.v2'
SHAPES=('box','open_box','sphere','cylinder','triangular_prism','bottle','tray','cup','bowl','cabinet','region','grain')
MASS={'box':.04,'open_box':.12,'sphere':.025,'cylinder':.04,'triangular_prism':.04,
      'bottle':.06,'tray':.15,'cup':.06,'bowl':.12,'cabinet':1.,'region':.001,'grain':.0003}
ROLES=('object','container','bottle','pouring_cup','grain','fixture','lid','target_region')
COLORS=('red','blue','green','yellow','orange','white','black','brown','gray','other')


def schema():
    def obj(fields,required=None):
        return {'type':'object','properties':fields,'required':list(fields) if required is None else required,'additionalProperties':False}
    def vec(n,lo=None,hi=None):
        number={'type':'number'}
        if lo is not None:number['minimum']=lo
        if hi is not None:number['maximum']=hi
        return {'type':'array','items':number,'minItems':n,'maxItems':n}
    name={'type':'string','pattern':'^[a-z][a-z0-9_]{0,39}$'}
    joint=obj({'name':name,'type':{'enum':['hinge','slide']},'axis':vec(3,-1,1),
               'anchor_in_object_m':vec(3,-.5,.5),'range':vec(2,0,3.2)})
    properties={'id':name,'shape':{'enum':list(SHAPES)},'role':{'enum':list(ROLES)},
                'color':{'enum':list(COLORS)},'position_m':vec(3),'size_m':vec(3,.003,.6),
                'yaw_rad':{'type':'number'},'roll_pitch_rad':vec(2,-math.pi,math.pi),
                'rgba':vec(4,0,1),'mobility':{'enum':['free','static','hinge','slide']},
                'parent_id':{'oneOf':[{'type':'null'},name]},'support_id':{'type':['string','null']},
                'wall_m':{'type':'number','minimum':.0015,'maximum':.02},'joint':joint,
                'head_center':{'oneOf':[{'type':'null'},vec(2,0,1)]},'evidence':{'type':'string','minLength':1,'maxLength':240}}
    item=obj(properties,[k for k in properties if k not in ('wall_m','joint','roll_pitch_rad')])
    ids={'type':'array','items':name,'uniqueItems':True,'maxItems':40}
    bindings=obj({**{k:ids for k in ('targets','containers','excluded','particles')},
                  'target_region':{'oneOf':[{'type':'null'},name]},'articulation':{'oneOf':[{'type':'null'},name]},
                  'stack_hands':obj({'base':{'enum':['left','right']},'upper':{'enum':['left','right']}})},
                  required=['targets','containers','excluded','particles','target_region','articulation'])
    return obj({'schema':{'const':VERSION},'task_id':{'enum':[f'di{i}' for i in range(1,8)]+[f'dm{i}' for i in range(1,9)]},
                'objects':{'type':'array','items':item,'minItems':2,'maxItems':40},'bindings':bindings})


def validate(scene):
    # Validate raw fields separately from deterministically added mass metadata.
    from rssr_protocol import Draft202012Validator
    raw=copy.deepcopy(scene)
    for o in raw.get('objects',[]):
        mass=o.pop('mass_kg',None)
        if mass is not None and mass!=MASS.get(o.get('shape')):raise ValueError('Fixed mass changed')
    Draft202012Validator(schema()).validate(raw)
    result=copy.deepcopy(raw);objects={o['id']:o for o in result['objects']}
    if len(objects)!=len(result['objects']):raise ValueError('Duplicate scene object IDs')
    reserved=('left_','right_','astribot_','rssr_','worktable','floor')
    joint_names=set()
    for o in result['objects']:
        if o['id'].startswith(reserved):raise ValueError('Object identity collides with robot/tool namespace')
        p=np.asarray(o['position_m']);size=np.asarray(o['size_m']);shape=o['shape']
        if not np.isfinite([*p,*size,o['yaw_rad'],*o.get('roll_pitch_rad',[0.,0.])]).all():raise ValueError('Nonfinite scene geometry')
        if not (-.6<=p[0]<=.6 and -1.15<=p[1]<=-.35 and .70<=p[2]<=1.6):raise ValueError('Object outside table workspace')
        if shape in ('sphere','grain') and np.ptp(size)>.00001:raise ValueError('Sphere diameter must agree on all axes')
        if shape in ('cup','bowl','cylinder','bottle') and abs(size[0]-size[1])>.00001:raise ValueError('Round primitive requires equal xy diameters')
        if shape in ('cup','bowl','open_box','cabinet'):
            if 'wall_m' not in o:raise ValueError(f"Object {o['id']}: shape {shape} requires numeric wall_m in [.003,.02], less than half of min(size_m)")
            if 2*o['wall_m']>=size.min():raise ValueError(f"Object {o['id']}: wall_m={o['wall_m']} must be less than half of min(size_m)={float(size.min())} for positive interior volume")
        if shape=='grain' and (size.max()>.012 or o['mobility']!='free'):raise ValueError('Grains are small free bodies')
        if shape in ('region','cabinet') and o['mobility']!='static':raise ValueError('Marker/cabinet must be fixed')
        if o['support_id'] not in (None,'table') and o['support_id'] not in objects:raise ValueError('Unknown support object')
        if o['parent_id'] is not None and o['parent_id'] not in objects:raise ValueError('Unknown articulated parent')
        if o['parent_id']==o['id'] or o['support_id']==o['id']:raise ValueError('Self-support/parent is invalid')
        if o['mobility']=='free' and o['parent_id'] is not None:raise ValueError('Free bodies cannot have a kinematic parent')
        if o['mobility'] in ('hinge','slide'):
            j=o.get('joint')
            if not j or j['type']!=o['mobility'] or j['name'] in joint_names:raise ValueError('Invalid articulation identity')
            if o['parent_id'] is None:raise ValueError('Articulated object needs an explicit fixture/parent')
            if abs(np.linalg.norm(j['axis'])-1)>.001 or j['range'][0]!=0 or not .01<=j['range'][1]<=(.4 if j['type']=='slide' else math.pi):raise ValueError('Invalid articulation axis/range')
            if np.max(np.abs(j['anchor_in_object_m']))>.6:raise ValueError('Joint anchor outside tool bounds')
            joint_names.add(j['name'])
        elif 'joint' in o:raise ValueError('Free/static object cannot override articulation')
        o['mass_kg']=MASS[shape]
    for o in result['objects']:
        current=o;seen=set()
        while current['parent_id'] is not None:
            if current['id'] in seen:raise ValueError('Cyclic scene hierarchy')
            seen.add(current['id']);current=objects[current['parent_id']]
    validate_bindings(result)
    return result


def validate_bindings(scene):
    objects={o['id']:o for o in scene['objects']};b=scene['bindings'];task=scene['task_id']
    for group in ('targets','containers','excluded','particles'):
        if any(i not in objects for i in b[group]):raise ValueError('Unknown task-bound object')
    from rssr_task_contracts import compile_contract
    compile_contract(task,scene,b)  # cardinality and required relations
    targets=[objects[i] for i in b['targets']];excluded=[objects[i] for i in b['excluded']]
    # A visible spare cup may legitimately have role=container while being a
    # distractor. Only task bindings identify receiving containers; changing a
    # shape/role label must not hide an independently manipulable object.
    bound_environment=set(b['containers'])|set(b['particles'])
    ordinary=[o for o in objects.values() if o['id'] not in bound_environment and
              (o['role']=='object' or o['mobility']=='free')]
    if any(o['mobility']!='free' for o in targets if o['role'] not in ('fixture','lid')):raise ValueError('Task targets must be physical free objects')
    if any(o['shape'] not in ('open_box','cup','bowl','tray') for o in [objects[i] for i in b['containers']]):raise ValueError('Container needs corresponding physical geometry')
    if task=='di1' and (len(targets)!=1 or targets[0]['color']!='blue' or targets[0]['shape']!='box'):raise ValueError('di1 requires the observed blue box')
    if task=='di2' and targets[0]['shape']!='triangular_prism':raise ValueError('di2 requires triangular geometry')
    if task=='di3':
        blocks=[o for o in ordinary if o['shape'] in ('box','triangular_prism')]
        volume=lambda o:np.prod(o['size_m'])*(.5 if o['shape']=='triangular_prism' else 1)
        if len(blocks)<2 or any(volume(o)>=volume(targets[0]) for o in blocks if o['id']!=targets[0]['id']):raise ValueError('di3 selected object is not uniquely largest')
    if task=='di4':
        references=[o for o in ordinary if o['shape']=='cylinder' and o['color']=='red']
        if len(references)!=1 or references[0]['head_center'] is None or targets[0]['head_center'] is None:raise ValueError('di4 needs observed left-of relation in head view')
        if targets[0]['head_center'][0]>=references[0]['head_center'][0]:raise ValueError('Selected object is not to the left of red cylinder')
    if task in ('di1','di2','di3','di4'):
        unused={o['id'] for o in ordinary}-{o['id'] for o in targets}
        if set(b['excluded'])!=unused:
            missing=sorted(unused-set(b['excluded']));extra=sorted(set(b['excluded'])-unused)
            raise ValueError(f'Unselected objects must all remain represented: missing excluded IDs={missing}; unexpected excluded IDs={extra}. Task container identity comes from bindings.containers, not the object role label.')
    if task=='di5' and ({o['id'] for o in ordinary}!=set(b['targets']+b['excluded']) or len(ordinary)!=3):raise ValueError('di5 requires exactly three observed cubes')
    if task=='di6':
        upper,lower=[objects[i] for i in b['containers']]
        if upper['head_center'] is None or lower['head_center'] is None or upper['head_center'][1]>=lower['head_center'][1]:raise ValueError('Basket order must match observed upper/lower image layout')
    if task=='di7':
        red=[o for o in ordinary if o['shape']=='box' and o['color']=='red']
        if len(red)!=1 or b['excluded']!=[red[0]['id']] or set(b['targets'])!={o['id'] for o in ordinary if o['id']!=red[0]['id']}:raise ValueError('di7 must include all objects except the red box')
    if task=='dm2' and targets[0]['shape']!='sphere':raise ValueError('dm2 needs a physical ball')
    if task=='dm3' and (any(o['shape']!='bottle' for o in targets) or set(b['targets'])!={o['id'] for o in objects.values() if o['role']=='bottle'}):raise ValueError('dm3 must include all observed bottles')
    if task=='dm5' and (targets[0]['shape']!='cup' or any(objects[i]['shape']!='grain' for i in b['particles'])):raise ValueError('dm5 needs pouring cup and explicit free grains')
    if task in ('dm4','dm7') and (b['target_region'] not in objects or objects[b['target_region']]['shape']!='region'):raise ValueError('Task needs a reconstructed target marker')
    if task in ('dm6','dm8'):
        joints=[o for o in objects.values() if o.get('joint',{}).get('name')==b['articulation']]
        if len(joints)!=1 or joints[0]['mobility']!=('slide' if task=='dm6' else 'hinge'):raise ValueError('Wrong drawer/lid articulation')
        if task=='dm6' and joints[0]['id']!=b['containers'][0]:raise ValueError('Containment must use moving drawer body')
    return True


def _vec(values):return ' '.join(f'{float(v):.12g}' for v in values)


def object_rotation(obj):
    return Rotation.from_euler('xyz', [*obj.get('roll_pitch_rad',[0.,0.]),obj['yaw_rad']]).as_matrix()


def lowest_geometry_offset(obj):
    """World z of the lowest primitive point, relative to its body origin.

    Used for initial tabletop support, not to force a dynamic object onto the
    table. Matches add_geometry for bottles, cylinders, spheres and prisms.
    """
    row=object_rotation(obj)[2];size=np.asarray(obj['size_m']);shape=obj['shape']
    if shape in ('sphere','grain'):return -float(size[0]/2)
    radial=float(np.linalg.norm(row[:2]));vertical=float(abs(row[2]))
    if shape=='cylinder':return -float(size[0]/2*radial+size[2]/2*vertical)
    if shape=='bottle':
        h=size[2]
        return min(float(-h*.14*row[2]-size[0]/2*radial-h*.36*vertical),
                   float(h*.36*row[2]-size[0]*.24*radial-h*.14*vertical))
    if shape=='triangular_prism':
        sx,sy,h=size
        vertices=np.array([(x,y,z) for z in (-h/2,h/2)
                          for x,y in ((-sx/2,-sy/2),(sx/2,-sy/2),(0,sy/2))])
        return float((vertices@row).min())
    return -float(np.abs(row)@(size/2))


def add_geometry(asset,body,obj):
    name=obj['id'];sx,sy,h=obj['size_m'];shape=obj['shape'];mass=obj['mass_kg']
    attrs={'contype':'4','conaffinity':'15','condim':'4','friction':'1 .01 .002',
           'solref':'.006 1','solimp':'.95 .99 .001','rgba':_vec(obj['rgba'])}
    def geom(suffix,kind,size,pos=(0,0,0),fraction=1.,**kwargs):
        return ET.SubElement(body,'geom',name=name+'_'+suffix,type=kind,size=_vec(size),pos=_vec(pos),mass=str(mass*fraction),**attrs,**kwargs)
    if shape in ('box','tray','region'):
        if shape=='region':attrs.update(contype='0',conaffinity='0')
        geom('geom','box',[sx/2,sy/2,h/2])
    elif shape in ('sphere','grain'):geom('geom','sphere',[sx/2])
    elif shape=='cylinder':geom('geom','cylinder',[sx/2,h/2])
    elif shape=='bottle':
        geom('body','cylinder',[sx/2,h*.36],[0,0,-h*.14],.85)
        geom('neck','cylinder',[sx*.24,h*.14],[0,0,h*.36],.15)
    elif shape=='triangular_prism':
        vertices=[(-sx/2,-sy/2,-h/2),(sx/2,-sy/2,-h/2),(0,sy/2,-h/2),
                  (-sx/2,-sy/2,h/2),(sx/2,-sy/2,h/2),(0,sy/2,h/2)]
        ET.SubElement(asset,'mesh',name=name+'_mesh',vertex=_vec(np.array(vertices).ravel()))
        ET.SubElement(body,'geom',name=name+'_geom',type='mesh',mesh=name+'_mesh',mass=str(mass),**attrs)
    elif shape in ('open_box','cabinet'):
        t=obj['wall_m']
        parts=[('bottom',(0,0,-h/2+t/2),(sx/2,sy/2,t/2)),
               ('xplus',(sx/2-t/2,0,0),(t/2,sy/2,h/2)),('xminus',(-sx/2+t/2,0,0),(t/2,sy/2,h/2)),
               ('back',(0,sy/2-t/2,0),(sx/2-t,t/2,h/2))]
        parts.append(('front',(0,-sy/2+t/2,0),(sx/2-t,t/2,h/2)) if shape=='open_box' else
                     ('top',(0,0,h/2-t/2),(sx/2,sy/2,t/2)))
        for suffix,pos,size in parts:geom(suffix,'box',size,pos,1/len(parts))
    elif shape in ('cup','bowl'):
        t=obj['wall_m'];radius=sx/2
        geom('bottom','cylinder',[radius,t/2],[0,0,-h/2+t/2],.2)
        for i in range(16):
            angle=2*math.pi*i/16;r=radius-t/2
            geom(f'wall_{i}','box',[t/2,r*math.tan(math.pi/16),h/2],
                 [r*math.cos(angle),r*math.sin(angle),0],.8/16,euler=_vec([0,0,angle]))
    else:raise ValueError('Unimplemented shape '+shape)


def append_objects(xml,scene):
    scene=validate(scene);world=xml.find('worldbody');asset=xml.find('asset')
    if asset is None:asset=ET.SubElement(xml,'asset')
    pending={o['id']:o for o in scene['objects']};created={};specs={o['id']:o for o in scene['objects']}
    while pending:
        progress=False
        for name,o in list(pending.items()):
            parent=o['parent_id']
            if parent is not None and parent not in created:continue
            p=np.asarray(o['position_m']);rotation=object_rotation(o)
            if parent:
                prior=specs[parent];rot=object_rotation(prior)
                p=rot.T@(p-prior['position_m']);rotation=rot.T@rotation
            quaternion=Rotation.from_matrix(rotation).as_quat()
            body=ET.SubElement(created[parent] if parent else world,'body',name=name,pos=_vec(p),quat=_vec(quaternion[[3,0,1,2]]))
            if o['mobility']=='free':ET.SubElement(body,'freejoint',name=name+'_free')
            elif o['mobility'] in ('hinge','slide'):
                j=o['joint'];ET.SubElement(body,'joint',name=j['name'],type=j['type'],axis=_vec(j['axis']),
                    pos=_vec(j['anchor_in_object_m']),limited='true',range=_vec(j['range']),
                    damping='.1' if j['type']=='slide' else '.01',frictionloss='.1' if j['type']=='slide' else '.001')
            add_geometry(asset,body,o);created[name]=body;del pending[name];progress=True
        if not progress:raise ValueError('Unresolvable scene hierarchy')
    return xml


def build(config,kind,folder,scene):
    import mujoco
    from pilot_physics import source_model,target_model
    if config.get('collision_integrity_revision') not in (None, 's1_cross_arm_v1',
                                                         's1_cross_arm_visual_gripper_v2'):
        raise ValueError('Unknown collision integrity revision')
    xml=source_model(config) if kind=='sim1' else target_model(config)
    visual=xml.find('visual')
    if visual is None:visual=ET.SubElement(xml,'visual')
    quality=visual.find('quality')
    if quality is None:quality=ET.SubElement(visual,'quality')
    quality.set('shadowsize','2048')
    ET.SubElement(xml.find('worldbody'),'light',pos='0 -.7 2.5',dir='0 0 -1',diffuse='.7 .7 .7')
    for g in xml.iter('geom'):
        if g.get('name')=='worktable_top':g.set('rgba','.88 .88 .86 1')
        if g.get('name','').startswith('physical_robot_'):g.set('rgba','0 0 0 0')
    append_objects(xml,scene)
    from rssr_engineered_assets import apply as apply_engineered_asset
    apply_engineered_asset(xml,config,kind,scene)
    from rssr_task_assets import apply as apply_task_asset
    apply_task_asset(xml,config,kind,scene)
    from rssr_model_tools import apply as apply_model_profile
    apply_model_profile(xml,config,kind,scene)
    if config.get('engineered_task_asset'):
        from rssr_engineered_assets import load as load_engineered_asset
        _, definition = load_engineered_asset(config)
        if definition.get('round_table_dispatch'):
            from rssr_round_table import apply as apply_round_table, REVISION
            if definition['round_table_dispatch'] != REVISION:
                raise ValueError('Unknown round-table collision policy')
            apply_round_table(xml, scene)
    # Engineered assets may replace the exact tabletop box with an 8-vertex mesh.
    # Canonicalize that equivalent mesh back to a box before MuJoCo compilation.
    from pilot_physics import _convert_exact_worktable_meshes_to_boxes
    if kind=='sim2': _convert_exact_worktable_meshes_to_boxes(xml)
    if kind=='sim2' and config.get('collision_integrity_revision') in (
            's1_cross_arm_v1', 's1_cross_arm_visual_gripper_v2'):
        if config.get('_diagnostic_probe'):
            raise ValueError('Collision integrity cannot override a diagnostic contact-removal probe')
        from rssr_collision_integrity import apply_s1_cross_arm_collision
        apply_s1_cross_arm_collision(xml, include_visual_gripper_meshes=(
            config['collision_integrity_revision']=='s1_cross_arm_visual_gripper_v2'))
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'scene.xml';ET.indent(xml);path.write_text(ET.tostring(xml,encoding='unicode'))
    return mujoco.MjModel.from_xml_path(str(path))
