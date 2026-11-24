"""Bounded model experiments. No task-specific recipes or learned defaults.

Profiles change numerical realization or gripper command scaling, never task
geometry, inertial parameters, force/joint limits, timing, or success criteria.
Contact removal is a diagnostic intervention and cannot produce task evidence.
"""
import copy
import itertools
import math
import xml.etree.ElementTree as ET
import numpy as np

REVISION = 'v17'
COUPLING_TOLERANCE_RAD = .002  # declared simulation fidelity gate, not calibration
PROBES = ('remove_object_contacts', 'remove_robot_contacts')
TIMESTEPS_S = (.0005, .001, .002)


def enabled(config):
    return config.get('learning_revision') == REVISION


def catalog(config, scene):
    from pilot_physics import target_model
    xml = target_model(dict(config, model_profile={}))
    couplings = {}
    for i,e in enumerate(xml.findall('./equality/joint')):
        if 'gripper' in e.get('joint1', ''):
            couplings['eq_'+str(i)] = {k:e.get(k,{'polycoef':'0 1 0 0 0','solref':'.02 1','solimp':'.9 .95 .001 .5 2'}.get(k)) for k in ('joint1','joint2','polycoef','solref','solimp')}
    return {'couplings':couplings,
            'box_objects':[o['id'] for o in scene['objects'] if o['shape'] in ('box','tray','open_box','cabinet','cup','bowl')],
            'profile_schema':{'coupling':{'equality_id_from_catalog':{'time_constant_s':[.001,.05], 'impedance':[.8,.9999]}},
                              'solver':{'timestep_s':list(TIMESTEPS_S)},
                              'box_collision':{'object_from_catalog':['primitive','convex']},
                              'gripper_opening_scale':[.6,1.]},
            'semantics':'Absolute fields merge into current profile. Empty profile reproduces the original model. Joint equality polynomials, joint/force limits, mass, friction, dimensions and task predicates stay fixed. Convex uses exactly the same eight corners for each box component.',
            'scope':{'coupling':'constraint solver only; no IK or object pose edits; time_constant_s must be at least twice the selected timestep',
                     'solver':'integration precision only; source timestamps, control targets and physical duration unchanged; smaller steps cost more simulation time',
                     'box_collision':'equivalent collision representation only',
                     'gripper_opening_scale':'source opening command only; no finger state forcing'},
            'diagnostic_probe':{'allowed':list(PROBES),'maximum_per_episode':2,
                'cost':'one normal candidate slot and one full rollout; no additional API calls',
                'restrictions':'Probe runs exactly the best trajectory, scene and model profile with the named contact class disabled. It cannot be selected, recorded as a success, or admitted to memory. Never combine a probe with another edit.'}}


def edit_profile(base, patch, available):
    if not isinstance(patch, dict) or set(patch)-{'coupling','box_collision','gripper_opening_scale','solver'}:
        raise ValueError('Unknown model profile field')
    out=copy.deepcopy(base)
    for field, changes in patch.items():
        if field=='solver':
            if not isinstance(changes,dict) or set(changes)!={'timestep_s'} or type(changes['timestep_s']) not in (int,float) or changes['timestep_s'] not in TIMESTEPS_S:
                raise ValueError('Solver timestep must be one of the declared precision levels')
            out[field]=copy.deepcopy(changes)
            continue
        if field=='gripper_opening_scale':
            if type(changes) not in (int,float) or not math.isfinite(changes) or not .6<=changes<=1.:
                raise ValueError('Opening scale outside .6..1')
            out[field]=changes
            continue
        if not isinstance(changes,dict):raise ValueError('Model field requires named entries')
        for name,value in changes.items():
            if field=='box_collision':
                if name not in available['box_objects'] or value not in ('primitive','convex'):
                    raise ValueError('Unknown object/representation')
                out.setdefault(field,{})[name]=value
            else:
                if name not in available['couplings'] or not isinstance(value,dict) or not value or set(value)-{'time_constant_s','impedance'}:
                    raise ValueError('Unknown coupling or numerical field')
                for key,v in value.items():
                    lo,hi=(.001,.05) if key=='time_constant_s' else (.8,.9999)
                    if type(v) not in (int,float) or not math.isfinite(v) or not lo<=v<=hi:
                        raise ValueError('Coupling parameter outside tool bounds')
                out.setdefault(field,{}).setdefault(name,{}).update(value)
    dt=out.get('solver',{}).get('timestep_s',.002)
    for value in out.get('coupling',{}).values():
        if value.get('time_constant_s',.05)<2*dt:
            raise ValueError('Coupling time constant must be >= 2 * solver timestep; choose a compatible precision level in the same model experiment')
    return out


def apply(xml,config,kind,scene):
    if not enabled(config):
        if config.get('model_profile') or config.get('_diagnostic_probe'):
            raise ValueError('Model experiments require learning_revision=v17')
        return
    profile=config.get('model_profile',{})
    # Validate on every model build, including resumes and non-agent callers.
    edit_profile({},profile,catalog(config,scene))
    if kind=='sim2':
        if profile.get('solver'):
            option=xml.find('option')
            if option is None:option=ET.SubElement(xml,'option')
            option.set('timestep',str(profile['solver']['timestep_s']))
        for i,e in enumerate(xml.findall('./equality/joint')):
            setting=profile.get('coupling',{}).get('eq_'+str(i))
            if not setting:continue
            ref=np.fromstring(e.get('solref','.02 1'),sep=' ')
            imp=np.fromstring(e.get('solimp','.9 .95 .001 .5 2'),sep=' ')
            if 'time_constant_s' in setting:ref[0]=setting['time_constant_s']
            if 'impedance' in setting:imp[:2]=setting['impedance']
            e.set('solref',' '.join(map(str,ref)));e.set('solimp',' '.join(map(str,imp)))
    apply_box_collisions(xml,profile.get('box_collision',{}))
    probe=config.get('_diagnostic_probe')
    if probe:
        if probe not in PROBES:raise ValueError('Unknown diagnostic intervention')
        object_geoms={id(g) for o in scene['objects'] for b in xml.findall(".//body[@name='"+o['id']+"']") for g in b.findall('geom')}
        for g in xml.iter('geom'):
            robot=(g.get('name','').startswith(('astribot_','left_','right_','physical_robot_')))
            if (probe=='remove_object_contacts' and id(g) in object_geoms) or (probe=='remove_robot_contacts' and robot):
                g.set('contype','0');g.set('conaffinity','0')


def apply_box_collisions(xml,representations,mesh_prefix='rssr_equivalent_'):
    """Equivalent convex vertices, shared by fixed assets and bounded repairs."""
    if any(v not in ('primitive','convex') for v in representations.values()):
        raise ValueError('Unknown collision representation')
    if 'convex' not in representations.values():return
    import mujoco
    original=mujoco.MjModel.from_xml_string(ET.tostring(xml,encoding='unicode'))
    asset=xml.find('asset')
    if asset is None:asset=ET.SubElement(xml,'asset')
    for name,rep in representations.items():
        if rep!='convex':continue
        body=xml.find(".//body[@name='"+name+"']")
        if body is None:raise ValueError('Collision object not built')
        # Retain the full compiled inertial frame, not just the scalar mass.
        if body.find('inertial') is None:
            b=original.body(name).id
            ET.SubElement(body,'inertial',pos=' '.join(map(str,original.body_ipos[b])),
                quat=' '.join(map(str,original.body_iquat[b])),mass=str(original.body_mass[b]),
                diaginertia=' '.join(map(str,original.body_inertia[b])))
        for geom in body.findall('geom'):
            if geom.get('type','sphere')!='box':continue
            half=np.fromstring(geom.get('size'),sep=' ')
            vertices=np.array(list(itertools.product((-1,1),repeat=3)))*half
            mesh=mesh_prefix+geom.get('name')
            ET.SubElement(asset,'mesh',name=mesh,vertex=' '.join(map(str,vertices.ravel())))
            geom.set('type','mesh');geom.set('mesh',mesh);geom.attrib.pop('size')


class HealthAudit:
    """Every-step numerical evidence; not a diagnosis or a success shortcut."""
    def __init__(self,model):
        import mujoco
        self.rows=[];self.maxima={};self.temporal={};self.worst_contact=None
        for i in range(model.neq):
            if model.eq_type[i]!=mujoco.mjtEq.mjEQ_JOINT:continue
            a,b=int(model.eq_obj1id[i]),int(model.eq_obj2id[i])
            if 'gripper' not in (model.joint(a).name or ''):continue
            if b<0:continue
            name='eq_'+str(i)
            self.rows.append((name,model.jnt_qposadr[a],model.jnt_qposadr[b],model.eq_data[i,:5].copy()))
            self.maxima[name]=0.

    def observe(self,m,d):
        import mujoco
        for name,a,b,poly in self.rows:
            # MuJoCo joint equality is relative to qpos0, including constant.
            q=d.qpos[b]-m.qpos0[b]
            value=abs(d.qpos[a]-m.qpos0[a]-sum(poly[k]*q**k for k in range(5)))
            value=float(value)
            if name not in self.temporal:
                self.temporal[name]={'initial_rad':value,'peak_time_s':float(d.time)}
            if value>self.maxima[name]:self.temporal[name]['peak_time_s']=float(d.time)
            self.temporal[name]['final_rad']=value
            self.maxima[name]=max(self.maxima[name],value)
        if d.ncon:
            i=int(np.argmin(d.contact.dist));c=d.contact[i];penetration=max(0.,-float(c.dist))
            if self.worst_contact is None or penetration>self.worst_contact['penetration_m']:
                force=np.zeros(6);mujoco.mj_contactForce(m,d,i,force)
                self.worst_contact={'time_s':float(d.time),'geoms':[m.geom(c.geom1).name,m.geom(c.geom2).name],
                    'penetration_m':penetration,'contact_force_norm_n':float(np.linalg.norm(force[:3]))}

    def report(self):
        return {'max_coupling_residual_rad':max(self.maxima.values(),default=0.),
                'per_coupling_residual_rad':self.maxima,'per_coupling_temporal':self.temporal,'worst_contact':self.worst_contact,
                'coupling_tolerance_rad':COUPLING_TOLERANCE_RAD,
                'scope':'all physics steps; geometry/contact diagnostics, not causal proof',
                'hardware_calibrated':False}
