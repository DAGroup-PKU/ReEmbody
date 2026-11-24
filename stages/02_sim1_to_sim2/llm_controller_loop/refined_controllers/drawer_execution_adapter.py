"""HOST v4 numerical execution adapter for a Luna-selected drawer policy.

The model chooses policy JSON. This host tool supplies reliable typed dispatch,
rigid-camera IK and the frozen source's complete task protocol. The source file
is unchanged. A documented AST adaptation regenerates handle reach/pull from the
new actual continuous pregrasp branch instead of replaying old planned joints.
No physical model, contact, margin, evaluator, replay, or clock is modified.
"""
import ast,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import drawer_numerical_tools as NT
import drawer_region_geometry_v3 as RG
import camera_space_route as CS
from joint_region_path import retime_new_plan
DR=NT.DR;D=Path(__file__).resolve().parent
SOURCE=NT.G.BASE
SOURCE_SHA='317c4f0d09cb3afa64187ce2608f4053422873d712f463ef2fc2bfa4b1cebf86'
import os as _os
ARCHDIR=Path(__file__).resolve().parents[2] / 'franka_dual_v1'
CATDIR=Path(_os.environ.get('REEMBODY_DRAWER_LOOK_CATALOG_DIR', str(Path(__file__).resolve().parent / 'look_catalog')))
CAT_HASHES={'left_candidates.json':'ed01a6c55ed30e42875ce0f489db292cfe74a323ed5dc50cbe011bf6a698def7','paired_candidates.json':'5ec398f1aa069d46701dddec1652d7cfa93c1c6720dfe7bba57ba4fe506e83f8','provenance.json':'0409be1c83ad06226e72608696968ad73307f2adee3eb0191042ad4c320286f3','right_candidates.json':'c0994ad90554458d598dc866308998583ee375a2c26e1405115e7191b488ad54'}

def roles(S):return NT.G.base.roles(S)
def grasps(S):return NT.G.base.grasps(S)

def slide_follow(S,h,obj):
    if obj!='bottom_handle_bar' or not all(S.contacts(h,obj)):return None
    return {'hand':h,'object':obj,'joint':S.objects['bottom_drawer']['joint']['name']}

def audit(S,plans,targets):
    slide=None
    for h,obj in targets.items():
        following=slide_follow(S,h,obj)
        if following is not None:slide=following
    return RG.audit(S,plans,targets,slide_follow=slide)

def require_facing(report):
    if not report.get('all_target_regions_in_FOV',False):raise DR.PlanError('Actual post-orientation target-region FOV fails')

def retime_fresh_paths(plans,label,policy):
    """Only newly generated planner paths, before any actual commands exist.

    The model explicitly selects labels. All selected joint curves retain at
    most the original loaded cap .5rad/s; gripper ramp and original physical
    controller/clock remain native. The paired result is rechecked afterward.
    """
    if label not in policy.get('retime_labels',[]):return plans
    velocity=float(policy['retime_joint_speed_rad_s'])
    if not (0.<velocity<=.5):raise ValueError('New-plan retiming may not exceed original loaded .5rad/s cap')
    return {h:retime_new_plan(policy['_current_q'][h],p,min(DR.STEP_MAX,velocity/30)) for h,p in plans.items()}

def execute_once(S,parts,label,*,native_together,targets_for,grips=None,allow=None,delay=None,policy=None):
    """One entry point for original source parts; no native replay after dispatch."""
    for part in parts:
        p={h:np.asarray(q,float) for h,q in part.items() if len(q)}
        if policy is not None:p=retime_fresh_paths(p,label,{**policy,'_current_q':{h:S.q_cmd(h) for h in p}})
        if p:
            report=audit(S,p,{h:targets_for(h) for h in p});require_facing(report)
            if report.get('rows'):
                S.note('HOST predicted region audit before dispatch '+str(dict(label=label,frames=report['command_frames'],FOV=True,visible_region_each_frame=report['all_target_regions_visible'],zero_target_ray_frames=sum(v['unoccluded_target_probes']==0 for r in report['rows'] for v in r['by_hand'].values()))))
        NT.checked_together(S,p,label,native_together=native_together,grips=grips,allow=allow,delay=delay)

def forward_handle_plans(S,hand,g_bar,rotation,pull,axis):
    """Regenerate complete future reach/pull from actual NEW pregrasp q_cmd.

    Future geometric planning is not real execution or a camera-tracking sample;
    slide prediction is applied only later when actual bilateral contact exists.
    """
    reach=S.plan_line({hand:(g_bar,rotation)},speed=.06,start={hand:S.q_cmd(hand)},label='reach',soft={hand:np.array([0.,0.,1.])})[hand]
    try:
        pull_path=S.plan_line({hand:(g_bar+pull*axis,rotation)},speed=.06,start={hand:reach[-1]},label='pull')[hand]
    except Exception:
        pull_path=S.plan_line({hand:(g_bar+pull*axis,rotation)},speed=.06,start={hand:reach[-1]},label='pull',soft={hand:np.array([0.,0.,1.])})[hand]
    S.note('HOST numeric adapter: complete handle reach/pull regenerated forward from actual continuous pregrasp branch; no old joint splice')
    return reach,pull_path

def source_protocol():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise ValueError('Frozen original drawer source changed')
    spec=importlib.util.spec_from_file_location('host_v2_original_drawer_protocol',SOURCE);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tree=ast.parse(SOURCE.read_text());run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    adapted=[];insertions=0
    for statement in run.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='reach' for t in statement.targets):
            injection=ast.parse('reach_plan, pull_plan = _HOST_FORWARD_HANDLE(S, dh, g_bar, Rh, pull, axis)').body[0]
            adapted.append(injection);insertions+=1
        adapted.append(statement)
    if insertions!=1:raise ValueError('Original source task protocol structure changed')
    run.body=adapted;module._HOST_FORWARD_HANDLE=forward_handle_plans
    exec(compile(ast.fix_missing_locations(ast.Module(body=[run],type_ignores=[])),str(D/'drawer_execution_adapter_v4.py')+'::source_protocol', 'exec'),module.__dict__)
    return module

def configure_insertion(source,policy):
    """Apply model-selected side-route geometry without changing task stages.

    Source SIDE_GAP is a waypoint offset, not a collision margin. The unchanged
    cabinet 3cm/full-body/paired guards validate every newly planned trajectory.
    Choosing source_side skips only the source's alternative direct-route IK
    search; acquisition, pull, insertion, release, withdrawal and close remain.
    Absence preserves the previous adapter behavior for dispatch regression tests.
    """
    route=policy.get('insertion_route')
    if route is None:return
    if route.get('family')!='source_side':raise ValueError('Unsupported model-selected insertion route')
    gap=float(route['side_gap_m']);turns=tuple(float(v) for v in route['side_turns_deg'])
    if not (.07<=gap<=.16):raise ValueError('Side waypoint gap outside numerical-tool range')
    if not turns or any(v not in (0.,20.,-20.) for v in turns):raise ValueError('Unsupported source side yaw')
    source.SIDE_GAP=gap;source.SIDE_TURNS=turns;source.STAGE_DISTS=()

def choose_initial(catalog,policy):
    rows=catalog if isinstance(catalog,list) else catalog['pairs']
    valid=[]
    for row in rows:
        facing=row.get('endpoint_facing',{})
        if facing and all(r.get('region_probes_inside_FOV',0)>0 and r.get('unoccluded_target_ray_probes',1)>0 for r in facing.values()):valid.append(row)
    if not valid:raise DR.PlanError('No native-safe aimed initial camera turn')
    explicit=policy.get('source0_candidate_ids')
    if explicit:
        match=[r for r in valid if r['candidate_ids']==explicit]
        if match:return match[0]
    rank=policy.get('rank','motion_then_angle')
    if rank=='motion_then_angle':return min(valid,key=lambda r:(r['motion_s'],max(f['center_axis_angle_deg'] for f in r['endpoint_facing'].values())))
    if rank=='angle_then_motion':return min(valid,key=lambda r:(max(f['center_axis_angle_deg'] for f in r['endpoint_facing'].values()),r['motion_s']))
    raise ValueError('Unsupported model-selected initial ranking')

def camera_waypoints(S,h,qgoal,route):
    ci=int(S.robot.arms[h].wrist_cam);a=NT.predicted_data(S,{h:S.q_cmd(h)}).cam_xpos[ci].copy();b=NT.predicted_data(S,{h:qgoal}).cam_xpos[ci].copy()
    axis=NT.G.unit(np.asarray(S.objects['bottom_drawer']['joint']['axis'],float));front=max(a@axis,b@axis)+float(route.get('front_offset_m',0.))
    family=route['family']
    if family=='direct':return []
    if family=='front_midpoint':
        mid=(a+b)/2;return [mid+(front-mid@axis)*axis]
    if family=='front_two_corner':return [a+(front-a@axis)*axis,b+(front-b@axis)*axis]
    raise ValueError('Unsupported model-selected physical camera route')

def run(S,policy):
    if not isinstance(policy,dict) or policy.get('schema')!='rssr.drawer.luna_policy.v2':raise ValueError('Luna must supply explicit v2 policy JSON')
    native_together=S.together;native_joint_plan=S.joint_plan
    block,bh,dh=NT.prepare_task_checks(S);sys.path.insert(0,str(ARCHDIR));from arch_away import install
    install(S,dh,lines=True,holding=True);install(S,bh,lines='posture',holding=False)
    catalog=NT.load_exact_initial_catalog(S,str(CATDIR),artifact_hashes=CAT_HASHES)
    if catalog is None:catalog=NT.initial_catalog(S,count=int(policy['initial']['fresh_candidate_count']))
    chosen=choose_initial(catalog,policy['initial'])
    NT.checked_together(S,chosen['plans'],'initial_object_look',native_together=native_together)
    source=source_protocol();source.OPTIONS.update(speed=float(policy['source_speed']),grip_s=tuple(policy['grip_s']),overlap=True,short_exit=True)
    source.WAIT_MOUNT=tuple(policy['wait_mount']);source.LEANS=(0.,)
    configure_insertion(source,policy)
    if 'insertion_route' in policy:S.note('Luna-selected HOST source-side insertion waypoints '+str(policy['insertion_route']))
    if 'handle_yaws_deg' in policy:source.BAR_YAWS=tuple(policy['handle_yaws_deg'])
    def target(h):return block if h==bh else 'bottom_handle_bar'
    def camera_plan(h,qgoal):
        route=policy['handle_route' if h==dh else 'block_route']
        path,summary,report=CS.plan(S,h,target(h),qgoal,waypoints=camera_waypoints(S,h,qgoal,route),camera_speed=float(route['camera_speed']),angular_speed=float(route['angular_speed']),cone_deg=float(route['cone_deg']),preserve_source_joint_branch=False)
        S.note('HOST rigid-camera route '+str({'hand':h,**summary}))
        if not summary['qualified_geometry']:raise DR.PlanError('Selected rigid-camera route fails actual original path/FOV geometry: '+str(summary))
        return path
    def dispatch(plans,label,grips=None,allow=None,delay=None):
        p={h:np.asarray(q,float) for h,q in plans.items() if len(q)}
        if label.startswith('approach'):
            for h in p:
                if h==dh or not audit(S,{h:p[h]},{h:target(h)})['all_target_regions_in_FOV']:p[h]=camera_plan(h,p[h][-1])
        return execute_once(S,[p],label,native_together=native_together,targets_for=target,grips=grips,allow=allow,delay=delay,policy=policy)
    def move_joints(targets,label,v_max=None,allow=None):
        plans={h:native_joint_plan(h,q,v_max=min(float(v_max) if v_max is not None else .72,.5 if h in S.attached else .72)) for h,q in targets.items()}
        return dispatch(plans,label,allow=allow)
    def line(goals,label,speed=.1,min_s=.5,allow=None,check=True,soft=None):
        plans=S.plan_line(goals,speed=speed,min_s=min_s,allow=allow,check=check,label=label,soft=soft)
        for h,path in plans.items():
            if not audit(S,{h:path},{h:target(h)})['all_target_regions_in_FOV']:
                if h in S.attached or slide_follow(S,h,target(h)) is not None:raise DR.PlanError('Held path leaves target-region FOV; model must choose another legal carry route')
                plans[h]=camera_plan(h,path[-1])
        return dispatch(plans,label,allow=allow)
    def run_plans(parts,label):return execute_once(S,parts,label,native_together=native_together,targets_for=target,policy=policy)
    original_grip=S.grip
    def grip(widths,label,duration=.6):
        result=original_grip(widths,label,duration)
        if label=='close' and all(S.contacts(bh,block)):S.attach(bh,block)
        return result
    S.together=dispatch;S.move_joints=move_joints;S.line=line;S.run_plans=run_plans;S.grip=grip
    # The complete original source protocol installs refine exactly once here.
    return source.run(S)
