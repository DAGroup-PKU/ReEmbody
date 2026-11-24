"""RSSR empty-bank pilot: raw Real1 -> approximate Sim1 -> object-success Sim2."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,copy,concurrent.futures,multiprocessing,pathlib,time,traceback,json
from common import read,write,sha,load_credentials,run_lock
from pilot_learning import Knowledge,digest
from pilot_prepare import selection,prepare

def file_manifest(code,config):
    files=[*code.glob('*.py'),*code.glob('prompts/*.txt'),*code.glob('tools/*.py')]
    if config.get('stage_protocol'):
        files += list((code/'schemas/rssr-stage-v1').glob('*.json'))
        files += [p for p in (code/'vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    if config.get('initialization_response_cache'):files.append(pathlib.Path(config['initialization_response_cache']))
    if config.get('fixed_camera_config'):
        from camera_mount_config import load as load_mount_config
        fixed=load_mount_config(config['fixed_camera_config'])
        files.append(pathlib.Path(config['fixed_camera_config']))
        files += [pathlib.Path(p) for p in fixed['asset_files']]
    models=pathlib.Path(config['model_root'])
    files += [models/'source/scene/task_frame_contract.json',models/'source/scene/s1_tabletop.xml',models/'implementation_revisions/physical_constraints_17/s1_tabletop_physical.xml']
    files += list((models/'source/agent_retarget').rglob('*.py'))
    files += [p for p in (models/'source').glob('*.py') if p.name in ['constrained_waist.py','tabletop_scene.py','task_pose_adapter.py','v21_geometry.py']]
    # Hash model assets, excluding older run histories/libraries/results.
    for sub in ['trackc/piper_isaac_sim/piper_description','trackc/piper_isaac_sim/piper_l_description']:
        base=models/'source/models'/sub
        files += [p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in ['.xml','.urdf','.stl','.obj','.dae']]
    import xml.etree.ElementTree as ET
    xml=models/'implementation_revisions/physical_constraints_17/s1_tabletop_physical.xml'
    tree=ET.parse(xml).getroot();compiler=tree.find('compiler');meshdir=pathlib.Path(compiler.get('meshdir','')) if compiler is not None else pathlib.Path('')
    for mesh in tree.findall('./asset/mesh'):
        p=pathlib.Path(mesh.get('file'));files.append(p if p.is_absolute() else (xml.parent/meshdir/p).resolve())
    return {str(p):sha(p) for p in sorted(set(files))}

def verify(config):
    frozen=read(pathlib.Path(config['batch_root'])/'protocol.json')
    for p,h in frozen['files'].items():
        if sha(p)!=h:raise RuntimeError('Frozen implementation/model changed: '+p)

def run_case(batch,row):
    root=pathlib.Path(batch['batch_root'])/'cases'/row['case_id'];root.mkdir(parents=True,exist_ok=True)
    with run_lock(root):
        if (root/'result.json').exists():return read(root/'result.json')
        config=dict(batch,run_root=str(root),import_case=str(root/'inputs'),case_id=row['case_id'],task=row['task'])
        write(root/'config.json',config);started=time.time()
        try:
            verify(config);load_credentials(config['env_file']);knowledge=Knowledge(pathlib.Path(batch['batch_root'])/'learning')
            write(root/'entry_knowledge.json',{s:knowledge.snapshot(s) for s in ['reconstruction','camera','retarget']})
            write(root/'status.json',{'stage':'preparing','at':time.time()});prepare(config,row)
            from pilot_scene import initialize,reconstruct
            from pilot_camera_workflow import run as cameras
            from pilot_physics import rollout
            from pilot_target import run as target_run
            from object_render import render
            write(root/'status.json',{'stage':'initial_scene','at':time.time()});scene=initialize(config,root/'reconstruction')
            write(root/'status.json',{'stage':'source_camera','at':time.time()});source_camera=cameras(config,knowledge,root,scene,final=False)
            write(root/'status.json',{'stage':'reconstruction','at':time.time()});reconstruction=reconstruct(config,knowledge,root/'reconstruction/refinement',scene,source_camera['profiles']['sim1'])
            scene=reconstruction['scene'];physical=root/'sim1/physics'
            source_report=read(physical/'report.json') if (physical/'report.json').exists() else rollout(config,'sim1',physical,scene)
            source={'trial':str(physical),'scene':scene,'report':source_report,'task_success_required':False,'source_action_modified':False};write(root/'sim1/selection.json',source)
            write(root/'status.json',{'stage':'retarget','at':time.time()});target=target_run(config,knowledge,root/'sim2',scene)
            write(root/'status.json',{'stage':'final_camera','at':time.time()});camera=cameras(config,knowledge,root,scene,profiles=source_camera['profiles'],final=True,selections={'sim1':source,'sim2':target})
            write(root/'status.json',{'stage':'rendering_sim_only','at':time.time()})
            render_config=dict(config,camera_alignment={'enabled':True})
            for kind,sel in [('sim1',source),('sim2',target)]:
                if not (root/(kind+'.mp4')).exists():render(render_config,kind,sel['trial'],root/(kind+'.mp4'))
            result={'case_id':row['case_id'],'status':'completed','source_task_success':bool(source_report['task_success']),'source_task_success_required':False,'reconstruction_passed':bool(reconstruction['objective']['passed']),'source_initial_camera_passed':bool(source_camera['passed']),'sim2_task_success':bool(target['passed']),'camera_passed':bool(camera['passed']),'rssr_to_sim2_passed':bool(reconstruction['objective']['passed'] and target['passed'] and camera['passed']),'real2_run':False,'hardware_ready':False,'wall_seconds':time.time()-started,'scene':str(root/'reconstruction/refinement/selection.json'),'sim2_selection':str(root/'sim2/selection.json'),'sim2_video':str(root/'sim2.mp4'),'camera_result':str(root/'camera_alignment/result.json'),'old_operators_imported':False,'old_experiences_imported':False,'old_target_trajectory_imported':False,'scope':'di1 task family, approximate scene and camera verified by model plus numeric geometry; task verified by simulator'}
        except Exception as e:
            result={'case_id':row['case_id'],'status':'execution_error','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc(),'wall_seconds':time.time()-started,'real2_run':False,'rssr_to_sim2_passed':None,'sim2_task_success':None}
        write(root/'result.json',result);write(root/'status.json',{'stage':result['status'],'at':time.time()});print(json.dumps(result,ensure_ascii=False),flush=True);return result

def summarize(config,results,started):
    complete=[r for r in results if r['status']=='completed'];root=pathlib.Path(config['batch_root']);state=read(root/'learning/knowledge.json')
    return {'at':time.time(),'planned':len(config['episodes']),'terminal':len(results),'completed':len(complete),'execution_errors':len(results)-len(complete),'not_finished':len(config['episodes'])-len(results),'successes':{k:sum(bool(r.get(k)) for r in complete) for k in ['reconstruction_passed','source_task_success','source_initial_camera_passed','sim2_task_success','camera_passed','rssr_to_sim2_passed']},'knowledge':{s:{'operators':sum(o['stage']==s for o in state['operators']),'experiences':sum(e['stage']==s for e in state['experiences']),'evidence':sum(e['stage']==s for e in state['evidence'])} for s in ['reconstruction','camera','retarget']},'budget':read(root/'budget.json') if (root/'budget.json').exists() else {'used':{}},'wall_seconds':time.time()-started,'cases':results,'real2_run':False}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args();config=read(args.config);root=pathlib.Path(config['batch_root']);root.mkdir(parents=True,exist_ok=True)
    if config.get('max_video_creates')!=0 or config.get('run_real2') is not False:raise ValueError('This pilot must never run Real2')
    with run_lock(root):
        rows=selection(config);protocol={'config':config,'selection':rows,'files':file_manifest(pathlib.Path(__file__).parent,config),'empty_bank_required':True,'old_results_allowed':False,'sim1_success_required':False,'sim2_object_success_required':True}
        dest=root/'protocol.json'
        if dest.exists() and read(dest)!=protocol:raise ValueError('Frozen pilot changed; use a distinct revision')
        if not dest.exists():write(dest,protocol)
        knowledge=Knowledge(root/'learning');knowledge.initialize();write(root/'selection.json',rows)
        if args.prepare_only:return
        started=time.time();write(root/'status.json',{'stage':'running','pid':os.getpid(),'at':started,'planned':len(rows)})
        results=[]
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.get('active_cases',2),mp_context=multiprocessing.get_context('spawn')) as pool:
            pending={pool.submit(run_case,config,row):row for row in rows}
            for f in concurrent.futures.as_completed(pending):
                try:result=f.result()
                except Exception as e:result={'case_id':pending[f]['case_id'],'status':'execution_error','error_type':type(e).__name__,'error':str(e)}
                results.append(result);write(root/'progress.json',summarize(config,results,started))
        final=summarize(config,results,started);write(root/'summary.json',final);write(root/'status.json',{'stage':'pilot_finished','at':time.time(),'full_learning_started':False,'real2_run':False,'completed':final['completed'],'execution_errors':final['execution_errors']})

if __name__=='__main__':main()
