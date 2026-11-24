"""Connect the frozen v22 interface to complete, object-free video conditions."""
import concurrent.futures,json,pathlib,shutil,subprocess,time
from common import read,write,sha,event

def prepare(config,code):
    root=pathlib.Path(config['run_root']);source=pathlib.Path(config['import_root']);root.mkdir(parents=True,exist_ok=True)
    fallback=pathlib.Path(config['fallback_import_root']) if config.get('fallback_import_root') else source
    selection=read(fallback/'selection.json')
    if config.get('case_ids'):
        selection['cases']=[i for i in selection['cases'] if i['case_id'] in config['case_ids']]
        if {i['case_id'] for i in selection['cases']}!=set(config['case_ids']):raise ValueError('Unknown selected case')
    write(root/'selection.json',selection)
    def one(item):
        c=root/item['case_id'];old=source/item['case_id'];c.mkdir(exist_ok=True)
        if not old.exists():old=fallback/item['case_id']
        if not (c/'case.json').exists():write(c/'case.json',item)
        if not (c/'request.json').exists():
            if (old/'request.json').exists():shutil.copy2(old/'request.json',c/'request.json')
            else:write(c/'request.json',{'schema_version':'cobot_s1_trajectory.v1','request_id':config['run_id']+'_'+c.name,'source':{'format':'atombench_lerobot_26d','parquet_path':item['source_parquet'],'frame_contract_sha256':'7d4b2933aa4f4e97c9b7a2272fa36136c0fd4989dbe44e6b76d3dc062bd14e33','units':{'length':'m','angle':'rad','time':'s','euler':'Rz(yaw)@Ry(pitch)@Rx(roll)'}},'target':'astribot_s1_v21_tabletop','memory':{'mode':'read_only'},'task':item['task']})
        if not (c/'source_action_state.npz').exists():
            if (old/'source_action_state.npz').exists():shutil.copy2(old/'source_action_state.npz',c/'source_action_state.npz')
            else:
                import pyarrow.parquet as pq,numpy as np
                table=pq.read_table(item['source_parquet']);np.savez_compressed(c/'source_action_state.npz',timestamp=np.array(table['timestamp'].to_pylist()),action=np.array(table['action'].to_pylist()),state=np.array(table['observation.state'].to_pylist()))
        if sha(pathlib.Path(item['source_parquet']))!=item['source_sha256']:raise RuntimeError('Source parquet changed')
        request=read(c/'request.json')
        if request.get('memory',{}).get('mode')!='read_only':raise RuntimeError('Memory must remain frozen')
        if (c/'response.json').exists():result=read(c/'response.json')
        elif (old/'response.json').exists():
            result=read(old/'response.json');shutil.copy2(old/'response.json',c/'response.json')
            write(c/'import_provenance.json',{'imported_existing_retarget':True,'path':str(old/'response.json'),'sha256':sha(old/'response.json'),'no_new_retarget_in_this_run':True})
        else:
            # For a new dataset manifest this is the same autonomous, read-only interface.
            for attempt in range(3):
                with (c/'retarget.log').open('a') as log:
                    subprocess.run(['bash',config['retarget_interface'],'--request',str(c/'request.json'),'--output',str(c/'response.json')],stdout=log,stderr=subprocess.STDOUT,timeout=7200)
                result=read(c/'response.json')
                if result.get('status') not in ['busy','checkpoint']:break
                time.sleep(10)
        if result.get('status')!='trajectory_passed':
            write(c/'preparation.json',{'ready':False,'reason':result.get('status')});return
        if result.get('provenance',{}).get('shared_memory_updated') is not False:raise RuntimeError('Unexpected memory update')
        for key,name in [('diagnostics_path','agent_diagnostics.json'),('body_certificate_path','agent_body_certificate.json')]:
            if not (c/name).exists():shutil.copy2(result['artifacts'][key],c/name)
        manifest=c/'render_manifest.json'
        if not manifest.exists() and (old/'render_manifest.json').exists() and read(old/'render_manifest.json').get('render_revision')==3:
            for name in ['render_manifest.json','agent_trajectory_full.npz','frame_action_mapping.npz','sim2_robot_only_scene.xml','real1_condition.mp4','sim2_robot_only.mp4','input_contact_sheet.jpg']:
                shutil.copy2(old/name,c/name)
            if (old/'frame_selection.json').exists():
                shutil.copy2(old/'frame_selection.json',c/'frame_selection.json')
                shutil.copytree(old/'native_inputs',c/'native_inputs',dirs_exist_ok=True)
        if not manifest.exists() or read(manifest).get('render_revision')!=3:
            event(root,'full_render_started',case=c.name)
            with (c/'render.log').open('a') as log:subprocess.run([config['python'],str(code/'render_inputs.py'),'--case',str(c),'--full'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        m=read(manifest)
        if m['objects_in_sim2'] is not False or not m['full_episode']:raise RuntimeError('Expected full robot-only conditions')
        for name,data in m['videos'].items():
            if sha(c/name)!=data['sha256']:raise RuntimeError('Condition video changed')
        if config.get('frame_quantum_adapter'):
            from adapter import adapt_frames
            adapt_frames(c,config);m=read(manifest)
        if not (c/'camera_geometry.json').exists():
            from geometry import camera_cues
            camera_cues(c)
        if old.parent==source and config.get('reuse_planner_from_import',True) and not (c/'planning').exists() and (old/'planning/result.json').exists():
            # K3.decision checks the source, prompt and video hashes again before reuse.
            shutil.copytree(old/'planning',c/'planning')
        if old.parent==source and config.get('reuse_generated_candidates_from_import') and not (c/'segments').exists() and (old/'segments').exists():
            # Keep raw candidates and provider task IDs. Re-run only the revised verifier.
            shutil.copytree(old/'segments',c/'segments')
            for segment in (c/'segments').iterdir():
                if not segment.is_dir():continue
                revision=pathlib.Path(config['import_root']).name
                if (segment/'selection.json').exists():(segment/'selection.json').rename(segment/('selection_prior_'+revision+'.json'))
                for attempt in segment.glob('attempt_*'):
                    if (attempt/'verification').exists() and not config.get('retain_previous_verifier_results'):(attempt/'verification').rename(attempt/('verification_prior_'+revision))
                    for inspection in attempt.glob('inspection_*'):
                        if inspection.is_dir() and '_prior_' not in inspection.name and not config.get('retain_previous_inspections'):inspection.rename(attempt/(inspection.name+'_prior_'+revision))
                    if (attempt/'outcome.json').exists():(attempt/'outcome.json').rename(attempt/('outcome_prior_'+revision+'.json'))
            write(c/'candidate_import.json',{'source':str(old/'segments'),'reuse_paid_video_tasks':True,'verifier_policy_revised':True})
        write(c/'preparation.json',{'ready':True,'trajectory_constraints_passed':True,'physical_task_verified':False,'full_sequence_frames':m['frames']})
        event(root,'inputs_ready',case=c.name,frames=m['frames'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=config['case_concurrency']) as pool:
        futures={pool.submit(one,item):item for item in selection['cases']}
        for f in concurrent.futures.as_completed(futures):
            item=futures[f]
            try:f.result()
            except Exception as e:
                write(root/item['case_id']/'preparation.json',{'ready':False,'reason':type(e).__name__});event(root,'preparation_failed',case=item['case_id'],error_type=type(e).__name__)
    return selection

def hashes(config,selection):
    root=pathlib.Path(config['run_root']);result={}
    for item in selection['cases']:
        c=root/item['case_id']
        for name in ['request.json','response.json','source_action_state.npz','agent_trajectory_full.npz','render_manifest.json','real1_condition.mp4','sim2_robot_only.mp4','frame_action_mapping.npz','sim2_robot_only_scene.xml','frame_selection.json','camera_geometry.json']:
            if (c/name).exists():result[c.name+'/'+name]=sha(c/name)
        for role,p in item['source_videos'].items():result[c.name+'/source_video/'+role]=sha(p)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,default=pathlib.Path(__file__).with_name('config.json'));a=p.parse_args()
    prepare(read(a.config),pathlib.Path(__file__).parent)
