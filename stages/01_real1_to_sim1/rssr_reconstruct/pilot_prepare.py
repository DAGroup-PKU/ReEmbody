"""Raw source inputs only. No v22 result, operator, recipe, or fitted camera import."""
import pathlib,sys,json,subprocess,shutil
import cv2,numpy as np,pyarrow.parquet as pq
from PIL import Image
from common import read,write,sha

TASK='Pick up the basket with the left hand and place the blue cube into the basket with the right hand.'
DATA='di1_Pick_up_basket_place_blue_block_into_basket'
ROLES={'head':'image_top','left_wrist':'image_left','right_wrist':'image_right'}

def selection(config):
    data=pathlib.Path(config['dataset_root'])/DATA;rows=[]
    for episode in config['episodes']:
        stem=f'episode_{episode:06d}';p=data/'data/chunk-000'/(stem+'.parquet')
        videos={role:str(data/'videos/chunk-000'/('observation.images.'+cam)/(stem+'.mp4')) for role,cam in ROLES.items()}
        row={'case_id':f'di1_ep{episode:06d}','task':TASK,'source_episode':episode,'source_parquet':str(p),'source_sha256':sha(p),'source_videos':videos,'source_video_sha256':{k:sha(v) for k,v in videos.items()}}
        rows.append(row)
    return rows

def prepare(config,row):
    case=pathlib.Path(config['import_case']);case.mkdir(parents=True,exist_ok=True)
    if (case/'prepared.json').exists():
        old=read(case/'prepared.json')
        if old['row']!=row:raise ValueError('Prepared source changed')
        for p,h in old['files'].items():
            if sha(case/p)!=h:raise ValueError('Prepared file changed '+p)
        return old
    table=pq.read_table(row['source_parquet'],columns=['observation.state','action','timestamp'])
    state=np.stack(table['observation.state'].to_numpy());action=np.stack(table['action'].to_numpy());ts=np.asarray(table['timestamp'].to_numpy(),float)
    if state.shape!=(len(ts),26) or action.shape!=(len(ts),14) or len(ts)<2 or not np.all(np.diff(ts)>0) or not all(np.isfinite(a).all() for a in [state,action,ts]):raise ValueError('Invalid source shape/time')
    dt=float(np.median(np.diff(ts)))
    if abs(dt-1/30)>.001:raise ValueError('This pilot adapter requires native 30 Hz source')
    if sha(row['source_parquet'])!=row['source_sha256']:raise ValueError('Raw input changed')
    np.savez_compressed(case/'source_action_state.npz',state=state,action=action,timestamp=ts)
    write(case/'case.json',dict(row,source_frames=len(ts),source_duration_s=float(ts[-1]-ts[0])))
    import render_inputs
    render_inputs.R=pathlib.Path(config['model_root']);render_inputs.build(case)
    # Three separate raw views are retained for first-frame visual observation.
    caps={k:cv2.VideoCapture(v) for k,v in row['source_videos'].items()}
    for role,cap in caps.items():
        if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-30)>.02:raise ValueError('Unsupported source video fps')
        ok,frame=cap.read()
        if not ok:raise ValueError('Missing first frame')
        cv2.imwrite(str(case/(role+'_first.jpg')),frame);cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    count=int(np.floor(len(ts)*24/30));indices=np.unique(np.rint(np.linspace(0,len(ts)-1,count)).astype(int))
    np.savez_compressed(case/'frame_action_mapping.npz',source_frame_index=indices,source_time_s=(ts-ts[0])[indices],video_frame_index=np.arange(len(indices)),video_pts_s=np.arange(len(indices))/24)
    raw=case/'real1_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),24,(1470,630))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    last=-1
    try:
        for k in indices:
            out=np.zeros((630,1470,3),np.uint8)
            for i,(role,cap) in enumerate(caps.items()):
                for _ in range(int(k)-last-1):cap.grab()
                ok,frame=cap.read()
                if not ok:raise ValueError('Missing source frame '+str(k))
                out[135:495,i*495:i*495+480]=cv2.resize(frame,(480,360));cv2.putText(out,role.upper(),(i*495+8,115),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,220,220),1)
            writer.write(out);last=int(k)
    finally:
        writer.release()
        for cap in caps.values():cap.release()
    subprocess.run([config['ffmpeg'],'-v','error','-y','-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',str(case/'real1_condition.mp4')],check=True)
    from pilot_ik import initial_program,plan
    planned=plan(config,case/'default_ik',initial_program());shutil.copy2(planned['trajectory'],case/'agent_trajectory_full.npz')
    files=['source_action_state.npz','frame_action_mapping.npz','agent_trajectory_full.npz','sim2_robot_only_scene.xml','real1_condition.mp4','case.json']+[r+'_first.jpg' for r in ROLES]
    report={'row':row,'files':{n:sha(case/n) for n in files},'old_result_imported':False,'default_ik':planned,'source_action_modified':False,'first_frame_views':list(ROLES)}
    write(case/'prepared.json',report);return report
