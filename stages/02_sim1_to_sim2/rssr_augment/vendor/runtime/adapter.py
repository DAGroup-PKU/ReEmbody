"""Deterministic frame selection and internal camera packing, with full provenance."""
import pathlib,shutil,subprocess
import numpy as np
from common import read,write,sha
from media import encode_args,probe

def compatible_indices(n,quantum=8):
    if n<97:raise ValueError('At least 97 source frames are required without padding')
    count=1+((n-1)//quantum)*quantum
    return np.rint(np.linspace(0,n-1,count)).astype(np.int64)

def adapt_frames(case,config):
    manifest=case/'frame_selection.json'
    if manifest.exists():return
    meta=read(case/'render_manifest.json');n=meta['frames'];idx=compatible_indices(n)
    native=case/'native_inputs';native.mkdir(exist_ok=True)
    for name in ['real1_condition.mp4','sim2_robot_only.mp4','frame_action_mapping.npz','render_manifest.json']:
        shutil.copy2(case/name,native/name)
    dropped=sorted(set(range(n))-set(idx.tolist()))
    expression='*'.join(f'not(eq(n,{i}))' for i in dropped) or '1'
    for name in ['real1_condition.mp4','sim2_robot_only.mp4']:
        subprocess.run(encode_args(config)+['-i',str(native/name),'-vf',f"select='{expression}',setpts=N/({config['fps']}*TB)",'-an','-fps_mode','passthrough','-c:v','libx264','-preset','fast','-crf','18',str(case/name)],check=True)
        if probe(case/name,config)['frames']!=len(idx):raise RuntimeError('Frame-selection count mismatch')
    with np.load(native/'frame_action_mapping.npz') as z:
        values={k:z[k][idx] if z[k].ndim and len(z[k])==n else z[k] for k in z.files}
    values['original_video_frame_index']=idx
    values['original_video_pts_s']=values['video_pts_s'].copy()
    values['video_frame_index']=np.arange(len(idx));values['video_pts_s']=np.arange(len(idx))/config['fps']
    np.savez_compressed(case/'frame_action_mapping.npz',**values)
    meta.update(frames=len(idx),duration_s=len(idx)/config['fps'],all_original_frames_retained=False,
      frame_policy='Monotonic original-frame selection to observed 8k+1 API length, including both endpoints; no interpolation or duplicated frames. Nominal 24 FPS playback is slightly shorter; original timestamps remain in mapping.')
    for name in ['real1_condition.mp4','sim2_robot_only.mp4']:
        meta['videos'][name]={'sha256':sha(case/name),'bytes':(case/name).stat().st_size}
    write(case/'render_manifest.json',meta)
    write(manifest,{'original_frames':n,'selected_frames':len(idx),'indices':idx.tolist(),'dropped_indices':dropped,
      'preserved_first_and_last':int(idx[0])==0 and int(idx[-1])==n-1,'duplicated_or_interpolated_frames':False,
      'playback_fps':config['fps'],'original_duration_s':n/config['fps'],'playback_duration_s':len(idx)/config['fps'],
      'playback_time_compression_s':len(dropped)/config['fps'],'original_timestamps_retained':True,
      'basis':'Observed provider 360->353,636->633,642->641. Runtime calibration, not an official exact-frame guarantee.',
      'original_hashes':{p.name:sha(p) for p in native.iterdir() if p.is_file()}})

def transform(source,out,graph,config):
    if out.exists():return out
    subprocess.run(encode_args(config)+['-i',str(source),'-filter_complex',graph,'-map','[out]','-an','-fps_mode','passthrough','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(out)],check=True)
    if probe(source,config)['frames']!=probe(out,config)['frames']:raise RuntimeError('Layout conversion changed frame count')
    return out

def pack_t(source,out,config):
    graph='[0:v]scale=1470:630,format=yuv444p,split=3[a][b][c];[a]crop=480:360:0:135,scale=960:720,pad=960:1088:0:0:black[top];[b]crop=480:360:495:135,scale=476:357,pad=960:360:0:0:black[left];[c]crop=480:360:990:135,scale=476:357[right];[left][right]overlay=484:0:format=yuv444[bottom];[top][bottom]overlay=0:728:format=yuv444[out]'
    return transform(source,out,graph,config)

def unpack_t(source,out,config):
    graph='[0:v]scale=960:1088,format=yuv444p,split=3[a][b][c];[a]crop=960:720:0:0,scale=480:360,pad=495:360:0:0:black[head];[b]crop=476:357:0:728,scale=480:360,pad=495:360:0:0:black[left];[c]crop=476:357:484:728,scale=480:360[right];[head][left][right]hstack=inputs=3,pad=1470:630:0:135:black[out]'
    return transform(source,out,graph,config)
