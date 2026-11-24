"""Frame-index media operations. Never interpolate or duplicate video frames."""
import hashlib,json,pathlib,subprocess
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from common import write

def probe(path,config):
    p=subprocess.run([config['ffprobe'],'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)],capture_output=True,text=True,check=True)
    s=json.loads(p.stdout)['streams'][0];a,b=map(float,s['avg_frame_rate'].split('/'))
    return {'width':int(s['width']),'height':int(s['height']),'fps':a/b,'frames':int(s['nb_read_frames']),'duration_s':float(s['duration'])}

def encode_args(config):return [config['ffmpeg'],'-hide_banner','-loglevel','error','-y']

def clip(source,out,start,end,config):
    if out.exists():return
    out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(encode_args(config)+['-i',str(source),'-vf',f'trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS','-an','-fps_mode','passthrough','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
    if probe(out,config)['frames']!=end-start:raise RuntimeError('Clip missing original frames')

def image_at(video,index,out):
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read();cap.release()
    if not ok:raise RuntimeError('Cannot decode selected reference frame')
    out.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out,quality=94)
    return out

def head_reference(video,index,out):
    """Crop an observed source frame; no invented pixels or scene re-rendering."""
    image_at(video,index,out)
    im=Image.open(out);w,h=im.size
    im.crop((0,round(h*135/630),round(w*480/1470),round(h*495/630))).save(out,quality=94)
    return out

def sample_indices(frames,fps,dense_times=(),maximum=64,rate=2):
    base=set(np.round(np.arange(0,frames,fps/rate)).astype(int).clip(0,frames-1));base.update([0,frames-1])
    dense=set()
    for t in dense_times:
        dense.update(int(np.clip(round((t+d)*fps),0,frames-1)) for d in [-.375,-.25,-.125,0,.125,.25,.375])
    allidx=sorted(base|dense)
    if len(allidx)>maximum:
        required=sorted(dense|{0,frames-1})
        if len(required)>maximum:required=[required[i] for i in np.linspace(0,len(required)-1,maximum).astype(int)]
        spare=sorted(base-set(required));slots=maximum-len(required)
        allidx=sorted(set(required)|set(spare[i] for i in np.linspace(0,max(0,len(spare)-1),min(slots,len(spare))).astype(int)))
    return [int(i) for i in allidx]

def visual_sequence(videos,out,config,dense_times=(),maximum=64,rate=2,explicit_indices=None):
    """At each time, one labelled image has rows Real1/Sim2[/Real2], each with three views."""
    ps={k:probe(v,config) for k,v in videos.items()};frames=min(p['frames'] for p in ps.values());fps=config['fps']
    # Inputs for a comparison must already be on the same clock; don't warp a bad generation.
    if any(abs(p['fps']-fps)>.01 for p in ps.values()):raise RuntimeError('Frame rates differ; refuse implicit resampling')
    indices=sample_indices(frames,fps,dense_times,maximum,rate) if explicit_indices is None else sorted(set(int(i) for i in explicit_indices if 0<=int(i)<frames));out.mkdir(parents=True,exist_ok=True)
    caps={k:cv2.VideoCapture(str(v)) for k,v in videos.items()};result=[]
    try:
        for index in indices:
            canvas=Image.new('RGB',(1104,310*len(videos)),(10,12,16));draw=ImageDraw.Draw(canvas)
            try:font=ImageFont.truetype('DejaVuSans.ttf',18)
            except OSError:font=ImageFont.load_default()
            for row,(name,cap) in enumerate(caps.items()):
                cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=cap.read()
                if not ok:raise RuntimeError('Frame extraction failed')
                im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));w,h=im.size
                # Only remove the known top/bottom letterbox; preserve all three panels.
                im=im.crop((0,round(h*135/630),w,round(h*495/630))).resize((1104,270))
                canvas.paste(im,(0,row*310+40))
                roles={'Real1':'SOURCE - appearance and objects','Sim2':'TARGET - S1 robot and cameras','Real2':'CANDIDATE - evaluate THIS row'}
                roles.update(config.get('visual_role_labels',{}))
                draw.text((8,row*310+2),f'{name}: {roles.get(name,name)} | t={index/fps:.3f}s',fill='white',font=font)
                for col,v in enumerate(['HEAD','LEFT_WRIST','RIGHT_WRIST']):draw.text((8+col*368,row*310+21),v,fill='white',font=font)
            p=out/f'frame_{index:06d}.jpg';canvas.save(p,quality=90);result.append({'frame':index,'time_s':index/fps,'path':str(p)})
    finally:
        for cap in caps.values():cap.release()
    write(out/'manifest.json',{'mode':'timestamped_image_sequence','samples':result,'videos':ps,'no_video_interpolation':True,'not_every_frame_visually_verified':True})
    return result

def format_check(video,condition,config):
    try:
        a=probe(video,config);b=probe(condition,config)
        checks={'fps':abs(a['fps']-config['fps'])<.01,'frame_count':a['frames']==b['frames'],'aspect':abs(a['width']/a['height']-b['width']/b['height'])<.02,'nonempty':a['frames']>0}
        return {'passed':all(checks.values()),'checks':checks,'generated':a,'condition':b}
    except Exception as e:return {'passed':False,'error_type':type(e).__name__}

def concatenate(parts,out,config):
    """parts: (video, drop_prefix). Drop overlap frames; no blending, insertion or speed change."""
    trimmed=[]
    for i,(p,drop) in enumerate(parts):
        meta=probe(p,config);q=out.parent/f'{out.stem}_join_{i:03d}.mp4';clip(p,q,drop,meta['frames'],config);trimmed.append(q)
    listing=out.parent/'concat.txt'
    if any("'" in str(p) for p in trimmed):raise ValueError('Unsupported path quote')
    listing.write_text(''.join("file '"+str(p)+"'\n" for p in trimmed))
    subprocess.run(encode_args(config)+['-f','concat','-safe','0','-i',str(listing),'-an','-c','copy','-movflags','+faststart',str(out)],check=True)
    expected=sum(probe(p,config)['frames']-drop for p,drop in parts)
    if probe(out,config)['frames']!=expected:raise RuntimeError('Stitch frame count mismatch')
