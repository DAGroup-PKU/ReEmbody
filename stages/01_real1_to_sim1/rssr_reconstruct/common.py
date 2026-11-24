"""Durable state, request accounting and immutable run configuration."""
import contextlib,fcntl,hashlib,json,os,pathlib,subprocess,threading,time,re

def read(path):
    return json.loads(pathlib.Path(path).read_text(encoding='utf8'))

def write(path,value):
    path=pathlib.Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp.'+str(threading.get_ident()))
    with tmp.open('w',encoding='utf8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
    tmp.replace(path)

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def event(root,name,**fields):
    row={'at':time.time(),'event':name,**fields}
    with (pathlib.Path(root)/'events.jsonl').open('a',encoding='utf8') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print(json.dumps(row,ensure_ascii=False),flush=True)

def load_credentials(path):
    # Existing deployment configuration executes only through its established shell loader.
    raw=subprocess.check_output(['bash','-c','set -a; source "$1"; env -0','_',str(path)],stderr=subprocess.DEVNULL)
    allowed={'KIMI_API_KEY','KIMI_BASE_URL','KIMI_MODEL','ARK_API_KEY'}
    for e in raw.split(b'\0'):
        if b'=' in e:
            k,v=e.split(b'=',1);name=k.decode()
            if name in allowed:os.environ.setdefault(name,v.decode())

class Budget:
    def __init__(self,root,limits):self.root=pathlib.Path(root);self.limits=limits
    def reserve(self,kind,case,tag):
        with (self.root/'budget.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/'budget.json';state=read(path) if path.exists() else {'used':{},'reservations':[]}
            n=state['used'].get(kind,0)
            limit=self.limits[kind]
            if limit is not None and n>=limit:raise RuntimeError(kind+'_budget_exhausted')
            state['used'][kind]=n+1
            state['reservations'].append({'kind':kind,'number':n+1,'case':case,'tag':tag,'at':time.time()})
            write(path,state);return n+1

@contextlib.contextmanager
def run_lock(root):
    with (pathlib.Path(root)/'run.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield

def seal(root,code,config,selection,input_hashes):
    files=sorted([*code.glob('*.py'),*code.glob('prompts/*.txt')])
    state={'config':config,'selection':selection,'code':{str(p.relative_to(code)):sha(p) for p in files},'inputs':input_hashes}
    path=pathlib.Path(root)/'frozen_run.json'
    if path.exists():
        if read(path)!=state:raise RuntimeError('Frozen inputs/configuration/code changed; use a new run_root')
    else:write(path,state)
    return sha(path)

def segments(n,maximum=360,minimum=96,overlap=24,fixed_length=False):
    if n<minimum:raise ValueError('Video shorter than provider minimum')
    if fixed_length:
        if n<maximum:raise ValueError('Episode shorter than calibrated reference window; padding is forbidden')
        result=[];start=0
        while True:
            end=start+maximum
            result.append({'start':start,'end':end,'drop_prefix':0 if not result else result[-1]['end']-start})
            if end==n:return result
            start=min(end-overlap,n-maximum)
    result=[];start=0
    while True:
        end=min(n,start+maximum)
        result.append({'start':start,'end':end,'drop_prefix':0 if not result else result[-1]['end']-start})
        if end==n:break
        start=end-overlap
        if n-start<minimum:start=n-minimum
    return result

CRITERIA=('scene_cross_view','robot_motion','object_motion','contact_and_release','temporal_order','phase_outcome','appearance_realism','camera_identity','temporal_continuity')
def semantic_accept(report,minimum=0.8):
    if report.get('video_visible') is not True:return False
    objects=report.get('object_observations',[])
    for view in ['head','left_wrist','right_wrist']:
        observations=[o for o in objects if isinstance(o,dict) and o.get('view')==view]
        if not observations or any(o.get('state') not in ['visible','occluded','outside_fov'] or not o.get('object') or not o.get('reason') or type(o.get('time_s')) not in (int,float) for o in observations):return False
    criteria=report.get('criteria',{})
    for key in CRITERIA:
        value=criteria.get(key,{})
        confidence=value.get('confidence')
        if value.get('pass') is not True or type(confidence) not in (int,float) or not minimum<=confidence<=1:return False
        evidence=value.get('evidence')
        if not isinstance(evidence,list) or not evidence:return False
        if any(not isinstance(e,dict) or type(e.get('time_s')) not in (int,float) or not e.get('observation') for e in evidence):return False
    if report.get('uncertainties') or report.get('violations'):return False
    return all(report.get('views',{}).get(k,{}).get('observed') is True for k in ['head','left_wrist','right_wrist'])

def verifier_action(report,minimum=0.8):
    if semantic_accept(report,minimum):return 'accept'
    if report.get('violations'):return 'regenerate'
    return 'inspect'

def inspection_times(report,duration,maximum=32):
    """Use the verifier's bounded uncertain intervals, never a human-picked frame."""
    intervals=[]
    for v in report.get('inspect_intervals_s',[]):
        if isinstance(v,list) and len(v)==2 and all(type(x) in (int,float) for x in v):intervals.append(v)
    for text in report.get('uncertainties',[]):
        for a,b in re.findall(r'(?:t\s*[=≈]?\s*)?(\d+(?:\.\d+)?)\s*[-–~至]\s*(\d+(?:\.\d+)?)',str(text)):intervals.append([float(a),float(b)])
    intervals=[(max(0,float(a)),min(duration,float(b))) for a,b in intervals if 0<=a<b<=duration+.1][:4]
    if not intervals:intervals=[(0,duration)]
    # Allocate fresh INTERIOR evidence first. Boundary-first sampling used to
    # request the very same eight frames when intervals matched uniform samples.
    times={0.,duration}
    for fraction in [.5,.25,.75,0.,1.,.125,.875]:
        for a,b in intervals:
            if len(times)>=maximum:break
            times.add(round(a+(b-a)*fraction,6))
    return sorted(times),intervals
