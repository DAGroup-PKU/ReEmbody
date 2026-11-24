"""K3 decisions through native video when available, timestamped vision otherwise."""
import base64,json,os,pathlib,threading,time
from openai import OpenAI
from common import read,write,event,sha
from media import visual_sequence

class K3DecisionError(RuntimeError):pass


def completion(client,config,model,messages,extra):
    """Use the configured route for timestamped images; never send video bytes."""
    if config.get('llm_api','chat_completions')=='responses':
        from responses_transport import create
        request={'model':model,'max_tokens':config.get('object_decision_max_tokens',16384),
                 'reasoning_effort':config.get('k3_reasoning_effort')}
        return create(client,request,messages,{'type':'json_object'})
    response=client.chat.completions.create(model=model,messages=messages,
        response_format={'type':'json_object'},temperature=1.,stream=False,extra_body=extra)
    return {'content':response.choices[0].message.content or '',
            'finish_reason':response.choices[0].finish_reason,
            'usage':response.usage.model_dump() if response.usage else None,
            'provider_model':getattr(response,'model',None),'provider_response_id':getattr(response,'id',None),
            'provider_api':'chat_completions'}

def parse_response(raw):
    text=raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    try:value=json.loads(text)
    except json.JSONDecodeError as error:
        # One observed gateway response swapped two closing delimiters: }}]}
        # instead of }]}}. Repair only a UNIQUE adjacent closing-token swap,
        # outside strings. Never change a key, value, word, number or boolean.
        outside=[];quoted=False;escaped=False
        for ch in text:
            outside.append(not quoted)
            if escaped:escaped=False
            elif ch=='\\' and quoted:escaped=True
            elif ch=='"':quoted=not quoted
        candidates=[]
        for i in range(max(0,error.pos-8),min(len(text)-1,error.pos+8)):
            if not (outside[i] and outside[i+1] and text[i:i+2] in ['}]',']}']):continue
            changed=text[:i]+text[i+1]+text[i]+text[i+2:]
            try:parsed=json.loads(changed)
            except json.JSONDecodeError:continue
            if isinstance(parsed,dict) and isinstance(parsed.get('video_visible'),bool):candidates.append((i,parsed))
        unique={json.dumps(v,sort_keys=True,ensure_ascii=False):(i,v) for i,v in candidates}
        if len(unique)!=1:raise error
        position,value=next(iter(unique.values()))
        value['_syntax_repair']={'kind':'unique_adjacent_closing_delimiter_swap','position':position,'values_unchanged':True}
    if not isinstance(value,dict) or not isinstance(value.get('video_visible'),bool):raise ValueError('Missing visual input assertion')
    return value

class K3:
    def __init__(self,config,budget):
        self.config=config;self.budget=budget;self.root=pathlib.Path(config['run_root']);self.slots=threading.Semaphore(config['k3_concurrency'])
        self.client=OpenAI(api_key=os.environ['KIMI_API_KEY'],base_url=os.environ['KIMI_BASE_URL'],timeout=config['k3_timeout_s'],max_retries=0)
        self.model=os.environ.get('KIMI_MODEL','Kimi-K3')
    def decision(self,folder,system,text,videos,dense_times=(),maximum=64,rate=2,explicit_indices=None):
        folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True);result=folder/'result.json'
        provenance={'system_sha256':__import__('hashlib').sha256(system.encode()).hexdigest(),'text':text,'videos':{k:sha(v) for k,v in videos.items()},'dense_times':list(dense_times),'maximum':maximum,'rate':rate,'model':self.model}
        if explicit_indices is not None:provenance['explicit_indices']=list(explicit_indices)
        if self.config.get('llm_api')=='responses':
            provenance.update(llm_api='responses',max_output_tokens=self.config.get('object_decision_max_tokens',16384))
        extra={'reasoning_effort':self.config['k3_reasoning_effort']} if self.config.get('k3_reasoning_effort') else {}
        if extra:provenance['extra_body']=extra
        if (folder/'input.json').exists() and read(folder/'input.json')!=provenance:raise RuntimeError('K3 cached decision input changed')
        write(folder/'input.json',provenance)
        if result.exists():return read(result)
        for prior in sorted(folder.glob('raw_*.json')):
            cached=read(prior)
            if cached.get('finish_reason')!='stop':continue
            try:parsed=parse_response(cached.get('content',''))
            except (ValueError,TypeError):continue
            parsed['_provenance']={'model':self.model,'mode':'timestamped_image_sequence','recovered_from':prior.name,'no_new_api_request':True,'model_verification_not_ground_truth':True}
            write(result,parsed);event(self.root,'k3_cached_response_recovered',decision=str(folder.relative_to(self.root)));return parsed
        for attempt in range(self.config['k3_attempts_per_decision']):
            receipt=folder/f'attempt_{attempt+1:02d}.json'
            if receipt.exists():continue
            effective_cap=maximum if attempt==0 else max(16,maximum//2)
            chosen=explicit_indices
            if chosen is not None and len(chosen)>effective_cap:
                chosen=[chosen[round(i*(len(chosen)-1)/(effective_cap-1))] for i in range(effective_cap)]
            samples=visual_sequence(videos,folder/f'frames_attempt_{attempt+1:02d}',self.config,dense_times,effective_cap,rate,chosen)
            content=[{'type':'text','text':text+'\n输入是自动提取的带时间戳视频帧序列，每个图像按时间递增。图像每行对应一个阶段，每行包含三个同步机位。仅依据可见证据回答。'}]
            for sample in samples:
                content.append({'type':'text','text':f"t={sample['time_s']:.3f}s"})
                content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(pathlib.Path(sample['path']).read_bytes()).decode()}})
            with self.slots:
                seq=self.budget.reserve('k3',folder.parts[-3] if len(folder.parts)>2 else 'probe',str(folder.relative_to(self.root)))
                write(receipt,{'reserved':True,'sequence':seq,'at':time.time()});start=time.time()
                event(self.root,'k3_started',decision=str(folder.relative_to(self.root)),sequence=seq,frames=len(samples))
                try:
                    response=completion(self.client,self.config,self.model,[{'role':'system','content':system},{'role':'user','content':content}],extra)
                    write(folder/f'raw_{attempt+1:02d}.json',response)
                    if response['finish_reason']!='stop':raise ValueError('Incomplete K3 response')
                    parsed=parse_response(response['content'])
                    parsed['_provenance']={'model':self.model,'mode':'timestamped_image_sequence','sampled_frames':len(samples),'latency_s':time.time()-start,'model_verification_not_ground_truth':True}
                    write(result,parsed);event(self.root,'k3_finished',decision=str(folder.relative_to(self.root)),sequence=seq,visible=parsed.get('video_visible'));return parsed
                except Exception as e:
                    write(folder/f'error_{attempt+1:02d}.json',{'error_type':type(e).__name__,'http_status':getattr(e,'status_code',None),'latency_s':time.time()-start})
                    event(self.root,'k3_error',decision=str(folder.relative_to(self.root)),error_type=type(e).__name__)
            if attempt+1<self.config['k3_attempts_per_decision']:time.sleep(10)
        raise K3DecisionError('K3 decision attempts exhausted')
