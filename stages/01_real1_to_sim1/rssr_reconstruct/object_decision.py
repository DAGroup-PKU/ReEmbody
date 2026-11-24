"""Schema-constrained K3 decisions for headless physics; one initial image only."""
import base64,json,os,pathlib,time
from openai import OpenAI
from common import write,read,sha,event

def decide(config,budget,folder,system,payload,images=()):
 folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
 request={'system':system,'payload':payload,'images':{str(p):sha(p) for p in images},'model':os.environ.get('KIMI_MODEL','Kimi-K3')}
 extra={'reasoning_effort':config['k3_reasoning_effort']} if config.get('k3_reasoning_effort') else {}
 if extra:request['extra_body']=extra
 if (folder/'input.json').exists() and read(folder/'input.json')!=request:raise RuntimeError('Decision inputs changed; new decision version required')
 write(folder/'input.json',request)
 if (folder/'result.json').exists():return read(folder/'result.json')
 content=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]
 for p in images:content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(pathlib.Path(p).read_bytes()).decode()}})
 client=OpenAI(api_key=os.environ['KIMI_API_KEY'],base_url=os.environ['KIMI_BASE_URL'],timeout=config['k3_timeout_s'],max_retries=0)
 for attempt in range(config['k3_attempts_per_decision']):
  p=folder/f'raw_{attempt:02d}.json'
  if p.exists():raw=read(p)
  else:
   receipt=folder/f'request_{attempt:02d}.json'
   if receipt.exists():continue
   seq=budget.reserve('k3',pathlib.Path(config['import_case']).name,str(folder));write(receipt,{'sequence':seq,'at':time.time()});event(pathlib.Path(config['run_root']),'object_k3_started',decision=folder.name,sequence=seq,images=len(images))
   try:
    response=client.chat.completions.create(model=request['model'],messages=[{'role':'system','content':system},{'role':'user','content':content}],response_format={'type':'json_object'},temperature=1.,stream=False,max_tokens=config.get('object_decision_max_tokens',2000),extra_body=extra)
    raw={'content':response.choices[0].message.content,'finish_reason':response.choices[0].finish_reason,'usage':response.usage.model_dump() if response.usage else None};write(p,raw)
   except Exception as e:write(folder/f'error_{attempt:02d}.json',{'type':type(e).__name__,'status':getattr(e,'status_code',None)});continue
  try:
   assert raw['finish_reason']=='stop';text=raw['content'].strip()
   if text.startswith('```'):text=text.split('\n',1)[1].rsplit('```',1)[0]
   result=json.loads(text);assert isinstance(result,dict);write(folder/'result.json',result);return result
  except (ValueError,AssertionError,TypeError):continue
 raise RuntimeError('K3 bounded decision failed')
