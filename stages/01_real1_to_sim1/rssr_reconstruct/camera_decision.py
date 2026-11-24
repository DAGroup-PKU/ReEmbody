"""Bounded camera-only K3 transport with explicit reasoning effort.

Separate from physical decision code so completed physical trials remain valid.
"""
import base64,json,os,pathlib,time
from openai import OpenAI
from common import write,read,sha,event

def decide(config,budget,folder,system,payload,images=()):
    folder=pathlib.Path(folder);folder.mkdir(parents=True,exist_ok=True)
    extra={'reasoning_effort':config.get('camera_reasoning_effort','low')}
    request={'system':system,'payload':payload,'images':{str(p):sha(p) for p in images},
             'model':os.environ.get('KIMI_MODEL','Kimi-K3'),'extra_body':extra,
             'max_tokens':config.get('camera_decision_max_tokens',16384)}
    if (folder/'input.json').exists() and read(folder/'input.json')!=request:raise RuntimeError('Camera decision inputs changed')
    write(folder/'input.json',request)
    if (folder/'result.json').exists():return read(folder/'result.json')
    content=[{'type':'text','text':json.dumps(payload,ensure_ascii=False)}]
    for p in images:content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(pathlib.Path(p).read_bytes()).decode()}})
    rawpath=folder/'raw_00.json';receipt=folder/'request_00.json'
    if rawpath.exists():raw=read(rawpath)
    else:
        if receipt.exists():raise RuntimeError('Camera request already attempted; new decision version required')
        seq=budget.reserve('k3',pathlib.Path(config['import_case']).name,str(folder));write(receipt,{'sequence':seq,'at':time.time()})
        event(pathlib.Path(config['run_root']),'object_k3_started',decision=folder.name,sequence=seq,images=len(images),reasoning_effort=extra['reasoning_effort'])
        client=OpenAI(api_key=os.environ['KIMI_API_KEY'],base_url=os.environ['KIMI_BASE_URL'],timeout=config.get('camera_timeout_s',600),max_retries=0)
        start=time.time()
        try:
            response=client.chat.completions.create(model=request['model'],messages=[{'role':'system','content':system},{'role':'user','content':content}],response_format={'type':'json_object'},temperature=1.,stream=False,max_tokens=request['max_tokens'],extra_body=extra)
            raw={'content':response.choices[0].message.content,'finish_reason':response.choices[0].finish_reason,'usage':response.usage.model_dump() if response.usage else None,'latency_s':time.time()-start};write(rawpath,raw)
        except Exception as error:
            write(folder/'error_00.json',{'type':type(error).__name__,'status':getattr(error,'status_code',None),'latency_s':time.time()-start});raise RuntimeError('Camera K3 request failed') from None
    if raw.get('finish_reason')!='stop':raise RuntimeError('Camera K3 response incomplete')
    from k3 import parse_response
    result=parse_response(raw.get('content') or '');write(folder/'result.json',result);return result
