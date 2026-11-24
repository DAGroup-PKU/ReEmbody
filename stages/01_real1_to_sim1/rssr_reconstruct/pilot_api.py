"""Shared bounded pipeline, full JSON responses; paid calls are durably reserved."""
import pathlib,os,hashlib,json
from common import Budget,read,write,sha
from object_decision import decide
from api_scheduler import api_slot

def initialization_key(case,request,max_tokens):
    # Only file locations are irrelevant. Preserve image order and every
    # prompt/payload/model setting; do not normalize numerical evidence.
    value=dict(request,images=list(request['images'].values()),case=case,max_tokens=max_tokens)
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def call(config,folder,prompt,payload,images=(),stage=None):
    if config.get('stage_protocol'):
        if config['stage_protocol']!='rssr.stage.v1':raise ValueError('Unsupported stage protocol')
        if stage is None:raise ValueError('Explicit stage required by JSON protocol')
        from protocol_decision import decide_stage
        budget=Budget(config.get('api_budget_root',config['batch_root']),{'k3':config['max_k3_requests'],'video_gen':0})
        # The transport locks each actual HTTP attempt. Cached decisions and
        # local validation must not wait behind long unrelated API requests.
        return decide_stage(config,budget,folder,stage,prompt,payload,images)
    cache_path=config.get('initialization_response_cache')
    if cache_path:
        request={'system':prompt,'payload':payload,'images':{str(p):sha(p) for p in images},'model':os.environ.get('KIMI_MODEL','Kimi-K3')}
        if config.get('k3_reasoning_effort'):request['extra_body']={'reasoning_effort':config['k3_reasoning_effort']}
        key=initialization_key(config['case_id'],request,config['object_decision_max_tokens'])
        match=read(cache_path)['entries'].get(key)
        if match:
            folder=pathlib.Path(folder)
            if match['scope']!='same_pilot_prelearning_visual_initialization':raise ValueError('Invalid initialization cache scope')
            if (folder/'input.json').exists() and read(folder/'input.json')!=request:raise ValueError('Cached decision input changed')
            write(folder/'input.json',request)
            write(folder/'cached_response_receipt.json',{'key':key,'source':match['source'],'source_input_sha256':match['input_sha256'],'source_result_sha256':match['result_sha256'],'new_request':False,'inherited_request_already_accounted':True,'not_learned_knowledge':True})
            write(folder/'result.json',match['result'])
            return match['result']
    budget=Budget(config['batch_root'],{'k3':config['max_k3_requests'],'video_gen':0})
    with api_slot(config):return decide(config,budget,folder,prompt,payload,images)

def propose(config,knowledge,stage,folder,payload,images=()):
    folder=pathlib.Path(folder)
    p=pathlib.Path(__file__).parent/'prompts/pilot_operator_repair.txt'
    prompt=p.read_text(encoding='utf8').replace('{stage}',stage)
    snapshot=knowledge.snapshot(stage,include_referenced=bool(config.get('stage_protocol')))
    if snapshot.get('admission_policy'):
        prompt+='\n本轮入库规则：优先复用已有模板调整参数。数值、参数名称或范围变化不构成新机制；不要为此另造算子。admission_state=candidate 只是在其他演示中改善过的待验证模板，可在本例正常候选预算内检验，不代表已正式入库。至少两条不同演示改善且无新增硬约束失败才转正；正式活跃库最多24个。继承库是显式初始化的先验，当前有效性仍以物理执行为准。'
    if stage=='retarget' and config.get('interaction_revision')=='v16':
        prompt+='\ninteraction_v16：数值残差下降仅是局部优化证据，不能转正为交互能力。至少两个不同演示获得接触/持物/任务判据改善，且无新增硬约束或任务失败，才可转正。场景改变后尚未完成视觉复核的证据不转正。经验中给出总尝试数及实际交互收益数，按当前repair_phase选取，避免反复使用只改善IK而抓取一直失败的方案。'
    if stage=='retarget' and config.get('learning_revision')=='v17':
        prompt+='\n'+(p.parent/'learning_v17.txt').read_text(encoding='utf8')
    if config.get('library_lifecycle'):
        prompt+='\n库允许根据真实执行证据增删改。近期在不同演示中重复失败的模板/上下文/参数配方会从检索中停用，审计历史保留；可提出修订候选，至少两条新的独立演示重新验证后恢复。不要把停用解释为普遍无效；读取当前上下文及证据，不要换名字反复调用同一失效方案。经验修订只可使用已经执行并验证的参数，不能直接把假设写入成功库。'
    if config.get('stage_protocol') and (folder/'full_input.json').exists():
        from rssr_protocol import digest
        saved=read(folder/'full_input.json');previous=saved['payload'];old=dict(previous);old.pop('knowledge',None)
        if digest(old)!=digest(payload):raise RuntimeError('Cannot resume operator decision with changed physical evidence')
        snapshot=previous['knowledge']
        prompt=saved['system']
    response=call(config,folder,prompt,dict(payload,knowledge=snapshot),images,stage=stage+'.repair')
    resolved=folder/'operator_resolution.json'
    if resolved.exists():
        record=read(resolved)
        if record['decision']!=response:raise RuntimeError('Resolved operator decision changed')
        # Rehydrate an episode-local proposed operator without promoting it to
        # shared knowledge during inference.
        trace=record['trace']
        if hasattr(knowledge,'transient') and trace.get('operator_id'):
            knowledge.transient[stage][trace['operator_id']]=trace['operator_definition']
        return record['patch'],trace
    if response.get('operator_id') and hasattr(knowledge,'transient'):
        # The wire catalog also contains immutable failed/history definitions.
        # Resolve against exactly what this decision saw, even if a concurrent
        # writer has since evicted that negative example from its short index.
        from rssr_protocol import _knowledge
        from rssr_knowledge import clean_operator
        from pilot_learning import validate_operator
        advertised=_knowledge(snapshot,payload.get('history',[]))['operators']
        chosen=next((o for o in advertised if o['id']==response['operator_id']),None)
        if chosen is None:raise ValueError('Operator was not advertised to this decision')
        checked=validate_operator(clean_operator(chosen),stage)
        if checked['id']!=chosen['id']:raise ValueError('Advertised operator identity changed')
        knowledge.transient[stage][checked['id']]=checked
    if 'base_candidate' in response:
        if stage!='retarget' or config.get('learning_revision')!='v17' or response['base_candidate'] not in ('incumbent','exploration'):
            raise ValueError('base_candidate is supported only by v17 retarget search')
    patch,trace=knowledge.resolve(stage,response,pathlib.Path(config['run_root']).name)
    if 'base_candidate' in response:trace=dict(trace,base_candidate=response['base_candidate'])
    write(resolved,{'decision':response,'patch':patch,'trace':trace})
    return patch,trace
