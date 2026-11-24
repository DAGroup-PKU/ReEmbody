"""Owned native stages and strict v3 publication admission; no HF writes."""
from pathlib import Path
import argparse, ast, json, os, sys
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
PROJECT_ROOT=HERE.parent
SCHEMA='rssr.v3_human_align.admission.v1'
if __package__:
 from . import owned_stage as common
else:
 sys.path.insert(0,str(HERE))
 import owned_stage as common
read,sha,verify=common.read,common.sha,common.verify_pins

def write(path,value):
 """Publish a new admission receipt atomically, without replacing any file."""
 path=Path(path)
 path.parent.mkdir(parents=True,exist_ok=True)
 temporary=path.with_name('.'+path.name+'.'+common.uuid.uuid4().hex+'.tmp')
 try:
  with temporary.open('x') as stream:
   json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
  os.link(temporary,path)
 finally:
  temporary.unlink(missing_ok=True)

def verify_receipt(path,accepted_key='accepted',pin_key='input_pins'):
 d=read(path)
 if pin_key in d:verify(d[pin_key])
 return d, bool(d.get(accepted_key) is True and d.get('checks') and all(v is True for v in d['checks'].values()))

CONTROLLER_FILES = ('run_case.py', 'short_refine_agent.py', 'short_native_audit.py', 'replay_load_audit.py')

def controller_binding(controller,strategy):
 """Require the complete invoked controller to belong to the frozen strategy."""
 pins=strategy.get('implementation_sha256',{})
 checked={}
 for name in CONTROLLER_FILES:
  path=(controller/name).resolve(strict=True)
  if not path.is_file():raise ValueError('Controller input must be a file: '+str(path))
  expected=pins.get(str(path))
  if expected is None:raise ValueError('Controller file is not pinned by strategy: '+str(path))
  actual=sha(path)
  if actual!=expected:raise ValueError('Pinned controller file changed: '+str(path))
  checked[str(path)]=actual
 # Parse only the literal version; importing the controller could load physics.
 tree=ast.parse((controller/'short_refine_agent.py').read_text())
 versions=[node.value.value for node in tree.body
           if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)
           and isinstance(node.value.value,str)
           and any(isinstance(target,ast.Name) and target.id=='VERSION' for target in node.targets)]
 if len(versions)!=1:raise ValueError('Controller must declare one literal VERSION')
 return dict(directory=str(controller),files=checked,agent_version=versions[0])

def admit(native,strategy_path,controller,source_row,*,implementation,python_executable,safety_owned_externally=False):
 native,strategy_path,controller,implementation=map(lambda p:Path(p).resolve(strict=True),(native,strategy_path,controller,implementation))
 python_executable=str(Path(python_executable).resolve(strict=True))
 if not implementation.is_dir() or not controller.is_dir():raise ValueError('implementation/controller must be explicit existing directories')
 if not os.access(python_executable,os.X_OK):raise ValueError('python_executable must be executable')
 if native==Path(source_row['source_native']).resolve():raise ValueError('Refinement output must differ from frozen source native')
 strategy=read(strategy_path);source_row=dict(source_row)
 controller_request=controller_binding(controller,strategy)
 frozen={**strategy['implementation_sha256'],**source_row['source_pins'],str(strategy_path):sha(strategy_path),str(Path(__file__).resolve()):sha(__file__),str(Path(common.__file__).resolve()):sha(common.__file__)}
 receipt_path=native/'receipt.json';nr=read(receipt_path)
 if nr['strategy_sha256']!=sha(strategy_path):raise ValueError('Native receipt strategy does not match')
 if nr.get('case_id')!=source_row['case_id']:raise ValueError('Native receipt case does not match source row')
 if nr.get('agent_version')!=controller_request['agent_version']:raise ValueError('Native receipt controller version does not match invoked controller')
 request_binding=dict(native_case=str(native),strategy=str(strategy_path),strategy_sha256=sha(strategy_path),
  source_id=source_row['source_id'],case_id=source_row['case_id'],physical_index=source_row['physical_index'],
  source_order=source_row.get('source_order'),source_native=str(Path(source_row['source_native']).resolve()),
  source_pins=dict(source_row['source_pins']),controller=controller_request,
  implementation=str(implementation),python_executable=python_executable,
  native_receipt_sha256=sha(receipt_path),safety_owned_externally=bool(safety_owned_externally))
 base=native.parent;dest=base/'admission_receipt.json'
 if dest.exists():
  d=read(dest)
  if d.get('input_pins',{}).get(str(Path(__file__).resolve()))!=sha(__file__):raise RuntimeError('Existing admission is not owned by this implementation; use a fresh output attempt: '+str(dest))
  for field,expected in [('native_case',str(native)),('strategy',str(strategy_path)),('source_id',source_row['source_id'])]:
   if d.get(field)!=expected:raise RuntimeError('Existing admission '+field+' differs from this request; use a fresh output attempt: '+str(dest))
  if d.get('request_binding')!=request_binding:raise RuntimeError('Existing admission request binding differs; use a fresh output attempt: '+str(dest))
  if any(d.get('input_pins',{}).get(path)!=digest for path,digest in frozen.items()):raise RuntimeError('Existing admission does not cover every current frozen input: '+str(dest))
  verify(d['input_pins']);return d
 verify(frozen)
 runtime=SimpleNamespace(implementation=implementation)
 # A task failure is retained as a failure; postprocessing cannot repair it.
 if nr.get('accepted') is not True:
  result=dict(schema=SCHEMA,accepted=False,checks={'native_full_task_accepted':False},
   source_id=source_row['source_id'],physical_index=source_row['physical_index'],case_id=source_row['case_id'],
   native_case=str(native),strategy=str(strategy_path),request_binding=request_binding,failure_reason=nr.get('failure_reason'),input_pins={**frozen,str(receipt_path):sha(receipt_path)},publication_artifacts={})
  write(dest,result);return result
 paths=dict(quality=base/'quality/quality_receipt.json',alias=base/'quality/state_alias_receipt.json',safety=base/'safety/every_step/clearance_receipt.json')
 common.run_stage(runtime,base/'processes/alias',[python_executable,str(PROJECT_ROOT/'validation/statealias_gate_v2.py'),'--native',str(native),'--output',str(paths['alias'])],paths['alias'],frozen)
 alias,aok=verify_receipt(paths['alias'])
 if not aok:
  result=dict(schema=SCHEMA,accepted=False,checks={'native_full_task_accepted':True,'defined_near_state_future_chunk_alias_gate':False},source_id=source_row['source_id'],physical_index=source_row['physical_index'],case_id=source_row['case_id'],native_case=str(native),strategy=str(strategy_path),request_binding=request_binding,failure_reason='state_alias_gate_rejected',alias_receipt=str(paths['alias']),alias_receipt_sha256=sha(paths['alias']),input_pins={**frozen,**alias['input_pins'],str(paths['alias']):sha(paths['alias']),str(receipt_path):sha(receipt_path)},publication_artifacts={})
  write(dest,result);return result
 force=base/'quality/exact_load_replay'
 common.run_stage(runtime,base/'processes/force_replay',[python_executable,str(controller/'replay_load_audit.py'),'--native-source',str(native),'--strategy',str(strategy_path),'--case-id',source_row['case_id'],'--output',str(force),'--horizon','14.965'],force/'native_replay_load_receipt.json',frozen)
 common.run_stage(runtime,base/'processes/quality',[python_executable,str(PROJECT_ROOT/'validation/quality_gate.py'),'--receipt',str(receipt_path),'--strategy',str(strategy_path),'--implementation',str(implementation),'--load-audit',str(force),'--source-receipt',str(Path(source_row['source_native'])/'receipt.json'),'--out',str(base/'quality')],paths['quality'],frozen)
 common.run_stage(runtime,base/'processes/alias',[python_executable,str(PROJECT_ROOT/'validation/statealias_gate_v2.py'),'--native',str(native),'--output',str(paths['alias'])],paths['alias'],frozen)
 if safety_owned_externally:
  if not paths['safety'].is_file():raise RuntimeError('Waiting for owned external clearance replay: '+str(paths['safety']))
 else:
  common.run_stage(runtime,base/'processes/safety',[python_executable,str(PROJECT_ROOT/'validation/clearance_replay_v2.py'),'--native',str(native),'--strategy',str(strategy_path),'--case-id',source_row['case_id'],'--implementation',str(implementation),'--output',str(paths['safety'].parent)],paths['safety'],frozen)
 quality,qok=verify_receipt(paths['quality'],'candidate_accepted','evidence_sha256')
 alias,aok=verify_receipt(paths['alias'])
 safety,sok=verify_receipt(paths['safety'],'accepted','pins')
 verify(frozen)
 checks=dict(native_full_task_accepted=True,strict_physical_first_grasp_force_and_source_quality=qok,
  defined_near_state_future_chunk_alias_gate=aok,every_step_cross_side_clearance_20mm=sok,
  safety_requirement_exact_20mm=float(safety.get('required_clearance_m',0))==.020,
  safety_covers_every_step_and_intervals=safety.get('samples')==29932 and safety.get('intervals')==29931,
  full_physical_duration_under_15s=0<float(quality['full_physics_time_s'])<15,
  safety_source_is_this_native=Path(safety['native_root']).resolve()==native,
  alias_source_is_this_native=Path(alias['native_case']).resolve()==native,
  exact_published_source_inherited=quality['checks'].get('all_source_initial_native_arrays_preserved_exactly') is True and quality['checks'].get('source_model_and_scene_xml_preserved_exactly') is True)
 pins=dict(frozen)
 for sub,key in [(quality,'evidence_sha256'),(alias,'input_pins'),(safety,'pins')]:pins.update(sub[key])
 for p in [receipt_path,*paths.values(),native/'replay/rollout.npz',native/'replay/immutable_hard_trace.npz',force/'native_replay_load_receipt.json',force/'native_replay_loads.npz',paths['safety'].parent/'clearance_trace.npz']:
  pins[str(p.resolve())]=sha(p)
 artifacts={}
 def artifact(relative,p):artifacts[relative]=dict(path=str(p.resolve()),sha256=sha(p))
 for k,p in paths.items():artifact(k+'_receipt.json',p)
 for rel,p in [('native_receipt.json',receipt_path),('strategy.json',strategy_path),('actual_force/receipt.json',force/'native_replay_load_receipt.json'),('actual_force/native_replay_loads.npz',force/'native_replay_loads.npz'),('safety/clearance_trace.npz',paths['safety'].parent/'clearance_trace.npz'),('initialization.json',native/'s1_command_initialization.json')]:artifact(rel,p)
 result=dict(schema=SCHEMA,accepted=all(checks.values()),checks=checks,source_id=source_row['source_id'],
  physical_index=source_row['physical_index'],case_id=source_row['case_id'],source_order=source_row['source_order'],
  native_case=str(native),strategy=str(strategy_path),request_binding=request_binding,source_native=source_row['source_native'],
  source_revision='098642c9e57dfa933db9d8976564a55bd740a1ea',source_prefix='150case_data_enhance_blender_v2_trajectory_refine',
  full_physics_time_s=quality['full_physics_time_s'],quality_receipt=str(paths['quality']),quality_receipt_sha256=sha(paths['quality']),alias_receipt=str(paths['alias']),alias_receipt_sha256=sha(paths['alias']),safety_receipt=str(paths['safety']),safety_receipt_sha256=sha(paths['safety']),
  input_pins=pins,publication_artifacts=artifacts,clearance_scope='All opposite-side arm/gripper/assumed wrist-camera hull pairs >=20mm; table follows original native hard checks',
  hardware_safety_claim=False,no_video_time_scaling=True)
 write(dest,result);return result

def main():
 p=argparse.ArgumentParser(description='严格准入原生refine结果 / Strict admission of a new native refinement attempt. No rendering or HF writes.')
 p.add_argument('--native',type=Path,required=True,help='New owned native output; admission is written only in its parent')
 p.add_argument('--strategy',type=Path,required=True)
 p.add_argument('--controller',type=Path,required=True,help='Explicit controller version directory')
 p.add_argument('--implementation',type=Path,required=True,help='Explicit rssr_augment implementation directory; no workspace default')
 p.add_argument('--python',dest='python_executable',type=Path,required=True,help='Explicit native interpreter path')
 p.add_argument('--source-manifest',type=Path,required=True)
 p.add_argument('--source-id',required=True)
 p.add_argument('--external-safety',action='store_true')
 a=p.parse_args()
 source=read(a.source_manifest)
 matches=[r for r in source['cases'] if r['source_id']==a.source_id]
 if len(matches)!=1:raise ValueError('source_id must occur exactly once in source manifest')
 row=dict(matches[0]);row['source_pins']=dict(row['source_pins'])
 row['source_pins'][str(a.source_manifest.resolve())]=sha(a.source_manifest)
 r=admit(a.native,a.strategy,a.controller,row,implementation=a.implementation,python_executable=a.python_executable,safety_owned_externally=a.external_safety)
 print(json.dumps(dict(accepted=r['accepted'],source_id=r['source_id'],failed_checks=[k for k,v in r['checks'].items() if not v])))

if __name__=='__main__':main()
