from pathlib import Path
import argparse
import mujoco, json, numpy as np
if __package__:
 from ._shared.paths import PROJECT_ROOT
else:
 from _shared.paths import PROJECT_ROOT


def inspect(native):
 p=Path(native)
 m=mujoco.MjModel.from_xml_path(str(p/'execute/scene.xml'));d=mujoco.MjData(m);z=np.load(p/'execute/rollout.npz');d.qpos[:]=z['qpos'][0];mujoco.mj_forward(m,d)
 for gi in range(m.ngeom):
  b=m.body(int(m.geom_bodyid[gi])).name or '';n=m.geom(gi).name or ''
  if 'gripper' in b or 'table' in n or 'table' in b:
   print(gi,n,b,'type',m.geom_type[gi],'rbound',m.geom_rbound[gi],'pos',d.geom_xpos[gi].tolist())
 print('cameras',[(m.camera(i).name,d.cam_xpos[i].tolist()) for i in range(m.ncam)])
 print('q0',z['qpos'][0].tolist())


if __name__=='__main__':
 parser=argparse.ArgumentParser(description='Read-only initial native geometry inspection; no stepping.')
 parser.add_argument('--native',type=Path,default=PROJECT_ROOT/'result/sim1-sim2/s1_short_refine_v2_20261018/revisions/issued_clock_v4/physical_01/refined_cases/stack_cups_ep000000/attempt_00/native')
 inspect(parser.parse_args().native)
