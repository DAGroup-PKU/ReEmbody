"""Full native-clock shell/table audit with independent cheap-axis witnesses.

Only the independent raw-zero SAT witness search is accelerated. Every support
projection uses every original compiled mesh vertex. Any failed cheap search
falls back to the frozen full-axis SAT implementation. Raw-negative distance
queries retain the inherited unconditional rejection behavior. No mj_step.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import mujoco
import numpy as np
import drawer_shell_environment_v2 as geometry


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FastShellEnvironmentGeometry(geometry.ShellEnvironmentGeometry):
    """Try box principal axes and previous world witness before full SAT.

    A previous axis is only a candidate. All support intervals and numerical
    guards are recomputed from the actual current FK and all source vertices.
    """
    def __init__(self,model,manifest):
        super().__init__(model,manifest)
        self._fast_vertices={}
        self._previous_world_axis={}
        self.fast_sat_stats=dict(calls=0,cheap_certificates=0,full_axis_fallbacks=0,
            full_axis_certificates=0,cheap_projected_source_vertex_count=0)

    def sat_separation_certificate(self,data,shell,env):
        self.fast_sat_stats['calls']+=1
        if int(self.model.geom_type[shell])!=int(mujoco.mjtGeom.mjGEOM_MESH):
            return None
        mesh=int(self.model.geom_dataid[shell])
        if mesh not in self._fast_vertices:
            start=int(self.model.mesh_vertadr[mesh]);count=int(self.model.mesh_vertnum[mesh])
            self._fast_vertices[mesh]=self.model.mesh_vert[start:start+count].astype(float)
        vertices=self._fast_vertices[mesh]
        shell_rotation=data.geom_xmat[shell].reshape(3,3)
        box_rotation=data.geom_xmat[env].reshape(3,3)
        relative_rotation=shell_rotation.T@box_rotation
        box_center=shell_rotation.T@(data.geom_xpos[env]-data.geom_xpos[shell])
        box_size=self.model.geom_size[env]
        scale=1.+max(float(np.max(np.abs(vertices))),float(np.max(np.abs(box_center))),float(np.max(box_size)))
        guard=geometry.SAT_PROJECTION_GUARD_M*scale
        world_axes=[box_rotation[:,i] for i in range(3)]
        previous=self._previous_world_axis.get((int(shell),int(env)))
        if previous is not None:
            world_axes.append(previous)
        valid=[]
        for world_axis in world_axes:
            norm=float(np.linalg.norm(world_axis))
            if not np.isfinite(world_axis).all() or norm<=geometry.AXIS_MIN_NORM:
                continue
            world_axis=np.asarray(world_axis)/norm
            local_axis=shell_rotation.T@world_axis
            local_norm=float(np.linalg.norm(local_axis))
            if not np.isfinite(local_axis).all() or local_norm<=geometry.AXIS_MIN_NORM:
                continue
            local_axis=local_axis/local_norm
            if local_axis[int(np.argmax(np.abs(local_axis)))]<0:
                local_axis=-local_axis
            if any(np.max(np.abs(local_axis-old))<1e-12 for old in valid):
                continue
            valid.append(local_axis)
        if valid:
            axes=np.asarray(valid)
            projections=vertices@axes.T
            mesh_lo,mesh_hi=projections.min(0),projections.max(0)
            box_mid=axes@box_center
            box_extent=np.abs(axes@relative_rotation)@box_size
            box_lo,box_hi=box_mid-box_extent,box_mid+box_extent
            gaps=np.maximum(mesh_lo-box_hi,box_lo-mesh_hi)
            best=int(np.argmax(gaps));bound=float(gaps[best]-2*guard)
            self.fast_sat_stats['cheap_projected_source_vertex_count']+=len(vertices)
            if bound>0:
                axis=axes[best];world_axis=shell_rotation@axis
                self._previous_world_axis[(int(shell),int(env))]=world_axis.copy()
                self.fast_sat_stats['cheap_certificates']+=1
                return dict(method='convex_mesh_box_projection_axis',
                    separation_lower_bound_m=bound,axis_in_shell_frame=axis.tolist(),axis_in_world=world_axis.tolist(),
                    mesh_projection_interval_m=[float(mesh_lo[best]),float(mesh_hi[best])],
                    box_projection_interval_m=[float(box_lo[best]),float(box_hi[best])],
                    support_outward_guard_m=guard,raw_axis_gap_m=float(gaps[best]),
                    source_vertices_projected=len(vertices),candidate_axis_count=len(axes),
                    axis_search='box3_principal_axes_and_previous_world_witness_then_frozen_full_sat',
                    interpretation='Every original compiled mesh vertex is projected on a normalized axis. Expanding both support intervals by the frozen scale-dependent guard leaves a strictly positive gap, a lower bound on Euclidean separation; this is not exact mesh distance.')
        self.fast_sat_stats['full_axis_fallbacks']+=1
        result=super().sat_separation_certificate(data,shell,env)
        if result is not None:
            self.fast_sat_stats['full_axis_certificates']+=1
            self._previous_world_axis[(int(shell),int(env))]=np.asarray(result['axis_in_world'],float)
        return result


def audit(native,clearance_trace,output):
    native=Path(native).resolve();clearance_trace=Path(clearance_trace).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    if mujoco.__version__!=geometry.REQUIRED_MUJOCO:
        raise ValueError('MuJoCo '+geometry.REQUIRED_MUJOCO+' required')
    started=time.perf_counter()
    model=mujoco.MjModel.from_xml_path(str(native/'execute/scene.xml'))
    manifest=geometry.select_geometry(model)
    manifest['environment_boxes']=[g for g in manifest['environment_boxes'] if g['category']=='table']
    manifest['selection_checks']={k:v for k,v in manifest['selection_checks'].items()
        if k not in ('cabinet_boxes_present','moving_bottom_drawer_back_present')}
    manifest['selection_rule']='All visible arm links/shoulder bases and complete visible gripper base/body, versus all native table boxes; intentional fingers/payload excluded.'
    if not all(manifest['selection_checks'].values()):raise ValueError('Missing required stack shell/table coverage')
    gate=FastShellEnvironmentGeometry(model,manifest);data=mujoco.MjData(model)
    with np.load(clearance_trace,allow_pickle=False) as z:
        times=z['time_s'].copy();qpos=z['qpos'].copy()
    with np.load(native/'execute/immutable_hard_trace.npz',allow_pickle=False) as z:
        native_times=z['time_s'].copy()
    if qpos.shape!=(len(times),model.nq) or not np.isfinite(qpos).all():
        raise ValueError('Full finite native qpos required for every declared time sample')
    coverage=times.shape==native_times.shape and times.dtype==native_times.dtype and times.tobytes()==native_times.tobytes()
    if not coverage:raise ValueError('Shell-table geometry must cover exact native time clock')
    bounds=[];pairs=[];first=None;raw_negative_queries=0
    for i,(t,q) in enumerate(zip(times,qpos)):
        data.qpos[:]=q;data.time=float(t);mujoco.mj_kinematics(model,data)
        bound,pair,rows=gate.at_forwarded_state(data)
        raw_negative_queries+=sum(row[4]<0 for row in rows)
        bounds.append(bound);pairs.append(pair)
        if bound<=0 and first is None:first=dict(sample=i,time_s=float(t),pair=int(pair),lower_bound_m=float(bound))
    output.mkdir(parents=True)
    trace=output/'shell_table_trace.npz'
    np.savez_compressed(trace,time_s=times,separation_lower_bound_m=np.asarray(bounds),pair_index=np.asarray(pairs))
    checks=dict(exact_native_0p5ms_clock=coverage,samples_29932=len(times)==29932,
        selected_geometry_complete=all(manifest['selection_checks'].values()),
        all_discrete_shell_table_separations_positive=first is None,
        all_bounds_finite=bool(np.isfinite(bounds).all()))
    i=int(np.argmin(bounds));pi=int(pairs[i]);ns=len(gate.shells);pair=gate.pairs[pi]
    inputs=[native/'execute/scene.xml',native/'execute/immutable_hard_trace.npz',clearance_trace,
        Path(__file__),Path(geometry.__file__),*geometry.xml_dependencies(native/'execute/scene.xml')]
    result=dict(schema='rssr.stack_cups.random300.shell_table.v2',accepted=all(checks.values()),checks=checks,
        native_case=str(native),samples=len(times),minimum_lower_bound_m=float(bounds[i]),
        minimum_time_s=float(times[i]),minimum_shell=gate.shells[int(pair[0])],
        minimum_environment=gate.environments[int(pair[1])-ns],first_nonpositive=first,
        manifest=manifest,trace=str(trace),trace_sha256=sha(trace),
        input_pins={str(p):sha(p) for p in inputs},fast_sat_stats=gate.fast_sat_stats,
        raw_negative_queries=raw_negative_queries,elapsed_s=time.perf_counter()-started,
        limitations=['Actual discrete native steps only; no continuous table interval or hardware proof.',
            'Convex hulls conservatively bound visual meshes; negative evidence rejects without claiming exact mesh penetration depth.',
            'The user20mm requirement applies between opposing robot arms/grippers; this table guard requires positive separation.',
            'Cheap SAT returns any certified positive lower bound, so its numeric minimum/pair may differ from the full-axis maximum witness while acceptance semantics and full native clock are retained.'])
    (output/'receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--clearance-trace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.native,a.clearance_trace,a.output)
    print(json.dumps(dict(accepted=r['accepted'],minimum_lower_bound_m=r['minimum_lower_bound_m'],elapsed_s=r['elapsed_s'],fast_sat_stats=r['fast_sat_stats'])),flush=True)
