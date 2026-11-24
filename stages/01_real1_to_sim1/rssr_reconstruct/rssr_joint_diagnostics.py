"""Read-only joint-limit peak context; observations are not causal labels."""
import numpy as np
import mujoco


def actuator_moment_column(model, data, dof):
    """Read one transmission column across dense 2.x and sparse 3.x layouts."""
    values = np.asarray(data.actuator_moment)
    if values.ndim == 2:
        return values[:, dof]
    if values.ndim != 1:
        raise ValueError('Unknown actuator-moment layout')
    result = np.zeros(model.nu)
    for actuator in range(model.nu):
        start = int(data.moment_rowadr[actuator])
        end = start + int(data.moment_rownnz[actuator])
        selected = np.asarray(data.moment_colind[start:end]) == dof
        result[actuator] = values[start:end][selected].sum()
    return result


class JointLimitTrace:
    def __init__(self, model, joints):
        self.joints=list(joints)
        self.names=[model.joint(j).name for j in joints]
        self.qa=model.jnt_qposadr[joints].copy()
        self.va=model.jnt_dofadr[joints].copy()
        self.limits=model.jnt_range[joints].copy()
        self.maxima=np.zeros(len(joints))
        self.peaks={}
        self.geometry_names=[model.geom(g).name for g in range(model.ngeom)]
        self.geometry_bodies=model.geom_bodyid.copy()
        self.actuator_names=[model.actuator(a).name for a in range(model.nu)]
        self.transmissions=[str(mujoco.mjtTrn(int(t))) for t in model.actuator_trntype]
        # Equality-connected passive fingers can load each other even when
        # their geometries live in sibling branches of the body tree.
        links={j:set() for j in range(model.njnt)}
        for i in range(model.neq):
            if model.eq_type[i]!=mujoco.mjtEq.mjEQ_JOINT:continue
            a,b=int(model.eq_obj1id[i]),int(model.eq_obj2id[i])
            if a>=0 and b>=0:links[a].add(b);links[b].add(a)
        self.related_bodies=[];self.coupled=[]
        for joint in joints:
            connected={joint};pending=[joint]
            while pending:
                for other in links[pending.pop()]-connected:
                    connected.add(other);pending.append(other)
            roots={int(model.jnt_bodyid[j]) for j in connected}
            related=set()
            for body in range(1,model.nbody):
                ancestor=body
                while ancestor:
                    if ancestor in roots:related.add(body);break
                    ancestor=int(model.body_parentid[ancestor])
            self.related_bodies.append(related)
            self.coupled.append([model.joint(j).name for j in sorted(connected)])

    def observe(self, model, data, excess, reference, reference_time_s):
        # The caller supplies the existing every-step criterion calculation.
        # Reading context neither changes that calculation nor steps the model.
        changed=np.flatnonzero(np.asarray(excess)>self.maxima)
        for index in changed:
            qa=int(self.qa[index]);lo,hi=self.limits[index]
            position=float(data.qpos[qa]);contacts=[]
            for contact in data.contact:
                if contact.dist>0:continue
                a,b=int(contact.geom1),int(contact.geom2)
                if not ({int(self.geometry_bodies[a]),int(self.geometry_bodies[b])} & self.related_bodies[index]):continue
                contacts.append({'geoms':[self.geometry_names[a],self.geometry_names[b]],
                                 'distance_m':float(contact.dist),
                                 'position_world_m':np.asarray(contact.pos).tolist()})
            contacts.sort(key=lambda x:x['distance_m'])
            # Read actual transmission moments. S1's two-finger actuator uses
            # a fixed tendon, so joint-only actuator IDs mislabel it passive.
            moments=actuator_moment_column(model,data,int(self.va[index]))
            drives=[{'actuator':self.actuator_names[a],'transmission':self.transmissions[a],
                     'moment_arm_native':float(moments[a]),'control_native':float(data.ctrl[a]),
                     'actuator_force_native':float(data.actuator_force[a])}
                    for a in np.flatnonzero(np.abs(moments)>1e-12)]
            self.peaks[self.names[index]]={
                'peak_time_s':float(data.time),'excess_native':float(excess[index]),
                'position_native':position,'limits_native':[float(lo),float(hi)],
                'violated_side':'lower' if position<lo else 'upper',
                'velocity_native_per_s':float(data.qvel[self.va[index]]),
                'reference_qpos_native':float(reference[qa]),
                'reference_time_s':float(reference_time_s),
                'actuator_transmissions_at_peak':drives,
                'equality_connected_joints':self.coupled[index],
                'related_contacts':contacts[:6], 'related_contact_count':len(contacts),
                'contact_evaluation_interval_s':[float(reference_time_s),float(data.time)],
                'contact_scope':'joint/equality-connected body subtrees; context of the integration step, not proof of the cause'}
        self.maxima=np.maximum(self.maxima,excess)

    def report(self):
        return {'per_joint_peak':self.peaks,
                'scope':'maxima observed at every physical step; only positive limit excesses listed',
                'reference_is_not_executed_qpos':True,
                'empty_transmission_list_does_not_exclude_equality_or_dynamic_loading':True,
                'empty_related_contacts_does_not_prove_absence_of_dynamic_cause':True}
