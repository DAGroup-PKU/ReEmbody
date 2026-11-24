"""Original issued-action columns, including fixed tendon grippers."""
import numpy as np
def desired_columns(model, issued_qpos):
    import mujoco
    columns, addresses = [], []
    for index in range(model.nu):
        actuator = model.actuator(index).name
        transmission = int(model.actuator_trntype[index])
        provenance = {'actuator_name': actuator}
        if transmission == mujoco.mjtTrn.mjTRN_JOINT:
            joint = model.joint(int(model.actuator_trnid[index, 0]))
            provenance['actuator_transmission'] = 'joint'
        elif transmission == mujoco.mjtTrn.mjTRN_TENDON:
            side = next((s for s in ('left', 'right')
                         if actuator == f'astribot_gripper_{s}_joint_L1'), None)
            if side is None:
                raise ValueError('Unsupported tendon target interface: ' + str(actuator))
            tendon = model.tendon(int(model.actuator_trnid[index, 0]))
            start, count = int(model.tendon_adr[tendon.id]), int(model.tendon_num[tendon.id])
            members = slice(start, start + count)
            names = [model.joint(int(j)).name for j in model.wrap_objid[members]]
            expected = {f'astribot_gripper_{side}_joint_L1', f'astribot_gripper_{side}_joint_R1'}
            if (tendon.name != f'gripper_{side}_split' or count != 2 or set(names) != expected or
                    not np.all(model.wrap_type[members] == mujoco.mjtWrap.mjWRAP_JOINT) or
                    not np.array_equal(model.wrap_prm[members], [.5, .5]) or
                    not np.array_equal(model.actuator_gear[index], [1., 0., 0., 0., 0., 0.])):
                raise ValueError('S1 gripper fixed-tendon mapping changed')
            finger_addresses = [int(model.joint(name).qposadr[0]) for name in names]
            if not np.array_equal(issued_qpos[:, finger_addresses[0]], issued_qpos[:, finger_addresses[1]]):
                raise ValueError('Issued left/right finger targets differ; cannot collapse the gripper action')
            joint = model.joint(actuator)
            provenance.update(actuator_transmission='fixed_tendon', tendon_name=tendon.name,
                              coupled_target_joints=names, fixed_tendon_coefficients=[.5, .5],
                              all_issued_finger_targets_equal=True)
        else:
            raise ValueError('Unsupported native actuator target interface: ' + str(actuator))
        if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError('Expected a scalar S1 hinge target: ' + joint.name)
        columns.append({'index': index, 'name': joint.name, 'unit': 'rad',
                        'semantics': 'desired_joint_position', 'target_joint_or_interface': joint.name,
                        **provenance})
        addresses.append(int(joint.qposadr[0]))
    return columns, addresses

