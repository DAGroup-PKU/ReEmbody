"""Cache immutable servo metadata without changing per-step control arithmetic."""
import numpy as np


class JointServo:
    def __init__(self, model, acts, kind):
        self.act = np.asarray([a for a, _, _ in acts], dtype=int)
        self.qa = np.asarray([q for _, q, _ in acts], dtype=int)
        self.va = np.asarray([v for _, _, v in acts], dtype=int)
        self.denominator = np.maximum(model.actuator_gainprm[self.act, 0], 1e-8)
        self.damping = -model.actuator_biasprm[self.act, 2] + model.dof_damping[self.va]
        self.limited = np.asarray(model.actuator_ctrllimited[self.act], dtype=bool)
        self.limited_act = self.act[self.limited]
        self.limits = model.actuator_ctrlrange[self.limited_act].copy()
        self.grippers = []
        if kind == 'sim2':
            for side in ('left', 'right'):
                name = 'astribot_gripper_' + side + '_joint_L1'
                self.grippers.append((model.actuator(name).id, model.joint(name).qposadr[0]))

    def apply(self, data, desired, velocity):
        # Preserve the original operation order. Bias forces remain live data.
        data.ctrl[self.act] = desired[self.qa] + (
            self.damping * velocity[self.qa] + data.qfrc_bias[self.va]
        ) / self.denominator
        data.ctrl[self.limited_act] = np.clip(
            data.ctrl[self.limited_act], self.limits[:, 0], self.limits[:, 1]
        )
        # This adapter intentionally follows the original post-clamp override.
        for actuator, qpos in self.grippers:
            data.ctrl[actuator] = 100 * desired[qpos] / .93
