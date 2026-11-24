"""Opt-in, scene-independent S1 cross-arm collision configuration.

The imported model gives both arms category 1 and affinity 6.  That pair
cannot collide with itself, so two different arms can pass through each other.
This adapter gives each arm and hand a separate category while preserving
contacts with category-2 tables and category-4 task objects.  Same-side
robot geoms remain filtered to avoid contacts between adjacent links.
"""

LEFT_ARM = 16
RIGHT_ARM = 32
LEFT_HAND = 64
RIGHT_HAND = 128
ENVIRONMENT = 2 | 4


def apply_s1_cross_arm_collision(xml, include_visual_gripper_meshes=False):
    changed = []
    visual_hand_meshes = []
    parents = {child: parent for parent in xml.iter() for child in parent}

    def arm_side(node):
        while node in parents:
            node = parents[node]
            if node.tag != 'body':
                continue
            body_name = node.get('name', '')
            if body_name.startswith('astribot_arm_left_'):
                return 'left'
            if body_name.startswith('astribot_arm_right_'):
                return 'right'
        return None

    for geom in xml.iter('geom'):
        name = geom.get('name', '')
        if name.startswith('physical_robot_astribot_arm_left_link_'):
            side, hand = 'left', False
        elif name.startswith('physical_robot_astribot_arm_right_link_'):
            side, hand = 'right', False
        elif name.startswith('astribot_gripper_') and (
                geom.get('contype', '0') != '0' or
                (include_visual_gripper_meshes and geom.get('type') == 'mesh')):
            side, hand = arm_side(geom), True
            if side is None:
                raise ValueError('Collidable S1 gripper has no left/right arm parent: '+name)
            if geom.get('contype', '0') == '0':
                visual_hand_meshes.append(name)
        else:
            continue
        if side == 'left':
            category = LEFT_HAND if hand else LEFT_ARM
            affinity = ENVIRONMENT | RIGHT_ARM | RIGHT_HAND
        else:
            category = RIGHT_HAND if hand else RIGHT_ARM
            affinity = ENVIRONMENT | LEFT_ARM | LEFT_HAND
        geom.set('contype', str(category))
        geom.set('conaffinity', str(affinity))
        changed.append(name)
    if not any('arm_left_link' in name for name in changed):
        raise ValueError('S1 left-arm collision geoms missing')
    if not any('arm_right_link' in name for name in changed):
        raise ValueError('S1 right-arm collision geoms missing')
    if not any(geom.get('contype') == str(LEFT_HAND) for geom in xml.iter('geom')):
        raise ValueError('S1 left-hand collision geoms missing')
    if not any(geom.get('contype') == str(RIGHT_HAND) for geom in xml.iter('geom')):
        raise ValueError('S1 right-hand collision geoms missing')
    if include_visual_gripper_meshes and len(visual_hand_meshes) != 12:
        raise ValueError('Expected exactly 12 S1 visual gripper meshes for physical collision')
    return changed


def _side(name):
    if name.startswith('astribot_gripper_right_left_geom'):
        return 'left'
    if name.startswith(('physical_robot_astribot_arm_left_link_', 'astribot_gripper_left_')):
        return 'left'
    if name.startswith(('physical_robot_astribot_arm_right_link_', 'astribot_gripper_right_')):
        return 'right'
    return None


class ContactIntegrityMonitor:
    """Read every MuJoCo step; track deep cross-arm and robot-scene contacts.

    Ordinary shallow grasp and fixture contacts remain allowed.  The monitor
    records an independent physical validity gate, not a task-specific rule.
    """

    def __init__(self, model, limit_m=.003):
        self.names = tuple(model.geom(i).name or '' for i in range(model.ngeom))
        def model_side(geom_id):
            name = self.names[geom_id]
            if not (name.startswith('physical_robot_astribot_arm_') or
                    name.startswith('astribot_gripper_')):
                return None
            body_id = int(model.geom_bodyid[geom_id])
            while body_id:
                body_name = model.body(body_id).name or ''
                if body_name.startswith('astribot_arm_left_'):
                    return 'left'
                if body_name.startswith('astribot_arm_right_'):
                    return 'right'
                body_id = int(model.body_parentid[body_id])
            return _side(name)
        self.sides = tuple(model_side(i) for i in range(model.ngeom))
        self.robot = tuple(bool(side) for side in self.sides)
        self.limit_m = float(limit_m)
        self.max_cross_arm_m = 0.0
        self.max_robot_scene_m = 0.0
        self.deep_cross_arm_steps = 0
        self.deep_robot_scene_steps = 0
        self.steps = 0

    def observe(self, data):
        cross, scene = 0.0, 0.0
        for contact in data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            depth = max(0.0, -float(contact.dist))
            if self.sides[a] and self.sides[b] and self.sides[a] != self.sides[b]:
                cross = max(cross, depth)
            elif self.robot[a] != self.robot[b]:
                # Includes task objects and fixtures. The physical contact
                # may be intentional; only deep interpenetration is invalid.
                scene = max(scene, depth)
        self.max_cross_arm_m = max(self.max_cross_arm_m, cross)
        self.max_robot_scene_m = max(self.max_robot_scene_m, scene)
        self.deep_cross_arm_steps += cross > self.limit_m
        self.deep_robot_scene_steps += scene > self.limit_m
        self.steps += 1
        return {'cross_arm_penetration_m': cross,
                'robot_scene_penetration_m': scene,
                'passed': cross <= self.limit_m and scene <= self.limit_m}

    def summary(self):
        return {'steps': self.steps, 'limit_m': self.limit_m,
                'max_cross_arm_penetration_m': self.max_cross_arm_m,
                'max_robot_scene_penetration_m': self.max_robot_scene_m,
                'deep_cross_arm_steps': self.deep_cross_arm_steps,
                'deep_robot_scene_steps': self.deep_robot_scene_steps,
                'passed': self.deep_cross_arm_steps == 0 and self.deep_robot_scene_steps == 0}
