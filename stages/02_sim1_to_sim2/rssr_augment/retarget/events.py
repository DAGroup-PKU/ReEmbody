"""Compile physical events from native predicate structure and object geometry."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math

import numpy as np

from .session import EventAction
from .executor import (desired_rotation, overhead_approach,
                       side_corridor_approach, front_staging_transport)
from .capability_compiler import (articulated_feature, articulation_axis_world,
                                  choose_articulation_feature, derive_pickup_objects,
                                  explicit_object_hands, other_hand_allowed)


@dataclass(frozen=True)
class EventDefinition:
    event_id: str
    action: EventAction
    evidence: str
    abandon_if: str
    source_predicate: str

    def as_packet(self) -> dict:
        packet = {"event_id": self.event_id, "physical_event": self.action.event,
                "hand": self.action.hand, "object_id": self.action.object_id,
                "target_id": self.action.target_id,
                "target_offset_m": list(self.action.target_offset_m),
                "max_speed_m_s": self.action.max_speed_m_s,
                "gripper_fraction": self.action.gripper_fraction,
                "contact_mode": self.action.recovery_mode or self.event_id.split(":")[-1], "evidence": self.evidence,
                "abandon_if": self.abandon_if,
                "source_predicate": self.source_predicate}
        if self.action.base_joint_target is not None:
            packet['base_joint_target'] = list(self.action.base_joint_target)
            packet['base_axis_order'] = list(self.action.base_axis_order)
            packet['base_certificate_sha256'] = self.action.base_certificate_sha256
        if self.action.bystander_clearance_m is not None:
            packet['bystander_clearance_m'] = self.action.bystander_clearance_m
        if self.action.support_pull_m is not None:
            packet['support_pull_m'] = self.action.support_pull_m
        return packet


def buffered_open_fraction(required: float) -> float:
    """A feasible optional reserve above an ordered-operation threshold."""
    threshold = float(required)
    return max(threshold, min(.98, threshold + .04))


def catalog(contract: dict, scene: dict,
            manipulators: tuple[str, ...] = ('left', 'right')) -> list[EventDefinition]:
    """Derive actionable events from the frozen native predicate contract."""
    if (not manipulators or len(manipulators) != len(set(manipulators)) or
            any(not isinstance(hand, str) or not hand for hand in manipulators)):
        raise ValueError('Robot adapter supplied invalid manipulator capabilities')
    primary = manipulators[0]
    explicit_object_hands(contract, manipulators)
    objects = {row["id"]: row for row in scene["objects"]}
    # These native checks constrain selection and the final evaluator, but do
    # not themselves require a new motion primitive. They remain visible to
    # RuleAgent and cannot be weakened by a generated rule.
    observational_predicates = {'outside', 'preserved', 'table_contact',
                                'hold_source_interval', 'hand_contact_observed',
                                'initial_outside_region',
                                'initial_not_upright_supported',
                                'particles_initially_inside'}
    known_predicates = observational_predicates | {'lift_with_contact', 'inside_region', 'inside',
                        'stable_supported_on', 'cup_nested', 'open_insert_close',
                        'released', 'initial_outside', 'object_contact',
                        'base_placed_before_stack', 'stable'}
    unknown = sorted({p['kind'] for p in contract['predicates']} - known_predicates)
    if unknown:
        raise ValueError('Capability gap: unsupported predicate kinds ' + repr(unknown))
    result: list[EventDefinition] = []

    def add(event_id, kind, hand, obj, target, evidence, abandon, predicate,
            offset=(0.0, 0.0, 0.0), opening=None, recovery_mode=None):
        if obj is not None and obj not in objects:
            raise ValueError("Task predicate names an unknown scene object")
        if target is not None and target not in objects:
            raise ValueError("Task predicate names an unknown target object")
        result.append(EventDefinition(event_id, EventAction(kind, hand, obj, target,
                                                            max_speed_m_s=(0.02 if kind=="contact" else
                                                                           0.04 if kind in ("clearance", "retreat") else 0.08), target_offset_m=offset,
                                                            gripper_fraction=1.0 if kind == "approach" else opening,
                                                            recovery_mode=recovery_mode),
                                      evidence, abandon, predicate))

    for pred in contract["predicates"]:
        kind, pid = pred["kind"], pred["id"]
        if kind == "lift_with_contact":
            obj, hand = pred["object"], pred["hand"]
            if hand not in manipulators:
                raise ValueError('Contract binds an unavailable manipulator')
            height = float(objects[obj]["size_m"][2])
            offset = (0.0, 0.0, 0.0)
            add(f"approach:{obj}", "approach", hand, obj, None,
                "finger center approaches object", "distance or joint margin stops improving",
                pid, offset)
            add(f"contact:{obj}", "contact", hand, obj, None,
                "both finger pads contact object", "contact stays absent or hard margin worsens",
                pid, offset, 0.25)
            add(f"grasp:{obj}", "grasp", hand, obj, None,
                "both finger pads contact while opening closes", "contact is lost",
                pid)
            add(f"lift:{obj}", "lift", hand, obj, None,
                f"object rises at least {pred['height_m']} m with bilateral contact",
                "contact is lost or lift margin stalls", pid, opening=0.0)
        elif kind in ("inside_region", "inside"):
            obj, target = pred["object"], pred.get("region", pred.get("container"))
            if kind == 'inside' and any(
                    p['kind'] == 'open_insert_close' and p['object'] == obj
                    and p['container'] == target for p in contract['predicates']):
                continue
            hand = next((p["hand"] for p in contract["predicates"]
                         if p["kind"] == "lift_with_contact" and p["object"] == obj),
                        primary)
            add(f"target_relation:{obj}", "transport", hand, obj, target,
                "object moves inside target region", "relation margin stalls or grasp is lost",
                pid, (0.0, 0.0, (objects[target]['size_m'][2]+objects[obj]['size_m'][2])/2+.002), 0.0)
        elif kind in ("stable_supported_on", "cup_nested"):
            obj, target = pred["object"], pred["support"]
            hand = next((p["hand"] for p in contract["predicates"]
                         if p["kind"] == "lift_with_contact" and p["object"] == obj),
                        primary)
            nested_release_height = (objects[target]['size_m'][2]+objects[obj]['size_m'][2])/2-float(pred.get('minimum_depth_m',0))-.005
            add(f"target_relation:{obj}", "align", hand, obj, target,
                "upper object approaches support relation", "relation margin stalls or grasp is lost",
                pid, (0.0, 0.0, nested_release_height if kind=='cup_nested' else (objects[target]['size_m'][2]+objects[obj]['size_m'][2])/2+.001), 0.0)
        elif kind == "open_insert_close":
            container, payload = pred["container"], pred["object"]
            handle = choose_articulation_feature(container, scene)
            axis = articulation_axis_world(objects[container])
            handle_offset = tuple(float(v) for v in .026 * axis)
            def permitted_hands(obj):
                bound = {p['hand'] for p in contract['predicates']
                         if p.get('kind') == 'lift_with_contact' and
                         p.get('object') == obj and p.get('hand') in manipulators}
                return sorted(bound) if bound else manipulators
            for hand in permitted_hands(handle):
                add(f"approach:articulation_feature:{hand}", "approach", hand, handle, None,
                    "finger center approaches articulation feature", "distance margin stalls", pid, handle_offset)
                add(f"contact:articulation_feature:{hand}", "contact", hand, handle, None,
                    "fingers touch articulation feature", "contact stays absent or hard margin worsens", pid,
                    offset=handle_offset, opening=0.2)
                add(f"grasp:articulation_feature:{hand}","grasp",hand,handle,None,
                    "bilateral feature contact persists for 0.1s","contact is lost",pid)
                add(f"open:articulation:{hand}", "open_drawer", hand, handle, container,
                    "articulation opening fraction reaches contract threshold", "opening does not increase",
                    pid, opening=0.0)
                add(f"close:articulation:{hand}", "close_drawer", hand, handle, container,
                    "articulation closing fraction reaches contract threshold", "closing does not improve",
                    pid, opening=0.0)
                add(f"push_close:{container}:{hand}", "push_close", hand, handle, container,
                    "articulation closing fraction decreases under measured fingertip contact",
                    "all finger contact is lost or closing fraction stops decreasing", pid)
                add(f"release:articulation_feature:{hand}","release",hand,handle,container,
                    "hand clears the articulated feature after closing","contact persists",pid,opening=1.0)
            for hand in permitted_hands(payload):
                add(f"insert:approach:{payload}:{hand}", "approach", hand, payload, None,
                    "finger center approaches object", "distance margin stalls", pid)
                add(f"insert:contact:{payload}:{hand}", "contact", hand, payload, None,
                    "fingers touch object", "contact stays absent", pid, opening=0.25)
                add(f"insert:grasp:{payload}:{hand}","grasp",hand,payload,None,
                    "bilateral object contact persists for 0.1s","contact is lost",pid)
                add(f"insert:lift:{payload}:{hand}", "lift", hand, payload, container,
                    "object clears the measured container rim before insertion", "contact is lost or lift stalls", pid,
                    offset=(0.0,0.0,(objects[container]['size_m'][2]+objects[payload]['size_m'][2])/2+.008), opening=0.0)
                add(f"insert:target_relation:{payload}:{hand}", "transport", hand, payload, container,
                    "object enters container", "inside margin stalls or grasp is lost",
                    next((p["id"] for p in contract["predicates"]
                          if p["kind"] == "inside" and p["object"] == payload), pid),
                    (0.0, 0.0, -objects[container]['size_m'][2]/2+
                     objects[container].get('wall_m',.006)+
                     objects[payload]['size_m'][2]/2+.002), 0.0)
        elif kind == "released":
            targets = pred.get("objects", [pred.get("object")])
            for obj in targets:
                if obj not in objects or objects[obj].get('role') == 'container':
                    continue
                hand = next((p["hand"] for p in contract["predicates"]
                             if p["kind"] == "lift_with_contact" and p["object"] == obj),
                            primary)
                add(f"release:{obj}", "release", hand, obj, None,
                    "finger contact ends after object support", "object falls or support is lost",
                    pid)
    # A relational goal without an explicit lift predicate still needs a
    # physical pickup path. Compile the existing primitives from object roles,
    # then expose every contract-permitted hand. The RuleAgent chooses from
    # measured reach/contact; no task or case name supplies the choice.
    for obj in sorted(derive_pickup_objects(contract)):
        if obj not in objects or any(e.action.event == 'approach' and
                                    e.action.object_id == obj for e in result):
            continue
        seed = next((e.action.hand for e in result if e.action.object_id == obj
                     and e.action.hand in manipulators), primary)
        add(f'approach:{obj}', 'approach', seed, obj, None,
            'finger center approaches task object', 'distance or joint margin stalls',
            'relation_pickup')
        add(f'contact:{obj}', 'contact', seed, obj, None,
            'both finger pads contact task object', 'contact stays absent',
            'relation_pickup', opening=.25)
        add(f'grasp:{obj}', 'grasp', seed, obj, None,
            'bilateral contact persists for 0.1s', 'contact is lost',
            'relation_pickup')
        add(f'lift:{obj}', 'lift', seed, obj, None,
            'task object rises while bilateral contact persists',
            'contact is lost or lift stalls', 'relation_pickup', opening=0.)
    alternate_hands = []
    for event in result:
        action = event.action
        if not other_hand_allowed(action, contract, manipulators):
            continue
        for alternate in manipulators:
            if alternate == action.hand:
                continue
            alternate_hands.append(replace(
                event, event_id=event.event_id + ':alternate:' + alternate,
                action=replace(action, hand=alternate),
                evidence=event.evidence + '; contract-permitted alternate manipulator'))
    result.extend(alternate_hands)
    # A common reactive recovery primitive for any securely carried object.
    # It is only physically available while that hand also contacts another
    # scene object. Which contact is an obstruction is decided from live task
    # evidence by the runtime agents, not by a scene-name repair recipe.
    carried = {(e.action.object_id, e.action.hand): e.source_predicate
               for e in result if e.action.event in ('lift', 'transport', 'align')}
    for (obj, hand), predicate_id in sorted(carried.items()):
        for mode in ('away', 'up', 'tangent_a', 'tangent_b'):
            add(f'clearance:{obj}:{hand}:{mode}', 'clearance', hand, obj, None,
                'non-carried gripper contact ends while bilateral object contact remains',
                'carried-object contact is lost or other contact persists',
                predicate_id, opening=0., recovery_mode=mode)
    # A contact with an environment fixture can also obstruct an empty hand
    # before grasp. Offer the same measured-normal directions as a distinct
    # short retreat; the physical precondition and immutable guard require
    # real non-target contact and reject this mode once grasp is secure.
    reachable = {(e.action.object_id, e.action.hand): e.source_predicate
                 for e in result if e.action.event in ('approach', 'contact')}
    for (obj, hand), predicate_id in sorted(reachable.items()):
        for mode in ('away', 'up', 'tangent_a', 'tangent_b'):
            add(f'retreat:{obj}:{hand}:{mode}', 'retreat', hand, obj, None,
                'non-target gripper contact ends before secure grasp',
                'contact persists or desired grasp becomes secure',
                predicate_id, opening=1., recovery_mode=mode)
    add("verify:stable", "hold", primary, None, None,
        "native relation, contact, release and stability windows remain satisfied",
        "protected relation is lost", "stable_verify")
    # Geometry-derived alternative contact regions let the policy change its
    # grasp strategy while the same physics session is paused.
    alternatives = []
    for event in result:
        if event.action.event not in ("approach", "contact"):
            continue
        size = objects[event.action.object_id]["size_m"]
        for label, sign in (("side_a", 1.0), ("side_b", -1.0)):
            offset = np.asarray(event.action.target_offset_m, dtype=float)
            offset[0] += sign * min(0.025, float(size[0]) * 0.2)
            alternatives.append(replace(event,
                event_id=event.event_id + ":" + label,
                action=replace(event.action, target_offset_m=tuple(offset)),
                evidence=event.evidence + "; contact region " + label))
    result.extend(alternatives)
    upper_contacts = []
    for event in result:
        if event.action.event != 'contact' or event.event_id.endswith((':side_a', ':side_b')):
            continue
        size = objects[event.action.object_id]['size_m']
        offset = np.asarray(event.action.target_offset_m, dtype=float)
        offset[2] += min(.025, .45 * float(size[2]))
        upper_contacts.append(replace(
            event, event_id=event.event_id + ':upper_surface',
            action=replace(event.action, target_offset_m=tuple(float(v) for v in offset)),
            evidence=event.evidence + '; geometry-scaled higher contact surface',
            abandon_if='bilateral contact absent or hard margin worsens'))
    result.extend(upper_contacts)
    # Generic object-relative route option for any graspable target,
    # including articulated features and single-manipulator adapters. The
    # RuleAgent chooses whether the measured geometry warrants this route.
    overhead = []
    for event in result:
        if (event.action.event != 'approach' or
                event.event_id.endswith((':side_a', ':side_b'))):
            continue
        overhead.append(replace(
            event, event_id=event.event_id + ':overhead',
            action=replace(event.action, recovery_mode='overhead'),
            evidence='open gripper stages above measured object geometry, '
                     'then approaches the contact region',
            abandon_if='non-target contact, joint limit, or route progress stalls'))
        # The usual center endpoint may bring an open gripper into a nearby
        # support surface. Expose one geometry-scaled higher endpoint; this is
        # a candidate for any rigid pickup object, never a scene-specific rule.
        size = objects[event.action.object_id]['size_m']
        elevated = np.asarray(event.action.target_offset_m, dtype=float)
        elevated[2] += min(.025, max(.010, float(size[2]) * .35))
        overhead.append(replace(
            event, event_id=event.event_id + ':overhead_high',
            action=replace(event.action, recovery_mode='overhead',
                           target_offset_m=tuple(float(v) for v in elevated)),
            evidence='open gripper stages above measured object geometry and '
                     'approaches a higher contact region when nearby support '
                     'contact blocks the center route',
            abandon_if='target contact remains absent, non-target contact, '
                       'joint limit, or route progress stalls'))
    result.extend(overhead)
    # A payload can lie across the fixture that parents an articulated
    # container. Offer a geometry-derived lateral staging event. It is an
    # option for RuleAgent, including on a single-manipulator adapter; no
    # case name or precomputed joint path selects it automatically.
    corridor_fixture = {}
    for predicate in contract['predicates']:
        if predicate['kind'] != 'open_insert_close':
            continue
        container = objects[predicate['container']]
        fixture = container.get('parent_id')
        if (fixture in objects and
                objects[fixture].get('mobility') == 'static' and
                objects[fixture].get('size_m')):
            corridor_fixture[(predicate['object'], predicate['id'])] = fixture
    corridor = []
    for event in result:
        action = event.action
        fixture = corridor_fixture.get((action.object_id, event.source_predicate))
        if (fixture is None or action.event != 'approach' or
                ':alternate:' in event.event_id or
                action.target_id is not None or action.recovery_mode is not None or
                np.max(np.abs(action.target_offset_m)) > 1e-9):
            continue
        corridor.append(replace(
            event, event_id=event.event_id + ':side_corridor',
            action=replace(action, target_id=fixture,
                           recovery_mode='side_corridor'),
            evidence=event.evidence + '; stages outside the measured fixture side',
            abandon_if='fixture contact, joint limit, or side-stage progress stalls'))
    result.extend(corridor)
    # A carried object may need to pass in front of an articulated container
    # before moving laterally across its opening. These are optional measured
    # waypoints for every contract-ordered container insertion, not case rules.
    front_routes = []
    ordered_insertions = {
        (p['object'], p['container']) for p in contract['predicates']
        if p['kind'] == 'open_insert_close' and
        objects[p['container']].get('size_m') and
        abs(float(articulation_axis_world(objects[p['container']])[2])) <= .5}
    for event in result:
        action = event.action
        if (action.event != 'transport' or
                (action.object_id, action.target_id) not in ordered_insertions or
                action.recovery_mode is not None or
                ':alternate:' in event.event_id):
            continue
        for mode in ('front_clearance', 'front_align'):
            front_routes.append(replace(
                event, event_id=event.event_id + ':' + mode,
                action=replace(action, recovery_mode=mode),
                evidence=('held object reaches the measured container front '
                          'clearance' if mode == 'front_clearance' else
                          'held object aligns laterally at the measured front '
                          'clearance'),
                abandon_if='loaded contact, articulation reserve or hard margin is lost'))
            if mode == 'front_clearance' and len(manipulators) > 1:
                joint = objects[action.target_id].get('joint') or {}
                limits = joint.get('range')
                if limits is not None:
                    travel = abs(float(limits[1]) - float(limits[0]))
                    reserve = min(.012, max(.004, .03 * travel))
                    front_routes.append(replace(
                        event,
                        event_id=event.event_id + ':front_clearance:support_reserve',
                        action=replace(action, recovery_mode=mode,
                                       support_pull_m=reserve),
                        evidence=('held object reaches the measured container front '
                                  'while the other manipulator maintains a bounded '
                                  'articulation reserve'),
                        abandon_if='support feature contact, articulation reserve or '
                                   'hard margin is lost'))
    result.extend(front_routes)
    # Closing an articulated slide may require slower contact-loaded motion
    # after the measured feature contact repeatedly breaks. Keep both
    # speed envelopes available; the agents choose from physical feedback.
    slower = []
    for event in result:
        if event.action.event not in ('close_drawer', 'push_close'):
            continue
        slower.append(replace(
            event, event_id=event.event_id + ':guarded_slow',
            action=replace(event.action, max_speed_m_s=.025),
            evidence=event.evidence + '; bounded slower contact-loaded motion',
            abandon_if='target contact is lost or articulation fraction stalls'))
    result.extend(slower)
    # A dependent transition can begin just after an opening threshold is
    # reached, while passive motion reduces that margin. Expose one optional
    # reserve target for any contract-ordered articulated opening. The agents
    # decide whether current measured drift warrants it.
    buffered = []
    for event in result:
        if event.action.event != 'open_drawer':
            continue
        buffered.append(replace(
            event, event_id=event.event_id + ':buffered',
            evidence=event.evidence + '; measured reserve above the contract threshold',
            abandon_if='feature contact is lost, joint limit, or opening stalls'))
    result.extend(buffered)
    # Keep the inactive manipulator clear of any moving articulated object.
    # The direction comes from the live joint axis; this is available to the
    # RuleAgent for every articulated task and never selected by a scene name.
    if len(manipulators) > 1:
        clearance = []
        for event in result:
            if event.action.event not in ('open_drawer', 'close_drawer') or event.action.bystander_clearance_m is not None:
                continue
            joint = objects[event.action.target_id].get('joint', {})
            limits = joint.get('range_m', joint.get('range'))
            travel = abs(float(limits[1])-float(limits[0])) if limits is not None else .2
            distance = min(.25, max(.08, travel + .02))
            clearance.append(replace(
                event, event_id=event.event_id + ':bystander_clearance',
                action=replace(event.action, bystander_clearance_m=distance,
                               max_speed_m_s=min(event.action.max_speed_m_s, .025)),
                evidence=event.evidence + '; inactive manipulator clears the moving object',
                abandon_if='hard margin worsens, loaded contact is lost, or articulation stalls'))
        result.extend(clearance)
    seen = set()
    unique = []
    for event in result:
        if event.event_id not in seen:
            seen.add(event.event_id)
            unique.append(event)
    return unique


def signed_margin(event: EventDefinition, obs: dict, initial: dict,
                  contract: dict) -> float:
    """Positive means the target event has been measured."""
    action = event.action
    kind, obj, hand = action.event, action.object_id, action.hand
    positions = obs["object_position_m"]
    feature = articulated_feature(obj, obs.get('object_geometry', {})) if obj else False
    if kind == "reference":
        return -1.0
    if kind == "hold":
        rows = list(obs.get("live_predicates", {}).values())
        return 1.0 if rows and all(r["passed"] for r in rows) else -1.0
    if kind == 'reposition_base':
        if action.base_joint_target is None:
            raise ValueError('Base event has no certified target')
        measured = np.asarray(obs['base_joint_qpos'], dtype=float)
        target = np.asarray(action.base_joint_target, dtype=float)
        if measured.shape != (3,) or target.shape != (3,):
            raise ValueError('Base event state has wrong dimensions')
        error = np.abs(measured - target)
        tolerance = np.array([.004, .004, .01])
        if np.all(error <= tolerance):
            return float(np.min(tolerance - error))
        # Stay negative until every axis reaches its actual controller
        # tolerance, but increase as each sequential axis makes progress.
        return -float(error[0] + error[1] + .10 * error[2])
    if kind in ('clearance', 'retreat'):
        extra = [c for c in obs.get('gripper_scene_contacts', {}).get(hand, [])
                 if c['object_id'] != obj]
        start = obs.get('recovery_start_m')
        target = obs.get('recovery_target_m')
        if start is None or target is None:
            movement_margin = -1.0
        else:
            direction = np.asarray(target) - np.asarray(start)
            direction /= max(float(np.linalg.norm(direction)), 1e-12)
            displacement = float(np.dot(
                np.asarray(obs['finger_center_m'][hand]) - np.asarray(start),
                direction))
            movement_margin = displacement - .015
        if kind == 'retreat':
            target_pad_contact = any(
                c['object_id'] == obj for c in
                obs.get('gripper_scene_contacts', {}).get(hand, []))
            return min(movement_margin,
                       1.0 if not extra and not target_pad_contact else -1.0)
        return min(movement_margin, 1.0 if not extra and
                   all(obs['finger_contact'][obj][hand]) else -1.0)
    if kind in ("approach", "contact"):
        # Cup pickup is the centered overhead skill executed by the robot.
        # Use the same actual target for progress, including generated side
        # and upper-surface offers; the native task judge stays unchanged.
        centered_cup=obs.get('object_geometry',{}).get(obj,{}).get('shape')=='cup'
        if centered_cup:
            action=replace(action,target_offset_m=(0.,0.,0.),recovery_mode='overhead')
        offset = np.asarray(action.target_offset_m)
        distance = (float(overhead_approach(action, obs)[1])
                    if kind == 'approach' and action.recovery_mode == 'overhead'
                    else float(side_corridor_approach(action, obs)[1])
                    if kind == 'approach' and action.recovery_mode == 'side_corridor'
                    else float(np.linalg.norm(obs["finger_center_m"][hand] -
                                              (positions[obj] + offset))))
        rotation=obs['hand_rotation_world'][hand]
        rotation_error=float(np.arccos(np.clip((np.trace(desired_rotation(obj,obs).T@rotation)-1)/2,-1.,1.)))
        position_margin=.008-distance
        if kind == "approach":
            if action.recovery_mode == 'side_corridor':
                return position_margin
            return min(position_margin,.15-rotation_error)
        pads = obs["finger_contact"][obj][hand]
        age=obs.get('bilateral_contact_seconds',{}).get(obj,{}).get(hand,0.)
        if feature and not all(pads):
            size=np.asarray(obs['object_geometry'][obj]['size_m'])
            jaw=rotation[:,0]
            width=float(np.sum(size*np.abs(obs['object_rotation_world'][obj].T@jaw)))
            remaining=max(0.,float(obs['gripper_opening_m'][hand])-width)
            return min(position_margin,.15-rotation_error,-remaining-1e-6)
        return (min(position_margin,.15-rotation_error,age-.10) if all(pads)
                else -distance-.25*float(obs['gripper_opening_m'][hand]))
    if kind == "grasp":
        pads = obs["finger_contact"][obj][hand]
        age=obs.get('bilateral_contact_seconds',{}).get(obj,{}).get(hand,0.)
        if obs.get('object_geometry',{}).get(obj,{}).get('shape')=='cup' and not all(pads):
            center_error=float(np.linalg.norm(obs['finger_center_m'][hand]-positions[obj]))
            rotation=obs['hand_rotation_world'][hand]
            angle=float(np.arccos(np.clip((np.trace(desired_rotation(obj,obs).T@rotation)-1)/2,-1.,1.)))
            # A progressing centering or bounded jaw closure must remain
            # distinguishable from an empty, motionless grasp attempt.
            return -center_error-.02*angle-.25*float(obs['gripper_opening_m'][hand])
        return age-.10 if all(pads) else -1.0
    if kind == "lift":
        height = next((float(p["height_m"]) for p in contract["predicates"]
                       if p["kind"] == "lift_with_contact" and p["object"] == obj),
                      0.025)
        pads = obs["finger_contact"][obj][hand]
        lift = float(positions[obj][2] - initial["object_position_m"][obj][2])
        if action.target_id is not None:
            margin=float(positions[obj][2]-positions[action.target_id][2]-action.target_offset_m[2])
        else:margin=lift-height
        return min(margin,0.0 if all(pads) else -height)
    if kind in ("transport", "align"):
        if kind == 'transport' and action.recovery_mode in ('front_clearance',
                                                            'front_align'):
            return .008 - float(front_staging_transport(action, obs)[1])
        live = obs.get("live_predicates", {}).get(event.source_predicate)
        if live is not None and live["signed_margin"] is not None:
            goalz=positions[action.target_id][2]+action.target_offset_m[2]
            height_margin=.006-abs(float(positions[obj][2]-goalz))
            goalxy=positions[action.target_id][:2]+np.asarray(action.target_offset_m)[:2]
            source=next((p for p in contract['predicates']
                         if p['id']==event.source_predicate),None)
            radial=source.get('radial_tolerance_m') if source is not None else None
            if radial is not None:
                radial=float(radial)
                if not np.isfinite(radial) or radial<=0:
                    raise ValueError('Invalid native radial tolerance')
            xy_tolerance=min(.008,radial/2) if radial is not None else .008
            xy_margin=xy_tolerance-float(np.linalg.norm(positions[obj][:2]-goalxy))
            margin=min(float(live['signed_margin']),height_margin,xy_margin)
            return margin if live['window_complete'] else min(-1e-9,margin)
        target = positions[action.target_id]
        planar = float(np.linalg.norm((positions[obj] - target)[:2]))
        return 0.04 - planar
    if kind == "release":
        pads = obs["finger_contact"][obj][hand]
        supported = bool(obs["support_contact"].get(obj)) or feature
        clearance=float(np.linalg.norm(obs['finger_center_m'][hand]-positions[obj])) if feature else float(obs['finger_center_m'][hand][2]-positions[obj][2])
        if feature:margin=clearance-.07
        else:margin=min(clearance-.10,float(np.linalg.norm((obs['finger_center_m'][hand]-positions[obj])[:2]))-.17)
        return min(margin,0. if not obs['hand_contact'][obj][hand] and supported else -.01)
    if kind in ("open_drawer", "close_drawer", "push_close"):
        geometry = obs.get('object_geometry', {}).get(action.target_id, {})
        joint = geometry.get('joint', {}).get('name')
        if joint not in obs['articulation_fraction']:
            raise ValueError('Articulation observation missing for target')
        fraction = obs['articulation_fraction'][joint]
        operation = next((row for row in contract['predicates']
                          if row['kind'] == 'open_insert_close' and
                          row['container'] == action.target_id and
                          row['joint'] == joint), None)
        if operation is None:
            raise ValueError('Articulation event lacks frozen task threshold')
        required_open = float(operation['open_fraction'])
        target_open = (buffered_open_fraction(required_open)
                       if event.event_id.split(':rule:', 1)[0].endswith(':buffered')
                       else required_open)
        return (fraction - target_open if kind == 'open_drawer'
                else float(operation['closed_fraction']) - fraction)
    raise ValueError("No measured margin for event")


def event_trigger(event: EventDefinition, initial: dict, contract: dict):
    def check(obs: dict) -> str | None:
        hard = obs.get('immutable_hard', {}).get('hard_constraints', {})
        if not (hard.get('waist_coupling_within_1mrad') and
                hard.get('limited_joints_within_2mrad_or_2mm')):
            return 'posture_regression'
        if obs["max_robot_table_penetration_m"] > 0.003 or \
                obs["min_joint_margin_native"] < -0.002:
            return "hard_regression"
        if event.action.event == "hold":
            return None
        if math.isfinite(signed_margin(event, obs, initial, contract)) and \
                signed_margin(event, obs, initial, contract) >= 0:
            return "event_reached"
        return None
    return check
