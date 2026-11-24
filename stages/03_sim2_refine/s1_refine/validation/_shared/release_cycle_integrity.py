"""Read-only, full-episode close cycles and original-policy release checks.

Unlike the original screening summary, opening detection does not depend on the
event name and counting continues after the first release.  Intentional release
uses the frozen native guard verbatim; its geometry-ready path is legitimate.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import numpy as np


def acquisition_retention(time_s, forces, local, diagnostics):
    """Keep the first stable loaded grasp, including the pre-25mm lift phase."""
    t = time_s; dt = float(np.median(np.diff(t))); rows = []
    for hi, (hand, obj) in enumerate(zip(('left', 'right'), ('cup_left_base', 'cup_right_upper'))):
        intended = (diagnostics['event_object'] == obj) & (diagnostics['event_hand'] == hand)
        acquisition = np.flatnonzero(intended & np.isin(diagnostics['event_kind'], ['contact', 'grasp']))
        if not len(acquisition):
            rows.append({'hand': hand, 'passed': False, 'reason': 'no_acquisition'}); continue
        first = int(acquisition[0])
        releases = np.flatnonzero(intended & (diagnostics['event_kind'] == 'release') & (np.arange(len(t)) >= first))
        end = int(releases[0]) if len(releases) else len(t)
        loaded = (forces[:, hi] > .0001).all(axis=-1)
        def spans(mask):
            edges = np.diff(np.r_[False, mask, False].astype(np.int8))
            return [(int(a), int(b), float(t[b - 1] - t[a] + dt))
                    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]
        scope = (np.arange(len(t)) >= first) & (np.arange(len(t)) < end)
        qualified = next((a for a, b, duration in spans(loaded & scope) if duration >= .1 - 1e-6), None)
        losses = []; motion = None
        if qualified is not None:
            retain = (np.arange(len(t)) >= qualified) & (np.arange(len(t)) < end)
            losses = [{'start_s': float(t[a]), 'end_s': float(t[b - 1] + dt), 'duration_s': duration}
                      for a, b, duration in spans((~loaded) & retain) if duration > .1 + 1e-6]
            motion = float(np.max(np.linalg.norm(local[qualified:end, hi] - local[qualified, hi], axis=1)))
        rows.append({'hand': hand, 'object': obj,
                     'first_stable_loaded_acquisition_s': float(t[qualified]) if qualified is not None else None,
                     'contact_loss_after_first_loaded_acquisition': losses,
                     'max_object_motion_from_first_loaded_acquisition_m': motion,
                     'passed': qualified is not None and not losses and motion <= .015})
    return {'schema': 'rssr.first_loaded_acquisition_retention.v1',
            'scope': 'first 100ms loaded bilateral acquisition until first legal release, including before the 25mm lift threshold',
            'maximum_contact_gap_s': .1, 'maximum_object_motion_in_gripper_m': .015,
            'hands': rows, 'passed': all(r['passed'] for r in rows)}


def evaluate(case, native, diagnostics, implementation):
    case = Path(case)
    sys.path.insert(0, str(implementation))
    from rssr_augment.retarget.events import catalog
    from rssr_augment.retarget.immutable_rule_guard import guard

    receipt = json.loads((case / 'receipt.json').read_text())
    contract = json.loads((case / 'execute/task_contract.json').read_text())
    packets = [json.loads(p.read_text()) for p in sorted((case / 'decisions').glob('*/packet.json'))]
    events = catalog(contract, {'objects': packets[0]['scene_objects']})
    t = native['time_s']
    kind, obj, hand = [diagnostics[k] for k in ('event_kind', 'event_object', 'event_hand')]
    angle, aperture = diagnostics['desired_gripper_angle_rad'], native['gripper_opening']
    rows = []
    for hi, (h, o) in enumerate(zip(('left', 'right'), ('cup_left_base', 'cup_right_upper'))):
        intent = (obj == o) & (hand == h)
        acquisition = intent & np.isin(kind, ['contact', 'grasp'])
        starts = np.flatnonzero(acquisition)
        if not len(starts):
            rows.append({'hand': h, 'object': o, 'passed': False, 'reason': 'no_acquisition'})
            continue
        first = int(starts[0])
        releases = np.flatnonzero(intent & (kind == 'release') & (np.arange(len(t)) >= first))
        release = int(releases[0]) if len(releases) else None

        # First pre-close centering may open an initially half-closed SDK hand.
        # A meaningful close starts only after a positive 0.06rad command rise.
        state = 'waiting'; trough = float(angle[first, hi]); peak = trough
        minimum_aperture = float(aperture[first, hi]); closes = []; opens = []
        for i in range(first + 1, len(t)):
            a, gap = float(angle[i, hi]), float(aperture[i, hi])
            if state == 'waiting':
                trough = min(trough, a)
                if a - trough >= .06 - 1e-12:
                    closes.append(i); state = 'closing'; peak = a; minimum_aperture = gap
            else:
                peak = max(peak, a); minimum_aperture = min(minimum_aperture, gap)
                if peak - a >= .12 - 1e-12 and gap - minimum_aperture >= .008 - 1e-12:
                    opens.append(i); state = 'waiting'; trough = a
        unplanned = [i for i in opens if release is None or i < release]
        later_acquisition = np.flatnonzero(acquisition & (np.arange(len(t)) >= (release if release is not None else len(t))))

        release_evidence = {'present': release is not None, 'original_guard_allowed': False}
        support_ok = False
        if release is not None:
            # The history rows record action end times.  Find the exact action
            # start boundary, then its saved native decision observation.
            history = receipt['event_history']
            ri = next(i for i, e in enumerate(history)
                      if e['event_id'].split(':')[0] == 'release'
                      and e.get('selected_object_id') == o and e.get('selected_hand') == h)
            release_start = float(history[ri - 1]['time_s']) if ri else 0.
            packet = min(packets, key=lambda p: abs(p['observation']['time_s'] - release_start))
            packet_gap = abs(packet['observation']['time_s'] - release_start)
            event = next(e for e in events if e.action.event == 'release' and e.action.object_id == o and e.action.hand == h)
            packet_allowed, packet_reasons = guard(event, packet['observation'], contract, events)
            # Packets are presentation records rounded to five decimal places.
            # Rechecking a 6mm physical guard on them can reject 5.997104mm as
            # 6.000000000000005mm.  The preceding native history row is the
            # actual same action boundary, with unrounded positions/predicates.
            # Keep the frozen guard and its thresholds completely unchanged.
            boundary = history[ri - 1] if ri else None
            trusted = copy.deepcopy(packet['observation'])
            boundary_valid = bool(boundary and all(k in boundary for k in
                                   ('object_position_m', 'live_predicates', 'immutable_hard')))
            if boundary_valid:
                raw_time = float(boundary['immutable_hard']['time_s'])
                boundary_valid = (abs(raw_time - release_start) <= 1e-4 and
                                  abs(raw_time - packet['observation']['time_s']) <= 1e-4 and
                                  all(np.array_equal(np.round(np.asarray(v), 5),
                                      np.asarray(packet['observation']['object_position_m'][k]))
                                      for k, v in boundary['object_position_m'].items()))
                for key in ('object_position_m', 'live_predicates', 'finger_contact',
                            'hand_contact', 'gripper_opening_m', 'gripper_scene_contacts',
                            'immutable_hard'):
                    if key in boundary: trusted[key] = copy.deepcopy(boundary[key])
                trusted['time_s'] = raw_time
            allowed, reasons = guard(event, trusted, contract, events)
            if not boundary_valid:
                reasons = list(reasons) + ['same_boundary_unrounded_native_evidence_missing_or_mismatched']
                allowed = False
            release_evidence.update(time_s=release_start, packet_time_s=packet['observation']['time_s'],
                                    packet_time_error_s=packet_gap, original_guard_allowed=bool(allowed and packet_gap <= 1e-4),
                                    original_guard_reasons=reasons,
                                    guard_input_precision='Unrounded native history action boundary; no epsilon or physical threshold change',
                                    native_history_row_index=ri - 1 if ri else None,
                                    native_history_row_sha256=hashlib.sha256(json.dumps(boundary,sort_keys=True,separators=(',',':')).encode()).hexdigest() if boundary else None,
                                    native_history_exact_time_s=trusted['time_s'],
                                    native_boundary_valid=boundary_valid,
                                    rounded_packet_guard_allowed=packet_allowed,
                                    rounded_packet_guard_reasons=packet_reasons,
                                    support_at_release=packet['observation']['support_contact'][o])

            # Original stack contract: base must settle on worktable; upper on
            # base. contacts[..., -1] is the runtime worktable column.  The
            # unchanged native final task judge still supplies relation,
            # contact-fraction, released and full 1s stability requirements.
            supported = (native['contacts'][:, hi, -1] > 0 if hi == 0
                         else native['object_contacts'][:, hi, 0] > 0)
            stable = next(p for p in contract['predicates'] if p['kind'] == 'stable')
            window = float(stable['window_s']); final_window = t >= t[-1] - window - 1e-8
            motion = float(np.linalg.norm(np.ptp(native['object_position'][final_window, hi], axis=0)))
            support_ok = bool(np.all(supported[final_window]) and motion <= float(stable['max_motion_m']) + 1e-12)
            supported_after = np.flatnonzero(supported & (np.arange(len(t)) >= release))
            release_evidence.update(expected_support='worktable' if hi == 0 else 'cup_left_base',
                                    first_post_release_support_s=float(t[supported_after[0]]) if len(supported_after) else None,
                                    final_support_continuous=bool(np.all(supported[final_window])),
                                    original_stability_window_s=window, final_window_motion_m=motion,
                                    original_maximum_motion_m=float(stable['max_motion_m']),
                                    supported_and_stable_at_end=support_ok)
        checks = {
            'one_meaningful_close_in_full_episode': len(closes) == 1,
            'no_unplanned_physical_reopens_in_any_event': not unplanned,
            'no_acquisition_after_first_release': not len(later_acquisition),
            'first_release_allowed_by_original_native_guard': release_evidence['original_guard_allowed'],
            'final_expected_support_and_original_stability': support_ok,
        }
        rows.append({'hand': h, 'object': o, 'checks': checks, 'passed': all(checks.values()),
                     'meaningful_close_s': [float(t[i]) for i in closes],
                     'actual_reopen_s': [float(t[i]) for i in opens],
                     'unplanned_actual_reopen_s': [float(t[i]) for i in unplanned],
                     'post_release_acquisition_samples': len(later_acquisition),
                     'release': release_evidence})
    return {'schema': 'rssr.full_episode_grasp_release_integrity.v2',
            'criteria': {'close_command_rise_rad': .06, 'reopen_command_fall_rad': .12,
                         'reopen_actual_aperture_rise_m': .008,
                         'cycle_window': 'first target acquisition to complete native episode, all event names',
                         'release_policy': 'unmodified frozen immutable_rule_guard.guard; support OR exact frozen goal geometry'},
            'hands': rows, 'passed': all(row['passed'] for row in rows)}
