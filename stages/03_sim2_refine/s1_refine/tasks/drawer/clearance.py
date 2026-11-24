"""Read-only every-step 20mm cross-side geometry proof for a drawer run."""
from pathlib import Path
import hashlib
import json
import numpy as np
from validation.clearance_cross_side import CrossSideNativeGate
from validation.clearance_gate import REQUIRED_CATEGORIES

class ClearanceRecorder:
    def __init__(self, required_m=.020):
        if required_m < .020:
            raise ValueError('Cross-side clearance may not be reduced below 20mm')
        self.required_m = float(required_m)
        self.rows = []
        self.sampled_minima = {}
        self.interval_minima = {}
        self.manifest = None
        self.gate = None
        self.previous = None

    def snapshotter(self, native_snapshotter):
        recorder = self
        class Snapshotter:
            def __init__(self, model):
                self.model = model
                self.native = native_snapshotter(model)
                self.gate = CrossSideNativeGate(model, recorder.required_m)
                recorder.gate = self.gate
                recorder.manifest = self.gate.geometry_manifest()
            def __call__(self, data):
                original = self.native(data)
                time_s = float(data.time)
                if recorder.rows and time_s == recorder.rows[-1][0]:
                    original['drawer_cross_side_clearance'] = recorder.rows[-1][4]
                    return original
                if recorder.rows and time_s <= recorder.rows[-1][0]:
                    raise ValueError('Nonmonotone actual native geometry clock')
                q = np.asarray(data.qpos, dtype=float).copy()
                point = self.gate.at_qpos(q)
                lower = np.full(len(REQUIRED_CATEGORIES), np.nan)
                if recorder.previous is not None:
                    previous_t, previous_q, previous_point = recorder.previous
                    margins = self.gate.interval_motion_margin(previous_q, q)
                    for i, category in enumerate(REQUIRED_CATEGORIES):
                        bound = min(previous_point['minima'][category]['distance_m'],
                                    point['minima'][category]['distance_m'])-margins[category]
                        lower[i] = bound
                        detail = {'lower_bound_m': float(bound), 'motion_margin_m': float(margins[category]),
                                  'start_s': previous_t, 'end_s': time_s}
                        if category not in recorder.interval_minima or bound < recorder.interval_minima[category]['lower_bound_m']:
                            recorder.interval_minima[category] = detail
                for category in REQUIRED_CATEGORIES:
                    value = dict(point['minima'][category], time_s=time_s)
                    if category not in recorder.sampled_minima or value['distance_m'] < recorder.sampled_minima[category]['distance_m']:
                        recorder.sampled_minima[category] = value
                sampled_passed = all(point['minima'][c]['distance_m'] >= recorder.required_m for c in REQUIRED_CATEGORIES)
                packet = {'sampled_passed': bool(sampled_passed), 'required_m': recorder.required_m,
                          'minima_m': {c: float(point['minima'][c]['distance_m']) for c in REQUIRED_CATEGORIES}}
                original['drawer_cross_side_clearance'] = packet
                recorder.rows.append((time_s, q, point, lower, packet))
                recorder.previous = (time_s, q, point)
                return original
        return Snapshotter

    def finish(self, output, expected_hard_times, expected_dt):
        output = Path(output)
        trace = output/'drawer_every_step_clearance.npz'
        times = np.asarray([r[0] for r in self.rows])
        sampled = np.asarray([[r[2]['minima'][c]['distance_m'] for c in REQUIRED_CATEGORIES] for r in self.rows])
        lower = np.stack([r[3] for r in self.rows])
        np.savez_compressed(trace, time_s=times, qpos=np.stack([r[1] for r in self.rows]),
                            required_categories=np.asarray(REQUIRED_CATEGORIES),
                            distance_lower_bound_m=sampled, interval_lower_bound_m=lower)
        checks = {'every_step_actual_native_clock_matches_hard_trace': bool(np.array_equal(times, expected_hard_times)),
                  'every_native_step_present': len(times)>1 and bool(np.allclose(np.diff(times),expected_dt,rtol=0,atol=1e-8)),
                  'sampled_cross_side_clearance_ge_20mm': bool(np.isfinite(sampled).all() and (sampled>=self.required_m).all()),
                  'defined_interpolated_path_clearance_ge_20mm': len(lower)>1 and bool(np.isfinite(lower[1:]).all() and (lower[1:]>=self.required_m).all())}
        result = {'schema':'rssr.sim2_refine.drawer.every_step_clearance.v1',
                  'accepted':all(checks.values()),'checks':checks,'required_clearance_m':self.required_m,
                  'samples':len(times),'sampled_minima':self.sampled_minima,'interval_minima':self.interval_minima,
                  'manifest':self.manifest,'trace':str(trace),
                  'trace_sha256':hashlib.sha256(trace.read_bytes()).hexdigest(),
                  'continuous_path_definition':'Piecewise scalar-joint-linear interpolation between every actual 0.5ms native qpos sample, bounded conservatively by geometric motion margins.',
                  'limitations':['Simulated geometry only; no claim about physical calibration or tracking error.',
                                 'Wrist-camera envelope remains the inherited assumed 90x40x40mm box; user threshold applies to opposing arms and grippers.',
                                 'Table is not subject to the 20mm clearance threshold; unchanged native penetration gate applies.']}
        (output/'clearance_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
