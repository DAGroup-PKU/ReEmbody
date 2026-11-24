"""Package admitted + rendered centred-layout drawer episodes as LeRobot v2.1
(v9_lean30: controller center_layout_v9_lean30, batch/render case dirs named
<case_id>__clean; schema identical to package_lerobot_v1 / the drawer template).

Schema (parquet columns, features, 450 frames at 30 Hz, 32-d measured state,
22-d absolute action, three 224x224 videos with 16:9 content + black bars) is
taken from the previous drawer release
result/lerobot/drawer_v30_v32partial_v34cutover_200_20261029_attempt_09_multirescue/draw.
Every episode is re-checked against its native batch result (task contract,
every-step hard constraints, 2 cm arm clearance, shell, alias, exact replay,
quality v7 + head visibility, grasp v3) and its render receipt before staging;
the official LeRobot reader then reads every episode before the atomic rename.
No upload.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[3]
TEMPLATE = PROJECT / 'result/lerobot/drawer_v30_v32partial_v34cutover_200_20261029_attempt_09_multirescue/draw'
TASK = 'Put_the_block_in_the_drawer.'
CAMERAS = {'head_rgbd': 'observation.images.cam_high', 'left_wrist_rgbd': 'observation.images.cam_left_wrist',
           'right_wrist_rgbd': 'observation.images.cam_right_wrist'}
SPEC = PROJECT / 'result/sim2-refine/drawer_center_layout_spec_20261101_attempt_01'


def require(c, m):
    if not c:
        raise ValueError(m)


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = sha256()
    with Path(p).open('rb') as s:
        for b in iter(lambda: s.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def jsonl(p, rows):
    Path(p).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {k: getattr(values, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')} | {'count': [len(values)]}


def aggregate_statistics(records):
    result = {}
    for key in records[0]['stats']:
        stats = [r['stats'][key] for r in records]
        counts = np.array([s['count'][0] for s in stats], dtype=np.float64)
        means = np.asarray([s['mean'] for s in stats])
        w = counts.reshape((-1,) + (1,) * (means.ndim - 1)) / counts.sum()
        mean = (means * w).sum(axis=0)
        var = ((np.asarray([s['std'] for s in stats]) ** 2 + (means - mean) ** 2) * w).sum(axis=0)
        result[key] = {'min': np.min([s['min'] for s in stats], axis=0).tolist(),
                       'max': np.max([s['max'] for s in stats], axis=0).tolist(),
                       'mean': mean.tolist(), 'std': np.sqrt(var).tolist(), 'count': [int(counts.sum())]}
    return result


def fixed_list(values):
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def write_parquet(path, columns, schema):
    table = pa.table({k: fixed_list(v) if v.ndim == 2 else pa.array(v) for k, v in columns.items()})
    require(table.schema == schema, 'LeRobot column schema differs from the drawer template')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial.parquet')
    pq.write_table(table, tmp, compression='zstd')
    saved = pq.read_table(tmp).to_pydict()
    for k, v in columns.items():
        require(np.array_equal(np.asarray(saved[k], dtype=v.dtype), v), f'parquet changed {k}')
    os.replace(tmp, path)


def video_statistics(path, frames=450, maximum=100):
    sel = set(np.linspace(0, frames - 1, min(maximum, frames), dtype=int).tolist())
    hist = np.zeros((3, 256), dtype=np.int64)
    decoded = samples = 0
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        require((st.width, st.height) == (224, 224) and abs(float(st.average_rate) - 30) < 1e-6, f'raster {path}')
        for fr in c.decode(st):
            if decoded in sel:
                im = fr.to_ndarray(format='rgb24')
                for ch in range(3):
                    hist[ch] += np.bincount(im[..., ch].reshape(-1), minlength=256)
                samples += 1
            decoded += 1
    require(decoded == frames, f'video frame count {path}: {decoded}')
    vals = np.arange(256) / 255
    mean = (hist * vals).sum(1) / hist.sum(1)
    var = (hist * vals ** 2).sum(1) / hist.sum(1) - mean ** 2
    res = {'mean': mean, 'std': np.sqrt(np.maximum(var, 0)),
           'min': np.array([np.flatnonzero(r)[0] / 255 for r in hist]),
           'max': np.array([np.flatnonzero(r)[-1] / 255 for r in hist])}
    return {k: v.reshape(3, 1, 1).tolist() for k, v in res.items()} | {'count': [samples]}


def collect(batch, renders, limit):
    rows = {json.loads(l)['case_id']: json.loads(l) for l in (SPEC / 'cases.jsonl').read_text().splitlines() if l}
    selected = []
    for res_path in sorted((renders / 'cases').glob('*/attempt_*/render_case_result.json')):
        rr = read(res_path)
        case_id, variant = res_path.parts[-3].split('__')
        require(variant == 'clean', f'v9_lean30 packages clean episodes only: {res_path}')
        native = Path(rr['native'])
        attempt = native.parent
        br = read(attempt / 'batch_result.json')
        receipt = read(native / 'receipt.json')
        grasp = read(attempt / 'actual_grasp_v3_receipt.json')
        quality = read(native / 'drawer_quality_receipt.json')
        vis = read(native / 'head_visibility_receipt.json')
        require(br['accepted'] is True and br['case_id'] == case_id and br['receipt_sha256'] == sha(native / 'receipt.json'),
                f'batch admission mismatch {case_id}')
        require(receipt['accepted'] is True and receipt['replay']['exact'] is True and
                receipt['replay']['every_step_hard_trace_exact'] is True and
                all(g['accepted'] for g in receipt['gates'].values()) and grasp['accepted'] is True and
                quality['accepted'] is True and vis['passed'] is True, f'native gates {case_id}')
        require(rows[case_id]['split'] == 'train', 'held-out case in training data')
        selected.append(dict(case_id=case_id, native=native, attempt=attempt, render=Path(rr['output']),
                             row=rows[case_id], receipt_sha256=sha(native / 'receipt.json')))
    selected.sort(key=lambda r: r['case_id'])
    require(len(selected) >= limit, f'only {len(selected)} rendered admitted episodes')
    return selected[:limit]


def package_episode(job):
    index, s, staging, schema_path = job
    staging = Path(staging)
    template = read(TEMPLATE / 'meta/info.json')
    schema = pq.read_schema(schema_path)
    src = s['render'] / 'source'
    meta = read(src / 'source.json')
    with np.load(src / 'actions_30fps.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    require(data['scalar_joint_qpos'].shape == (450, 32) and data['qpos'].shape == (450, 22) and
            data['scalar_joint_names'].tolist() == template['features']['observation.state']['names'][0] and
            data['joint_names'].tolist() == template['features']['action']['names'][0] and
            np.array_equal(data['video_time_s'], np.arange(450) / 30), f'state/action layout {s["case_id"]}')
    require(meta['native_accepted'] is True and meta['native_attempt'] == str(s['native']), 'render source binding')
    values = {
        'observation.state': data['scalar_joint_qpos'].astype(np.float32),
        'action': data['qpos'].astype(np.float32),
        'timestamp': data['video_time_s'].astype(np.float32),
        'frame_index': np.arange(450, dtype=np.int64),
        'episode_index': np.full(450, index, dtype=np.int64),
        'index': np.arange(index * 450, (index + 1) * 450, dtype=np.int64),
        'task_index': np.zeros(450, dtype=np.int64),
        'physical_index': np.full(450, int(s['row']['sampler_index']), dtype=np.int64),
        'source_case_index': np.full(450, 48, dtype=np.int64),
        'appearance_index': np.zeros(450, dtype=np.int64),
        **{k: data[k].astype(np.float64) for k in ('physical_query_time_s', 'state_time_s', 'command_time_s')},
        **{k: data[k].astype(np.int64) for k in ('state_source_index', 'command_source_index')},
    }
    require(all(np.isfinite(v).all() for v in values.values()), 'nonfinite feature')
    parquet = staging / f'data/chunk-000/episode_{index:06d}.parquet'
    write_parquet(parquet, values, schema)
    stats = {k: statistics(v) for k, v in values.items()}
    files = {str(parquet.relative_to(staging)): {'sha256': sha(parquet), 'bytes': parquet.stat().st_size}}
    for cam, feature in CAMERAS.items():
        srcv = s['render'] / 'video' / f'{cam}.mp4'
        dest = staging / f'videos/chunk-000/{feature}/episode_{index:06d}.mp4'
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(srcv, dest)
        require(sha(srcv) == sha(dest), 'video copy changed')
        stats[feature] = video_statistics(dest)
        files[str(dest.relative_to(staging))] = {'sha256': sha(dest), 'bytes': dest.stat().st_size}
    prov = dict(task=TASK, case_id=s['case_id'], visual_side='center', split='train',
                sampler_index=s['row']['sampler_index'], block_center_xy_m=s['row']['block_center_xy_m'],
                cabinet_center_xy_m=s['row']['cabinet_center_xy_m'], spec=str(SPEC),
                spec_scene_sha256=s['row']['scene_sha256'], native=str(s['native']),
                native_receipt_sha256=s['receipt_sha256'], batch_attempt=str(s['attempt']),
                render_attempt=str(s['render']), source=str(src), source_metadata_sha256=sha(src / 'source.json'),
                camera_order=list(CAMERAS), camera_mounts='astribot_sdk_nominal_image_up',
                controller='rssr.drawer.center_layout.v9_lean30', handle_hand='right', payload_hand='left',
                left_grasp_extra_pitch_deg=30.0, left_grasp_raise_m=0.015, dart=False,
                drawer_closer='right (handle hand)')
    record = {'episode_index': index, 'length': 450, 'tasks': [TASK], 'stats': stats, 'provenance': prov, 'files': files}
    write(staging / f'evidence/episode_{index:06d}.json', record)
    return record


def build(batch, renders, output, limit, staging_root=None):
    sel = collect(batch, renders, limit)
    output = output.absolute()
    require(not output.exists(), 'fresh output required')
    # Stage outside the shared parent directory (only the final rename touches it).
    staging = (Path(staging_root) if staging_root else output.parent) / (output.name + '.building.' + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    template = read(TEMPLATE / 'meta/info.json')
    schema = pq.read_schema(TEMPLATE / 'data/chunk-000/episode_000000.parquet')
    (staging / 'data/chunk-000').mkdir(parents=True)
    from multiprocessing import Pool
    jobs = [(i, x, str(staging), str(TEMPLATE / 'data/chunk-000/episode_000000.parquet')) for i, x in enumerate(sel)]
    with Pool(int(os.environ.get('PACKAGE_WORKERS', '12'))) as pool:
        records = pool.map(package_episode, jobs)
    print(json.dumps({'packaged': len(records)}), flush=True)
    n = len(records)
    info = deepcopy(template)
    info.update(campaign_id=output.parent.name, total_episodes=n, total_frames=n * 450, total_tasks=1,
                total_videos=3 * n, total_chunks=1, chunks_size=1000, splits={'train': f'0:{n}'},
                maximum_episode_frames=450, native_duration_limit_s=15., hardware_sdk_ready=False,
                raster_policy='upright 224x126 16:9 content in 224x224 with 49px black bars',
                camera_mounts='Astribot SDK kinematic.xml nominal mounts, image rolled 180 deg to match real/training image up',
                layout='cabinet centred on robot forward axis; block disk left of centre between cabinet and robot',
                controller='rssr.drawer.center_layout.v9_lean30')
    meta = staging / 'meta'
    write(meta / 'info.json', info)
    jsonl(meta / 'tasks.jsonl', [{'task_index': 0, 'task': TASK}])
    jsonl(meta / 'episodes.jsonl', [{'episode_index': i, 'tasks': [TASK], 'length': 450} for i in range(n)])
    jsonl(meta / 'episodes_stats.jsonl', [{'episode_index': i, 'stats': r['stats']} for i, r in enumerate(records)])
    jsonl(meta / 'source_cases.jsonl', [{'episode_index': i, **r['provenance']} for i, r in enumerate(records)])
    write(meta / 'stats.json', aggregate_statistics(records))
    write(staging / 'manifest.json', dict(schema='rssr.drawer.v9_lean30_local_lerobot.v1', episodes=n, frames=n * 450,
                                          controller='rssr.drawer.center_layout.v9_lean30',
                                          fps=30, state_dimension=32, action_dimension=22,
                                          camera_order=list(CAMERAS), camera_features=list(CAMERAS.values()),
                                          camera_resolution=[224, 224], hf_writes=False,
                                          spec=str(SPEC), native_batch=str(batch), renders=str(renders)))
    (staging / 'README.md').write_text(
        '# Drawer (centred layout): put the block in the third drawer from the top\n\n'
        f'LeRobot v2.1. {n} admitted native episodes, 450 frames at 30 Hz, three 224x224 cameras (16:9 content, '
        'black bars). Cabinet centred on the robot forward axis, block sampled in a disk left of centre between '
        'cabinet and robot. Right hand opens and closes the drawer; left hand places the block. Fixed head pitch '
        '0.98 rad; SDK nominal camera mounts. State: 32 measured S1 joints (rad); action: 22 absolute joint '
        'targets (rad).\n\nv9_lean30: the left gripper grasps the block pitched 30 deg further down (about its pinch '
        'axis, grasp point 1.5 cm higher on the block) and approaches along its pitched finger axis, so the block stays '
        'in the left wrist camera view; no DART.\n')
    vendor = PROJECT / 'pi0.5/training/C498_originalS1_60k_20261025_attempt_01/data_reader_02/vendor'
    sys.path.insert(0, str(vendor))
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(repo_id='local/drawer_center_layout', root=staging, video_backend='pyav',
                        delta_timestamps={'action': [i / 30 for i in range(50)]})
    require(ds.num_episodes == n and len(ds) == n * 450, 'official reader counts')
    checked = 0
    for e in range(n):
        for off in (0, 225, 449):
            smp = ds[e * 450 + off]
            require(tuple(smp['observation.state'].shape) == (32,) and tuple(smp['action'].shape) == (50, 22) and
                    int(smp['episode_index']) == e and int(smp['frame_index']) == off and
                    all(tuple(smp[k].shape) == (3, 224, 224) for k in CAMERAS.values()), f'reader rejected {e}/{off}')
            checked += 1
    write(staging / 'official_reader_receipt.json', dict(schema='rssr.drawer.center_layout_official_reader.v1',
                                                         accepted=True, episodes=n, frames=n * 450,
                                                         samples_checked=checked, hf_writes=False))
    require(not output.exists(), 'output appeared')
    os.rename(staging, output)
    return dict(output=str(output), episodes=n)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--batch', type=Path, required=True)
    ap.add_argument('--renders', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--episodes', type=int, default=100)
    ap.add_argument('--staging-root', type=Path, default=None, help='same filesystem as --output')
    a = ap.parse_args()
    print(json.dumps(build(a.batch, a.renders, a.output, a.episodes, a.staging_root)))


if __name__ == '__main__':
    main()
