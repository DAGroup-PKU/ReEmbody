#!/usr/bin/env python3
"""Package rendered FR3 Duo site episodes as one LeRobot v2.1 dataset (S1 v8/v9 packaging: sim2-real2/
package_v8_fixed80_cup_block_v3.py). Never uploads.

Every episode is re-read from its selected run (record.json, episode.npz, scene.xml, sha256-pinned by the selection)
and its verified render (verification.json row ok, video sha256 = render receipt). Per 30 Hz frame k = 0..N:
  observation.state (16): measured per arm 7 joint positions (rad) + Robotiq 2F-85 pad opening (m), left then right;
  action (16): the command executed from frame k to k + 1, same layout (absolute joint targets, rad; gripper opening
               target, m); the last frame repeats the last command;
  observation.images.cam_high / cam_left_wrist / cam_right_wrist: head D455 and wrist D405 renders, 224x168 (4:3)
               black-padded to 224x224, H.264, frame k at k / 30 s.
The official LeRobot v2.1 reader (pi0.5 vendor copy, as S1) then opens the dataset and samples 3 frames per episode.

    t1_retarget/bin/python package_lerobot.py --selection <selection.json> --render <production dir> \
        --out <fresh dataset dir> --campaign <id> --description <text>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import av
import mujoco
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGE2_DIR = REPO_ROOT / 'stages/02_sim1_to_sim2'
if str(STAGE2_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE2_DIR))
from franka_dual_v1.duo_robot import DualFR3  # noqa: E402

CAMERAS = {'head.mp4': 'observation.images.cam_high', 'left_wrist.mp4': 'observation.images.cam_left_wrist',
           'right_wrist.mp4': 'observation.images.cam_right_wrist'}
TASK = 'Stack_the_cups_together.'  # S1's task string (v2: set per task with --task-string, the S1 datasets' strings)
NAMES = [f'{h}_fr3v2_joint{i}' for h in ('left',) for i in range(1, 8)] + ['left_gripper_opening_m'] + \
        [f'{h}_fr3v2_joint{i}' for h in ('right',) for i in range(1, 8)] + ['right_gripper_opening_m']
PAD = 28  # rows of black above and below the 224x168 content
BAR_MAX = 8  # v3: largest value in the bars' inner rows; H.264 ringing reached 6 in 3 of the 300 block videos (1-2 frames each), a moved bar would show the scene


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for piece in iter(lambda: f.read(4 << 20), b''):
            h.update(piece)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def stats(v):
    x = np.asarray(v, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return {k: getattr(x, k)(axis=0).tolist() for k in ('min', 'max', 'mean', 'std')} | {'count': [len(x)]}


def aggregate(items):
    out = {}
    for key in items[0]:
        g = [i[key] for i in items]
        n = np.asarray([x['count'][0] for x in g], np.float64)
        mu = np.asarray([x['mean'] for x in g], np.float64)
        var = np.asarray([x['std'] for x in g], np.float64) ** 2
        w = (n / n.sum()).reshape((-1,) + (1,) * (mu.ndim - 1))
        m = np.sum(mu * w, axis=0)
        out[key] = dict(min=np.min([x['min'] for x in g], axis=0).tolist(), max=np.max([x['max'] for x in g], axis=0).tolist(),
                        mean=m.tolist(), std=np.sqrt(np.maximum(np.sum((var + (mu - m) ** 2) * w, axis=0), 0)).tolist(),
                        count=[int(n.sum())])
    return out


def fixed(v):
    x = np.asarray(v, np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1), type=pa.float32()), x.shape[1])


def video_stats(path, n):
    """Checks raster, fps, codec, letterbox, frame clock and count; returns S1-style per-channel stats."""
    chosen = set(np.linspace(0, n - 1, 64, dtype=int).tolist())
    hist = np.zeros((3, 256), np.int64)
    k = 0
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.codec_context.thread_count = 1
        require((s.width, s.height) == (224, 224) and abs(float(s.average_rate) - 30) < 1e-6
                and s.codec_context.name == 'h264', f'raster/fps/codec: {path}')
        for fr in c.decode(s):
            if k in chosen:
                rgb = fr.to_ndarray(format='rgb24')
                require(rgb[:PAD - 6].max() <= BAR_MAX and rgb[224 - PAD + 6:].max() <= BAR_MAX, f'letterbox moved: {path}')
                for ch in range(3):
                    hist[ch] += np.bincount(rgb[..., ch].reshape(-1), minlength=256)
            require(abs(float(fr.pts * s.time_base) - k / 30) < 2e-6, f'video clock differs from frame clock: {path}')
            k += 1
    require(k == n, f'{path}: {k} frames, episode has {n}')
    v = np.arange(256, dtype=np.float64) / 255
    mean = (hist * v).sum(1) / hist.sum(1)
    var = (hist * v ** 2).sum(1) / hist.sum(1) - mean ** 2
    return dict(min=np.asarray([np.flatnonzero(h)[0] / 255 for h in hist]).reshape(3, 1, 1).tolist(),
                max=np.asarray([np.flatnonzero(h)[-1] / 255 for h in hist]).reshape(3, 1, 1).tolist(),
                mean=mean.reshape(3, 1, 1).tolist(), std=np.sqrt(np.maximum(var, 0)).reshape(3, 1, 1).tolist(),
                count=[len(chosen)])


def source_case_index(case):
    # cups and blocks come from S1 episodes ("stack_cups_ep000000"); the drawer has a single source case (drawer_s1_center)
    return int(case.rsplit('_ep', 1)[1]) if '_ep' in case else 0


def columns(run, episode, offset):
    z = np.load(run / 'episode.npz')
    m = mujoco.MjModel.from_xml_path(str(run / 'scene.xml'))
    d = mujoco.MjData(m)
    robot = DualFR3(m)
    qpos, act = z['qpos'], z['action']
    n = len(qpos)
    require(len(act) == n - 1 and np.allclose(z['time'], np.arange(n) / 30, atol=1e-5), f'clock: {run}')
    state = np.zeros((n, 16))
    for k in range(n):
        d.qpos[:] = qpos[k]
        mujoco.mj_kinematics(m, d)
        for i, h in enumerate(('left', 'right')):
            state[k, 8 * i:8 * i + 7] = qpos[k, robot.arms[h].qadr]
            state[k, 8 * i + 7] = robot.width(d, h)
    action = np.r_[act, act[-1:]]
    v = read(run / 'record.json')['variation']
    cols = {'observation.state': state.astype(np.float32), 'action': action.astype(np.float32),
            'timestamp': (np.arange(n) / 30).astype(np.float32), 'frame_index': np.arange(n, dtype=np.int64),
            'episode_index': np.full(n, episode, np.int64), 'index': np.arange(offset, offset + n, dtype=np.int64),
            'task_index': np.zeros(n, np.int64),
            'source_case_index': np.full(n, source_case_index(v['source_case']), np.int64),
            'variation_index': np.full(n, int(v['id'].rsplit('_', 1)[1]), np.int64)}
    require(all(np.all(np.isfinite(x)) for x in cols.values()), f'nonfinite: {run}')
    return cols, n


def feature_info(names_len):
    vid = dict(dtype='video', shape=[3, 224, 224], names=['channels', 'height', 'width'],
               info={'video.fps': 30, 'video.height': 224, 'video.width': 224, 'video.channels': 3,
                     'video.codec': 'h264', 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'has_audio': False})
    one = lambda t: dict(dtype=t, shape=[1], names=None)  # noqa: E731
    return {'observation.state': dict(dtype='float32', shape=[names_len], names=[NAMES]),
            'action': dict(dtype='float32', shape=[names_len], names=[NAMES]),
            **{f: vid for f in CAMERAS.values()},
            'timestamp': one('float32'), 'frame_index': one('int64'), 'episode_index': one('int64'),
            'index': one('int64'), 'task_index': one('int64'), 'source_case_index': one('int64'),
            'variation_index': one('int64')}


def official_reader(root, lengths, reader_vendor: Path | None = None):
    if reader_vendor is not None and str(reader_vendor) not in sys.path:
        sys.path.insert(0, str(reader_vendor))
    import lerobot
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    data = LeRobotDataset(repo_id='local/reembody_fr3_duo_site', root=root, video_backend='pyav',
                          delta_timestamps={'action': [i / 30 for i in range(50)]})
    require(data.num_episodes == len(lengths) and len(data) == sum(lengths), 'reader counts differ')
    checked, start = 0, 0
    for e, n in enumerate(lengths):
        for f in (0, n // 2, n - 1):
            s = data[start + f]
            require(tuple(s['observation.state'].shape) == (16,) and tuple(s['action'].shape) == (50, 16)
                    and int(s['episode_index']) == e and int(s['frame_index']) == f
                    and all(tuple(s[k].shape) == (3, 224, 224) for k in CAMERAS.values()), f'reader sample {e}/{f}')
            checked += 1
        start += n
    return dict(schema='rssr.fr3_duo.official_reader.v1', accepted=True, reader=str(Path(lerobot.__file__).resolve().parent),
                episodes=len(lengths), frames=int(sum(lengths)), samples_checked=checked, state_shape=[16],
                action_horizon_shape=[50, 16], camera_shape=[3, 224, 224])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--render', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--campaign', required=True)
    ap.add_argument('--description', default='')
    ap.add_argument('--task-string', default=TASK, help="S1's task string for this task (v2, 2026-11-06)")
    a = ap.parse_args()
    require(not a.out.exists(), f'fresh output required: {a.out}')
    sel = read(a.selection)
    ver = read(a.render / 'verification.json')
    require(ver['all_ok'] and ver['selection'] == str(a.selection) and ver['episodes'] == len(sel['episodes']),
            'render verification does not cover this selection')
    vrow = {r['episode_index']: r for r in ver['rows']}
    a.out.mkdir(parents=True)
    rows, ep_stats, src_rows, lengths, offset = [], [], [], [], 0
    for e in sel['episodes']:
        i = e['episode_index']
        run = Path(e['run'])
        require(sha(run / 'record.json') == e['record_sha256'] and sha(run / 'episode.npz') == e['episode_sha256'],
                f'run changed since selection: {run}')
        require(vrow[i]['ok'] and vrow[i]['variation'] == e['variation'], f'render not verified: {i}')
        ep = a.render / f'ep{i:03d}'
        receipt = read(ep / 'render' / 'render_receipt.json')
        cols, n = columns(run, i, offset)
        table = pa.table({k: fixed(v) if v.ndim == 2 else pa.array(v) for k, v in cols.items()})
        pq_path = a.out / f'data/chunk-000/episode_{i:06d}.parquet'
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, pq_path, compression='zstd')
        back = pq.read_table(pq_path).to_pydict()
        for k, v in cols.items():
            require(np.array_equal(np.asarray(back[k], dtype=v.dtype), v), f'parquet round trip {i}/{k}')
        st = {k: stats(v) for k, v in cols.items()}
        pins = {}
        for fname, feat in CAMERAS.items():
            srcv = ep / 'render' / fname
            dst = a.out / f'videos/chunk-000/{feat}/episode_{i:06d}.mp4'
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(srcv, dst)
            pins[feat] = sha(dst)
            require(pins[feat] == sha(srcv), f'video copy {i}/{fname}')
            st[feat] = video_stats(dst, n)
        ep_stats.append(dict(episode_index=i, stats=st))
        rec = read(run / 'record.json')
        src_rows.append(dict(episode_index=i, length=n, task=a.task_string, variation=e['variation'],
                             source_case=rec['variation']['source_case'], layout=rec['variation'].get('layout'),
                             run=str(run), record_sha256=e['record_sha256'], episode_sha256=e['episode_sha256'],
                             options=rec.get('options'), duration_s=e['duration_s'],
                             render_receipt=str(ep / 'render' / 'render_receipt.json'),
                             render_receipt_sha256=sha(ep / 'render' / 'render_receipt.json'),
                             lighting_seed=vrow[i].get('seed'), render_device=vrow[i].get('device'),
                             blender=receipt.get('blender_version', receipt.get('blender')), videos=pins))
        rows.append(dict(episode_index=i, tasks=[a.task_string], length=n))
        lengths.append(n)
        offset += n
        print(json.dumps(dict(episode=i, variation=e['variation'], frames=n)), flush=True)
    count = len(rows)
    info = dict(codebase_version='v2.1', robot_type='franka_fr3_duo_robotiq_2f85', total_episodes=count,
                total_frames=int(offset), total_tasks=1, total_videos=3 * count, total_chunks=1, chunks_size=1000,
                fps=30, splits={'train': f'0:{count}'},
                data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
                video_path='videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4',
                features=feature_info(16), campaign_id=a.campaign, description=a.description,
                action_contract=dict(semantics='absolute_desired_joint_target', unit='rad; gripper opening m',
                                     gripper='Robotiq 2F-85 pad opening (m), 0 = close with the force limit'),
                raster_policy='224x168 4:3 content with 28 px black bars above/below in a 224x224 canvas',
                cameras=dict(cam_high='RealSense D455 on the arch behind the robot (fitted to the user frame)',
                             cam_left_wrist='RealSense D405, nominal 2F-85 bracket mount (not hardware-calibrated)',
                             cam_right_wrist='RealSense D405, nominal 2F-85 bracket mount (not hardware-calibrated)'),
                maximum_episode_frames=int(max(lengths)), minimum_episode_frames=int(min(lengths)))
    write(a.out / 'meta/info.json', info)
    jsonl(a.out / 'meta/tasks.jsonl', [dict(task_index=0, task=a.task_string)])
    jsonl(a.out / 'meta/episodes.jsonl', rows)
    jsonl(a.out / 'meta/episodes_stats.jsonl', ep_stats)
    jsonl(a.out / 'meta/source_cases.jsonl', src_rows)
    write(a.out / 'meta/stats.json', aggregate([r['stats'] for r in ep_stats]))
    write(a.out / 'packaging_receipt.json', dict(
        schema='rssr.fr3_duo.local_lerobot_package.v1', package_complete=True, episodes=count, frames=int(offset),
        videos=3 * count, fps=30, selection=str(a.selection), selection_sha256=sha(a.selection),
        render=str(a.render), verification_sha256=sha(a.render / 'verification.json'),
        code_sha256=sha(__file__), uploaded_to_hf=False))
    write(a.out / 'official_reader_receipt.json', official_reader(a.out, lengths))
    print(json.dumps(dict(out=str(a.out), episodes=count, frames=int(offset))))


if __name__ == '__main__':
    main()
