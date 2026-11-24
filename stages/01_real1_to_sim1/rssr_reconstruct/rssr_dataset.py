"""Deterministic, content-addressed Cobot manifests; no result-based sampling."""
import argparse
import collections
import pathlib
import random

from common import read, sha, write
from rssr_protocol import digest, parse

ROLES = {'head': 'image_top', 'left_wrist': 'image_left', 'right_wrist': 'image_right'}
TASK_IDS = tuple([f'di{i}' for i in range(1, 8)] + [f'dm{i}' for i in range(1, 9)])


def inventory(dataset_root):
    root = pathlib.Path(dataset_root).resolve()
    tasks, grouped = {}, {}
    for task_id in TASK_IDS:
        matches = sorted(root.glob(task_id + '_*'))
        if len(matches) != 1:
            raise ValueError(f'{task_id}: expected exactly one raw dataset, got {len(matches)}')
        folder = matches[0]
        info = read(folder / 'meta/info.json')
        descriptions = [parse(line) for line in (folder / 'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
        if len(descriptions) != 1 or info['total_tasks'] != 1:
            raise ValueError('Ambiguous task description: ' + task_id)
        features = info['features']
        if features['action']['shape'] != [14] or features['observation.state']['shape'] != [26] or info['fps'] != 30:
            raise ValueError('Unsupported raw action/state/video contract: ' + task_id)
        paths = sorted(folder.glob('data/chunk-*/episode_*.parquet'))
        if len(paths) != info['total_episodes']:
            raise ValueError('Metadata and actual episode count differ: ' + task_id)
        task = descriptions[0]['task']
        tasks[task_id] = {'task': task, 'dataset': str(folder), 'episodes': len(paths),
                          'info_sha256': sha(folder/'meta/info.json'),
                          'tasks_sha256': sha(folder/'meta/tasks.jsonl')}
        rows = []
        for source in paths:
            episode = int(source.stem.split('_')[-1])
            videos = {role: folder / 'videos' / source.parent.name / ('observation.images.' + camera) / (source.stem + '.mp4')
                      for role, camera in ROLES.items()}
            if not all(p.is_file() for p in videos.values()):
                raise ValueError('Missing raw view for ' + str(source))
            rows.append({'case_id': f'{task_id}_ep{episode:06d}', 'task_id': task_id, 'task_family': task_id,
                         'task': task, 'source_episode': episode, 'source_parquet': str(source),
                         'source_sha256': sha(source), 'source_videos': {k: str(v) for k, v in videos.items()},
                         'source_video_sha256': {k: sha(v) for k, v in videos.items()}})
        grouped[task_id] = rows
    return tasks, grouped


def build(dataset_root, seed=20260916, verification_count=10, expected_training_count=1500):
    tasks, groups = inventory(dataset_root)
    # Round-robin task order prevents spending an entire initial bank on one task.
    rows = [groups[t][i] for i in range(max(map(len, groups.values()))) for t in TASK_IDS if i < len(groups[t])]
    ids = [r['case_id'] for r in rows]
    if len(ids) != len(set(ids)) or len(rows) != expected_training_count:
        raise ValueError(f'Expected {expected_training_count} unique training episodes, found {len(rows)}')
    rng = random.Random(seed)
    order = list(TASK_IDS)
    rng.shuffle(order)
    verify_rows = []
    for i in range(verification_count):
        candidates = [r for r in groups[order[i % len(order)]] if r['case_id'] not in {v['case_id'] for v in verify_rows}]
        if not candidates:
            raise ValueError('Not enough distinct verification episodes')
        verify_rows.append(rng.choice(candidates))
    common = {'schema': 'rssr.dataset.v1', 'dataset_root': str(pathlib.Path(dataset_root).resolve()),
              'seed': seed, 'task_metadata': tasks, 'selection_basis': 'metadata and deterministic seed; no outcome filtering'}
    train = dict(common, split='train', rows=rows, task_counts=dict(collections.Counter(r['task_id'] for r in rows)))
    train['manifest_id'] = digest(train)
    verify = dict(common, split='verification', rows=verify_rows,
                  training_manifest_id=train['manifest_id'], overlap_policy='seen_demonstration_frozen_library',
                  overlap_with_training=[r['case_id'] for r in verify_rows],
                  unseen_generalization_claim=False,
                  explanation='All 1500 available demonstrations are used for training; these ten measure frozen-bank replay inference, not held-out generalization.')
    verify['manifest_id'] = digest(verify)
    return train, verify


def validate_manifest(manifest, verify_assets=False):
    value = dict(manifest)
    identity = value.pop('manifest_id')
    if digest(value) != identity:
        raise ValueError('Dataset manifest content changed')
    rows = manifest['rows']
    if not rows or len({r['case_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate cases')
    for row in rows:
        if set(row['source_videos']) != set(ROLES) or set(row['source_video_sha256']) != set(ROLES):
            raise ValueError('Exactly three named raw camera views required')
        if verify_assets:
            if sha(row['source_parquet']) != row['source_sha256']:
                raise ValueError('Raw action changed: ' + row['case_id'])
            for role, path in row['source_videos'].items():
                if sha(path) != row['source_video_sha256'][role]:
                    raise ValueError('Raw video changed: ' + row['case_id'] + ':' + role)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True, help='JSON with dataset_root, output_dir and optional seed')
    args = parser.parse_args()
    request = read(args.request)
    train, verification = build(request['dataset_root'], request.get('seed', 20260916))
    root = pathlib.Path(request['output_dir'])
    for name, data in [('train.json', train), ('verification.json', verification)]:
        path = root / name
        if path.exists() and read(path) != data:
            raise ValueError('Refusing to replace another manifest: ' + str(path))
        write(path, data)
    response = {'schema': 'rssr.dataset.response.v1', 'training_cases': len(train['rows']),
                'verification_cases': len(verification['rows']), 'tasks': len(train['task_metadata']),
                'train': str(root/'train.json'), 'verification': str(root/'verification.json'),
                'verification_is_held_out': False, 'api_requests': 0}
    write(root/'response.json', response)
    import json
    print(json.dumps(response))


if __name__ == '__main__':
    main()
