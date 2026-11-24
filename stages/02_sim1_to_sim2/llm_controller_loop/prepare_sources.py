#!/usr/bin/env python3
"""Prepare every requested FR3 v2 source without resampling or stepping physics.

Uses the release's exact record Variation and options. DS.build only compiles the
model; the optional initial-state check calls reset+forward, never mj_step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

FRAMEWORK = Path(__file__).resolve().parents[1] / 'franka_dual_v1'
DEFAULT_AUDIT = Path(__file__).with_name('source_audit_20261109.json')
SOURCE_PREFIX = '150case_data_enhance_blender_franka_v2'
MAPPING = {'cup': 'cup', 'block': 'block', 'draw': 'v2_draw_new'}
MIRROR = os.environ.get('REEMBODY_LEGACY_MIRROR_PREFIX', '')


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def canonical_xml_sha(xml):
    # Remote/local path relocation is represented identically, with no numeric
    # rounding or removal of physical/model attributes.
    relocated = xml.replace(MIRROR, str(Path(__file__).resolve().parents[3])) if MIRROR else xml
    return digest_bytes(ET.canonicalize(relocated, strip_text=True).encode())


def check_build(row):
    import numpy as np
    import mujoco
    sys.path.insert(0, str(FRAMEWORK.parent))
    from franka_dual_v1 import duo_scene as DS

    result = dict(source_id=row['source_id'], checks={}, errors=[])
    try:
        v = DS.Variation.from_json(row['variation'])
        model, scene, model_sha, xml = DS.build(v)
        source_xml = Path(row['source_native_scene']).read_text()
        source_sha = digest_bytes(source_xml.encode())
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
        with np.load(row['source_native_episode'], allow_pickle=False) as episode:
            initial_qpos_equal = bool(np.array_equal(data.qpos, episode['qpos'][0]))
            initial_qvel_equal = bool(np.array_equal(data.qvel, episode['qvel'][0]))
            initial_time_equal = float(data.time) == float(episode['time'][0]) == 0.
        checks = dict(
            variation_roundtrip_exact=v.to_json() == row['variation'],
            build_xml_sha_consistent=model_sha == digest_bytes(xml.encode()),
            xml_bytes_equal_to_local_source=source_xml == xml,
            xml_semantics_equal=canonical_xml_sha(source_xml) == canonical_xml_sha(xml),
            reset_home_qpos_equal_source_t0=initial_qpos_equal,
            reset_home_qvel_equal_source_t0=initial_qvel_equal,
            reset_home_time_equal_source_t0=initial_time_equal,
        )
        result.update(checks=checks, rebuilt_scene_sha256=model_sha,
                      source_local_scene_sha256=source_sha,
                      rebuilt_scene_canonical_sha256=canonical_xml_sha(xml),
                      source_scene_canonical_sha256=canonical_xml_sha(source_xml),
                      initial_state_qpos_sha256=digest_bytes(data.qpos.tobytes()),
                      initial_state_qvel_sha256=digest_bytes(data.qvel.tobytes()),
                      model_dimensions=dict(nq=model.nq, nv=model.nv, nu=model.nu,
                                            nbody=model.nbody, ngeom=model.ngeom,
                                            ncam=model.ncam),
                      task_id=scene['task_id'], bindings=scene['bindings'],
                      physics_steps_executed=0)
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    result['reproducible'] = bool(result['checks']) and all(result['checks'].values()) and not result['errors']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=DEFAULT_AUDIT)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    if audit['source_prefix'] != SOURCE_PREFIX or not audit['all_source_native_and_receipt_checks_passed']:
        raise ValueError('Source audit must pass and bind the exact requested FR3 prefix')
    rows, task_files = [], {}
    for task, source_folder in MAPPING.items():
        selected = sorted((r for r in audit['episodes'] if r['task_folder'] == source_folder),
                          key=lambda r: r['episode_index'])
        if len(selected) != 100 or [r['episode_index'] for r in selected] != list(range(100)):
            raise ValueError(f'{task}: expected all source episode indices 0..99')
        variations, source_ids = [], []
        task_rows = []
        for source in selected:
            run = Path(source['native_run'])
            if sha(run / 'record.json') != source['files']['native_record']['sha256']:
                raise ValueError(f'Native source record changed: {run}')
            rec = json.loads((run / 'record.json').read_text())
            source_id = f'{SOURCE_PREFIX}/{source_folder}/episode_{source["episode_index"]:06d}'
            row = dict(source_id=source_id, output_task=task, source_task_folder=source_folder,
                       source_episode_index=source['episode_index'], variation=rec['variation'],
                       source_options=rec.get('options', {}), source_duration_s=source['duration_s'],
                       source_native_run=str(run), source_native_scene=str(run / 'scene.xml'),
                       source_native_episode=str(run / 'episode.npz'),
                       source_native_record=str(run / 'record.json'),
                       source_native_record_sha256=source['files']['native_record']['sha256'],
                       source_native_episode_sha256=source['files']['native_episode']['sha256'],
                       source_native_scene_local_sha256=source['files']['native_scene']['sha256'],
                       source_native_scene_execution_sha256=source['files']['native_scene']['original_execution_sha256'],
                       source_release_row=source['source_row'], source_status='all_preserved')
            rows.append(row)
            task_rows.append(row)
            variations.append(rec['variation'])
            source_ids.append(source_id)
        if len({v['id'] for v in variations}) != 100 or len(set(source_ids)) != 100:
            raise ValueError(f'{task}: duplicate logical sources or native variation IDs')
        directory = args.output / 'sources' / task
        for name, content in [('variations.json', variations), ('source_ids.json', source_ids),
                              ('episodes.json', task_rows)]:
            write(directory / name, content)
        task_files[task] = dict(source_task_folder=source_folder, episodes=100,
                               native_task=variations[0]['task'],
                               variations_json=str((directory / 'variations.json').resolve()),
                               source_ids_json=str((directory / 'source_ids.json').resolve()),
                               episodes_json=str((directory / 'episodes.json').resolve()),
                               source_duration_over_20s=sum(r['source_duration_s'] > 20 for r in task_rows),
                               dropped_sources=0, sampled_new_sources=0)
    if len(rows) != 300:
        raise ValueError('Requested scope must preserve exactly 300 sources')
    builds = {}
    write(args.output / 'preparation_progress.json', dict(status='checking_builds', total=300, checked=0))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(check_build, r): r['source_id'] for r in rows}
        for future in as_completed(futures):
            source_id = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(source_id=source_id, reproducible=False, checks={},
                              errors=[f'{type(error).__name__}: {error}'])
            builds[source_id] = result
            if len(builds) % 25 == 0:
                progress = dict(status='checking_builds', total=300, checked=len(builds),
                                nonreproducible=sum(not r['reproducible'] for r in builds.values()))
                write(args.output / 'preparation_progress.json', progress)
                print(json.dumps(progress), flush=True)
    for row in rows:
        row['build_check'] = builds[row['source_id']]
    failed = [r['source_id'] for r in rows if not r['build_check']['reproducible']]
    summary = dict(schema='rssr.franka_v3.exact_prepared_sources.v1',
                   at_utc=datetime.now(timezone.utc).isoformat(),
                   source_prefix=SOURCE_PREFIX, output_prefix='150case_data_enhance_blender_franka_v3',
                   source_audit=str(args.audit.resolve()), source_audit_sha256=sha(args.audit),
                   preparation_code=str(Path(__file__).resolve()), preparation_code_sha256=sha(__file__),
                   framework_root=str(FRAMEWORK),
                   framework_pins={str(p): sha(p) for p in FRAMEWORK.glob('*.py')},
                   tasks=task_files, episodes=rows, total_sources=300, dropped_sources=0,
                   sampled_new_sources=0, physics_steps_executed=0,
                   exact_source_options_preserved=True, exact_source_variations_preserved=True,
                   all_sources_reproducible=not failed, nonreproducible_source_ids=failed)
    write(args.output / 'prepared_sources.json', summary)
    for task in task_files:
        directory = args.output / 'sources' / task
        write(directory / 'episodes.json', [r for r in rows if r['output_task'] == task])
        task_files[task]['variations_sha256'] = sha(directory / 'variations.json')
        task_files[task]['source_ids_sha256'] = sha(directory / 'source_ids.json')
        task_files[task]['episodes_sha256'] = sha(directory / 'episodes.json')
    write(args.output / 'prepared_sources.json', summary)
    write(args.output / 'preparation_progress.json', dict(status='complete', total=300, checked=300,
                                                        nonreproducible=len(failed)))
    print(json.dumps(dict(prepared_sources=str((args.output / 'prepared_sources.json').resolve()),
                          total_sources=300, all_sources_reproducible=not failed,
                          nonreproducible_source_ids=failed)), flush=True)


if __name__ == '__main__':
    main()
