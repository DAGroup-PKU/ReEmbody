#!/usr/bin/env python3
"""Run all 100 exact sources in separate native processes, archiving as ready.

Requires a genuine same-controller pilot that already passed the unchanged
native/task/replay and duration checks. Every source remains in the manifest;
failures are retained, never replaced by newly sampled scenes. A finished
source can enter export/render preparation while other sources still execute.
This coordinator never uploads and never edits a controller or physics gate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading

HERE = Path(__file__).resolve().parent
import sys as _sys
PYTHON = _sys.executable
STORAGE = json.loads((HERE / 'artifact_storage.json').read_text())
TEMP = Path(STORAGE['temporary_root'])
PFS = Path(STORAGE['data_root'])


def read(path):
    return json.loads(Path(path).read_text())


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', choices=('cup', 'block', 'draw'), required=True)
    p.add_argument('--api-request', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--adopt-trial', type=Path, action='append', default=[], help='Additional completed same-controller source, preserving its true pass/fail status; never run it twice')
    p.add_argument('--name', required=True)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--prepare-renders', action='store_true')
    args = p.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('Use 1..16 independent native workers; coordinate total CPU demand before launching')
    if not args.name or len(args.name) > 48 or any(not (c.isalnum() or c in '_-') for c in args.name):
        raise ValueError('Invalid fresh batch name')
    delivery = module(HERE / 'stream_delivery.py', 'native_batch_delivery')
    pilot, pilot_source = delivery.validate_trial(args.pilot)
    api = args.api_request.resolve(strict=True)
    controller = api / 'controller.py'
    if Path(pilot['api']['folder']).resolve() != api or pilot_source['output_task'] != args.task:
        raise ValueError('Pilot belongs to another task or API controller')
    if delivery.sha(controller) != pilot['controller_sha256']:
        raise ValueError('Pilot and batch controller bytes differ')
    adopted = {pilot_source['source_id']: (args.pilot.resolve(), pilot, pilot_source)}
    prepared = read(HERE / 'prepared_sources.json')
    for path in args.adopt_trial:
        # Terminal failures still count as attempted sources. They need the
        # same immutable provenance, never a forged positive admission.
        path = path.resolve(strict=True)
        rec, source = read(path / 'trial_receipt.json'), read(path / 'source.json')
        if rec.get('schema') != 'rssr.franka_v3.luna_native_trial_receipt.v1' or not rec.get('api', {}).get('model'):
            raise ValueError('Unrecognized terminal source receipt')
        matches = [r for r in prepared['episodes'] if r['source_id'] == rec['source_id']]
        if matches != [source] or delivery.sha(path / 'controller.py') != rec['controller_sha256']:
            raise ValueError('Terminal source or controller binding changed')
        for filename, digest in rec['input_pins'].items():
            if delivery.sha(filename) != digest:
                raise ValueError('Terminal trial input changed: ' + filename)
        for relative, digest in rec['artifacts'].items():
            if delivery.sha(path / relative) != digest:
                raise ValueError('Terminal trial artifact changed: ' + relative)
        if rec['native_and_duration_accepted']:
            rec, source = delivery.validate_trial(path)
        if source['output_task'] != args.task or Path(rec['api']['folder']).resolve() != api or rec['controller_sha256'] != pilot['controller_sha256']:
            raise ValueError('Additional adopted trial belongs to another task or controller')
        if source['source_id'] in adopted:
            raise ValueError('Duplicate adopted source')
        adopted[source['source_id']] = (path.resolve(), rec, source)
    rows = sorted((r for r in prepared['episodes'] if r['output_task'] == args.task), key=lambda r: r['source_episode_index'])
    if len(rows) != 100 or [r['source_episode_index'] for r in rows] != list(range(100)):
        raise ValueError('Exactly all 100 original sources are required')
    # Fresh launch only. An interrupted coordinator is reconciled from its
    # durable per-source PIDs and receipts; invoking it again cannot duplicate it.
    root = TEMP / (args.task + '_native') / args.name
    root.mkdir(parents=True, exist_ok=False)
    controls = PFS / 'batch_records' / args.task / args.name
    controls.mkdir(parents=True, exist_ok=False)
    owner = dict(schema='rssr.franka_v3.exact_native_batch.v1', task=args.task, api=str(api),
        controller_sha256=pilot['controller_sha256'], source_ids=[r['source_id'] for r in rows],
        source_count=100, no_resampling=True, temporary_root=str(root), controls=str(controls),
        workers=args.workers, separate_process_per_source=True, never_auto_restart=True,
        coordinator_sha256=delivery.sha(__file__), per_worker_temporary_reservation_bytes=128 * 2**20)
    (controls / 'owner.json').write_text(json.dumps(owner, indent=2) + '\n')
    lock = threading.Lock()
    capacity = threading.Condition()
    reservations = 0
    stop_new = threading.Event()
    completed = {}

    def event(value):
        value = dict(at_utc=datetime.now(timezone.utc).isoformat(), **value)
        with lock:
            with (controls / 'events.jsonl').open('a') as f:
                f.write(json.dumps(value, ensure_ascii=False) + '\n')
            if value.get('source_id'):
                completed[value['source_id']] = value
            status = dict(expected=100, finished=sum(v.get('event') in ('terminal', 'terminal_missing_receipt') for v in completed.values()),
                native_and_duration_accepted=sum(v.get('native_and_duration_accepted', False) for v in completed.values()),
                complete=False, coordinator_finished=False, entries=completed)
            tmp = controls / 'status.tmp'
            tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
            tmp.replace(controls / 'status.json')
        print(json.dumps({k: value[k] for k in ('event', 'source_episode_index', 'pid', 'native_and_duration_accepted',
            'full_physics_duration_s', 'returncode', 'workers', 'expected_sources') if k in value}, ensure_ascii=False), flush=True)

    def postprocess(trial, receipt, source):
        archive = subprocess.run([PYTHON, str(HERE / 'archive_terminal.py'), str(trial)], capture_output=True, text=True)
        if archive.returncode:
            stop_new.set()
            with capacity:
                capacity.notify_all()
            return dict(archive_error=(archive.stderr or archive.stdout)[-1000:])
        result = dict(archive=json.loads(archive.stdout))
        if args.prepare_renders and receipt['native_and_duration_accepted'] and shutil.disk_usage(TEMP).free < 2 * 2**30:
            result['render_preparation'] = dict(deferred=True, reason='Bound temporary cache; native data safely archived to PFS')
        elif args.prepare_renders and receipt['native_and_duration_accepted']:
            export = subprocess.run([PYTHON, str(HERE / 'stream_delivery.py'), 'prepare', '--trial', str(trial)], capture_output=True, text=True)
            result['render_preparation'] = json.loads(export.stdout) if export.returncode == 0 else dict(error=(export.stderr or export.stdout)[-1000:])
        return result

    def execute_reserved(source):
        index = source['source_episode_index']
        if source['source_id'] in adopted:
            path, rec, src = adopted[source['source_id']]
            event(dict(event='terminal', source_id=source['source_id'], source_episode_index=index,
                adopted_existing_trial=str(path), native_and_duration_accepted=rec['native_and_duration_accepted'], trial=str(path)))
            return
        trial = root / f'episode_{index:06d}'
        log = controls / f'episode_{index:06d}.log'
        argv = [PYTHON, str(HERE / 'native_trial.py'), '--task', args.task, '--episode-index', str(index),
                '--controller', str(controller), '--api-request', str(api), '--output', str(trial)]
        with log.open('x') as output:
            process = subprocess.Popen(argv, stdout=output, stderr=subprocess.STDOUT,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl'))
            event(dict(event='launched', source_id=source['source_id'], source_episode_index=index, pid=process.pid,
                argv=argv, trial=str(trial)))
            code = process.wait()
        if not (trial / 'trial_receipt.json').is_file():
            archived = None
            if trial.is_dir():
                proof = dict(schema='rssr.franka_v3.owned_partial_process_exit.v1', wait_returned=True,
                    pid=process.pid, returncode=code, source_id=source['source_id'], output_task=args.task,
                    controller_sha256=pilot['controller_sha256'], publication_admitted=False)
                with (trial / 'terminal_process_exit.json').open('x') as f:
                    json.dump(proof, f, indent=2)
                proc = subprocess.run([PYTHON, str(HERE / 'archive_terminal.py'), str(trial)], capture_output=True, text=True)
                archived = json.loads(proc.stdout) if proc.returncode == 0 else dict(error=(proc.stderr or proc.stdout)[-1000:])
                if proc.returncode:
                    stop_new.set()
            event(dict(event='terminal_missing_receipt', source_id=source['source_id'], source_episode_index=index,
                returncode=code, trial=str(trial), retry_performed=False, partial_archive=archived))
            return
        receipt = read(trial / 'trial_receipt.json')
        processed = postprocess(trial, receipt, source)
        event(dict(event='terminal', source_id=source['source_id'], source_episode_index=index, returncode=code,
            trial=str(trial), original_native_admitted=receipt['original_native_admitted'],
            native_and_duration_accepted=receipt['native_and_duration_accepted'],
            full_physics_duration_s=receipt['full_physics_duration_s'], **processed))

    def execute(source):
        nonlocal reservations
        if source['source_id'] in adopted:
            return execute_reserved(source)
        waiting_reported = False
        acquired = False
        while True:
            with capacity:
                if stop_new.is_set():
                    break
                available = shutil.disk_usage(TEMP).free - reservations * 128 * 2**20
                if available >= (512 + 128) * 2**20:
                    reservations += 1
                    acquired = True
                    break
                if not waiting_reported:
                    event(dict(event='waiting_temporary_space', source_id=source['source_id'], source_episode_index=source['source_episode_index']))
                    waiting_reported = True
                capacity.wait(timeout=2)
        if stop_new.is_set():
            if acquired:
                with capacity:
                    reservations -= 1
                    capacity.notify_all()
            event(dict(event='not_started_after_archive_failure', source_id=source['source_id'], source_episode_index=source['source_episode_index']))
            return
        try:
            return execute_reserved(source)
        finally:
            with capacity:
                reservations -= 1
                capacity.notify_all()

    event(dict(event='started', task=args.task, workers=args.workers, expected_sources=100, owner=str(controls / 'owner.json')))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in rows}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                source = futures[future]
                event(dict(event='coordinator_error', source_id=source['source_id'], error=type(error).__name__ + ': ' + str(error)))
    accepted = [r for r in completed.values() if r.get('native_and_duration_accepted')]
    result = dict(schema='rssr.franka_v3.exact_native_batch_result.v1', expected_sources=100,
        all_sources_retained=True, native_and_duration_accepted=len(accepted),
        all_native_and_duration_accepted=len(accepted) == 100, publication_admitted=False,
        entries=completed, never_auto_retried=True, coordinator_finished=True,
        all_sources_processed=sum(r.get('event') in ('terminal', 'terminal_missing_receipt') for r in completed.values()) == 100,
        final_source_states={state: sum(r.get('event') == state for r in completed.values()) for state in {r.get('event') for r in completed.values()}})
    (controls / 'batch_result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (controls / 'status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(batch_result=str(controls / 'batch_result.json'), accepted=len(accepted), expected=100)))


if __name__ == '__main__':
    main()
