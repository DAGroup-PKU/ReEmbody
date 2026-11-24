"""Ten-worker, restartable official campaign with physical attempt retention."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rssr_augment.shared.family_autoresearch.store import Store, atomic_json, file_sha
from .attempts import attempts, completed_attempts, next_attempt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=Path, required=True)
    parser.add_argument('--scope', choices=('pilot', 'full'), required=True)
    parser.add_argument('--max-wall-seconds', type=int, default=2400)
    args = parser.parse_args()
    if not 60 <= args.max_wall_seconds <= 3000:
        raise ValueError('Window duration must be 60–3000 seconds')
    strategy = json.loads(args.strategy.read_text())
    run = Path(strategy['new_run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    if manifest['strategy_sha256'] != file_sha(args.strategy):
        raise ValueError('Frozen strategy changed')
    for path, sha in strategy['implementation_sha256'].items():
        if file_sha(path) != sha:
            raise ValueError('Frozen implementation changed: ' + path)
    cases = (strategy['campaign']['pilot_cases'] if args.scope == 'pilot'
             else strategy['cases'])
    if len(cases) != len(set(cases)):
        raise ValueError('Duplicate case IDs')
    workers = strategy['campaign']['max_case_concurrency']
    if not isinstance(workers, int) or not 1 <= workers <= 10:
        raise ValueError('Invalid frozen case concurrency')
    store = Store(run)
    store.validate_events()
    if any(v['status'] == 'started' for v in store.read()['reservations'].values()):
        raise RuntimeError('Unsettled operations exist; inspect live processes before resuming')
    store.set_status('running_' + args.scope,
                     'Frozen ten-worker case scheduling; each episode remains independent')
    deadline = time.monotonic() + args.max_wall_seconds
    lock = threading.Lock()
    results = []

    def progress(stage: str) -> dict:
        with lock:
            state = store.read()
            completed = [r for r in results if 'native' in r]
            by_family = Counter(r['case_id'].split('_ep', 1)[0]
                                for r in completed if r['accepted'])
            full_complete = args.scope == 'full' and len(completed) == len(cases)
            summary = {
                'schema': 'rssr.event150.concurrent10_progress.v1',
                'scope': args.scope, 'strategy_seq': strategy['seq'],
                'stage': stage, 'expected_cases': len(cases),
                'processed': len(results), 'native_complete': len(completed),
                'accepted': sum(bool(r['accepted']) for r in completed),
                'accepted_by_family': {f: by_family[f] for f in
                                       ('drawer', 'stack_blocks', 'stack_cups')},
                'errors': sum(r.get('status') == 'execution_error' for r in results),
                'case_results': [{k: v for k, v in r.items()
                                  if k in ('case_id', 'accepted', 'status',
                                           'event_decisions', 'attempt')}
                                 for r in results],
                'spent': state['spent'],
                'unsettled': [k for k, v in state['reservations'].items()
                              if v['status'] == 'started'],
                'full150_completed': full_complete,
                'cohort_completed': full_complete,
                'goal_achieved': (full_complete and
                                  all(by_family[f] >= required for f, required in
                                      strategy['campaign']['minimum_goal_successes_per_family'].items()) and
                                  not any(r.get('status') == 'execution_error'
                                          for r in results)),
                'concurrency': workers,
                'experience_visibility': 'completed same-structure samples at case start',
            }
            atomic_json(run / f'{args.scope}_progress.json', summary)
            print(json.dumps({k: summary[k] for k in
                              ('stage', 'processed', 'native_complete',
                               'accepted', 'accepted_by_family', 'spent')},
                             ensure_ascii=False), flush=True)
            return summary

    progress('running')

    def run_one(cid: str) -> dict:
        complete = completed_attempts(run, cid)
        if len(complete) > 1:
            raise ValueError('Multiple scored physical attempts: ' + cid)
        if complete:
            return {**json.loads((complete[0] / 'receipt.json').read_text()),
                    'attempt': complete[0].name}
        failures = [path / 'failure.json' for path in attempts(run, cid)
                    if (path / 'failure.json').is_file()]
        if failures:
            return json.loads(failures[-1].read_text())
        if time.monotonic() >= deadline:
            return {'case_id': cid, 'accepted': False, 'status': 'deferred'}
        remaining = store.case_remaining(cid)
        if (remaining['full'] is not None and remaining['full'] < 1 or
                remaining['api'] is not None and remaining['api'] < 1):
            return {'case_id': cid, 'accepted': False,
                    'status': 'case_budget_exhausted'}
        output = next_attempt(run, cid)
        log = run / 'logs' / f'{cid}_{output.name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('x') as stream:
            code = subprocess.run([
                sys.executable, '-m', strategy['executor_module'] + '.run_case',
                '--strategy', str(args.strategy), '--case-id', cid,
                '--output', str(output)], cwd=Path(__file__).resolve().parents[2],
                stdout=stream, stderr=subprocess.STDOUT).returncode
        complete = completed_attempts(run, cid)
        if code == 0 and len(complete) == 1:
            return {**json.loads((complete[0] / 'receipt.json').read_text()),
                    'attempt': complete[0].name}
        result = {'case_id': cid, 'accepted': False, 'status': 'execution_error',
                  'attempt': output.name, 'returncode': code, 'log': str(log)}
        atomic_json(output / 'failure.json', result)
        return result

    # Flat scheduling permits up to ten independent Sim2 episodes. A case
    # reads only samples committed before its own start; the profile lock
    # serializes concurrent completion without imposing a task-family recipe.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(run_one, cid) for cid in cases]):
            result = future.result()
            with lock:
                results.append(result)
            progress('running')
    store.validate_events()
    final = progress('window_complete')
    atomic_json(run / f'{args.scope}_summary.json', final)
    store.set_status('window_complete_' + args.scope)


if __name__ == '__main__':
    main()
