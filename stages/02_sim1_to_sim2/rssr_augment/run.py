"""Safe-by-default branch entry: initialize, plan, execute or resume explicitly."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

# Works both as a module and by absolute server path.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from rssr_augment.common import BRANCH, MODULE, PYTHON, read, pin, within_branch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init', help='Create a fresh isolated run without experiments')
    init.add_argument('--name', required=True)
    init.add_argument('--mode', choices=['cold_start', 'warm_start'], required=True)
    init.add_argument('--no-prior-pilot', action='store_true')
    init.add_argument('--physical-index', type=int)
    for name in ['sim2', 'check']:
        p = sub.add_parser(name)
        p.add_argument('--run', type=Path, required=True)
        p.add_argument('--case-id')
        if name == 'sim2':
            p.add_argument('--execute', action='store_true')
            p.add_argument('--window-seconds', type=int, default=2400)
    augment = sub.add_parser('plan-augmentation')
    augment.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'init':
        from rssr_augment.bootstrap import initialize
        from rssr_augment.physical import sample_profile
        cfg = read(BRANCH / 'config/pipeline.json')
        physical = None
        if args.physical_index is not None:
            if not 0 <= args.physical_index < cfg['sim2_augmentation']['variants_per_demo']:
                parser.error('Physical index outside configured variants')
            from rssr_augment.common import BASE_RUN
            from rssr_augment.bootstrap import selected_cases
            cases = selected_cases(cfg, read(BASE_RUN / 'manifest.json')['cases'])
            physical = {'profiles': {cid: sample_profile(cid, args.physical_index, cfg['sim2_augmentation'])
                                     for cid in cases}}
        run = initialize(args.name, args.mode, not args.no_prior_pilot, physical, cfg)
        print(json.dumps({'run': str(run), 'knowledge': read(run / 'knowledge_inventory.json'),
                          'api_requests': 0, 'physical_evaluations': 0}, ensure_ascii=False))
        return
    run = within_branch(args.run)
    strategy = read(run / 'strategy.json')
    if args.command == 'plan-augmentation':
        from rssr_augment.physical import sample_profile
        settings = strategy['lyc_dev_v2']['config']['sim2_augmentation']
        jobs = [sample_profile(cid, i, settings) for cid in strategy['cases']
                for i in range(settings['variants_per_demo'])]
        path = run / 'augmentation_plan.json'
        pin(path, {'schema': 'rssr.lyc_dev_v2.augmentation_plan.v1', 'source_run': str(run),
                   'profiles': jobs, 'count': len(jobs), 'executed': 0,
                   'source_demo_split_before_augmentation': True,
                   'physical_success_cannot_be_inherited': True})
        print(json.dumps({'plan': str(path), 'count': len(jobs), 'executed': 0}))
        return
    from rssr_augment.bootstrap import validate_startup
    case = args.case_id or strategy['cases'][0]
    validate_startup(run / 'strategy.json', case, allow_legacy_snapshot=args.command == 'check')
    if args.command == 'check':
        print(json.dumps({'startup_valid': True, 'run': str(run), 'case_id': case,
                          'knowledge': read(run / 'knowledge_inventory.json'), 'api_requests': 0,
                          'physical_evaluations': 0}, ensure_ascii=False))
        return
    command = [str(PYTHON), '-u', '-m', MODULE + ('.run_case' if args.case_id else '.run_campaign'),
               '--strategy', str(run / 'strategy.json')]
    if args.case_id:
        from rssr_augment.retarget.attempts import completed_attempts, next_attempt
        completed = completed_attempts(run, args.case_id)
        if completed:
            print(json.dumps({'already_completed': True, 'receipt': str(completed[0] / 'receipt.json')}))
            return
        command += ['--case-id', args.case_id, '--output', str(next_attempt(run, args.case_id))]
    else:
        command += ['--scope', 'full', '--max-wall-seconds', str(args.window_seconds)]
    print(json.dumps({'command': command, 'execute': args.execute}, ensure_ascii=False), flush=True)
    if args.execute:
        from rssr_augment.shared.family_autoresearch.store import Store
        Store(run).validate_events()
        if any(r['status'] == 'started' for r in Store(run).read()['reservations'].values()):
            raise RuntimeError('Unsettled work exists: preserve cost and inspect before resuming')
        with (run / 'launcher.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
