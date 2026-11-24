"""Read-only integration checks; temporary outputs only, no physics or models."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sim2_refine_cli', ROOT / 'run.py')
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def rejects(call, message):
    try:
        call()
    except (ValueError, FileExistsError):
        return
    raise AssertionError(message)


def main():
    rows = [json.loads(line) for line in
            (ROOT / 'controllers/published_selection.jsonl').read_text().splitlines()]
    versions = {}
    for row in rows:
        versions.setdefault(row['controller_version'], row)
    before = {row['strategy']['path']: api.sha(row['strategy']['path'])
              for row in versions.values()}
    with tempfile.TemporaryDirectory(prefix='rssr_refine_entrypoint_') as temporary:
        work = Path(temporary)
        for version, row in versions.items():
            args = argparse.Namespace(episode_index=row['episode_index'], strategy=None, controller=None)
            strategy, selected, _ = api.prepare_strategy(args)
            assert selected == version
            assert strategy['sim2_refine_migration']['source_implementation_sha256'] == \
                api.read(row['strategy']['path'])['implementation_sha256']
            assert not any('/work/v3_human_align/' in p for p in strategy['implementation_sha256'])
            assert any('/validation/_shared/' in p for p in strategy['implementation_sha256'])
            api.verify(strategy['implementation_sha256'])
        args = argparse.Namespace(episode_index=0, strategy=None, controller=None,
            case_id=None, output=work / 'candidate', python=api.DEFAULT_PYTHON,
            implementation=api.DEFAULT_IMPLEMENTATION, dry_run=True)
        assert api.run_case(args)['controller'] == 'v4'
        assert not args.output.exists()
        args.output.mkdir()
        rejects(lambda: api.run_case(args), 'Existing experiment output was accepted')
        jobs = work / 'jobs.json'
        api.write_new(jobs, {'jobs': [dict(job_id='episode0', episode_index=0),
                                    dict(job_id='episode54', episode_index=54)]})
        args = argparse.Namespace(jobs=jobs, output=work / 'batch', workers=10,
            python=api.DEFAULT_PYTHON, implementation=api.DEFAULT_IMPLEMENTATION, dry_run=True)
        assert len(api.batch(args)['jobs']) == 2 and not args.output.exists()
        # Use a real published admission, including all native and quality evidence.
        receipt = Path(rows[0]['admission']['original_path'])
        args = argparse.Namespace(receipt=[receipt], output=work / 'selected.json', expected_cases=1)
        assert api.merge(args)['complete'] is True
        rejects(lambda: api.merge(args), 'Existing selection was overwritten')
        args.output = work / 'duplicate.json'
        args.receipt = [receipt, receipt]
        rejects(lambda: api.merge(args), 'Duplicate logical sources were accepted')
        rejected = work / 'unadmitted.json'
        native = api.read(receipt)
        native['checks']['native_full_task_accepted'] = False
        api.write_new(rejected, native)
        args.receipt = [rejected]
        rejects(lambda: api.merge(args), 'Rejected trajectory was admitted')
        changed = work / 'changed_evidence.json'
        native = api.read(receipt)
        native['publication_artifacts']['strategy.json']['sha256'] = '0' * 64
        api.write_new(changed, native)
        args.receipt = [changed]
        rejects(lambda: api.merge(args), 'Changed publication artifact was accepted')
        assert not args.output.exists()
    assert before == {path: api.sha(path) for path in before}
    print(json.dumps(dict(passed=True, versions=list(versions), source_strategies_unchanged=True,
        tests=['pin_migration_all_versions', 'single_candidate_dry_run', 'batch_dry_run',
               'real_admission_selection', 'overwrite_refusal', 'duplicate_source_refusal',
               'failed_gate_refusal', 'changed_evidence_refusal'], physics_started=False)))


if __name__ == '__main__':
    main()
