"""Owned process stages extracted from the frozen outer RefineAgent.

Receipt/recovery rules are preserved; the working directory is this project.
No stage is started merely by importing this module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Keep the historical process-owner marker and receipt structure.
SCHEMA = 'rssr.short_refine.outer_agent.v4'


def now():
    return datetime.now(timezone.utc).isoformat()



def read(path):
    return json.loads(Path(path).read_text())



def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)



def verify_pins(values):
    for path, expected in values.items():
        if not Path(path).is_file() or sha(path) != expected:
            raise RuntimeError('Frozen input or implementation changed: '+path)



def identity(pid):
    try:
        proc = Path('/proc')/str(pid)
        # comm may contain spaces or parentheses; fields after its last ')'
        # begin at stat field 3. starttime is stat field 22.
        stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=int(pid), start_ticks=stat[19],
                    command_sha256=hashlib.sha256((proc/'cmdline').read_bytes()).hexdigest())
    except (OSError, IndexError, ValueError):
        return None



def is_same_process(saved):
    return saved is not None and identity(saved['pid']) == saved



def environment(implementation):
    result = dict(os.environ, PYTHONPATH=str(implementation), OPENBLAS_NUM_THREADS='1',
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', MUJOCO_GL='egl', MALLOC_ARENA_MAX='2')
    # The candidate controller does not call an LLM or use API worker slots.
    return result



def run_stage(args, stage, command, required_output, frozen):
    """Run or adopt exactly this stage; never overwrite partial native output."""
    stage = Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    receipt_path, process_path = stage/'process_receipt.json', stage/'process.json'
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt['command'] != command:
            raise RuntimeError('Completed stage command changed: '+str(stage))
        if not Path(required_output).is_file() or sha(required_output) != receipt.get('output_sha256'):
            raise RuntimeError('Completed stage artifact missing or changed: '+str(required_output))
        verify_pins(frozen)
        return receipt
    verify_pins(frozen)
    if process_path.exists():
        process = read(process_path)
        if process['command'] != command:
            raise RuntimeError('Existing stage belongs to another command: '+str(stage))
        while is_same_process(process.get('identity')):
            time.sleep(1)
        # A crash may occur after child completion but before the parent writes
        # its exit status. Recover only from the required complete artifact.
        if not Path(required_output).is_file():
            raise RuntimeError('Interrupted stage has no complete artifact; preserve it for inspection: '+str(stage))
        receipt = dict(command=command, returncode=None, recovered_from_complete_artifact=True,
                       completed=now(), required_output=str(required_output), output_sha256=sha(required_output))
        atomic(receipt_path, receipt)
        return receipt
    # An existing native output without owned process metadata must not be
    # mistaken for this proposed policy.
    if Path(required_output).exists():
        raise RuntimeError('Unowned complete stage output exists: '+str(required_output))
    with (stage/'stdout.log').open('ab') as stream:
        proc = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment(args.implementation),
                                stdout=stream, stderr=subprocess.STDOUT)
        atomic(process_path, dict(command=command, identity=identity(proc.pid), started=now(),
                                  owner=SCHEMA, parent_pid=os.getpid()))
        returncode = proc.wait()
    receipt = dict(command=command, returncode=returncode, recovered_from_complete_artifact=False,
                   completed=now(), required_output=str(required_output),
                   output_sha256=sha(required_output) if Path(required_output).is_file() else None)
    atomic(receipt_path, receipt)
    verify_pins(frozen)
    if returncode != 0 or not Path(required_output).is_file():
        raise RuntimeError(f'{stage.name} process failed ({returncode}); inspect {stage}/stdout.log')
    return receipt



