"""Select a completed physical attempt without discarding interrupted attempts.

This helper is used by the generic concurrent campaign runner. It does not
change the retarget agent, RuleAgent, controller, or native judge.
"""
from __future__ import annotations

import re
from pathlib import Path

ATTEMPT_NAME = re.compile(r'attempt_(\d{2})$')


def attempts(run: Path, case_id: str) -> list[Path]:
    root = Path(run) / 'cases' / case_id
    rows = [(int(match.group(1)), path) for path in root.glob('attempt_*')
            if path.is_dir() and (match := ATTEMPT_NAME.fullmatch(path.name))]
    return [path for _, path in sorted(rows)]


def completed_attempts(run: Path, case_id: str) -> list[Path]:
    return [path for path in attempts(run, case_id)
            if (path / 'receipt.json').is_file() and
            (path / 'scene_profile_sample_link.json').is_file()]


def selected_attempt(run: Path, case_id: str) -> Path:
    complete = completed_attempts(run, case_id)
    if len(complete) > 1:
        raise ValueError('Multiple scored physical attempts: ' + case_id)
    return complete[0] if complete else Path(run) / 'cases' / case_id / 'attempt_00'


def next_attempt(run: Path, case_id: str) -> Path:
    existing = attempts(run, case_id)
    index = 0 if not existing else int(existing[-1].name.split('_')[1]) + 1
    if index > 99:
        raise ValueError('Too many physical attempts: ' + case_id)
    return Path(run) / 'cases' / case_id / f'attempt_{index:02d}'
