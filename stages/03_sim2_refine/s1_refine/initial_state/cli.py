#!/usr/bin/env python3
"""Use the pinned v3 registry unless an explicit registry is selected."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    # Pass all restore options, including --help, to the byte-identical helper.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--registry-dir', type=Path)
    selected, remaining = parser.parse_known_args()
    pointer = json.loads((HERE / 'registry_pointer.json').read_text())
    helper = HERE / pointer['restore_helper_file']
    if sha(helper) != pointer['restore_helper_sha256']:
        raise RuntimeError('Pinned restore helper changed: ' + str(helper))
    registry = selected.registry_dir
    if registry is None:
        registry = Path(pointer['registry_dir'])
        for field in ('manifest', 'cases', 'record_completion'):
            path = registry / pointer[field + '_file']
            if sha(path) != pointer[field + '_sha256']:
                raise RuntimeError('Default registry pin changed: ' + str(path))
    arguments = [sys.executable, str(helper), '--registry-dir', str(registry), *remaining]
    os.execv(sys.executable, arguments)


if __name__ == '__main__':
    main()
