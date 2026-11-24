"""Independent check of a render_fleet output against its selection (run after the fleet).

Per episode: the selected run's record/episode sha256 are those the selection pinned; the render source was exported
from that run (source.json input pins); the receipt covers every frame of the source clock at 224x224 with 12 samples;
each video's sha256 equals the receipt's and it decodes to exactly that many frames; the config the render used is
the one in the episode folder. Writes verification.json.

    python verify_renders.py --selection <selection.json> --output <fleet output dir>
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import os as _os, shutil as _shutil
FFMPEG = _os.environ.get('REEMBODY_FFMPEG') or _os.environ.get('RSSR_FFMPEG') or _shutil.which('ffmpeg') or 'ffmpeg'
VIDEOS = ('head.mp4', 'left_wrist.mp4', 'right_wrist.mp4')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frames(p):
    r = subprocess.run([FFMPEG, '-v', 'error', '-i', str(p), '-map', '0:v:0', '-f', 'framemd5', '-'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-300:]
    return sum(1 for line in r.stdout.splitlines() if line and not line.startswith('#'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    sel = json.loads(a.selection.read_text())
    rows, total = [], 0
    for e in sel['episodes']:
        w = a.output / f"ep{e['episode_index']:03d}"
        run = Path(e['run'])
        src = json.loads((w / 'source/source.json').read_text())
        rc = json.loads((w / 'render/render_receipt.json').read_text())
        n = src['frames']
        chk = dict(
            run_pins=sha(run / 'record.json') == e['record_sha256'] and sha(run / 'episode.npz') == e['episode_sha256'],
            source_from_run=src['inputs'][str(run / 'record.json')] == e['record_sha256']
            and src['inputs'][str(run / 'episode.npz')] == e['episode_sha256'] and src['variation'] == e['variation'],
            source_artifacts=all(sha(w / 'source' / f) == h for f, h in src['artifacts'].items()),
            receipt_clock=rc['frames'] == list(range(n)) and rc['resolution'] == [224, 224] and rc['raw_resolution'] == [224, 168]
            and rc['samples'] == 12 and rc['variation'] == e['variation'],
            videos_sha=all(sha(w / 'render' / v) == rc['artifacts'][v] for v in VIDEOS),
            videos_decode=all(frames(w / 'render' / v) == n for v in VIDEOS),
            config_pinned=any(h == sha(w / 'config.json') for p, h in rc['input_pins'].items() if p.endswith('config.json')))
        rows.append(dict(episode_index=e['episode_index'], variation=e['variation'], frames=n, ok=all(chk.values()),
                         device=json.loads((w / 'render_device.json').read_text()), seed=rc['lighting_recipe']['seed'],
                         added_lights=len(rc['lighting_recipe']['added_lights']), render_s=round(rc['elapsed_s']), **chk))
        total += n
    out = dict(schema='rssr.fr3_duo.site_render_verification.v1', selection=str(a.selection), episodes=len(rows),
               all_ok=all(r['ok'] for r in rows), frames_per_camera=total, rows=rows)
    (a.output / 'verification.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('episodes', 'all_ok', 'frames_per_camera')}))
    for r in rows:
        if not r['ok']:
            print('FAIL', r)


if __name__ == '__main__':
    main()
