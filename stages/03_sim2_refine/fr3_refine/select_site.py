"""Select the first N admitted episodes of a site synthesis run, in variation-index order (S1 v8/v9 select 100 the
same way), for render_fleet.py. Writes selection.json: per episode its index, variation, run directory, frame
count and the sha256 of its record and episode, plus the run's controller and framework hashes.

    python select_site.py --round <aug run dir> --n 100 --out <selection.json>
    python select_site.py --round <running aug dir> --n 100 --prefix --out <partial.json>

--prefix (while the run is still going): only the variations before the first one without a record are considered;
their admitted episodes are the first ones of the final selection, with the same indices. On a refine run
(sim2-refine/tasks/fr3_stack_cups/refine.py) admitted means refined (its round.json rows likewise).
v3: --exclude <json>: variations left out (posture_filter.py: joints at the grasp off the hand's main posture).
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=Path, required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--prefix', action='store_true')
    ap.add_argument('--exclude', type=Path)
    a = ap.parse_args()
    order = [v['id'] for v in json.loads((a.round / 'variations.json').read_text())]
    if a.prefix:
        rnd = dict(controller_sha256=hashlib.sha256((a.round / 'controller.py').read_bytes()).hexdigest(), framework_sha256=None)
        rows = {}
        for vid in order:
            f = a.round / 'runs' / vid / 'record.json'
            if not f.exists():
                break
            ok = json.loads(f.read_text())['admitted']
            if (f.parent / 'refine.json').exists():  # a refine run: only refined episodes (not Sim2 rollbacks)
                ok = ok and json.loads((f.parent / 'refine.json').read_text())['refined']
            rows[vid] = dict(admitted=ok)
        order = order[:len(rows)]
    else:
        rnd = json.loads((a.round / 'round.json').read_text())
        rows = {r['variation']: r for r in rnd['runs']}
    eps = []
    skip = set(json.loads(a.exclude.read_text())['excluded']) if a.exclude else set()
    for vid in order:
        if not rows[vid]['admitted'] or vid in skip:
            continue
        run = a.round / 'runs' / vid
        rec = json.loads((run / 'record.json').read_text())
        assert rec['admitted'] and rec['measurements']['replay']['bit_exact']
        eps.append(dict(episode_index=len(eps), variation=vid, run=str(run), duration_s=rec['measurements']['duration_s'],
                        record_sha256=sha(run / 'record.json'), episode_sha256=sha(run / 'episode.npz')))
        if len(eps) == a.n:
            break
    if len(eps) < a.n and not a.prefix:
        raise SystemExit(f'only {len(eps)} admitted episodes in {a.round}')
    last = order.index(eps[-1]['variation']) if len(eps) == a.n else len(order) - 1
    out = dict(schema='rssr.fr3_duo.site_selection.v1', round=str(a.round), controller_sha256=rnd['controller_sha256'],
               framework_sha256=rnd['framework_sha256'], rule=f'first {a.n} admitted in variation-index order',
               variations_considered=last + 1, admitted_among_considered=len(eps), partial=bool(a.prefix and len(eps) < a.n),
               episodes=eps)
    a.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('rule', 'variations_considered', 'admitted_among_considered')}))


if __name__ == '__main__':
    main()
