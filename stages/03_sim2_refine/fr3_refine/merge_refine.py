"""Merge refine runs of one task made in batches over consecutive initial conditions of the same sampler stream (v2
drawer: draws 0-219 on worker A, 220-319 on worker B) into one refine directory in the duo_rounds layout, so that
select_site.py picks the first 100 refined in draw order exactly as for one batch. The batches must share the refine
controller (sha256). Runs are copied (record.json, episode.npz, scene.xml, refine.json); nothing is re-run.

    python merge_refine.py --out <fresh dir> <refine dir> [<refine dir> ...]
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('dirs', type=Path, nargs='+')
    a = ap.parse_args()
    a.out.mkdir(parents=True)
    ctl = {sha(d / 'controller.py') for d in a.dirs}
    assert len(ctl) == 1, f'different refine controllers: {ctl}'
    rounds = [json.loads((d / 'round.json').read_text()) for d in a.dirs]
    order, rows = [], []
    for d, rnd in zip(a.dirs, rounds):
        vs = json.loads((d / 'variations.json').read_text())
        order += vs
        rows += rnd['runs']
        for v in vs:
            src, dst = d / 'runs' / v['id'], a.out / 'runs' / v['id']
            dst.mkdir(parents=True)
            for f in ('record.json', 'episode.npz', 'scene.xml', 'refine.json'):
                shutil.copy2(src / f, dst / f)
    ids = [v['id'] for v in order]
    assert len(set(ids)) == len(ids) and ids == sorted(ids), 'draws must be distinct and in order'
    shutil.copy2(a.dirs[0] / 'controller.py', a.out / 'controller.py')
    (a.out / 'variations.json').write_text(json.dumps(order, indent=1) + '\n')
    n_ok = sum(r['admitted'] for r in rows)
    merged = dict(rounds[0], split=' + '.join(r['split'] for r in rounds), n=len(rows), admitted=n_ok,
                  admitted_fraction=round(n_ok / len(rows), 4), runs=rows, merged_from=[str(d) for d in a.dirs],
                  framework_sha256_per_batch=[r['framework_sha256'] for r in rounds])
    (a.out / 'round.json').write_text(json.dumps(merged, indent=1) + '\n')
    print(json.dumps(dict(batches=len(a.dirs), runs=len(rows), refined=n_ok)))


if __name__ == '__main__':
    main()
