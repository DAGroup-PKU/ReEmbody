"""duo_stack_blocks_v2.py (this directory) with both arms kept off the camera arch by their redundancy (../code/arch_away.py,
2026-11-05): after every IK solve, the joints move in the null space of the hand pose away from the arch. The task
sequence, targets and every check are duo_stack_blocks_v2.py's."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import arch_away  # noqa: E402

_spec = importlib.util.spec_from_file_location('duo_stack_blocks_v2_base', str(HERE / 'duo_stack_blocks_v2.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
globals().update({k: v for k, v in vars(base).items() if not k.startswith('__') and k != 'run'})


def run(S):
    if 'OPTIONS' in globals():
        base.OPTIONS = globals()['OPTIONS']
    for hand in ('left', 'right'):
        arch_away.install(S, hand)
    out = base.run(S)
    S.note('arch_away: ' + '; '.join(f'{h} {S.arch_away_all[h].stats}' for h in S.arch_away_all))
    return out
