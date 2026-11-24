"""duo_drawer_v2d.py (this directory) with both arms kept off the camera arch by their redundancy (../code/arch_away.py,
2026-11-05): after an IK solve, the joints move in the null space of the hand pose away from the arch (the handle hand
along every line, the block hand at its key poses, keeping its posture along lines). The task sequence, targets and every check are duo_drawer.py's."""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import arch_away  # noqa: E402

_spec = importlib.util.spec_from_file_location('duo_drawer_v2d_base', str(HERE / 'duo_drawer_v2d.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)
globals().update({k: v for k, v in vars(base).items() if not k.startswith('__') and k != 'run'})


def run(S):
    if 'OPTIONS' in globals():
        base.OPTIONS = globals()['OPTIONS']
    _, bh, dh = base.roles(S)
    arch_away.install(S, dh)  # the handle hand: key poses and every line
    # the block hand: key poses relaxed; along lines it keeps its posture (its way into the drawer is tight, and pulled
    # back toward each line's start posture the factory-mounted wrist cannot hold the block's orientation on the way in)
    # the block hand: key poses pushed until it holds the block (pushed key poses while carrying swung the arm and let
    # the block twist in the grip); along lines it keeps its posture, pushed only on the way in from the side
    # (duo_drawer.py switches line_mode there)
    arch_away.install(S, bh, lines='posture', holding=False)
    out = base.run(S)
    S.note('arch_away: ' + '; '.join(f'{h} {S.arch_away_all[h].stats}' for h in S.arch_away_all))
    return out
