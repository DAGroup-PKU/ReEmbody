"""Complete source clocks, including terminal states near a video tick."""
from __future__ import annotations

import numpy as np

from .common import FPS, sample_clock


def complete_source_clock(state_times, command_times, duration, fps=FPS):
    video, states, commands, queries = sample_clock(state_times, command_times, duration, fps=fps)
    if states[-1] == len(state_times) - 1:
        return video, states, commands, queries

    # Floating point accumulation can put the final simulation state just
    # beyond the last video tick. Preserve every preceding sample and append
    # the next regular video tick with the true terminal state and its action.
    # This avoids assigning a later release to an earlier picture.
    if float(state_times[-1]) != float(duration):
        raise ValueError('Complete source duration must equal the terminal physical time')
    next_video = len(video) / fps
    if next_video < duration:
        raise ValueError('Complete source clock cannot reach its terminal physical state')
    command_times = np.asarray(command_times, dtype=np.float64)
    terminal_command = int(np.searchsorted(command_times, duration + 1e-10, side='right') - 1)
    if terminal_command < 0:
        raise ValueError('Terminal physical state has no causal action')
    gap = duration - command_times[terminal_command]
    if gap < -1e-10 or gap > np.max(np.diff(command_times)) + 1e-8:
        raise ValueError('Terminal physical state lost causal action alignment')
    return (np.append(video, next_video), np.append(states, len(state_times) - 1),
            np.append(commands, terminal_command), np.append(queries, duration))
