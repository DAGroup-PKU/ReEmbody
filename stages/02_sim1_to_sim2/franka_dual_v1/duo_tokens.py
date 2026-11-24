"""LLM tokens the coding agent (Claude Code, claude-opus-5-5) spent writing and revising the FR3 Duo controllers.

Read from the Claude Code session transcript (one JSON line per message; assistant messages carry `usage`; a message
streamed in several lines is counted once, by message id). From --since on, every assistant message is attributed to
the tasks whose controller or results its tool calls touch (by file / folder name; split evenly when several),
to 'shared' when they touch only framework files or nothing task-specific. Per revision round, the tokens between the end of the previous round of
that task (or --since) and the launch of the round are summed. Tasks were developed interleaved, so the per-round
sums are by attribution, not by wall-clock slices.

Reported per message group: output tokens (generated, thinking included) and input tokens (uncached + cache writes +
cache reads, i.e. all context processed).

    python -m franka_dual_v1.duo_tokens --transcript <jsonl> --since 2026-11-03T12:30:00Z --results <controllers dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

KEYS = {'stack_cups': ('duo_stack_cups', '/stack_cups/', "'stack_cups'", 'stack_cups_dev', 'task stack_cups'),
        'stack_blocks': ('duo_stack_blocks', '/stack_blocks/', "'stack_blocks'", 'stack_blocks_dev', 'task stack_blocks'),
        'drawer': ('duo_drawer', '/drawer/', "'drawer_ep", 'drawer_dev', 'task drawer', 'bar_sweep')}


def messages(path: Path, since: str):
    last = {}
    order = []
    with path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get('type') != 'assistant' or d.get('timestamp', '') < since:
                continue
            msg = d['message']
            mid = msg.get('id') or d.get('uuid')
            if mid not in last:
                order.append(mid)
            text = json.dumps([c.get('input') for c in msg.get('content', []) if c.get('type') == 'tool_use'])
            prev = last.get(mid)
            last[mid] = dict(t=d['timestamp'], usage=msg.get('usage') or {},
                             text=(prev['text'] if prev else '') + text)
    return [last[m] for m in order]


def tasks_of(text: str) -> list:
    """Tasks a message's tool calls touch; a message touching several is split evenly among them."""
    return [t for t, keys in KEYS.items() if any(k in text for k in keys)] or ['shared']


def tokens(u):
    inp = int(u.get('input_tokens', 0)) + int(u.get('cache_creation_input_tokens', 0)) + \
        int(u.get('cache_read_input_tokens', 0))
    return dict(output=int(u.get('output_tokens', 0)), input=inp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transcript', type=Path, required=True)
    ap.add_argument('--since', required=True, help='UTC ISO time the controller work started')
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    msgs = messages(a.transcript, a.since)
    for m in msgs:
        m['tasks'] = tasks_of(m['text'])
        m['tok'] = tokens(m['usage'])

    def share(m, task):
        return 1.0 / len(m['tasks']) if task in m['tasks'] else 0.0
    out = dict(schema='rssr.fr3_duo.tokens.v1', transcript=str(a.transcript), since=a.since,
               model='claude-opus-5-5', n_messages=len(msgs), totals={}, rounds={})
    for k in ('stack_cups', 'stack_blocks', 'drawer', 'shared'):
        out['totals'][k] = dict(messages=round(sum(share(m, k) for m in msgs), 1),
                                output=round(sum(share(m, k) * m['tok']['output'] for m in msgs)),
                                input=round(sum(share(m, k) * m['tok']['input'] for m in msgs)))
    out['totals']['all'] = dict(messages=len(msgs), output=sum(m['tok']['output'] for m in msgs),
                                input=sum(m['tok']['input'] for m in msgs))
    for task in ('stack_cups', 'stack_blocks', 'drawer'):
        hist = a.results / task / 'history.json'
        if not hist.exists():
            continue
        rows, prev = [], a.since
        for h in json.loads(hist.read_text()):
            start = h.get('started_utc')
            if not start:
                rows.append(dict(round=h['round'], note='no start time recorded'))
                continue
            sel = [m for m in msgs if prev <= m['t'] < start]
            rows.append(dict(round=h['round'], interval=[prev, start],
                             messages=round(sum(share(m, task) for m in sel), 1),
                             output=round(sum(share(m, task) * m['tok']['output'] for m in sel)),
                             input=round(sum(share(m, task) * m['tok']['input'] for m in sel))))
            prev = h.get('finished_utc') or start
        out['rounds'][task] = rows
    s = json.dumps(out, indent=1) + '\n'
    if a.out:
        a.out.write_text(s)
    print(s)


if __name__ == '__main__':
    main()
