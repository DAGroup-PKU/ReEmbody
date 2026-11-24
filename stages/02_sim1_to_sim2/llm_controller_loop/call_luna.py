#!/usr/bin/env python3
"""Run exactly one persisted Luna request from a prepared local JSON context.

Context may be a plain evidence object, or an envelope containing `evidence`,
`prompt`, and `strategy_schema`. The latter allows the execution adapter to
provide its exact code/parameter contract. It never sends credential contents.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from luna_api import MODEL, loads, request_strategy


DEFAULT_PROMPT = """You are the actual GPT-6 Luna trajectory optimization agent.
Use the supplied source trajectory, task, execution interface, measured scene
geometry and actual trial feedback to author the next COMPLETE refinement
strategy or complete executable controller. Your exact API output is retained;
the host must not replace your choices with hand-authored motion logic.

The user's full campaign covers every episode of all three source tasks. This
request covers only the task selected in the execution contract and its exact
bound source variations; do not require the other task agents to finish first.
At the beginning of each grasp action rotate the wrist camera to look at the
object being grasped. For cups and blocks, then move above that object and
perform a top-down grasp. The user explicitly permits the horizontal drawer
handle to be approached and grasped from the FRONT according to its structure;
the block in the drawer task still requires an overhead and top-down grasp.
During motion the wrist camera must continuously face its grasp target.
The final physical action duration must be at most 20 seconds. Follow the
provided camera calibration and actual controllable robot degrees of freedom;
do not assume an independent virtual camera or simply edit video extrinsics.
Preserve the original task outcome, scene/physics, source identity and all
provided physical/admission constraints. Use actual simulated execution and
independent replay feedback when supplied. Never claim unmeasured success.
Do not implement duration compliance by cropping, timestamp relabeling, hiding
settling time, or speeding up a failed old trajectory without re-execution.

Respect the exact execution API and schema supplied by the host. If required
measurements or capabilities are missing, report the specific missing items;
do not fabricate them. Return exactly one propose_trajectory function call.
"""

DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "schema": {"type": "string", "enum": ["rssr.trajectory.luna.strategy.v1"]},
        "strategy_kind": {"type": "string", "enum": ["controller_python", "parameter_schedule", "missing_capability"]},
        "rationale": {"type": "string"},
        "python_code": {"type": ["string", "null"], "description": "Complete controller module, unmodified from your own output; null when a schedule is required instead."},
        "strategy_json": {"type": ["string", "null"], "description": "Complete JSON strategy settings encoded as a string for the supplied executor; null for controller code."},
        "expected_physical_duration_s": {"type": ["number", "null"], "minimum": 0, "maximum": 20},
        "constraint_reasoning": {"type": "array", "items": {"type": "string"}},
        "missing_capability": {"type": ["string", "null"]},
    },
    "required": ["schema", "strategy_kind", "rationale", "python_code", "strategy_json",
                 "expected_physical_duration_s", "constraint_reasoning", "missing_capability"],
    "additionalProperties": False,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="Prepared local JSON evidence or envelope")
    parser.add_argument("--output", type=Path, required=True, help="Fresh request directory; must not exist")
    parser.add_argument("--effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=16000)
    parser.add_argument("--profile-source", choices=("franka-latest", "legacy"), default="franka-latest",
                        help="Explicit existing credential route; no automatic fallback (default: franka-latest)")
    args = parser.parse_args()
    context = loads(args.context.read_text())
    if not isinstance(context, dict):
        raise ValueError("Context must be a JSON object")
    evidence = context.get("evidence", context)
    if not isinstance(evidence, dict):
        raise ValueError("Evidence must be a JSON object")
    prompt = DEFAULT_PROMPT
    if context.get("prompt"):
        if not isinstance(context["prompt"], str):
            raise ValueError("Additional prompt must be a string")
        prompt += "\n\nExecution contract and requested output:\n" + context["prompt"]
    schema = context.get("strategy_schema", DEFAULT_SCHEMA)
    if not isinstance(schema, dict):
        raise ValueError("Strategy schema must be an object")
    if args.output.exists():
        raise FileExistsError("Refusing to resubmit or overwrite an existing output directory")
    request_strategy(prompt, evidence, schema, args.output, execute=True,
                     effort=args.effort, max_output_tokens=args.max_output_tokens,
                     profile_source=args.profile_source)
    response = loads((args.output / "response.json").read_text())
    print(json.dumps(dict(output_dir=str(args.output.resolve()),
                          strategy_path=str((args.output / "strategy.json").resolve()),
                          requested_model=MODEL, provider_model=response.get("provider_model"),
                          provider_status=response.get("provider_status"), usage=response.get("usage")),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
