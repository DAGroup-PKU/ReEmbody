# Stage 2: Sim1 → Sim2

Target-embodiment execution and controller synthesis:
- `franka_dual_v1/` — Dual-Arm Franka FR3 + Robotiq 2F-85 runtime & production controllers (`duo_stack_cups_away.py`, `duo_stack_blocks_v2_away.py`, `duo_drawer_v2d_away.py`)
- `rssr_augment/` — Shared task-contract / Astribot S1 runtime
- `llm_controller_loop/` — LLM-guided controller revision & wrist-gaze audit (`call_luna.py`, `native_trial.py`)
