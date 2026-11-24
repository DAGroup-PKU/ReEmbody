# Stage 2: Sim1 → Sim2 (Cross-Embodiment Transfer & Controller Synthesis)

This stage converts reconstructed source scenes ($S_j$), contact evidence ($E_j$), and bound task-family contracts ($\Phi_j$) into physically verified target-robot trajectories (`Sim2`) across two real target embodiments:

1. **`franka_dual_v1/`** — Dual-Arm Franka FR3v2 + Robotiq 2F-85 on the FR3 Duo mount (`fr3_duo_mount_v0_3`), with calibrated head (D455) and wrist (D405) cameras, null-space arch avoidance (`arch_away.py`), dense mesh-surface clearance (`exact_clearance.py`), and bit-exact replay admission (`duo_run.py`).
2. **`rssr_augment/`** — Shared physical/task-contract runtime and Astribot S1 target execution (`2×7`-DoF arms, `4`-DoF waist, `2`-DoF head, coupled linkage grippers).
3. **`llm_controller_loop/`** — LLM-guided controller synthesis and wrist-camera gaze optimization loop (`call_luna.py`, `luna_api.py`, `prepare_sources.py`, `build_contexts.py`, `audit_grasp_camera.py`, `native_trial.py`, `native_batch.py`, `franka_controller_contract.json`), plus reference gaze-refined controllers (`refined_controllers/`).

## Production FR3 Duo Controllers (`franka_dual_v1/controllers/`)

- **Stack Cups:** `duo_stack_cups_away.py` (wraps `duo_stack_cups.py`)
- **Stack Blocks:** `duo_stack_blocks_v2_away.py` (wraps `duo_stack_blocks_v2.py`)
- **Open / Insert / Close Drawer:** `duo_drawer_v2d_away.py` (wraps `duo_drawer_v2d.py` with 3 cm `CabinetGuard`)

## Quick Example Commands

```bash
# Run FR3 Duo Sim2 synthesis on a variations split (requires third_party assets)
MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
python -m franka_dual_v1.duo_rounds \
  --variations <variations.json> \
  --controller stages/02_sim1_to_sim2/franka_dual_v1/controllers/duo_stack_cups_away.py \
  --out outputs/sim2_cups --workers 4

# Export an admitted FR3 episode for Stage 4 Blender rendering
MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
python -m franka_dual_v1.site_export --run <admitted_run_dir> --out <export_dir>
```

For the optional LLM controller revision loop (`llm_controller_loop/call_luna.py`), configure standard environment variables: `OPENAI_API_KEY` (or `REEMBODY_LLM_API_KEY`), optionally `OPENAI_BASE_URL` and `REEMBODY_LLM_MODEL`.
