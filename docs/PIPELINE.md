# ReEmbody Data Pipeline (`Real1 → Sim1 → Sim2 → Real2`)

This guide details the four stages of the ReEmbody cross-embodiment data generation pipeline, how data flows between stages, and the exact commands for each module.

---

## Stage 1: Real1 → Sim1 (Source Scene Reconstruction & Evidence Grounding)

- **Code:** [`stages/01_real1_to_sim1/`](../stages/01_real1_to_sim1/)
- **Inputs:** Source robot demonstration (`Real1`) consisting of synchronized 3-view RGB videos (`head`, `left_wrist`, `right_wrist`) and source joint states/actions (e.g., dual-arm AgileX COBOT-Magic).
- **Outputs:** Reconstructed MuJoCo scene (`source_reference/scene.xml`), parametric task-object geometry (`asset/definition.json`), scene metadata (`source_scene.json`), and source replay arrays (`inputs/source_action_state.npz`) used to extract hand–object contacts, grasp/release timing, and bound task-family relation contracts $\Phi_j$.
- **Key modules:**
  - `rssr_reconstruct/rssr_dataset.py` — source video/parquet inventory and checksum verification.
  - `rssr_reconstruct/pilot_prepare.py` — synchronized 3-view frame extraction.
  - `rssr_reconstruct/pilot_scene.py` & `rssr_scene.py` — parametric object fitting (cups, blocks, articulated drawer) and multi-view verification.
  - `rssr_reconstruct/pilot_camera_workflow.py` — source camera pose mapping.
  - `rssr_reconstruct/rssr_task_contracts.py` — frozen task-family relation contract templates $\Phi_0$ and binding $\Phi_j = \mathrm{Bind}(\Phi_0, S_j)$.

Read-only preflight check on a campaign request JSON:
```bash
python stages/01_real1_to_sim1/run.py --request <campaign-request.json> --preflight-only
```

---

## Stage 2: Sim1 → Sim2 (Target Embodiment Execution & LLM Controller Loop)

- **Code:** [`stages/02_sim1_to_sim2/`](../stages/02_sim1_to_sim2/)
- **Supported target embodiments:**
  1. **Dual-Arm Franka FR3 + Robotiq 2F-85** (`franka_dual_v1/`): kinematic mount geometry from `franka_description` (`fr3_duo_mount_v0_3`), FR3v2 joint limits, factory Robotiq 2F-85 pinch-point alignment, calibrated D455 head + D405 wrist cameras, null-space arch avoidance (`arch_away.py`), exact mesh-surface clearance auditing (`exact_clearance.py`), and bit-exact replay verification (`duo_run.py`).
  2. **Astribot S1** (`rssr_augment/`): 2×7-DoF arms + 4-DoF waist + 2-DoF head, coupled linkage gripper checks, head visibility checks, and adaptive retargeting runtime.
- **Production controllers (`franka_dual_v1/controllers/`):**
  - `duo_stack_cups_away.py` (wraps `duo_stack_cups.py`)
  - `duo_stack_blocks_v2_away.py` (wraps `duo_stack_blocks_v2.py`)
  - `duo_drawer_v2d_away.py` (wraps `duo_drawer_v2d.py` with 3 cm `CabinetGuard` surface clearance)
- **LLM-guided controller revision loop (`llm_controller_loop/`):**
  - `prepare_sources.py` & `build_contexts.py` — packages bound contract $\Phi_j$, geometry, and measured trial diagnostics into structured prompts.
  - `call_luna.py` & `luna_api.py` — OpenAI Responses-compatible strict-schema API driver with immutable request/response provenance (`OPENAI_API_KEY` / `REEMBODY_LLM_API_KEY`).
  - `native_trial.py`, `native_batch.py`, `audit_grasp_camera.py` — executes synthesized controllers in MuJoCo, checks relation/physical/replay admission, and audits wrist-camera target visibility and duration caps ($\le 20\,\text{s}$).

Example single-split FR3 Duo execution:
```bash
MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
python -m franka_dual_v1.duo_rounds \
  --variations <variations.json> \
  --controller stages/02_sim1_to_sim2/franka_dual_v1/controllers/duo_stack_cups_away.py \
  --out outputs/sim2_cups --workers 4
```

---

## Stage 3: Sim2 → Refine (Augmentation, Refinement & Admission Gates)

- **Code:** [`stages/03_sim2_refine/`](../stages/03_sim2_refine/)
- **FR3 Duo refinement suite (`fr3_refine/`):**
  - `sample_v2.py` — generates placement-augmented train/held-out variations (cup placement within 15 cm radius, $\pm 15^\circ$ yaw; blocks within 4 cm; drawer cabinet $\pm 1$ cm & block 4 cm).
  - `refine.py` & `refine_v2.py` — sweeps speed/overlap/staging refinement profiles per initial scene and admits the first candidate passing all four gates:
    1. Bound task-family relation contract $\Phi_j$ (event order, support contact, terminal stability, release).
    2. Physical constraints (table penetration $\le 3\,\text{mm}$, joint-limit excess $\le 2\,\text{mrad}$, arm speed $\le 2\,\text{rad/s}$, tracking error $\le 0.3\,\text{rad}$, exact arch/inter-arm clearance).
    3. Independent bit-exact replay from reset under the same simulator, assets, and evaluator.
    4. Trajectory filters (no missed/repeated grasps, bounded idle duration).
  - `select_site.py`, `posture_filter.py`, `drawer_clearance.py`, `export_replay_v2.py` — episode selection, grasp-posture filtering, surface clearance reporting, and real-robot replay export.
- **Astribot S1 refinement & admission suite (`s1_refine/`):**
  - `validation/` — `quality_gate.py`, `clearance_gate.py`, `clearance_replay_v3.py`, `statealias_gate_v2.py`, `drawer_shell_environment_v2.py`.
  - `orchestration/` — `admission.py`, `owned_stage.py`.

---

## Stage 4: Sim2 → Real2 (Photorealistic Rendering & LeRobot v2.1 Export)

- **Code:** [`stages/04_sim2_to_real2/`](../stages/04_sim2_to_real2/)
- **Step 4a — MuJoCo-to-Blender geometry & pose export:**
  ```bash
  MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
  python -m franka_dual_v1.site_export --run <admitted_run_dir> --out <export_dir>
  ```
- **Step 4b — Per-episode lighting & material config generation:**
  ```bash
  python stages/04_sim2_to_real2/fr3_site_render/make_config_v2.py \
    --site-dir assets/fr3_duo_site \
    --cup-blend assets/fr3_duo_site/measured_cups.blend \
    --episode-index 0 --variation <variation_id> --out <config.json>
  ```
- **Step 4c — Synchronized 3-view Blender Cycles rendering (`224×224`, 30 fps, H.264):**
  ```bash
  <blender-python> stages/04_sim2_to_real2/fr3_site_render/render_site_v2.py \
    --source <export_dir> --config <config.json> --output <render_dir> \
    --device OPTIX --samples 12
  ```
- **Step 4d — Render verification & LeRobot v2.1 packaging:**
  ```bash
  python stages/04_sim2_to_real2/fr3_site_render/verify_renders.py \
    --selection <selection.json> --render <renders_root>
  python stages/04_sim2_to_real2/fr3_site_render/package_lerobot_v2.py \
    --selection <selection.json> --render <renders_root> \
    --out <lerobot_dataset_dir> --campaign reembody_fr3 --task-string "Stack_the_cups_together."
  ```

Every packaged FR3 Duo episode contains:
- `observation.state` (`float32[16]`): left/right 7 joint positions (rad) + Robotiq 2F-85 opening (m)
- `action` (`float32[16]`): absolute joint position targets (rad) + gripper opening target (m) at 30 Hz
- `observation.images.{cam_high, cam_left_wrist, cam_right_wrist}`: `224×224` H.264 MP4 (`224×168` 4:3 content letterboxed with 28 px top/bottom padding)

Astribot S1 datasets (`s1_render/package_lerobot_v9_lean30.py`) export 32-D states, 22-D actions, and `224×224` letterboxed 16:9 views (`224×126` with 49 px padding).
