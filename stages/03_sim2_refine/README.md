# Stage 3: Sim2 → Refine (Augmentation, Trajectory Refinement & Admission Gates)

This stage performs initial-placement augmentation, multi-profile trajectory refinement (removing detours, enabling collision-free bimanual overlap, bounded joint speedups), and independent admission verification.

## Components

### 1. Dual-Arm Franka FR3 Refinement (`fr3_refine/`)

| File | Purpose |
| :--- | :--- |
| `sample_v2.py` | Placement-augmented train / held-out variation sampler for cups, blocks, and drawer |
| `refine.py`, `refine_v2.py` | Multi-profile refinement sweep across all 3 tasks (admits first candidate passing all gates) |
| `arch_away.py` | Null-space redundancy controller maintaining exact clearance from the camera arch |
| `merge_refine.py` | Merges sharded refine outputs into a single indexed split |
| `select_site.py`, `posture_filter.py`, `scan_grasps.py` | Admitted episode selection & grasp-posture filtering |
| `drawer_clearance.py`, `drawer_clearance_all.py` | Surface-sampled hand-to-cabinet clearance auditing |
| `export_replay_v2.py`, `export_replay_run.py` | 30 Hz trajectory export for physical robot replay |

### 2. Astribot S1 Refinement & Admission Gates (`s1_refine/`)

- `validation/` — `quality_gate.py`, `clearance_gate.py`, `clearance_fast.py`, `clearance_cross_side.py`, `clearance_replay_v3.py`, `statealias_gate_v2.py`, `drawer_shell_environment_v2.py`.
- `orchestration/` — `admission.py`, `owned_stage.py`.
- `tasks/drawer/` — Astribot S1 drawer rule-refinement controller, servo, contract, and quality gate.
- `tests/` — Unit tests (`check_entrypoint.py`, `test_admission_binding.py`).

## Admission Criteria

Every admitted trajectory must pass four gates:
1. Bound task-family relation contract $\Phi_j$ (event order, lift/support predicates, terminal hold & release).
2. Physical constraints (table penetration, joint-limit margins, joint speeds, tracking error, inter-arm & arch/cabinet clearance).
3. Independent replay from reset under the identical simulator, assets, and contract evaluator.
4. Trajectory quality filters (no missed/repeated grasps, bounded idle duration).
