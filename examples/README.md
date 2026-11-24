# Bundled End-to-End Examples (`Real1 → Sim1 → Sim2 → Real2`)

This directory contains four complete, verified cross-embodiment data transfer examples covering all three manipulation tasks from the paper (**Stack Cups**, **Stack Blocks**, and **Open / Insert / Close Drawer**) and both real target embodiments (**Dual-Arm Franka FR3** and **Astribot S1**).

## Included Examples

1. **`stack_cups_fr3_ep000/`** — AgileX COBOT-Magic → Dual-Arm Franka FR3 + Robotiq 2F-85 (519 frames, 17.3 s)
2. **`stack_blocks_fr3_ep000/`** — AgileX COBOT-Magic → Dual-Arm Franka FR3 + Robotiq 2F-85 (523 frames, 17.4 s)
3. **`drawer_fr3_ep000/`** — AgileX COBOT-Magic → Dual-Arm Franka FR3 + Robotiq 2F-85 (470 frames, 15.7 s)
4. **`stack_cups_s1_ep000/`** — AgileX COBOT-Magic → Astribot S1 (450 frames, 15.0 s)

## Uniform Per-Example Directory Layout

```text
examples/<example_id>/
├── metadata.json                  # Provenance, frame count, duration, task prompt
├── 01_real1_source/
│   ├── episode_000000.parquet     # Recorded Real1 source states & actions
│   └── videos/                    # Source cam_high, cam_left_wrist, cam_right_wrist MP4s
├── 02_sim1_reconstructed/
│   ├── source_scene.json          # Fitted object poses & workspace metadata
│   ├── asset/definition.json      # Parametric object geometry
│   ├── source_reference/scene.xml # Reconstructed Sim1 MuJoCo scene
│   └── inputs/source_action_state.npz
├── 03_sim2_target/
│   ├── scene.xml                  # Target MuJoCo scene (portable relative mesh paths)
│   ├── controller.py              # Target controller (FR3 examples)
│   ├── episode.npz                # Admitted native trajectory (qpos, ctrl, timestamps)
│   ├── record.json                # Full contract, physical & independent-replay admission record
│   └── admission_receipt.json     # Independent admission receipt
└── 04_real2_rendered/
    ├── render_receipt.json        # Blender Cycles settings & SHA-256 checksums
    ├── keyframes/                 # Rendered start/end PNG keyframes (FR3 examples)
    └── lerobot_v2_1/              # Self-contained single-episode LeRobot v2.1 dataset
        ├── meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl, stats.json}
        ├── data/chunk-000/episode_000000.parquet
        └── videos/chunk-000/observation.images.{cam_high,cam_left_wrist,cam_right_wrist}/episode_000000.mp4
```

## Verifying & Running the Examples

```bash
# Validate all 4 examples (parquet, video frame alignment, XML/JSON schemas)
python scripts/verify_examples.py

# Also compile FR3 scene.xml files in MuJoCo 3.3.7 and verify FK over episode.npz
python scripts/verify_examples.py --check-mujoco

# Inspect one example in detail
python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode inspect

# Re-export Sim2 trajectory -> Blender source bundle + render config
python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode export-blender --out outputs/demo_cups
```
