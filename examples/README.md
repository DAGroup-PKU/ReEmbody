# Examples

Four complete real-world `Real1 → Sim1 → Sim2 → Real2` examples:
- `stack_cups_fr3_ep000/` — Stack Cups (COBOT-Magic → Dual-Arm FR3, 519 frames)
- `stack_blocks_fr3_ep000/` — Stack Blocks (COBOT-Magic → Dual-Arm FR3, 523 frames)
- `drawer_fr3_ep000/` — Open / Insert / Close Drawer (COBOT-Magic → Dual-Arm FR3, 470 frames)
- `stack_cups_s1_ep000/` — Stack Cups (COBOT-Magic → Astribot S1, 450 frames)

Each directory follows the same 4-stage layout:
- `01_real1_source/` — source parquet + 3 camera MP4s
- `02_sim1_reconstructed/` — reconstructed MuJoCo scene + fitted object spec
- `03_sim2_target/` — target `scene.xml`, controller, `episode.npz`, and admission receipt
- `04_real2_rendered/` — render receipt, keyframes, and self-contained LeRobot v2.1 episode

```bash
python scripts/verify_examples.py [--check-mujoco]
python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode inspect
```
