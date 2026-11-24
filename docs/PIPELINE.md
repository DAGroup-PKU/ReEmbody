# Pipeline Reference (`Real1 → Sim1 → Sim2 → Real2`)

## 1. Real1 → Sim1 (`stages/01_real1_to_sim1/`)
Reconstructs source scenes ($S_j$), contact/timing evidence ($E_j$), and bound task contracts ($\Phi_j$) from real COBOT-Magic 3-view demonstrations.
```bash
python stages/01_real1_to_sim1/run.py --request <request.json> --preflight-only
```

## 2. Sim1 → Sim2 (`stages/02_sim1_to_sim2/`)
Runs target controllers in MuJoCo 3.3.7 (`franka_dual_v1/` for Dual-Arm FR3, `rssr_augment/` for Astribot S1) and supports LLM-guided controller revision (`llm_controller_loop/call_luna.py` via `OPENAI_API_KEY`).
```bash
MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
python -m franka_dual_v1.duo_rounds \
  --variations <variations.json> \
  --controller stages/02_sim1_to_sim2/franka_dual_v1/controllers/duo_stack_cups_away.py \
  --out outputs/sim2_cups --workers 4
```

## 3. Sim2 → Refine (`stages/03_sim2_refine/`)
Placement sampling (`sample_v2.py`), multi-profile refinement (`refine.py`, `refine_v2.py`), clearance checks, and independent replay admission (`fr3_refine/`, `s1_refine/`).

## 4. Sim2 → Real2 (`stages/04_sim2_to_real2/`)
Exports admitted trajectories to Blender Cycles (`bpy==4.5.14`), renders 3 synchronized `224×224` 30 fps views, verifies frame counts, and exports LeRobot v2.1 datasets.
```bash
# Export MuJoCo rollout -> Blender bundle
MUJOCO_GL=egl PYTHONPATH=stages/02_sim1_to_sim2 \
python -m franka_dual_v1.site_export --run <run_dir> --out <export_dir>

# Generate per-episode render config
python stages/04_sim2_to_real2/fr3_site_render/make_config_v2.py \
  --site-dir assets/fr3_duo_site --cup-blend assets/fr3_duo_site/measured_cups.blend \
  --episode-index 0 --variation <variation_id> --out <config.json>

# Render 3-view videos (or add `--preview 0` for keyframes only)
<bpy-python> stages/04_sim2_to_real2/fr3_site_render/render_site_v2.py \
  --source <export_dir> --config <config.json> --output <render_dir> --device OPTIX

# Verify & package as LeRobot v2.1
python stages/04_sim2_to_real2/fr3_site_render/verify_renders.py --selection <selection.json> --render <renders_dir>
python stages/04_sim2_to_real2/fr3_site_render/package_lerobot_v2.py \
  --selection <selection.json> --render <renders_dir> --out <dataset_dir> \
  --campaign reembody_fr3 --task-string "Stack_the_cups_together."
```
