# Stage 4: Sim2 → Real2 (Blender Cycles Rendering & LeRobot v2.1 Export)

This stage renders admitted target-robot trajectories inside calibrated Blender Cycles workspaces across three synchronized moving cameras (`cam_high`, `cam_left_wrist`, `cam_right_wrist` at `224×224`, 30 fps, H.264), verifies video integrity and frame-clock alignment, and packages the output into standard **LeRobot v2.1** datasets.

## Components

### 1. Dual-Arm Franka FR3 Site Rendering & Packaging (`fr3_site_render/`)

| File | Purpose |
| :--- | :--- |
| `make_site_scene.py` | Procedural Blender workspace builder (table, cloth, carpet, pillar, arch, cart, lights) |
| `style_fit.py`, `calibrate_colours.py` | Real-image exposure fitting and linear RGB color calibration under site lighting |
| `make_config.py`, `make_config_v2.py` | Deterministic per-episode lighting & exposure randomization + S1 object appearance config |
| `render_site.py`, `render_site_v2.py` | Blender Cycles 3-view renderer (`bpy` 4.5 LTS, OptiX GPU or CPU, `224×168` padded to `224×224`) |
| `render_local_gpu.py` | Local multi-episode rendering queue runner |
| `verify_renders.py` | Verifies `render_receipt.json`, SHA-256 digests, and decoded frame counts via `ffmpeg` |
| `package_lerobot_v2.py` | Packages verified renders + MuJoCo trajectories into LeRobot v2.1 (16-D state/action) |

### 2. Astribot S1 Rendering & Packaging (`s1_render/`)

- `rssr_render/{common.py, source_clock.py}` — causal 30 Hz state/action/video clock alignment.
- `camera.py`, `environment.py`, `drawer_table_v2.py`, `action_columns.py`, `renderer_center_v1.py`, `export_source_center_v1.py`, `export_source_sdkcam_v1.py`, `package_lerobot_v9_lean30.py` — Astribot S1 Cycles renderer and 32-D state / 22-D action LeRobot v2.1 exporter.

### 3. Pinned Appearance & Camera Configs (`configs/`)

- `drawer_appearance_v9b_uniform_cabinet_upright.json`
- `render_config_center_sdk_v1.json`

## Example Usage (Single Episode)

```bash
# 1) Generate per-episode render config
python stages/04_sim2_to_real2/fr3_site_render/make_config_v2.py \
  --site-dir assets/fr3_duo_site \
  --cup-blend assets/fr3_duo_site/measured_cups.blend \
  --episode-index 0 --variation stack_cups_v2_train_0000 --out outputs/config.json

# 2) Render preview frames (or omit --preview for full MP4 videos)
<blender-python> stages/04_sim2_to_real2/fr3_site_render/render_site_v2.py \
  --source <blender_source_dir> --config outputs/config.json \
  --output outputs/render_ep000 --device CPU --samples 8 --preview 0
```
