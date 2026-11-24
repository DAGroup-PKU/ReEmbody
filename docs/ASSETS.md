# Assets & External Robot Descriptions

## 1. Bundled Assets (`assets/`)

Lightweight task objects, calibrated site scenes, and URDF metadata are bundled directly in the repository and indexed with SHA-256 checksums in [`assets/manifest.json`](../assets/manifest.json):

| Path | Contents | Purpose |
| :--- | :--- | :--- |
| `assets/s1_task_objects/` | Measured cups, blocks, and articulated drawer XMLs & `definition.json` specs | Shared task object geometry for Sim1/Sim2 physics |
| `assets/fr3_duo_site/` | `site_scene.blend`, `site_parameters.json`, `look.json`, `measured_cups.blend`, `desk_scene_bgfit_v3_runtime.blend`, environment textures/HDRI | Calibrated Blender Cycles workspace for FR3 Duo rendering |
| `assets/urdf/fr3_duo_factory.urdf` | Factory Dual-Arm FR3 + Robotiq 2F-85 URDF | Kinematic reference for real-robot mount geometry |

## 2. External Upstream Robot Meshes (`third_party/`)

Large robot visual/collision meshes (~95 MB) are fetched directly from their upstream open-source repositories instead of being duplicated in git:

```bash
bash scripts/setup_third_party_assets.sh
```

This checks out:
1. **`franka_description`** at pinned commit `7aeeddc449edf8d62b594f9e36a81da53e7796f9` → `third_party/franka_description/`
   - Provides `fr3_duo_mount_v0_3` mount/cover meshes and `franka_head_v0_2` meshes.
2. **`mujoco_menagerie`** (sparse checkout: `franka_fr3_v2`, `robotiq_2f85`, `franka_emika_panda`) → `third_party/mujoco_menagerie/`
   - Provides FR3v2 arm and Robotiq 2F-85 gripper MJCF/mesh assets.

## 3. Environment Variable Overrides

All asset and tool locations can be redirected without editing code:

| Environment Variable | Default | Description |
| :--- | :--- | :--- |
| `REEMBODY_THIRD_PARTY_DIR` | `<repo>/third_party` | Root directory containing `franka_description/` and `mujoco_menagerie/` |
| `REEMBODY_FRANKA_DESCRIPTION_DIR` | `$REEMBODY_THIRD_PARTY_DIR/franka_description` | Explicit path to pinned `franka_description` |
| `REEMBODY_MUJOCO_MENAGERIE_DIR` | `$REEMBODY_THIRD_PARTY_DIR/mujoco_menagerie` | Explicit path to `mujoco_menagerie` |
| `REEMBODY_S1_MJCF_DIR` | `$REEMBODY_THIRD_PARTY_DIR/astribot_s1_mjcf` | Optional Astribot S1 robot MJCF assets |
| `REEMBODY_S1_TASK_ASSETS_DIR` | `<repo>/assets/s1_task_objects` | Measured task object MJCF/JSON directory |
| `REEMBODY_FR3_SITE_ASSETS_DIR` | `<repo>/assets/fr3_duo_site` | FR3 Duo Blender site asset directory |
| `REEMBODY_FFMPEG` | `ffmpeg` on `$PATH` | Explicit `ffmpeg` binary for video encoding/verification |
| `REEMBODY_BLENDER_PYTHON` | current `sys.executable` | Python executable with `bpy==4.5.14` installed |
| `OPENAI_API_KEY` / `REEMBODY_LLM_API_KEY` | *(unset)* | API key for optional LLM controller synthesis loop |
| `OPENAI_BASE_URL` / `REEMBODY_LLM_BASE_URL` | *(OpenAI default)* | Optional OpenAI-compatible Responses endpoint |
| `REEMBODY_LLM_MODEL` | `gpt-5` | Model name for LLM controller synthesis loop |
