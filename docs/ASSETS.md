# Assets & Environment Variables

## Bundled Assets (`assets/`)
- `assets/s1_task_objects/` — measured cup, block, and drawer MJCF/JSON specs
- `assets/fr3_duo_site/` — calibrated FR3 Duo Blender scene, lighting, textures, and `measured_cups.blend`
- `assets/urdf/fr3_duo_factory.urdf` — factory Dual-Arm FR3 + Robotiq 2F-85 URDF
- `assets/manifest.json` — SHA-256 checksums

## Upstream Robot Meshes (`third_party/`)
Fetch pinned `franka_description` (`7aeeddc449edf8d62b594f9e36a81da53e7796f9`) and `mujoco_menagerie` (`franka_fr3_v2`, `robotiq_2f85`):
```bash
bash scripts/setup_third_party_assets.sh
```

## Optional Overrides
- `REEMBODY_THIRD_PARTY_DIR` (or `REEMBODY_FRANKA_DESCRIPTION_DIR`, `REEMBODY_MUJOCO_MENAGERIE_DIR`, `REEMBODY_S1_MJCF_DIR`)
- `REEMBODY_FFMPEG`, `REEMBODY_BLENDER_PYTHON`
- `OPENAI_API_KEY` / `REEMBODY_LLM_API_KEY`, `OPENAI_BASE_URL`, `REEMBODY_LLM_MODEL`
