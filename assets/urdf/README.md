# Dual-Arm Franka FR3 + Robotiq 2F-85 URDF

`fr3_duo_factory.urdf` is the measured factory URDF for the Dual-Arm Franka FR3 Duo with Robotiq 2F-85 grippers.

Robot link and gripper visual/collision meshes are not committed to git due to size (~95 MB). Run `bash scripts/setup_third_party_assets.sh` from the repository root to fetch the pinned upstream `franka_description` (`7aeeddc449edf8d62b594f9e36a81da53e7796f9`) and `mujoco_menagerie` (`franka_fr3_v2`, `robotiq_2f85`) assets into `third_party/`.
