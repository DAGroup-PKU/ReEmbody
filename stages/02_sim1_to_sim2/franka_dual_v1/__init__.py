"""Dual-arm Franka (MuJoCo Menagerie Panda + Franka Hand) physics layer for RSSR."""
import os

# segmentation renders (metrics.py) need the GL backend chosen before mujoco is first imported
os.environ.setdefault('MUJOCO_GL', 'egl')
