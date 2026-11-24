#!/usr/bin/env python3
"""Verify integrity of all bundled ReEmbody Real1 -> Sim1 -> Sim2 -> Real2 examples.

Usage:
    python scripts/verify_examples.py
    python scripts/verify_examples.py --check-mujoco
"""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from reembody.paths import resolve_franka_description, resolve_mujoco_menagerie  # noqa: E402

EXAMPLES_DIR = REPO_ROOT / "examples"
CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def count_video_frames(mp4: Path) -> tuple[int, int, int]:
    with av.open(str(mp4)) as container:
        stream = container.streams.video[0]
        w, h = int(stream.width), int(stream.height)
        n = sum(1 for _ in container.decode(stream))
    return n, w, h


def check_mujoco_fr3_scene(scene_xml: Path, episode_npz: Path) -> dict:
    import mujoco

    franka_desc = resolve_franka_description()
    menagerie = resolve_mujoco_menagerie()
    if not franka_desc.is_dir() or not (menagerie / "franka_fr3_v2").is_dir():
        raise FileNotFoundError(
            f"Third-party robot assets not found ({franka_desc}, {menagerie}). "
            "Run `bash scripts/setup_third_party_assets.sh` or set REEMBODY_THIRD_PARTY_DIR."
        )

    xml_text = scene_xml.read_text()
    xml_text = xml_text.replace("../../../third_party/franka_description/", str(franka_desc) + "/")
    xml_text = xml_text.replace("../../../third_party/mujoco_menagerie/", str(menagerie) + "/")
    xml_text = xml_text.replace("../../../assets/", str(REPO_ROOT / "assets") + "/")

    with tempfile.NamedTemporaryFile("w", suffix=".xml", dir=str(scene_xml.parent), delete=False) as tf:
        tf.write(xml_text)
        tmp_path = Path(tf.name)
    try:
        model = mujoco.MjModel.from_xml_path(str(tmp_path))
        data = mujoco.MjData(model)
        with np.load(episode_npz, allow_pickle=False) as ep:
            qpos = ep["qpos"]
            qvel = ep["qvel"]
            action = ep["action"]
            if qpos.shape[1] != model.nq:
                raise ValueError(f"qpos width {qpos.shape[1]} != model.nq {model.nq}")
            if qvel.shape[1] != model.nv:
                raise ValueError(f"qvel width {qvel.shape[1]} != model.nv {model.nv}")
            if action.shape[1] != model.nu:
                raise ValueError(f"action width {action.shape[1]} != model.nu {model.nu}")
            for idx in (0, len(qpos) // 2, len(qpos) - 1):
                data.qpos[:] = qpos[idx]
                data.qvel[:] = qvel[idx]
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.xpos).all():
                    raise ValueError(f"Non-finite FK xpos at frame {idx}")
        return {"nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu), "steps": int(len(qpos))}
    finally:
        tmp_path.unlink(missing_ok=True)


def verify_one_example(ex_dir: Path, *, check_mujoco: bool = False) -> dict:
    meta = json.loads((ex_dir / "metadata.json").read_text())
    name = ex_dir.name
    is_fr3 = "franka" in meta["target_embodiment"]

    # Stage 1: Real1 source
    r1_dir = ex_dir / "01_real1_source"
    r1_table = pq.read_table(r1_dir / "episode_000000.parquet")
    assert r1_table.num_rows > 0, f"{name}: empty Real1 parquet"
    r1_vids = {}
    for cam in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
        n, w, h = count_video_frames(r1_dir / "videos" / f"{cam}.mp4")
        assert n > 0, f"{name}: empty Real1 video {cam}"
        r1_vids[cam] = f"{n}f@{w}x{h}"

    # Stage 2: Sim1 reconstructed
    s1_dir = ex_dir / "02_sim1_reconstructed"
    json.loads((s1_dir / "source_scene.json").read_text())
    json.loads((s1_dir / "asset/definition.json").read_text())
    ET.parse(s1_dir / "source_reference/scene.xml")
    with np.load(s1_dir / "inputs/source_action_state.npz", allow_pickle=False) as s1_npz:
        assert len(s1_npz.files) > 0, f"{name}: empty Sim1 npz"

    # Stage 3: Sim2 target
    s2_dir = ex_dir / "03_sim2_target"
    ET.parse(s2_dir / "scene.xml")
    mj_info = None
    if is_fr3:
        py_compile.compile(str(s2_dir / "controller.py"), doraise=True)
        rec = json.loads((s2_dir / "record.json").read_text())
        assert rec.get("admitted") is True, f"{name}: Sim2 record not admitted"
        with np.load(s2_dir / "episode.npz", allow_pickle=False) as ep:
            for key in ("qpos", "qvel", "action", "time"):
                assert key in ep.files, f"{name}: missing {key} in episode.npz"
                assert np.isfinite(ep[key]).all(), f"{name}: non-finite {key}"
        if check_mujoco:
            mj_info = check_mujoco_fr3_scene(s2_dir / "scene.xml", s2_dir / "episode.npz")
    else:
        json.loads((s2_dir / "task_contract.json").read_text())
        with np.load(s2_dir / "initial_state.npz", allow_pickle=False) as init:
            assert "qpos" in init.files and np.isfinite(init["qpos"]).all()

    # Stage 4: Real2 rendered LeRobot v2.1
    lr_dir = ex_dir / "04_real2_rendered/lerobot_v2_1"
    info = json.loads((lr_dir / "meta/info.json").read_text())
    assert info["total_episodes"] == 1, f"{name}: expected 1 episode in mini-dataset"
    lr_table = pq.read_table(lr_dir / "data/chunk-000/episode_000000.parquet")
    n_frames = lr_table.num_rows
    assert n_frames == int(info["total_frames"]) == int(meta["frames"]), (
        f"{name}: frame mismatch parquet={n_frames} info={info['total_frames']} meta={meta['frames']}"
    )
    state_arr = np.asarray(lr_table["observation.state"].to_pylist(), dtype=np.float32)
    action_arr = np.asarray(lr_table["action"].to_pylist(), dtype=np.float32)
    assert np.isfinite(state_arr).all() and np.isfinite(action_arr).all()

    for ck in CAM_KEYS:
        vid_path = lr_dir / "videos/chunk-000" / ck / "episode_000000.mp4"
        vn, vw, vh = count_video_frames(vid_path)
        assert vn == n_frames, f"{name}: {ck} has {vn} frames != {n_frames}"
        assert (vw, vh) == (224, 224), f"{name}: {ck} resolution {(vw, vh)} != (224, 224)"

    return {
        "example": name,
        "target": meta["target_embodiment"],
        "real1_frames": r1_table.num_rows,
        "real2_frames": n_frames,
        "state_dim": int(state_arr.shape[1]),
        "action_dim": int(action_arr.shape[1]),
        "mujoco": mj_info,
        "status": "OK",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify bundled ReEmbody end-to-end examples.")
    ap.add_argument(
        "--check-mujoco",
        action="store_true",
        help="Also compile FR3 target scenes in MuJoCo 3.3.7 and verify FK over episode.npz (requires third_party assets).",
    )
    ap.add_argument(
        "--example",
        default=None,
        help="Verify only one specific example folder name under examples/.",
    )
    args = ap.parse_args()

    example_dirs = sorted(
        p for p in EXAMPLES_DIR.iterdir() if p.is_dir() and (p / "metadata.json").is_file()
    )
    if args.example:
        example_dirs = [p for p in example_dirs if p.name == args.example]
        if not example_dirs:
            raise SystemExit(f"Unknown example: {args.example}")

    results = []
    for ex_dir in example_dirs:
        res = verify_one_example(ex_dir, check_mujoco=args.check_mujoco)
        results.append(res)
        mj_str = (
            f" | MuJoCo nq={res['mujoco']['nq']} nu={res['mujoco']['nu']}"
            if res["mujoco"]
            else ""
        )
        print(
            f"[OK] {res['example']:24s} | {res['target']:32s} | "
            f"Real1={res['real1_frames']:3d}f -> Real2={res['real2_frames']:3d}f "
            f"(state={res['state_dim']}D, action={res['action_dim']}D, 3x224x224){mj_str}"
        )

    print(f"\nVerified {len(results)}/{len(example_dirs)} examples successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
