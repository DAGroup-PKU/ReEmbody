#!/usr/bin/env python3
"""Runnable walkthrough demo for the ReEmbody Real1 -> Sim1 -> Sim2 -> Real2 data pipeline.

Modes:
  inspect           Print full Stage 1-4 metadata, contract summary, and tensor/video specs (no extra deps).
  export-blender    Run Stage 2 -> Stage 4 MuJoCo-to-Blender export (`site_export.py` + `make_config_v2.py`).
  render-preview    Run Stage 4 Blender Cycles renderer (`render_site_v2.py`) on the exported bundle.

Examples:
  python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode inspect
  python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode export-blender --out outputs/demo_cups
  python scripts/run_example_pipeline.py --example stack_cups_fr3_ep000 --mode render-preview --out outputs/demo_cups --device CPU --samples 4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from reembody.paths import (  # noqa: E402
    ASSETS_DIR,
    EXAMPLES_DIR,
    resolve_franka_description,
    resolve_mujoco_menagerie,
)


def prepare_portable_scene(scene_xml: Path, dst_xml: Path) -> Path:
    """Resolve relative third_party/assets paths inside an example scene.xml into an output dir."""
    franka_desc = resolve_franka_description()
    menagerie = resolve_mujoco_menagerie()
    if not franka_desc.is_dir() or not (menagerie / "franka_fr3_v2").is_dir():
        raise FileNotFoundError(
            f"Missing third-party robot assets ({franka_desc}, {menagerie}). "
            "Run `bash scripts/setup_third_party_assets.sh` or set REEMBODY_THIRD_PARTY_DIR."
        )
    text = scene_xml.read_text()
    text = text.replace("../../../third_party/franka_description/", str(franka_desc) + "/")
    text = text.replace("../../../third_party/mujoco_menagerie/", str(menagerie) + "/")
    text = text.replace("../../../assets/", str(ASSETS_DIR) + "/")
    dst_xml.parent.mkdir(parents=True, exist_ok=True)
    dst_xml.write_text(text)
    return dst_xml


def cmd_inspect(ex_dir: Path) -> None:
    meta = json.loads((ex_dir / "metadata.json").read_text())
    print(f"=== ReEmbody Example: {meta['example_id']} ===")
    print(f"Task               : {meta['task']} ({', '.join(meta.get('task_prompt', []))})")
    print(f"Source -> Target   : {meta['source_embodiment']} -> {meta['target_embodiment']}")
    print(f"Source case        : {meta['source_case']}")
    print(f"Frames / Duration  : {meta['frames']} frames @ {meta['fps']} fps ({meta['duration_s']} s)")
    print("\nPipeline artifacts:")
    for stage_key, desc in meta["stages"].items():
        print(f"  - {stage_key:22s}: {desc}")
    rec_path = ex_dir / "03_sim2_target/record.json"
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text())
        print("\nSim2 Admission Summary:")
        print(f"  - admitted         : {rec.get('admitted')}")
        if "variation" in rec:
            print(f"  - variation        : {rec['variation'].get('id')} ({rec['variation'].get('task')})")
        if "duration_s" in rec:
            print(f"  - sim duration (s) : {rec['duration_s']}")
        if "clearance_m" in rec:
            print(f"  - min clearance (m): {rec['clearance_m']}")


def cmd_export_blender(ex_dir: Path, out_dir: Path) -> Path:
    meta = json.loads((ex_dir / "metadata.json").read_text())
    if "franka" not in meta["target_embodiment"]:
        raise SystemExit("`export-blender` demo currently targets the Dual-Arm Franka FR3 examples.")

    s2_dir = ex_dir / "03_sim2_target"
    run_dir = out_dir / "sim2_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    prepare_portable_scene(s2_dir / "scene.xml", run_dir / "scene.xml")
    for fname in ("episode.npz", "record.json"):
        target = run_dir / fname
        if not target.exists():
            os.symlink(s2_dir / fname, target)

    rec = json.loads((s2_dir / "record.json").read_text())
    src_case = rec.get("variation", {}).get("source_case", meta["source_case"])
    cases_dir = out_dir / "sim1_cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_link = cases_dir / src_case
    if not case_link.exists():
        case_link.parent.mkdir(parents=True, exist_ok=True)
        sim1_target = ex_dir / "02_sim1_reconstructed" / src_case
        if not sim1_target.is_dir():
            sim1_target = ex_dir / "02_sim1_reconstructed"
        os.symlink(sim1_target, case_link)

    export_dir = out_dir / "blender_source"
    if not export_dir.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT / "stages/02_sim1_to_sim2") + os.pathsep + env.get("PYTHONPATH", "")
        env["REEMBODY_SOURCE_CASES_DIR"] = str(cases_dir)
        env.setdefault("MUJOCO_GL", "egl")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "franka_dual_v1.site_export",
                "--run",
                str(run_dir),
                "--out",
                str(export_dir),
            ],
            check=True,
            env=env,
        )

    variation_id = rec.get("variation", {}).get("id", meta["task"] + "_ep000")
    cfg_path = out_dir / "render_config.json"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "stages/04_sim2_to_real2/fr3_site_render/make_config_v2.py"),
            "--site-dir",
            str(ASSETS_DIR / "fr3_duo_site"),
            "--cup-blend",
            str(ASSETS_DIR / "fr3_duo_site/measured_cups.blend"),
            "--episode-index",
            "0",
            "--variation",
            variation_id,
            "--out",
            str(cfg_path),
        ],
        check=True,
    )
    print(f"\nExported Blender source bundle -> {export_dir}")
    print(f"Generated render config        -> {cfg_path}")
    return out_dir


def cmd_render_preview(
    ex_dir: Path,
    out_dir: Path,
    *,
    device: str,
    samples: int,
    full: bool,
    blender_python: str | None,
) -> None:
    export_dir = out_dir / "blender_source"
    cfg_path = out_dir / "render_config.json"
    if not export_dir.is_dir() or not cfg_path.is_file():
        cmd_export_blender(ex_dir, out_dir)

    render_out = out_dir / "render_out"
    if render_out.exists():
        raise SystemExit(f"Render output directory already exists (use a fresh --out): {render_out}")

    py = blender_python or os.environ.get("REEMBODY_BLENDER_PYTHON") or sys.executable
    cmd = [
        py,
        str(REPO_ROOT / "stages/04_sim2_to_real2/fr3_site_render/render_site_v2.py"),
        "--source",
        str(export_dir),
        "--config",
        str(cfg_path),
        "--output",
        str(render_out),
        "--device",
        device,
        "--samples",
        str(samples),
    ]
    if not full:
        cmd.extend(["--preview", "0"])
    subprocess.run(cmd, check=True)
    print(f"\nRendered preview outputs -> {render_out}")


def main() -> int:
    ap = argparse.ArgumentParser(description="ReEmbody Real1 -> Sim1 -> Sim2 -> Real2 example walkthrough.")
    ap.add_argument(
        "--example",
        default="stack_cups_fr3_ep000",
        choices=[
            "stack_cups_fr3_ep000",
            "stack_blocks_fr3_ep000",
            "drawer_fr3_ep000",
            "stack_cups_s1_ep000",
        ],
    )
    ap.add_argument(
        "--mode",
        default="inspect",
        choices=["inspect", "export-blender", "render-preview"],
    )
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "outputs/demo_run")
    ap.add_argument("--device", default="CPU", choices=["CPU", "OPTIX"])
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--full", action="store_true", help="Render all frames & encode MP4 videos (otherwise renders start/end keyframes).")
    ap.add_argument("--blender-python", default=None, help="Python executable with `bpy` installed.")
    args = ap.parse_args()

    ex_dir = EXAMPLES_DIR / args.example
    if args.mode == "inspect":
        cmd_inspect(ex_dir)
    elif args.mode == "export-blender":
        cmd_export_blender(ex_dir, args.out)
    elif args.mode == "render-preview":
        cmd_render_preview(
            ex_dir,
            args.out,
            device=args.device,
            samples=args.samples,
            full=args.full,
            blender_python=args.blender_python,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
