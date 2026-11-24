# Stage 1: Real1 → Sim1 (Source Scene Reconstruction & Evidence Grounding)

This stage reconstructs a MuJoCo source scene $S_j$ from a recorded three-view source demonstration (`Real1`: head, left-wrist, right-wrist videos + source joint states/actions), replays the source actions to extract contact and timing evidence $E_j$, and binds a frozen task-family relation contract $\Phi_j = \mathrm{Bind}(\Phi_0, S_j)$.

## Key Modules (`rssr_reconstruct/`)

| File | Responsibility |
| :--- | :--- |
| `rssr_dataset.py` | Raw video inventory and input checksum validation |
| `pilot_prepare.py` | Synchronized three-view frame extraction |
| `pilot_scene.py` | Parametric scene initialization and iterative three-view review |
| `rssr_scene.py` | MuJoCo task object and support geometry builder |
| `pilot_camera_workflow.py` | Source camera mapping |
| `rssr_task_contracts.py` | Frozen task-family relation templates $\Phi_0$ and contract binding |
| `rssr_campaign.py` | End-to-end campaign entrypoint (`run.py` delegates here) |
| `prompts/`, `schemas/` | Structured reconstruction prompts and JSON validation schemas |

## Usage

Read-only preflight validation of a campaign request:

```bash
python stages/01_real1_to_sim1/run.py --request <campaign-request.json> --preflight-only
```

> **Note:** Full execution of `rssr_campaign.py` requires an OpenAI-compatible vision-language API configured via standard environment variables (`OPENAI_API_KEY` or `KIMI_API_KEY`) and source robot MJCF descriptions. See [`examples/*/02_sim1_reconstructed/`](../../examples/) for concrete Stage 1 outputs.
