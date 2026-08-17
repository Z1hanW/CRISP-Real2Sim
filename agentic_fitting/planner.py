from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


PLAN_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["scene_assessment", "actions", "stop"],
    "properties": {
        "scene_assessment": {"type": "string"},
        "stop": {"type": "boolean"},
        "actions": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "type",
                    "primitive_ids",
                    "target_shape",
                    "confidence",
                    "rationale",
                ],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["keep", "merge", "split", "refit", "complete", "drop"],
                    },
                    "primitive_ids": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "minItems": 1,
                    },
                    "target_shape": {
                        "type": "string",
                        "enum": ["unchanged", "surface", "box", "sphere", "ellipsoid", "cylinder"],
                    },
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "rationale": {"type": "string"},
                },
            },
        },
    },
}


def _prompt(summary: dict[str, Any]) -> str:
    return f"""You are the visual planner in a constrained real-to-sim fitting loop.

Input evidence:
- global_overview.png: gray fused point cloud with colored primitives and numeric IDs.
- global_residuals.png: point-to-primitive residuals; blue is supported and red is missed.
- per_frame_3d.png: selected per-frame observations against the same global primitives.
- rgb/: selected original frames.
- clusters/: per-frame normal-cluster maps; the same color denotes a matched global segment.
- evidence.json: exact metrics and primitive parameters.

Objectives, in order:
1. Every retained primitive must be consistent across per-frame observations.
2. Explain the whole static scene with as few primitives as possible.
3. Preserve or improve point-cloud coverage and residuals.
4. Complete a partially observed object only when RGB, repeated frames, and curvature support a
   familiar full shape. A spherical cap may become a sphere. An isolated planar patch should
   remain a thin surface, not inflate into a volume.

You do not choose numeric parameters. Propose only semantic actions; a deterministic geometry
executor will fit numbers and reject any action whose measured objective does not improve.

Rules:
- Use IDs exactly as shown in evidence.json.
- Do not reference an ID in more than one non-keep action.
- merge requires at least two IDs.
- split/refit/complete/drop should normally use one ID.
- complete may target sphere, ellipsoid, or cylinder only.
- Prefer merge over many refits when adjacent fragments represent one physical surface.
- Two adjacent orthogonal surface primitives may merge into one solid box when that box explains
  both observed faces. This is compression, not unsupported completion.
- evidence.json includes deterministic merge_candidates. Their estimated coverage is measured on
  points currently supported by that pair. Prefer disjoint high-preservation candidates, while
  using RGB and per-frame evidence to reject pairs belonging to different objects.
- Drop only clear duplicates or unsupported floaters.
- Return stop=true with no actions when the current fit is already the best conservative model.

Current evidence summary:
{json.dumps(summary, indent=2)}
"""


def call_codex_planner(
    evidence_dir: Path,
    summary: dict[str, Any],
    attached_images: list[Path],
    *,
    codex_bin: Path = Path("/usr/local/bin/codex"),
    timeout_seconds: int = 900,
) -> dict[str, Any]:
    schema_path = evidence_dir / "plan_schema.json"
    plan_path = evidence_dir / "plan.json"
    event_log = evidence_dir / "codex_events.jsonl"
    schema_path.write_text(json.dumps(PLAN_SCHEMA, indent=2), encoding="utf-8")

    command = [
        str(codex_bin),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(plan_path),
        "--json",
        "-C",
        str(evidence_dir),
    ]
    for image in attached_images:
        command.extend(["-i", str(image)])
    command.extend(["--", "-"])
    completed = subprocess.run(
        command,
        input=_prompt(summary),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    event_log.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex planner failed with exit code {completed.returncode}; see {event_log}"
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return plan
