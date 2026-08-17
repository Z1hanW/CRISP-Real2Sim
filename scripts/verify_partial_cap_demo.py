from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the end-to-end partial-sphere completion demo."
    )
    parser.add_argument("--demo-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    demo_dir = args.demo_dir.expanduser().resolve()
    manifest = json.loads((demo_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((demo_dir / "run" / "report.json").read_text(encoding="utf-8"))
    with np.load(
        demo_dir / "run" / "final" / "scene_mesh_sqs" / "sqs_params.npz",
        allow_pickle=False,
    ) as data:
        params = np.asarray(data["params"], dtype=np.float64)

    accepted_actions = [
        audit["action"]
        for iteration in report["iterations"]
        for audit in iteration.get("action_audit", [])
        if audit.get("metric_decision") == "accepted"
    ]
    completions = [
        action
        for action in accepted_actions
        if action["type"] == "complete" and action["target_shape"] == "sphere"
    ]
    if not completions:
        raise AssertionError("No accepted complete->sphere action was recorded")

    sphere_mask = (
        np.all(params[:, :2] > 0.75, axis=1)
        & (
            np.max(params[:, 2:5], axis=1)
            / np.maximum(np.min(params[:, 2:5], axis=1), 1.0e-8)
            < 1.12
        )
    )
    sphere_rows = params[sphere_mask]
    if len(sphere_rows) != 1:
        raise AssertionError(f"Expected one completed sphere, found {len(sphere_rows)}")

    expected_center = np.asarray(
        manifest["ground_truth"]["sphere_center"],
        dtype=np.float64,
    )
    expected_radius = float(manifest["ground_truth"]["sphere_radius"])
    center_error = float(np.linalg.norm(sphere_rows[0, 8:11] - expected_center))
    radius_error = float(abs(np.mean(sphere_rows[0, 2:5]) - expected_radius))
    if center_error > 0.04:
        raise AssertionError(f"Sphere center error {center_error:.5f} exceeds 0.04")
    if radius_error > 0.04:
        raise AssertionError(f"Sphere radius error {radius_error:.5f} exceeds 0.04")

    initial = report["initial_metrics"]
    final = report["final_metrics"]
    if final["coverage"] < 0.98:
        raise AssertionError(f"Final coverage is only {final['coverage']:.5f}")
    if (
        initial["per_frame_p10_coverage"]
        - final["per_frame_p10_coverage"]
        > 0.01
    ):
        raise AssertionError("Per-frame P10 regressed beyond the global gate")

    result = {
        "accepted_completion": completions[0],
        "center_error": center_error,
        "radius_error": radius_error,
        "initial_coverage": initial["coverage"],
        "final_coverage": final["coverage"],
        "initial_per_frame_p10": initial["per_frame_p10_coverage"],
        "final_per_frame_p10": final["per_frame_p10_coverage"],
        "primitive_count": final["primitive_count"],
    }
    (demo_dir / "verification.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
