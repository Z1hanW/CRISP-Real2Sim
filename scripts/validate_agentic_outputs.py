from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


SEQUENCES = ("handstand", "stairs", "wall-kicking")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate real-demo and partial-completion agentic fitting outputs."
    )
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--cluster-root", type=Path, required=True)
    parser.add_argument("--partial-cap-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _validate_sequence(
    demo_root: Path,
    cluster_root: Path,
    sequence: str,
) -> dict[str, Any]:
    sequence_root = demo_root / sequence
    report_path = sequence_root / "report.json"
    _require(report_path.is_file(), f"{sequence}: missing {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    initial = report["initial_metrics"]
    final = report["final_metrics"]

    _require(
        report.get("input_has_frame_offsets") is True,
        f"{sequence}: exact frame offsets were not used",
    )
    _require(
        int(final["primitive_count"]) < int(initial["primitive_count"]),
        f"{sequence}: primitive count did not decrease",
    )
    frame_drop = float(
        initial["per_frame_p10_coverage"] - final["per_frame_p10_coverage"]
    )
    coverage_drop = float(initial["coverage"] - final["coverage"])
    _require(
        frame_drop <= 0.0100001,
        f"{sequence}: cumulative per-frame P10 drop {frame_drop:.6f} exceeds 0.01",
    )
    _require(
        coverage_drop <= 0.0200001,
        f"{sequence}: cumulative coverage drop {coverage_drop:.6f} exceeds 0.02",
    )

    result_root = sequence_root / "final" / "scene_mesh_sqs"
    params_path = result_root / "sqs_params.npz"
    obj_path = result_root / "scene_mesh_sqs.obj"
    urdf_path = result_root / "scene_mesh_sqs.urdf"
    for path in (params_path, obj_path, urdf_path):
        _require(path.is_file() and path.stat().st_size > 0, f"{sequence}: missing {path}")
    with np.load(params_path, allow_pickle=False) as data:
        params = np.asarray(data["params"])
    _require(
        params.shape == (int(final["primitive_count"]), 11),
        f"{sequence}: params shape {params.shape} disagrees with report",
    )
    piece_paths = sorted((result_root / "pieces").glob("part_*.obj"))
    _require(
        len(piece_paths) == int(final["primitive_count"]),
        f"{sequence}: {len(piece_paths)} OBJ pieces disagree with report",
    )

    iterations_root = sequence_root / "iterations"
    iteration_dirs = sorted(path for path in iterations_root.iterdir() if path.is_dir())
    _require(iteration_dirs, f"{sequence}: no planner iterations")
    for iteration_dir in iteration_dirs:
        for name in ("evidence.json", "codex_events.jsonl", "validated_plan.json"):
            path = iteration_dir / name
            _require(path.is_file() and path.stat().st_size > 0, f"{sequence}: missing {path}")

    final_evidence_path = sequence_root / "final_evidence" / "evidence.json"
    _require(final_evidence_path.is_file(), f"{sequence}: missing final evidence")
    final_evidence = json.loads(final_evidence_path.read_text(encoding="utf-8"))
    per_frame = final_evidence.get("selected_per_frame_evidence", [])
    _require(
        len(per_frame) >= 4,
        f"{sequence}: insufficient selected per-frame evidence ({len(per_frame)})",
    )
    for frame in per_frame:
        _require("coverage" in frame, f"{sequence}: per-frame coverage missing")
        _require(
            "visible_primitive_ids" in frame,
            f"{sequence}: visible primitive ids missing",
        )

    cluster_dir = cluster_root / sequence / "gv"
    for name in ("per_frame_segments.npz", "segments.json", "evidence_manifest.json"):
        path = cluster_dir / name
        _require(path.is_file() and path.stat().st_size > 0, f"{sequence}: missing {path}")

    return {
        "initial_primitives": int(initial["primitive_count"]),
        "final_primitives": int(final["primitive_count"]),
        "initial_coverage": float(initial["coverage"]),
        "final_coverage": float(final["coverage"]),
        "coverage_drop": coverage_drop,
        "initial_per_frame_p10": float(initial["per_frame_p10_coverage"]),
        "final_per_frame_p10": float(final["per_frame_p10_coverage"]),
        "per_frame_p10_drop": frame_drop,
        "median_residual": float(final["median_residual"]),
        "objective": float(final["objective"]),
        "planner_iterations": len(iteration_dirs),
        "selected_evidence_frames": len(per_frame),
        "exported_piece_count": len(piece_paths),
    }


def main() -> None:
    args = _parse_args()
    demo_root = args.demo_root.expanduser().resolve()
    cluster_root = args.cluster_root.expanduser().resolve()
    partial_cap_dir = args.partial_cap_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    sequence_results = {
        sequence: _validate_sequence(demo_root, cluster_root, sequence)
        for sequence in SEQUENCES
    }

    partial_verification_path = partial_cap_dir / "verification.json"
    _require(
        partial_verification_path.is_file(),
        f"partial completion: missing {partial_verification_path}",
    )
    partial = json.loads(partial_verification_path.read_text(encoding="utf-8"))
    action = partial["accepted_completion"]
    _require(
        action["type"] == "complete" and action["target_shape"] == "sphere",
        "partial completion: accepted action is not complete->sphere",
    )
    _require(
        float(partial["center_error"]) <= 0.04,
        "partial completion: center error exceeds tolerance",
    )
    _require(
        float(partial["radius_error"]) <= 0.04,
        "partial completion: radius error exceeds tolerance",
    )
    _require(
        float(partial["final_coverage"]) >= 0.98,
        "partial completion: final coverage below 0.98",
    )
    _require(
        float(partial["final_per_frame_p10"]) >= 0.98,
        "partial completion: final per-frame P10 below 0.98",
    )

    result = {
        "schema_version": 1,
        "status": "passed",
        "requirements": {
            "exact_per_frame_evidence": True,
            "fewer_primitives_on_all_real_demos": True,
            "global_coverage_drop_at_most_0_02": True,
            "cumulative_per_frame_p10_drop_at_most_0_01": True,
            "codex_plans_and_events_present": True,
            "sqs_obj_urdf_exports_consistent": True,
            "partial_observation_completion_verified": True,
        },
        "sequences": sequence_results,
        "partial_completion": partial,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
