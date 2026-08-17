from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .evidence import build_evidence
from .executor import PrimitiveRecord, execute_plan
from .geometry import load_pointcloud
from .io import export_result, generate_baseline, load_params
from .metrics import accept_candidate, evaluate_fit
from .planner import call_codex_planner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codex-guided, metric-gated superquadric fitting from a fused point cloud."
    )
    parser.add_argument("--pointcloud", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-params", type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--cluster-root", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--codex-bin", type=Path, default=Path("/usr/local/bin/codex"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-primitives", type=int, default=90)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--minimum-improvement", type=float, default=0.001)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _metrics_for_records(
    cloud: Any,
    records: list[PrimitiveRecord],
    *,
    threshold: float | None,
) -> dict[str, Any]:
    params = np.stack([record.params for record in records]).astype(np.float32)
    completion = {
        index: record.completion_confidence
        for index, record in enumerate(records)
        if record.completion_confidence > 0.0
    }
    return evaluate_fit(
        cloud,
        params,
        threshold=threshold,
        completion_confidence=completion,
    )


def _ordered_actions(plan: dict[str, Any]) -> list[dict[str, Any]]:
    actions = [
        action
        for action in plan.get("actions", [])
        if str(action.get("type")) != "keep"
    ]
    return sorted(
        actions,
        key=lambda action: max(int(value) for value in action["primitive_ids"]),
        reverse=True,
    )


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key != "per_frame_coverages"
    }


def _resolve_action_ids(
    action: dict[str, Any],
    origins: list[set[int]],
) -> tuple[dict[str, Any] | None, str]:
    original_ids = [int(value) for value in action["primitive_ids"]]
    id_to_currents: dict[int, list[int]] = {}
    for current_id, source_ids in enumerate(origins):
        for original_id in source_ids:
            id_to_currents.setdefault(original_id, []).append(current_id)
    missing = [value for value in original_ids if value not in id_to_currents]
    if missing:
        return None, f"original primitive ids no longer exist: {missing}"
    ambiguous = {
        value: id_to_currents[value]
        for value in original_ids
        if len(id_to_currents[value]) != 1
    }
    if ambiguous:
        return None, f"original primitive ids are ambiguous after split: {ambiguous}"
    current_ids = [id_to_currents[value][0] for value in original_ids]
    if len(set(current_ids)) != len(current_ids):
        return None, "multiple original ids resolve to the same current primitive"
    resolved = dict(action)
    resolved["primitive_ids"] = current_ids
    return resolved, ""


def _origins_after_accepted_action(
    origins: list[set[int]],
    resolved_action: dict[str, Any],
    *,
    proposed_count: int,
) -> list[set[int]]:
    primitive_ids = sorted(
        set(int(value) for value in resolved_action["primitive_ids"])
    )
    action_type = str(resolved_action["type"])
    first_id = primitive_ids[0]
    merged_origins = set().union(*(origins[value] for value in primitive_ids))
    replacement_count = 0 if action_type == "drop" else 2 if action_type == "split" else 1

    updated: list[set[int]] = []
    for current_id, source_ids in enumerate(origins):
        if current_id == first_id:
            updated.extend(set(merged_origins) for _ in range(replacement_count))
        elif current_id not in primitive_ids:
            updated.append(set(source_ids))
    if len(updated) != proposed_count:
        raise RuntimeError(
            "stable primitive-id mapping disagrees with executor output: "
            f"{len(updated)} origins for {proposed_count} records"
        )
    return updated


def main() -> None:
    args = _parse_args()
    args.pointcloud = args.pointcloud.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.repo_root = args.repo_root.expanduser().resolve()
    if args.output_dir.exists() and args.force:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cloud = load_pointcloud(args.pointcloud)
    baseline_params_path = args.baseline_params
    if baseline_params_path is None:
        baseline_params_path = generate_baseline(
            args.repo_root,
            args.python_bin,
            args.pointcloud,
            args.output_dir,
            device=args.device,
            max_primitives=args.max_primitives,
        )
    params = load_params(baseline_params_path.expanduser().resolve())
    records = [PrimitiveRecord(row.copy()) for row in params]
    current_metrics = _metrics_for_records(cloud, records, threshold=args.threshold)
    initial_metrics = current_metrics.copy()
    print(
        json.dumps(
            {
                "event": "baseline",
                "metrics": current_metrics,
                "pointcloud": str(args.pointcloud),
            }
        ),
        flush=True,
    )

    history: list[dict[str, Any]] = []
    for iteration in range(args.max_iterations):
        iteration_dir = args.output_dir / "iterations" / f"{iteration:02d}"
        evidence = build_evidence(
            cloud,
            np.stack([record.params for record in records]),
            current_metrics,
            iteration_dir,
            image_root=args.image_root,
            cluster_root=args.cluster_root,
            iteration=iteration,
        )
        evidence["summary"]["previous_iterations"] = [
            {
                "iteration": item["iteration"],
                "decision": item["decision"],
                "reason": item.get("reason", ""),
                "plan": item.get("plan", {}),
            }
            for item in history
        ]
        plan = call_codex_planner(
            iteration_dir,
            evidence["summary"],
            evidence["attached_images"],
            codex_bin=args.codex_bin,
        )
        (iteration_dir / "validated_plan.json").write_text(
            json.dumps(plan, indent=2),
            encoding="utf-8",
        )
        if plan.get("stop") or not plan.get("actions"):
            item = {
                "iteration": iteration,
                "decision": "stopped",
                "reason": "planner requested stop or returned no actions",
                "plan": plan,
            }
            history.append(item)
            print(json.dumps({"event": "iteration", **item}), flush=True)
            break

        action_audit: list[dict[str, Any]] = []
        accepted_count = 0
        origins = [{index} for index in range(len(records))]
        claimed_original_ids: set[int] = set()
        for action in _ordered_actions(plan):
            original_ids = set(int(value) for value in action["primitive_ids"])
            overlap = claimed_original_ids.intersection(original_ids)
            if overlap:
                action_audit.append(
                    {
                        "action": action,
                        "status": "rejected",
                        "reason": f"planner reused original primitive ids: {sorted(overlap)}",
                    }
                )
                continue
            claimed_original_ids.update(original_ids)
            resolved_action, resolve_error = _resolve_action_ids(action, origins)
            if resolved_action is None:
                action_audit.append(
                    {
                        "action": action,
                        "status": "rejected",
                        "reason": resolve_error,
                    }
                )
                continue
            action_plan = {
                "scene_assessment": plan.get("scene_assessment", ""),
                "stop": False,
                "actions": [resolved_action],
            }
            proposed, execution_audit = execute_plan(
                cloud,
                records,
                action_plan,
                threshold=float(current_metrics["threshold"]),
            )
            audit_entry = execution_audit[0] if execution_audit else {
                "action": resolved_action,
                "status": "rejected",
                "reason": "executor returned no audit entry",
            }
            audit_entry["action"] = action
            if resolved_action["primitive_ids"] != action["primitive_ids"]:
                audit_entry["resolved_primitive_ids"] = resolved_action["primitive_ids"]
            if audit_entry["status"] != "executed" or len(proposed) == 0:
                action_audit.append(audit_entry)
                continue

            candidate_metrics = _metrics_for_records(
                cloud,
                proposed,
                threshold=float(current_metrics["threshold"]),
            )
            accepted, reason = accept_candidate(
                current_metrics,
                candidate_metrics,
                minimum_improvement=args.minimum_improvement,
                reference=initial_metrics,
            )
            audit_entry["metric_decision"] = "accepted" if accepted else "rejected"
            audit_entry["metric_reason"] = reason
            audit_entry["candidate_metrics"] = _compact_metrics(candidate_metrics)
            action_audit.append(audit_entry)
            if accepted:
                origins = _origins_after_accepted_action(
                    origins,
                    resolved_action,
                    proposed_count=len(proposed),
                )
                records = proposed
                current_metrics = candidate_metrics
                accepted_count += 1

        accepted = accepted_count > 0
        reason = (
            f"{accepted_count} action(s) passed independent metric gates"
            if accepted
            else "no action passed execution and metric gates"
        )
        item = {
            "iteration": iteration,
            "decision": "accepted" if accepted else "rejected",
            "reason": reason,
            "plan": plan,
            "action_audit": action_audit,
            "candidate_metrics": current_metrics,
        }
        history.append(item)
        print(json.dumps({"event": "iteration", **item}), flush=True)

    final_dir = args.output_dir / "final_evidence"
    build_evidence(
        cloud,
        np.stack([record.params for record in records]),
        current_metrics,
        final_dir,
        image_root=args.image_root,
        cluster_root=args.cluster_root,
        iteration=len(history),
        include_merge_candidates=False,
    )
    report = {
        "schema_version": 1,
        "pointcloud": str(args.pointcloud),
        "baseline_params": str(baseline_params_path),
        "input_has_frame_offsets": cloud.frame_offsets is not None,
        "initial_metrics": initial_metrics,
        "final_metrics": current_metrics,
        "iterations": history,
    }
    export_result(args.output_dir, records, report)
    print(
        json.dumps(
            {
                "event": "complete",
                "output": str(args.output_dir),
                "metrics": current_metrics,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
