from __future__ import annotations

import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .executor import PrimitiveRecord
from .geometry import params_to_meshes


def load_params(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            params = np.asarray(data["params"], dtype=np.float32)
    else:
        params = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if params.ndim != 2 or params.shape[1] != 11:
        raise ValueError(f"Expected (N, 11) SQS parameters, got {params.shape} from {path}")
    return params


def generate_baseline(
    repo_root: Path,
    python_bin: Path,
    pointcloud_path: Path,
    output_dir: Path,
    *,
    device: str,
    max_primitives: int,
) -> Path:
    input_root = output_dir / "_baseline_input"
    baseline_root = output_dir / "baseline"
    sequence = "scene"
    target = input_root / sequence / "gv" / "nksr_input" / "pointcloud_world.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    try:
        target.symlink_to(pointcloud_path.resolve())
    except OSError:
        shutil.copy2(pointcloud_path, target)
    command = [
        str(python_bin),
        str(repo_root / "vis_scripts/viser_m/fit_sqs_from_world_points.py"),
        "--input-root",
        str(input_root),
        "--output-root",
        str(baseline_root),
        "--sequences",
        sequence,
        "--device",
        device,
        "--max-primitives",
        str(max_primitives),
        "--force",
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (output_dir / "baseline.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"Baseline fitter failed; see {output_dir / 'baseline.log'}")
    return baseline_root / sequence / "gv" / "scene_mesh_sqs" / "sqs_params.npz"


def _write_urdf(path: Path, names: list[str]) -> None:
    robot = ET.Element("robot", name="agentic_scene_sqs")
    link = ET.SubElement(robot, "link", name="scene")
    for name in names:
        for tag in ("visual", "collision"):
            section = ET.SubElement(link, tag)
            ET.SubElement(section, "origin", xyz="0 0 0", rpy="0 0 0")
            geometry = ET.SubElement(section, "geometry")
            ET.SubElement(geometry, "mesh", filename=f"pieces/{name}")
    ET.ElementTree(robot).write(path, encoding="utf-8", xml_declaration=True)


def export_result(
    output_dir: Path,
    records: list[PrimitiveRecord],
    report: dict[str, Any],
) -> None:
    result_dir = output_dir / "final" / "scene_mesh_sqs"
    pieces_dir = result_dir / "pieces"
    pieces_dir.mkdir(parents=True, exist_ok=True)
    params = np.stack([record.params for record in records]).astype(np.float32)
    confidences = np.asarray(
        [record.completion_confidence for record in records],
        dtype=np.float32,
    )
    provenance = np.asarray(
        [record.provenance for record in records],
        dtype=f"<U{max(1, max((len(record.provenance) for record in records), default=1))}",
    )
    np.save(result_dir / "sqs_params.npy", params)
    np.savez_compressed(
        result_dir / "sqs_params.npz",
        params=params,
        completion_confidence=confidences,
        provenance=provenance,
        source=np.asarray("codex_agentic_fitting"),
    )

    meshes = params_to_meshes(params)
    names = []
    for primitive_id, mesh in enumerate(meshes):
        name = f"part_{primitive_id:03d}.obj"
        mesh.export(pieces_dir / name)
        names.append(name)
    if meshes:
        merged = trimesh.util.concatenate(meshes)
    else:
        merged = trimesh.Trimesh(
            vertices=np.empty((0, 3), dtype=np.float32),
            faces=np.empty((0, 3), dtype=np.int64),
            process=False,
        )
    merged.export(result_dir / "scene_mesh_sqs.obj")
    _write_urdf(result_dir / "scene_mesh_sqs.urdf", names)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    initial = report["initial_metrics"]
    final = report["final_metrics"]
    lines = [
        "# Agentic Superquadric Fitting Report",
        "",
        "| metric | baseline | final |",
        "|---|---:|---:|",
        f"| primitives | {initial['primitive_count']} | {final['primitive_count']} |",
        f"| coverage | {initial['coverage']:.4f} | {final['coverage']:.4f} |",
        f"| median residual | {initial['median_residual']:.5f} | {final['median_residual']:.5f} |",
        f"| p90 residual | {initial['p90_residual']:.5f} | {final['p90_residual']:.5f} |",
        f"| surface precision | {initial['surface_precision']:.4f} | {final['surface_precision']:.4f} |",
        f"| per-frame P10 coverage | {initial['per_frame_p10_coverage']:.4f} | {final['per_frame_p10_coverage']:.4f} |",
        f"| objective | {initial['objective']:.5f} | {final['objective']:.5f} |",
        "",
        "## Iterations",
        "",
    ]
    for item in report["iterations"]:
        lines.append(
            f"- iteration {item['iteration']}: {item['decision']} "
            f"({item.get('reason', '')})"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
