from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a deterministic multi-frame partial-sphere completion demo."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260817)
    return parser.parse_args()


def _render_rgb(path: Path, frame_index: int, shift: int) -> None:
    fig, ax = plt.subplots(figsize=(5.12, 3.84), dpi=100)
    ax.set_xlim(0, 512)
    ax.set_ylim(384, 0)
    ax.set_facecolor("#d9dde0")
    ax.add_patch(Rectangle((0, 275), 512, 109, color="#8d9499"))
    ax.add_patch(
        Circle(
            (256 + shift, 258),
            92,
            facecolor="#3f82c5",
            edgecolor="#24527f",
            linewidth=4,
        )
    )
    ax.add_patch(Rectangle((0, 258), 512, 126, color="#737a80"))
    ax.plot([0, 512], [258, 258], color="#4f555a", linewidth=4)
    ax.text(
        18,
        30,
        f"frame {frame_index}",
        color="#202428",
        fontsize=12,
        va="top",
    )
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _render_cluster(path: Path, shift: int) -> None:
    fig, ax = plt.subplots(figsize=(5.12, 3.84), dpi=100)
    ax.set_xlim(0, 512)
    ax.set_ylim(384, 0)
    ax.set_facecolor("black")
    ax.add_patch(
        Circle(
            (256 + shift, 258),
            92,
            facecolor="#348bd4",
            edgecolor="none",
        )
    )
    ax.add_patch(Rectangle((0, 258), 512, 126, color="#56a64b"))
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    input_dir = output_dir / "input"
    rgb_dir = input_dir / "rgb"
    cluster_dir = input_dir / "clusters"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    center = np.asarray([0.0, 0.0, 1.50], dtype=np.float64)
    radius = 0.75
    maximum_phi = np.deg2rad(52.0)
    frame_indices = np.asarray([0, 10, 20, 30], dtype=np.int32)
    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    frame_offsets = [0]

    for frame_position, frame_index in enumerate(frame_indices):
        floor_xy = rng.uniform(-2.0, 2.0, size=(4000, 2))
        floor = np.column_stack(
            [floor_xy, rng.normal(scale=0.0025, size=len(floor_xy))]
        )
        floor_normals = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]]),
            len(floor),
            axis=0,
        )

        cos_phi = rng.uniform(np.cos(maximum_phi), 1.0, size=3200)
        phi = np.arccos(cos_phi)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=len(phi))
        directions = np.column_stack(
            [
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ]
        )
        cap = center + radius * directions
        cap += rng.normal(scale=0.0025, size=cap.shape)
        cap_normals = directions + rng.normal(scale=0.003, size=directions.shape)
        cap_normals /= np.linalg.norm(cap_normals, axis=1, keepdims=True)

        frame_points = np.concatenate([floor, cap]).astype(np.float32)
        frame_normals = np.concatenate([floor_normals, cap_normals]).astype(np.float32)
        order = rng.permutation(len(frame_points))
        all_points.append(frame_points[order])
        all_normals.append(frame_normals[order])
        frame_offsets.append(frame_offsets[-1] + len(frame_points))

        shift = (frame_position - 1) * 8
        _render_rgb(rgb_dir / f"{int(frame_index):06d}.jpg", int(frame_index), shift)
        _render_cluster(
            cluster_dir / f"frame_{int(frame_index):06d}_clusters.png",
            shift,
        )

    pointcloud_path = input_dir / "pointcloud_world.npz"
    np.savez_compressed(
        pointcloud_path,
        points=np.concatenate(all_points),
        normals=np.concatenate(all_normals),
        frame_indices=frame_indices,
        frame_offsets=np.asarray(frame_offsets, dtype=np.int64),
    )

    baseline = np.asarray(
        [
            [-2.398, -2.398, 2.02, 2.02, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-2.398, -2.398, 0.66, 0.66, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 2.20],
        ],
        dtype=np.float32,
    )
    baseline_path = input_dir / "baseline.npz"
    np.savez_compressed(baseline_path, params=baseline)

    manifest = {
        "schema_version": 1,
        "description": "A floor plus the upper 52-degree cap of a partially occluded sphere.",
        "pointcloud": str(pointcloud_path),
        "baseline_params": str(baseline_path),
        "image_root": str(rgb_dir),
        "cluster_root": str(cluster_dir),
        "ground_truth": {
            "sphere_center": center.tolist(),
            "sphere_radius": radius,
            "maximum_observed_phi_degrees": 52.0,
            "frame_indices": frame_indices.tolist(),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
