from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass
class PointCloud:
    points: np.ndarray
    normals: np.ndarray
    frame_indices: np.ndarray | None
    frame_offsets: np.ndarray | None
    extras: dict[str, np.ndarray]


def load_pointcloud(path: Path) -> PointCloud:
    with np.load(path, allow_pickle=False) as data:
        points = np.asarray(data["points"], dtype=np.float32)
        normals = np.asarray(data["normals"], dtype=np.float32)
        frame_indices = (
            np.asarray(data["frame_indices"], dtype=np.int32)
            if "frame_indices" in data.files
            else None
        )
        frame_offsets = (
            np.asarray(data["frame_offsets"], dtype=np.int64)
            if "frame_offsets" in data.files
            else None
        )
        extras = {
            key: np.asarray(data[key])
            for key in data.files
            if key not in {"points", "normals", "frame_indices", "frame_offsets"}
        }

    finite = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1)
    finite &= np.linalg.norm(points, axis=1) > 1.0e-8
    finite &= np.linalg.norm(normals, axis=1) > 1.0e-8
    if not finite.all():
        # Filtering destroys exact frame spans, so retain a frame id per point first.
        if frame_offsets is not None and frame_indices is not None:
            frame_ids = np.full(points.shape[0], -1, dtype=np.int32)
            for idx in range(min(len(frame_indices), len(frame_offsets) - 1)):
                frame_ids[frame_offsets[idx] : frame_offsets[idx + 1]] = idx
            frame_ids = frame_ids[finite]
            counts = np.bincount(frame_ids[frame_ids >= 0], minlength=len(frame_indices))
            frame_offsets = np.concatenate(
                [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
            )
            order = np.argsort(frame_ids, kind="stable")
            points = points[finite][order]
            normals = normals[finite][order]
        else:
            points = points[finite]
            normals = normals[finite]

    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    if frame_offsets is not None and (
        frame_indices is None
        or len(frame_offsets) != len(frame_indices) + 1
        or int(frame_offsets[-1]) != len(points)
    ):
        frame_offsets = None
    return PointCloud(points, normals, frame_indices, frame_offsets, extras)


def sample_indices(count: int, maximum: int, seed: int = 0) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, maximum, replace=False)).astype(np.int64)


def signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    return np.sign(values) * np.abs(values) ** exponent


def sample_superquadric(
    params: np.ndarray,
    *,
    latitudes: int = 22,
    longitudes: int = 44,
) -> np.ndarray:
    row = np.asarray(params, dtype=np.float64)
    eps1 = float(np.clip(row[0], 0.1, 2.0))
    eps2 = float(np.clip(row[1], 0.1, 2.0))
    eta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, latitudes)
    omega = np.linspace(-np.pi, np.pi, longitudes, endpoint=False)
    eta_grid, omega_grid = np.meshgrid(eta, omega, indexing="ij")
    cos_eta = signed_power(np.cos(eta_grid), eps1)
    vertices = np.stack(
        [
            cos_eta * signed_power(np.sin(omega_grid), eps2),
            signed_power(np.sin(eta_grid), eps1),
            cos_eta * signed_power(np.cos(omega_grid), eps2),
        ],
        axis=-1,
    ).reshape(-1, 3)
    vertices *= np.maximum(row[2:5], 1.0e-4)
    rotation = Rotation.from_euler("ZYX", row[5:8]).as_matrix()
    return (vertices @ rotation.T + row[8:11]).astype(np.float32)


def params_to_mesh(params: np.ndarray, *, latitudes: int = 22, longitudes: int = 44) -> trimesh.Trimesh:
    vertices = sample_superquadric(params, latitudes=latitudes, longitudes=longitudes)
    faces: list[list[int]] = []
    for lat in range(latitudes - 1):
        for lon in range(longitudes):
            nxt = (lon + 1) % longitudes
            a = lat * longitudes + lon
            b = lat * longitudes + nxt
            c = (lat + 1) * longitudes + lon
            d = (lat + 1) * longitudes + nxt
            faces.append([a, c, b])
            faces.append([b, c, d])
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )


def params_to_meshes(params: np.ndarray) -> list[trimesh.Trimesh]:
    return [params_to_mesh(row) for row in np.asarray(params)]


def surface_samples_by_primitive(
    params: np.ndarray,
    *,
    max_per_primitive: int = 700,
) -> tuple[np.ndarray, np.ndarray]:
    samples = []
    ids = []
    for primitive_id, row in enumerate(np.asarray(params)):
        points = sample_superquadric(row)
        keep = sample_indices(len(points), max_per_primitive, seed=primitive_id + 17)
        samples.append(points[keep])
        ids.append(np.full(len(keep), primitive_id, dtype=np.int32))
    if not samples:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)
    return np.concatenate(samples), np.concatenate(ids)


def assign_points_to_primitives(
    points: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return point_to_primitives_distance(points, params)


def point_to_primitive_distance(points: np.ndarray, params: np.ndarray) -> np.ndarray:
    row = np.asarray(params, dtype=np.float64)
    rotation = Rotation.from_euler("ZYX", row[5:8]).as_matrix()
    local = (np.asarray(points, dtype=np.float64) - row[8:11]) @ rotation
    scale = np.maximum(row[2:5], 1.0e-6)
    eps = np.clip(row[:2], 0.1, 2.0)

    if np.all(eps < 0.3):
        q = np.abs(local) - scale
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return np.abs(outside + inside).astype(np.float32)
    if eps[0] < 0.3 and eps[1] > 0.7:
        radial = np.linalg.norm(local[:, [0, 2]], axis=1)
        q = np.column_stack([radial - 0.5 * (scale[0] + scale[2]), np.abs(local[:, 1]) - scale[1]])
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return np.abs(outside + inside).astype(np.float32)

    normalized_radius = np.linalg.norm(local / scale, axis=1)
    return (np.abs(normalized_radius - 1.0) * np.min(scale)).astype(np.float32)


def point_to_primitives_distance(
    points: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    params = np.asarray(params)
    if len(params) == 0:
        return np.full(len(points), -1, dtype=np.int32), np.full(len(points), np.inf)
    best_distance = np.full(len(points), np.inf, dtype=np.float32)
    best_id = np.full(len(points), -1, dtype=np.int32)
    for primitive_id, row in enumerate(params):
        distance = point_to_primitive_distance(points, row)
        improved = distance < best_distance
        best_distance[improved] = distance[improved]
        best_id[improved] = primitive_id
    return best_id, best_distance


def _rotation_with_local_y(axis: np.ndarray) -> np.ndarray:
    y_axis = axis / max(float(np.linalg.norm(axis)), 1.0e-8)
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(y_axis, reference))) > 0.92:
        reference = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(y_axis, reference)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1.0e-8)
    z_axis = np.cross(x_axis, y_axis)
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def _box_from_rotation(
    points: np.ndarray,
    rotation: np.ndarray,
    *,
    surface: bool,
) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        rotation[:, -1] *= -1.0
    center0 = np.median(points, axis=0)
    local = (points - center0) @ rotation
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    half = np.maximum(0.5 * (hi - lo), 0.02)
    if surface:
        minor = int(np.argmin(half))
        half[minor] = min(float(half[minor]), 0.025)
    local_center = 0.5 * (lo + hi)
    center = center0 + rotation @ local_center
    euler = Rotation.from_matrix(rotation).as_euler("ZYX")
    return np.asarray([-2.398, -2.398, *half, *euler, *center], dtype=np.float32)


def _reference_box_rotations(reference_params: np.ndarray) -> list[np.ndarray]:
    params = np.atleast_2d(np.asarray(reference_params, dtype=np.float64))
    rotations = [Rotation.from_euler("ZYX", row[5:8]).as_matrix() for row in params]
    normals = [
        rotation[:, int(np.argmin(row[2:5]))]
        for row, rotation in zip(params, rotations)
    ]
    for first_idx in range(len(normals)):
        for second_idx in range(first_idx + 1, len(normals)):
            first = normals[first_idx] / max(np.linalg.norm(normals[first_idx]), 1.0e-8)
            second = normals[second_idx] / max(np.linalg.norm(normals[second_idx]), 1.0e-8)
            if abs(float(np.dot(first, second))) > 0.35:
                continue
            shared = np.cross(first, second)
            shared /= max(np.linalg.norm(shared), 1.0e-8)
            second = np.cross(shared, first)
            second /= max(np.linalg.norm(second), 1.0e-8)
            rotations.append(np.stack([first, second, shared], axis=1))
    return rotations


def fit_box(
    points: np.ndarray,
    *,
    surface: bool = False,
    reference_params: np.ndarray | None = None,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        raise ValueError("At least three points are required")
    center0 = np.median(points, axis=0)
    centered = points - center0
    _, _, axes_t = np.linalg.svd(centered, full_matrices=False)
    rotations = [axes_t.T]
    if reference_params is not None:
        rotations.extend(_reference_box_rotations(reference_params))

    keep = sample_indices(len(points), 30_000, seed=313)
    score_points = points[keep]
    candidates = [
        _box_from_rotation(points, rotation, surface=surface)
        for rotation in rotations
    ]
    scores = []
    for candidate in candidates:
        distance = point_to_primitive_distance(score_points, candidate)
        scores.append(
            float(np.median(distance))
            + 0.45 * float(np.quantile(distance, 0.90))
        )
    return candidates[int(np.argmin(scores))]


def fit_ellipsoid(points: np.ndarray) -> np.ndarray:
    row = fit_box(points)
    row[:2] = 1.0
    return row


def fit_cylinder(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    center = np.median(points, axis=0)
    centered = points - center
    _, _, axes_t = np.linalg.svd(centered, full_matrices=False)
    major_axis = axes_t[0]
    rotation = _rotation_with_local_y(major_axis)
    local = centered @ rotation
    axial_lo, axial_hi = np.quantile(local[:, 1], [0.01, 0.99])
    radial = np.linalg.norm(local[:, [0, 2]], axis=1)
    radius = max(float(np.quantile(radial, 0.9)), 0.02)
    local_center = np.array([0.0, 0.5 * (axial_lo + axial_hi), 0.0])
    center = center + rotation @ local_center
    half_length = max(0.5 * float(axial_hi - axial_lo), 0.02)
    euler = Rotation.from_matrix(rotation).as_euler("ZYX")
    return np.asarray(
        [0.12, 1.0, radius, half_length, radius, *euler, *center],
        dtype=np.float32,
    )


def fit_sphere(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 16:
        raise ValueError("At least 16 points are required for sphere completion")
    active = np.arange(len(points))
    center = np.mean(points, axis=0)
    radius = float(np.median(np.linalg.norm(points - center, axis=1)))
    for _ in range(3):
        p = points[active]
        system = np.column_stack([2.0 * p, np.ones(len(p))])
        target = np.sum(p * p, axis=1)
        solution, *_ = np.linalg.lstsq(system, target, rcond=None)
        center = solution[:3]
        radius_sq = float(solution[3] + np.dot(center, center))
        radius = np.sqrt(max(radius_sq, 1.0e-8))
        residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
        cutoff = max(float(np.quantile(residual, 0.85)), 1.0e-5)
        active = np.flatnonzero(residual <= cutoff)

    residual = np.abs(np.linalg.norm(points - center, axis=1) - radius)
    directions = points - center
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-8)
    direction_center = np.mean(directions, axis=0)
    angular_radius = float(
        np.rad2deg(
            np.quantile(
                np.arccos(np.clip(directions @ direction_center / max(np.linalg.norm(direction_center), 1.0e-8), -1.0, 1.0)),
                0.9,
            )
        )
    )
    centered = points - np.mean(points, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    plane_residual = float(singular[-1] / np.sqrt(max(len(points), 1)))
    relative_residual = float(np.median(residual) / max(radius, 1.0e-8))
    curvature_gain = float(plane_residual / max(np.median(residual), 1.0e-8))
    confidence = np.clip(
        0.45 * (1.0 - relative_residual / 0.08)
        + 0.35 * min(angular_radius / 45.0, 1.0)
        + 0.20 * min(curvature_gain / 2.0, 1.0),
        0.0,
        1.0,
    )
    row = np.asarray([1.0, 1.0, radius, radius, radius, 0.0, 0.0, 0.0, *center], dtype=np.float32)
    diagnostics = {
        "radius": float(radius),
        "relative_residual": relative_residual,
        "angular_radius_deg": angular_radius,
        "plane_residual": plane_residual,
        "curvature_gain": curvature_gain,
        "completion_confidence": float(confidence),
    }
    return row, diagnostics


def primitive_summary(params: np.ndarray) -> list[dict[str, Any]]:
    result = []
    for primitive_id, row in enumerate(np.asarray(params)):
        eps = np.clip(row[:2], 0.1, 2.0)
        if np.all(eps > 0.75) and np.max(row[2:5]) / max(np.min(row[2:5]), 1.0e-5) < 1.25:
            shape = "sphere"
        elif eps[0] < 0.3 and eps[1] > 0.7:
            shape = "cylinder"
        elif np.all(eps < 0.3):
            shape = "box"
        else:
            shape = "ellipsoid"
        result.append(
            {
                "id": primitive_id,
                "shape": shape,
                "eps": row[:2].round(4).tolist(),
                "half_scales": row[2:5].round(4).tolist(),
                "center": row[8:11].round(4).tolist(),
                "volume_proxy": float(np.prod(2.0 * row[2:5])),
            }
        )
    return result
