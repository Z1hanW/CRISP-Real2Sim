from __future__ import annotations

import numpy as np

from agentic_fitting.geometry import (
    PointCloud,
    fit_box,
    fit_sphere,
    point_to_primitive_distance,
)
from agentic_fitting.metrics import accept_candidate, evaluate_fit


def test_box_distance_is_analytic() -> None:
    params = np.asarray(
        [-2.398, -2.398, 1.0, 2.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    points = np.asarray([[0.0, 0.0, 0.05], [1.0, 1.5, 0.0], [1.2, 0.0, 0.0]])
    distance = point_to_primitive_distance(points, params)
    np.testing.assert_allclose(distance, [0.0, 0.0, 0.2], atol=1.0e-5)


def test_partial_sphere_completion() -> None:
    rng = np.random.default_rng(4)
    theta = rng.uniform(0.0, 2.0 * np.pi, 6000)
    phi = rng.uniform(0.0, np.deg2rad(55.0), 6000)
    radius = 1.4
    center = np.asarray([0.5, -0.3, 1.2])
    points = center + radius * np.column_stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    )
    points += rng.normal(scale=0.004, size=points.shape)
    params, diagnostics = fit_sphere(points)
    np.testing.assert_allclose(params[8:11], center, atol=0.03)
    assert abs(float(params[2]) - radius) < 0.03
    assert diagnostics["completion_confidence"] >= 0.62
    assert diagnostics["angular_radius_deg"] >= 20.0


def test_orthogonal_partial_faces_fit_one_box_with_reference_axes() -> None:
    rng = np.random.default_rng(12)
    xy = rng.uniform([-1.0, -0.6], [1.0, 0.6], size=(5000, 2))
    top = np.column_stack([xy, np.full(len(xy), 0.5)])
    xz = rng.uniform([-1.0, -0.5], [1.0, 0.5], size=(5000, 2))
    front = np.column_stack([xz[:, 0], np.full(len(xz), -0.6), xz[:, 1]])
    points = np.concatenate([top, front]).astype(np.float32)
    points += rng.normal(scale=0.002, size=points.shape)

    references = np.asarray(
        [
            [-2.398, -2.398, 1.0, 0.6, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
            [-2.398, -2.398, 1.0, 0.5, 0.02, np.pi / 2.0, 0.0, 0.0, 0.0, -0.6, 0.0],
        ],
        dtype=np.float32,
    )
    merged = fit_box(points, reference_params=references)
    distance = point_to_primitive_distance(points, merged)
    assert float(np.quantile(distance, 0.90)) < 0.02
    np.testing.assert_allclose(
        np.sort(merged[2:5]),
        np.asarray([0.5, 0.6, 1.0]),
        atol=0.04,
    )


def test_per_frame_metrics_are_reported() -> None:
    rng = np.random.default_rng(7)
    xy = rng.uniform(-1.0, 1.0, size=(8000, 2))
    points = np.column_stack([xy, rng.normal(scale=0.002, size=len(xy))]).astype(np.float32)
    normals = np.repeat(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32), len(points), axis=0)
    cloud = PointCloud(
        points=points,
        normals=normals,
        frame_indices=np.asarray([0, 1], dtype=np.int32),
        frame_offsets=np.asarray([0, 4000, 8000], dtype=np.int64),
        extras={},
    )
    params = np.asarray(
        [[-2.398, -2.398, 1.05, 1.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    metrics = evaluate_fit(cloud, params, threshold=0.03)
    assert metrics["coverage"] > 0.98
    assert metrics["per_frame_p10_coverage"] > 0.98


def test_acceptance_limits_cumulative_per_frame_drop() -> None:
    baseline = {
        "coverage": 0.98,
        "per_frame_p10_coverage": 0.96,
        "median_residual": 0.02,
        "objective": 1.0,
        "primitive_count": 10,
    }
    current = {
        **baseline,
        "per_frame_p10_coverage": 0.954,
        "primitive_count": 9,
    }
    candidate = {
        **current,
        "per_frame_p10_coverage": 0.948,
        "primitive_count": 8,
        "objective": 1.01,
    }
    accepted, reason = accept_candidate(
        current,
        candidate,
        minimum_improvement=0.001,
        reference=baseline,
    )
    assert not accepted
    assert "from baseline" in reason
