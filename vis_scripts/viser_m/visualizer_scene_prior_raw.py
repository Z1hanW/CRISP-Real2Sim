#!/usr/bin/env python3
"""Visualize raw scene-reconstruction priors as point clouds and cameras."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import viser
import viser.transforms as tf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize raw step3 scene reconstruction output."
    )
    parser.add_argument("--data", type=Path, required=True, help="Raw prior .npz.")
    parser.add_argument("--port", type=int, default=9143)
    parser.add_argument("--max-points-per-layer", type=int, default=1_200_000)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--camera-stride", type=int, default=10)
    parser.add_argument("--image-downsample", type=int, default=6)
    parser.add_argument("--point-size", type=float, default=0.004)
    parser.add_argument("--camera-scale", type=float, default=None)
    parser.add_argument("--depth-min", type=float, default=1.0e-6)
    parser.add_argument("--depth-max", type=float, default=np.inf)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional per-frame mask directory to remove from the point cloud.",
    )
    parser.add_argument("--mask-key", default="dyn_mask")
    parser.add_argument(
        "--mask-dilate",
        type=int,
        default=3,
        help="Dilate the resized mask by this many pixels before carving.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold for non-bool masks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--confidence-quantiles",
        type=float,
        nargs="*",
        default=(0.0, 0.2),
        help=(
            "Depth confidence quantiles to visualize. Values >1 are treated as "
            "percentages. q=0 is the unfiltered raw sample."
        ),
    )
    parser.add_argument(
        "--up",
        choices=("+x", "-x", "+y", "-y", "+z", "-z"),
        default="-z",
    )
    return parser.parse_args()


def _normalize_quantiles(values: list[float] | tuple[float, ...]) -> list[float]:
    quantiles: list[float] = []
    for value in values:
        q = float(value)
        if q > 1.0:
            q /= 100.0
        q = min(max(q, 0.0), 1.0)
        if not any(abs(q - existing) < 1.0e-9 for existing in quantiles):
            quantiles.append(q)
    return quantiles or [0.0]


def _get_poses(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "cam_c2w" in data.files:
        return data["cam_c2w"].astype(np.float32)
    if "world_to_camera" in data.files:
        return np.linalg.inv(data["world_to_camera"].astype(np.float64)).astype(
            np.float32
        )
    raise KeyError("Raw prior must contain either 'cam_c2w' or 'world_to_camera'.")


def _get_intrinsics(data: np.lib.npyio.NpzFile, num_frames: int) -> np.ndarray:
    if "intrinsics_per_frame" in data.files:
        intrinsics = data["intrinsics_per_frame"].astype(np.float32)
    elif "intrinsic" in data.files:
        intrinsics = np.repeat(data["intrinsic"][None].astype(np.float32), num_frames, 0)
    else:
        raise KeyError("Raw prior must contain 'intrinsics_per_frame' or 'intrinsic'.")
    if intrinsics.shape[0] != num_frames:
        raise ValueError(
            f"Expected {num_frames} intrinsic matrices, got {intrinsics.shape[0]}."
        )
    return intrinsics


def _crop_to_supported_aspect_ratio(
    mask: np.ndarray,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> np.ndarray:
    height, width = mask.shape[:2]
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return mask[:, left : left + crop_width]

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return mask[top : top + crop_height]

    return mask


def _read_mask(path: Path, key: str, threshold: float) -> np.ndarray:
    data = np.load(path)
    if key in data.files:
        mask = data[key]
    elif len(data.files) == 1:
        mask = data[data.files[0]]
    else:
        raise KeyError(f"{path} does not contain key {key!r}; keys={data.files}")

    mask = np.asarray(mask)
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask in {path}, got {mask.shape}")
    if mask.dtype == np.bool_:
        return mask
    return mask.astype(np.float32) > float(threshold)


def _load_carve_masks(
    mask_dir: Path | None,
    source_indices: np.ndarray,
    target_shape: tuple[int, int],
    key: str,
    threshold: float,
    dilate: int,
) -> np.ndarray | None:
    if mask_dir is None:
        return None
    mask_dir = mask_dir.expanduser().resolve()
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    target_h, target_w = target_shape
    kernel = None
    if dilate > 0:
        k = int(dilate) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)

    masks = np.zeros((len(source_indices), target_h, target_w), dtype=bool)
    missing: list[str] = []
    for frame_idx, source_idx in enumerate(source_indices):
        path = mask_dir / f"dyn_mask_{int(source_idx)}.npz"
        if not path.exists():
            fallback = mask_dir / f"dyn_mask_{frame_idx}.npz"
            path = fallback if fallback.exists() else path
        if not path.exists():
            missing.append(path.name)
            continue

        mask = _read_mask(path, key=key, threshold=threshold)
        mask = _crop_to_supported_aspect_ratio(mask)
        resized = cv2.resize(
            mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        )
        if kernel is not None:
            resized = cv2.dilate(resized, kernel, iterations=1)
        masks[frame_idx] = resized.astype(bool)

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} masks under {mask_dir}; first missing: {preview}"
        )

    removed = masks.reshape(len(source_indices), -1).mean(axis=1)
    print(
        "[raw-prior] carve mask loaded: "
        f"{mask_dir} mean coverage={removed.mean():.4f} "
        f"range=({removed.min():.4f}, {removed.max():.4f})",
        flush=True,
    )
    return masks


def _orthonormalize_rotation(rotation: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(rotation.astype(np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result.astype(np.float32)


def _estimate_conf_threshold(confidence: np.ndarray | None, quantile: float) -> float:
    if confidence is None or quantile <= 0.0:
        return -np.inf
    flat = confidence.reshape(-1)
    stride = max(1, flat.size // 5_000_000)
    sample = flat[::stride]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return -np.inf
    return float(np.quantile(sample, quantile))


def _unproject_sampled_layer(
    images: np.ndarray,
    depths: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
    confidence: np.ndarray | None,
    confidence_threshold: float,
    carve_masks: np.ndarray | None,
    frame_indices: np.ndarray,
    max_points: int,
    depth_min: float,
    depth_max: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    points_by_frame: list[np.ndarray] = []
    colors_by_frame: list[np.ndarray] = []
    per_frame_budget = max(1, int(np.ceil(max_points / max(1, len(frame_indices)))))

    for frame_count, frame_idx in enumerate(frame_indices, start=1):
        depth = depths[frame_idx]
        valid = np.isfinite(depth) & (depth > depth_min)
        if np.isfinite(depth_max):
            valid &= depth < depth_max
        if confidence is not None and np.isfinite(confidence_threshold):
            valid &= confidence[frame_idx] >= confidence_threshold
        if carve_masks is not None:
            valid &= ~carve_masks[frame_idx]

        valid_flat = np.flatnonzero(valid.reshape(-1))
        if valid_flat.size == 0:
            continue
        if valid_flat.size > per_frame_budget:
            valid_flat = rng.choice(valid_flat, size=per_frame_budget, replace=False)

        height, width = depth.shape
        ys, xs = np.divmod(valid_flat, width)
        z = depth.reshape(-1)[valid_flat].astype(np.float32)

        k = intrinsics[frame_idx]
        fx, fy = float(k[0, 0]), float(k[1, 1])
        cx, cy = float(k[0, 2]), float(k[1, 2])
        cam_points = np.stack(
            (
                (xs.astype(np.float32) - cx) / fx * z,
                (ys.astype(np.float32) - cy) / fy * z,
                z,
            ),
            axis=1,
        )

        pose = poses_c2w[frame_idx]
        world_points = cam_points @ pose[:3, :3].T + pose[:3, 3]
        points_by_frame.append(world_points.astype(np.float32))
        colors_by_frame.append(images[frame_idx].reshape(height * width, 3)[valid_flat])

        if frame_count % 25 == 0 or frame_count == len(frame_indices):
            print(
                f"[raw-prior] sampled {frame_count}/{len(frame_indices)} frames "
                f"for threshold {confidence_threshold:.4g}",
                flush=True,
            )

    if not points_by_frame:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(points_by_frame, axis=0)
    colors = np.concatenate(colors_by_frame, axis=0).astype(np.uint8)

    if points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        colors = colors[keep]

    finite = np.isfinite(points).all(axis=1)
    return points[finite], colors[finite]


def _color_ramp(t: float) -> tuple[int, int, int]:
    t = min(max(float(t), 0.0), 1.0)
    return (
        int(40 + 200 * t),
        int(180 - 80 * t),
        int(255 - 200 * t),
    )


def _camera_scale(poses_c2w: np.ndarray, override: float | None) -> float:
    if override is not None:
        return float(override)
    centers = poses_c2w[:, :3, 3]
    lo = np.nanpercentile(centers, 5, axis=0)
    hi = np.nanpercentile(centers, 95, axis=0)
    spread = float(np.linalg.norm(hi - lo))
    return max(spread * 0.04, 0.02)


def main() -> None:
    args = _parse_args()
    if not args.data.exists():
        raise FileNotFoundError(args.data)

    print(f"[raw-prior] loading {args.data}", flush=True)
    data = np.load(args.data, allow_pickle=False)
    images = data["images"].astype(np.uint8)
    depths = data["depths"].astype(np.float32)
    poses_c2w = _get_poses(data)
    intrinsics = _get_intrinsics(data, images.shape[0])
    confidence = data["depth_conf"].astype(np.float32) if "depth_conf" in data.files else None
    source_indices = (
        data["source_frame_indices"].astype(np.int32)
        if "source_frame_indices" in data.files
        else np.arange(images.shape[0], dtype=np.int32)
    )

    num_frames, height, width = images.shape[:3]
    if depths.shape[:3] != (num_frames, height, width):
        raise ValueError(
            f"Depth shape {depths.shape} does not match images shape {images.shape}."
        )
    if poses_c2w.shape[0] != num_frames:
        raise ValueError(f"Expected {num_frames} poses, got {poses_c2w.shape[0]}.")

    frame_indices = np.arange(0, num_frames, max(1, args.frame_stride), dtype=np.int32)
    quantiles = _normalize_quantiles(args.confidence_quantiles)
    carve_masks = _load_carve_masks(
        mask_dir=args.mask_dir,
        source_indices=source_indices,
        target_shape=(height, width),
        key=args.mask_key,
        threshold=args.mask_threshold,
        dilate=max(0, args.mask_dilate),
    )

    server = viser.ViserServer(port=args.port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")
    server.scene.set_up_direction(args.up)

    camera_scale = _camera_scale(poses_c2w, args.camera_scale)
    server.scene.add_frame(
        "/raw_prior/world",
        axes_length=camera_scale * 3.0,
        axes_radius=camera_scale * 0.04,
    )

    point_handles = []
    with server.gui.add_folder("Raw step3"):
        server.gui.add_markdown(f"Source: `{args.data.name}`")
        server.gui.add_markdown(
            f"Frames: `{num_frames}` | image: `{width}x{height}` | "
            f"frame stride: `{max(1, args.frame_stride)}`"
        )
        if args.mask_dir is not None:
            server.gui.add_markdown(
                f"Carved mask: `{args.mask_dir.name}` | dilate: `{max(0, args.mask_dilate)}`"
            )

    for q_idx, quantile in enumerate(quantiles):
        threshold = _estimate_conf_threshold(confidence, quantile)
        label = "q0_raw" if quantile <= 0.0 else f"q{int(round(quantile * 100)):02d}_conf"
        print(
            f"[raw-prior] building layer {label}: confidence threshold {threshold}",
            flush=True,
        )
        points, colors = _unproject_sampled_layer(
            images=images,
            depths=depths,
            intrinsics=intrinsics,
            poses_c2w=poses_c2w,
            confidence=confidence,
            confidence_threshold=threshold,
            carve_masks=carve_masks,
            frame_indices=frame_indices,
            max_points=max(1, args.max_points_per_layer),
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            seed=args.seed + q_idx * 1009,
        )
        print(f"[raw-prior] layer {label}: {points.shape[0]} points", flush=True)
        handle = server.scene.add_point_cloud(
            name=f"/raw_prior/scene/{label}",
            points=points,
            colors=colors,
            point_size=args.point_size,
            point_shape="rounded",
            precision="float32",
            visible=quantile <= 0.0,
        )
        point_handles.append((label, handle, threshold, points.shape[0]))

    camera_handles = []
    center_colors = np.array(
        [_color_ramp(i / max(1, num_frames - 1)) for i in range(num_frames)],
        dtype=np.uint8,
    )
    camera_centers = poses_c2w[:, :3, 3].astype(np.float32)
    camera_centers_handle = server.scene.add_point_cloud(
        "/raw_prior/cameras/centers",
        points=camera_centers,
        colors=center_colors,
        point_size=max(args.point_size * 3.0, camera_scale * 0.08),
        point_shape="circle",
        precision="float32",
    )
    camera_handles.append(camera_centers_handle)

    camera_indices = np.arange(0, num_frames, max(1, args.camera_stride), dtype=np.int32)
    for display_i, frame_idx in enumerate(camera_indices):
        rgb = images[frame_idx]
        k = intrinsics[frame_idx]
        fov = 2.0 * np.arctan2(height / 2.0, float(k[1, 1]))
        aspect = width / height
        pose = poses_c2w[frame_idx]
        source_idx = int(source_indices[frame_idx])
        color = _color_ramp(frame_idx / max(1, num_frames - 1))
        frustum = server.scene.add_camera_frustum(
            name=f"/raw_prior/cameras/frustum_{int(frame_idx):04d}_src_{source_idx:04d}",
            fov=float(fov),
            aspect=float(aspect),
            scale=camera_scale,
            image=np.ascontiguousarray(
                rgb[:: max(1, args.image_downsample), :: max(1, args.image_downsample)]
            ),
            wxyz=tf.SO3.from_matrix(_orthonormalize_rotation(pose[:3, :3])).wxyz,
            position=pose[:3, 3],
            color=color,
            line_width=1.5,
        )
        camera_handles.append(frustum)
        if display_i % 25 == 0:
            print(
                f"[raw-prior] added camera frustum {display_i + 1}/{len(camera_indices)}",
                flush=True,
            )

    with server.gui.add_folder("Layers"):
        for label, handle, threshold, count in point_handles:
            suffix = "raw sample" if label == "q0_raw" else f"conf >= {threshold:.4g}"
            checkbox = server.gui.add_checkbox(
                f"{label} ({count:,})",
                initial_value=handle.visible,
                hint=suffix,
            )

            @checkbox.on_update
            def _(_, checkbox=checkbox, handle=handle) -> None:
                handle.visible = checkbox.value

        camera_checkbox = server.gui.add_checkbox("Cameras", initial_value=True)

        @camera_checkbox.on_update
        def _(_) -> None:
            with server.atomic():
                for handle in camera_handles:
                    handle.visible = camera_checkbox.value

        point_size_slider = server.gui.add_slider(
            "Point size x1000",
            min=1,
            max=30,
            step=1,
            initial_value=max(1, int(round(args.point_size * 1000))),
        )

        @point_size_slider.on_update
        def _(_) -> None:
            size = float(point_size_slider.value) / 1000.0
            for _, handle, _, _ in point_handles:
                handle.point_size = size

    print(
        f"[raw-prior] ready: http://localhost:{args.port} "
        "(raw scene depth + cameras only)",
        flush=True,
    )
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
