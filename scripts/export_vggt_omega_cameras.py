#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def _video_shape(video_path: Path) -> tuple[int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        frame_count += 1
    cap.release()
    if frame_count <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video metadata for {video_path}")
    return frame_count, width, height


def _select_intrinsic(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "intrinsic" in data.files:
        intrinsic = np.asarray(data["intrinsic"], dtype=np.float32)
        if intrinsic.ndim == 2:
            return intrinsic
        if intrinsic.ndim == 3:
            return intrinsic[0]
    if "intrinsics_per_frame" in data.files:
        intrinsics = np.asarray(data["intrinsics_per_frame"], dtype=np.float32)
        if intrinsics.ndim != 3:
            raise ValueError(f"Expected intrinsics_per_frame shape (T,3,3), got {intrinsics.shape}")
        return np.nanmedian(intrinsics, axis=0).astype(np.float32)
    raise KeyError("VGGT-Omega prior must contain intrinsic or intrinsics_per_frame")


def _expand_cam_c2w(cam_c2w: np.ndarray, source_indices: np.ndarray | None, frame_count: int) -> np.ndarray:
    cam_c2w = np.asarray(cam_c2w, dtype=np.float32)
    if cam_c2w.ndim != 3 or cam_c2w.shape[-2:] != (4, 4):
        raise ValueError(f"Expected cam_c2w shape (T,4,4), got {cam_c2w.shape}")
    if cam_c2w.shape[0] == frame_count:
        return cam_c2w

    if source_indices is None:
        source_indices = np.linspace(0, frame_count - 1, cam_c2w.shape[0]).round().astype(np.int32)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int32)
    if source_indices.shape[0] != cam_c2w.shape[0]:
        raise ValueError(
            f"source_frame_indices length {source_indices.shape[0]} does not match cam_c2w length {cam_c2w.shape[0]}"
        )

    order = np.argsort(source_indices)
    source_indices = source_indices[order]
    cam_sorted = cam_c2w[order]
    full_indices = np.arange(frame_count, dtype=np.int32)
    right = np.searchsorted(source_indices, full_indices, side="left")
    left = np.clip(right - 1, 0, len(source_indices) - 1)
    right = np.clip(right, 0, len(source_indices) - 1)
    use_right = np.abs(source_indices[right] - full_indices) < np.abs(full_indices - source_indices[left])
    nearest = np.where(use_right, right, left)
    return cam_sorted[nearest]


def export_one(raw_path: Path, video_path: Path, output_root: Path) -> None:
    frame_count, width, height = _video_shape(video_path)
    with np.load(raw_path, allow_pickle=False) as data:
        images = np.asarray(data["images"])
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected images shape (T,H,W,3), got {images.shape} in {raw_path}")
        raw_h, raw_w = int(images.shape[1]), int(images.shape[2])
        intrinsic = _select_intrinsic(data)
        source_indices = data["source_frame_indices"] if "source_frame_indices" in data.files else None
        cam_c2w = _expand_cam_c2w(data["cam_c2w"], source_indices, frame_count)

    ax = float(width) / float(raw_w)
    ay = float(height) / float(raw_h)
    fx = float(intrinsic[0, 0]) * ax
    fy = float(intrinsic[1, 1]) * ay
    cx = float(intrinsic[0, 2]) * ax
    cy = float(intrinsic[1, 2]) * ay
    camera = {
        "img_focal": (fx + fy) / 2.0,
        "img_center": (cx, cy),
        "cam_c2w": cam_c2w.astype(np.float32),
    }

    seq_dir = output_root / raw_path.stem
    seq_dir.mkdir(parents=True, exist_ok=True)
    np.save(seq_dir / "camera.npy", camera)
    print(
        f"[vggt-camera] {raw_path.stem}: raw={raw_w}x{raw_h} video={width}x{height} "
        f"frames={frame_count} focal={camera['img_focal']:.3f} -> {seq_dir / 'camera.npy'}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export HMR-compatible camera.npy files from VGGT-Omega priors.")
    parser.add_argument("--split-root", type=Path, required=True, help="CRISP split root; videos are read from <root>_videos.")
    parser.add_argument("--raw-priors-root", type=Path, required=True)
    parser.add_argument("--camera-output-root", type=Path, required=True)
    parser.add_argument("--pattern", default="*.npz")
    args = parser.parse_args()

    video_root = Path(f"{str(args.split_root).rstrip('/')}_videos")
    if not video_root.is_dir():
        raise FileNotFoundError(video_root)
    raw_paths = sorted(args.raw_priors_root.glob(args.pattern))
    if not raw_paths:
        raise FileNotFoundError(f"No raw priors matching {args.pattern} under {args.raw_priors_root}")

    missing: list[str] = []
    for raw_path in raw_paths:
        video_path = video_root / f"{raw_path.stem}.mp4"
        if not video_path.is_file():
            missing.append(video_path.name)
            continue
        export_one(raw_path, video_path, args.camera_output_root)

    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} videos under {video_root}; first: {preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
