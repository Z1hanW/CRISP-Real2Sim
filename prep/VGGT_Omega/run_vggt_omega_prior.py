#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _add_repo_to_path(repo_path: Path | None) -> None:
    if repo_path is None:
        return
    repo_path = repo_path.expanduser().resolve()
    if not repo_path.exists():
        raise FileNotFoundError(
            f"VGGT-Omega repo not found: {repo_path}. "
            "Clone https://github.com/facebookresearch/vggt-omega.git and set VGGT_OMEGA_REPO."
        )
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))


def _sequence_name(input_path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    return input_path.stem if input_path.is_file() else input_path.name


def _image_paths_from_dir(input_dir: Path) -> list[Path]:
    image_paths = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not image_paths:
        raise FileNotFoundError(f"No image frames found in {input_dir}")
    return image_paths


def _image_paths_from_video(video_path: Path, out_dir: Path, frame_stride: int, max_frames: int) -> tuple[list[Path], list[int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    image_paths: list[Path] = []
    source_frame_indices: list[int] = []
    frame_idx = 0
    saved_idx = 0
    stride = max(1, int(frame_stride))
    max_frames = int(max_frames)
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_path = out_dir / f"{saved_idx:08d}.png"
            if not cv2.imwrite(str(frame_path), frame_bgr):
                raise RuntimeError(f"Failed to write temporary frame: {frame_path}")
            image_paths.append(frame_path)
            source_frame_indices.append(frame_idx)
            saved_idx += 1
            if max_frames > 0 and saved_idx >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if not image_paths:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return image_paths, source_frame_indices


def _strip_batch(arr: torch.Tensor | np.ndarray) -> np.ndarray:
    out = arr.detach().float().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)
    if out.ndim >= 1 and out.shape[0] == 1:
        out = out[0]
    return np.asarray(out)


def _images_to_uint8(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] == 3:
        arr = np.transpose(arr, (0, 2, 3, 1))
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Expected images shaped (T,H,W,3) or (T,3,H,W), got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        arr = (arr * 255.0).clip(0, 255)
    return arr.astype(np.uint8)


def _depths_to_float32(depth: np.ndarray) -> np.ndarray:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Expected depth shaped (T,H,W) or (T,H,W,1), got {arr.shape}")
    return np.clip(arr, 1.0e-6, 1.0e4).astype(np.float32)


def _intrinsic_compatible(intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics shaped (T,3,3), got {arr.shape}")

    compat = np.eye(3, dtype=np.float32)
    compat[0, 0] = float(np.median(arr[:, 0, 0]))
    compat[1, 1] = float(np.median(arr[:, 1, 1]))
    compat[0, 2] = float(np.median(arr[:, 0, 2]))
    compat[1, 2] = float(np.median(arr[:, 1, 2]))
    return compat, arr.astype(np.float32)


def _camera_to_world(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(extrinsics, dtype=np.float32)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3 or arr.shape[-2:] != (3, 4):
        raise ValueError(f"Expected camera-from-world extrinsics shaped (T,3,4), got {arr.shape}")

    world_to_camera = np.repeat(np.eye(4, dtype=np.float32)[None], arr.shape[0], axis=0)
    world_to_camera[:, :3, :4] = arr
    cam_c2w = np.linalg.inv(world_to_camera).astype(np.float32)
    return cam_c2w, world_to_camera.astype(np.float32)


def _load_model(checkpoint: Path, device: str, enable_alignment: bool):
    from vggt_omega.models import VGGTOmega

    if not checkpoint.exists():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint}")
    model = VGGTOmega(enable_alignment=enable_alignment).eval()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    model.load_state_dict(state)
    return model.to(device)


def run(args: argparse.Namespace) -> Path:
    repo_path = args.repo_path if args.repo_path else None
    _add_repo_to_path(repo_path)

    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    input_arg_path = args.input_path.expanduser()
    seq_name = _sequence_name(input_arg_path, args.sequence_name)
    input_path = input_arg_path.resolve()
    output_path = args.output_path
    if output_path is None:
        output_path = args.raw_priors_root / f"{seq_name}.npz"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        if input_path.is_dir():
            image_paths = _image_paths_from_dir(input_path)
            source_frame_indices = list(range(len(image_paths)))
        elif input_path.suffix.lower() in VIDEO_SUFFIXES:
            temp_dir = tempfile.TemporaryDirectory(prefix=f"vggt_omega_{seq_name}_")
            image_paths, source_frame_indices = _image_paths_from_video(
                input_path,
                Path(temp_dir.name),
                frame_stride=args.frame_stride,
                max_frames=args.max_frames,
            )
        elif input_path.suffix.lower() in IMAGE_SUFFIXES:
            image_paths = [input_path]
            source_frame_indices = [0]
        else:
            raise ValueError(f"Unsupported input path for VGGT-Omega: {input_path}")

        model = _load_model(args.checkpoint.expanduser().resolve(), args.device, args.enable_alignment)
        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=args.image_mode,
            image_resolution=args.image_resolution,
        ).to(args.device)

        with torch.inference_mode():
            predictions = model(images)
            extrinsics, intrinsics = encoding_to_camera(
                predictions["pose_enc"],
                predictions["images"].shape[-2:],
            )

        pred_images = _strip_batch(predictions.get("images", images))
        depths = _depths_to_float32(_strip_batch(predictions["depth"]))
        images_np = _images_to_uint8(pred_images)
        intrinsic, intrinsics_per_frame = _intrinsic_compatible(_strip_batch(intrinsics))
        cam_c2w, world_to_camera = _camera_to_world(_strip_batch(extrinsics))

        if not (len(images_np) == len(depths) == len(cam_c2w)):
            raise RuntimeError(
                f"VGGT-Omega output length mismatch: images={len(images_np)} depths={len(depths)} cams={len(cam_c2w)}"
            )

        save_dict: dict[str, np.ndarray] = {
            "images": images_np,
            "depths": depths,
            "intrinsic": intrinsic.astype(np.float32),
            "cam_c2w": cam_c2w,
            "scale": np.asarray(1.0, dtype=np.float32),
            "intrinsics_per_frame": intrinsics_per_frame,
            "world_to_camera": world_to_camera,
            "source_frame_indices": np.asarray(source_frame_indices, dtype=np.int32),
            "scene_reconstruction_backend": np.asarray("vggt_omega"),
            "vggt_omega_image_resolution": np.asarray(int(args.image_resolution), dtype=np.int32),
            "vggt_omega_image_mode": np.asarray(str(args.image_mode)),
        }
        if "depth_conf" in predictions:
            save_dict["depth_conf"] = _strip_batch(predictions["depth_conf"]).astype(np.float32)

        np.savez(output_path, **save_dict)
        print(f"[vggt_omega] saved CRISP raw prior: {output_path}")
        print(
            f"[vggt_omega] sequence={seq_name} frames={len(images_np)} "
            f"image_shape={tuple(images_np.shape[1:3])} depth_shape={tuple(depths.shape[1:])}"
        )
        return output_path
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VGGT-Omega and save a CRISP-compatible raw scene prior."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input video, frame directory, or image.")
    parser.add_argument("--sequence-name", default=None, help="Override sequence name; defaults to input stem/name.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Local VGGT-Omega checkpoint path.")
    parser.add_argument(
        "--repo-path",
        type=Path,
        default=Path(os.environ.get("VGGT_OMEGA_REPO", "")) if os.environ.get("VGGT_OMEGA_REPO") else None,
        help="Path to a clone of facebookresearch/vggt-omega. Optional if vggt_omega is already installed.",
    )
    parser.add_argument(
        "--raw-priors-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results/init/vslam/raw_vggt_omega_priors",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--image-mode", choices=("balanced", "max_size"), default="balanced")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional debugging cap. Keep 0 for normal CRISP runs so frame counts stay aligned.",
    )
    parser.add_argument(
        "--enable-alignment",
        action="store_true",
        help="Use VGGTOmega(enable_alignment=True), required for the text-aligned checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
