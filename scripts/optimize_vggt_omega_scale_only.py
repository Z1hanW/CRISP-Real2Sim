#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import minimize_scalar


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Optimize one global depth scale for a VGGT-Omega CRISP scene without changing human pose."
    )
    parser.add_argument("--seq", default="stair_75")
    parser.add_argument(
        "--raw-prior",
        type=Path,
        default=None,
        help="Raw VGGT-Omega prior npz. Defaults to results/init/vslam/raw_vggt_omega_priors/<seq>.npz.",
    )
    parser.add_argument(
        "--scene-npz",
        type=Path,
        default=None,
        help="Existing 6_align scene npz used only for comparing the saved scale.",
    )
    parser.add_argument(
        "--hmr-root",
        type=Path,
        default=REPO_ROOT / "results/init/hmr_vggt_omega",
    )
    parser.add_argument(
        "--dyn-mask-root",
        type=Path,
        default=REPO_ROOT / "results/init/dyn_mask",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "_analysis/scale_only_vggt_omega",
    )
    parser.add_argument("--use-person-mask", type=int, default=1)
    parser.add_argument("--use-depth-conf", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth raw frame for scale fitting.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many used frames; 0 means no limit.")
    parser.add_argument("--max-samples-per-frame", type=int, default=12000)
    parser.add_argument("--min-valid-pixels", type=int, default=20)
    parser.add_argument("--huber-delta-m", type=float, default=0.10)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--scale-min", type=float, default=0.1)
    parser.add_argument("--scale-max", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=75)
    return parser.parse_args()


def scalar_str(value: Any) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    return str(item)


def resolve_scene_npz(seq: str, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        REPO_ROOT / "results/output/scene" / f"{seq}_vggt_omega_gv_sgd_cvd_hr.npz",
        Path("/data/far_offload/CRISP-Real2Sim/results/output/scene") / f"{seq}_vggt_omega_gv_sgd_cvd_hr.npz",
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return None


def resize_nearest(arr: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == shape_hw:
        return arr
    return cv2.resize(arr, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)


def load_person_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=True) as data:
        if "dyn_mask" not in data.files:
            return None
        mask = np.asarray(data["dyn_mask"])
    if mask.ndim == 3:
        mask = mask[0]
    mask = resize_nearest(mask.astype(np.uint8), shape_hw).astype(bool)
    return mask


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denom = float(np.sum(weights))
    if denom <= 1.0e-12:
        return float(np.mean(values))
    return float(np.sum(values * weights) / denom)


def huber_loss(residual: np.ndarray, delta: float) -> np.ndarray:
    abs_r = np.abs(residual)
    return np.where(abs_r <= delta, 0.5 * residual * residual, delta * (abs_r - 0.5 * delta))


def evaluate_scale(scale: float, x: np.ndarray, y: np.ndarray, w: np.ndarray, delta: float) -> dict[str, float]:
    residual = scale * x - y
    abs_r = np.abs(residual)
    rel = abs_r / np.maximum(np.abs(y), 1.0e-6)
    return {
        "scale": float(scale),
        "weighted_huber_m2": weighted_mean(huber_loss(residual, delta), w),
        "weighted_rmse_m": float(np.sqrt(max(weighted_mean(residual * residual, w), 0.0))),
        "weighted_mae_m": weighted_mean(abs_r, w),
        "median_abs_m": float(np.median(abs_r)),
        "p90_abs_m": float(np.percentile(abs_r, 90)),
        "median_rel": float(np.median(rel)),
        "p90_rel": float(np.percentile(rel, 90)),
    }


def main() -> None:
    args = parse_args()
    seq = str(args.seq)
    raw_prior = (
        args.raw_prior.expanduser().resolve()
        if args.raw_prior is not None
        else (REPO_ROOT / "results/init/vslam/raw_vggt_omega_priors" / f"{seq}.npz").resolve()
    )
    if not raw_prior.is_file():
        raise FileNotFoundError(f"Missing raw prior: {raw_prior}")
    scene_npz = resolve_scene_npz(seq, args.scene_npz)

    hmr_depth_dir = args.hmr_root.expanduser().resolve() / seq / "depth_out"
    if not hmr_depth_dir.is_dir():
        raise FileNotFoundError(f"Missing HMR depth directory: {hmr_depth_dir}")

    out_dir = args.out_dir.expanduser().resolve() / seq
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    raw = np.load(raw_prior, allow_pickle=True)
    depths = np.asarray(raw["depths"], dtype=np.float32)
    source_frame_indices = np.asarray(
        raw["source_frame_indices"] if "source_frame_indices" in raw.files else np.arange(depths.shape[0]),
        dtype=np.int64,
    )
    depth_conf = np.asarray(raw["depth_conf"], dtype=np.float32) if "depth_conf" in raw.files else None
    shape_hw = tuple(int(v) for v in depths.shape[1:3])

    saved_scale = None
    saved_scene = ""
    if scene_npz is not None and scene_npz.is_file():
        with np.load(scene_npz, allow_pickle=True) as data:
            saved_scale = float(np.asarray(data["scale"]).reshape(-1)[0]) if "scale" in data.files else None
            saved_scene = str(scene_npz)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    frame_stats: list[dict[str, Any]] = []

    frame_stride = max(1, int(args.frame_stride))
    max_frames = max(0, int(args.max_frames))
    for raw_i in range(0, depths.shape[0], frame_stride):
        if max_frames > 0 and len(frame_stats) >= max_frames:
            break
        source_i = int(source_frame_indices[raw_i])
        hmr_path = hmr_depth_dir / f"mesh_depth_{source_i}.npy"
        if not hmr_path.is_file():
            continue
        target = resize_nearest(np.load(hmr_path).astype(np.float32), shape_hw)
        source = depths[raw_i]
        valid = (target > 0.0) & (source > 0.0) & np.isfinite(target) & np.isfinite(source)

        if int(args.use_person_mask) == 1:
            mask_path = args.dyn_mask_root.expanduser().resolve() / seq / "person" / f"dyn_mask_{source_i}.npz"
            person_mask = load_person_mask(mask_path, shape_hw)
            if person_mask is not None:
                valid &= person_mask

        valid_count = int(valid.sum())
        if valid_count < int(args.min_valid_pixels):
            continue

        flat_idx = np.flatnonzero(valid.reshape(-1))
        if int(args.max_samples_per_frame) > 0 and flat_idx.size > int(args.max_samples_per_frame):
            flat_idx = rng.choice(flat_idx, size=int(args.max_samples_per_frame), replace=False)

        x = source.reshape(-1)[flat_idx].astype(np.float32)
        y = target.reshape(-1)[flat_idx].astype(np.float32)

        if int(args.use_depth_conf) == 1 and depth_conf is not None:
            conf = depth_conf[raw_i].reshape(-1)[flat_idx].astype(np.float32)
            conf = np.where(np.isfinite(conf) & (conf > 0.0), conf, 0.0)
            if np.any(conf > 0.0):
                med = float(np.median(conf[conf > 0.0]))
                if med > 1.0e-12:
                    conf = conf / med
                weight = np.clip(conf, 0.05, 20.0).astype(np.float32)
            else:
                weight = np.ones_like(x, dtype=np.float32)
        else:
            weight = np.ones_like(x, dtype=np.float32)

        ratio = y / np.maximum(x, 1.0e-8)
        frame_stats.append(
            {
                "raw_frame": int(raw_i),
                "source_frame": int(source_i),
                "valid_pixels": valid_count,
                "sampled_pixels": int(flat_idx.size),
                "scale_median": float(np.median(ratio)),
                "scale_mean": float(np.mean(ratio)),
                "scale_p10": float(np.percentile(ratio, 10)),
                "scale_p90": float(np.percentile(ratio, 90)),
            }
        )
        xs.append(x)
        ys.append(y)
        ws.append(weight)
        progress_every = max(0, int(args.progress_every))
        if progress_every > 0 and len(frame_stats) % progress_every == 0:
            print(
                f"[scale-only] {seq}: used_frames={len(frame_stats)} "
                f"latest_raw={raw_i} latest_source={source_i} samples={sum(len(v) for v in xs)}",
                flush=True,
            )

    if not xs:
        raise RuntimeError(f"No valid frame samples for {seq}")

    x_all = np.concatenate(xs).astype(np.float64)
    y_all = np.concatenate(ys).astype(np.float64)
    w_all = np.concatenate(ws).astype(np.float64)
    finite = np.isfinite(x_all) & np.isfinite(y_all) & np.isfinite(w_all) & (x_all > 0.0) & (y_all > 0.0) & (w_all > 0.0)
    x_all, y_all, w_all = x_all[finite], y_all[finite], w_all[finite]

    ratio_all = y_all / np.maximum(x_all, 1.0e-12)
    frame_medians = np.asarray([row["scale_median"] for row in frame_stats], dtype=np.float64)
    median_of_frame_medians = float(np.median(frame_medians))
    nearest_frame_idx = int(np.argmin(np.abs(frame_medians - median_of_frame_medians)))
    frame_median_nearest = float(frame_medians[nearest_frame_idx])
    pixel_median = float(np.median(ratio_all))
    weighted_ls = float(np.sum(w_all * x_all * y_all) / max(np.sum(w_all * x_all * x_all), 1.0e-12))
    unweighted_ls = float(np.sum(x_all * y_all) / max(np.sum(x_all * x_all), 1.0e-12))

    s_min = float(args.scale_min)
    s_max = float(args.scale_max)
    if not (s_min < s_max):
        raise ValueError(f"Invalid scale bounds: {s_min}, {s_max}")

    def objective(scale: float) -> float:
        return weighted_mean(huber_loss(scale * x_all - y_all, float(args.huber_delta_m)), w_all)

    opt = minimize_scalar(objective, bounds=(s_min, s_max), method="bounded", options={"xatol": 1.0e-8})
    huber_scale = float(opt.x)

    candidates = {
        "six_align_saved": saved_scale,
        "frame_median_nearest": frame_median_nearest,
        "median_of_frame_medians": median_of_frame_medians,
        "global_pixel_median": pixel_median,
        "unweighted_l2": unweighted_ls,
        "weighted_l2": weighted_ls,
        "weighted_huber_optimized": huber_scale,
    }
    metrics = {
        name: evaluate_scale(float(scale), x_all, y_all, w_all, float(args.huber_delta_m))
        for name, scale in candidates.items()
        if scale is not None and np.isfinite(float(scale))
    }

    summary = {
        "sequence": seq,
        "raw_prior": str(raw_prior),
        "scene_npz": saved_scene,
        "hmr_depth_dir": str(hmr_depth_dir),
        "method": (
            "Scale-only VideoMimic-inspired alignment: fixed VGGT-Omega scene/HMR pose, "
            "optimize one scalar s minimizing robust depth residual s*raw_vggt_depth - hmr_mesh_depth "
            "on valid human mesh pixels."
        ),
        "use_person_mask": int(args.use_person_mask),
        "use_depth_conf": int(args.use_depth_conf),
        "frame_stride": frame_stride,
        "max_frames": max_frames,
        "huber_delta_m": float(args.huber_delta_m),
        "scale_bounds": [s_min, s_max],
        "frames_used": len(frame_stats),
        "samples_used": int(x_all.size),
        "existing_six_align_scale": saved_scale,
        "optimized_scale": huber_scale,
        "scale_candidates": candidates,
        "metrics_on_sample": metrics,
        "optimizer": {
            "success": bool(opt.success),
            "fun": float(opt.fun),
            "nfev": int(opt.nfev),
            "message": str(opt.message),
        },
    }

    summary_path = out_dir / "scale_only_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / "scale_only_samples_and_frame_stats.npz",
        sample_raw_depth=x_all.astype(np.float32),
        sample_hmr_depth=y_all.astype(np.float32),
        sample_weight=w_all.astype(np.float32),
        frame_scale_median=frame_medians.astype(np.float32),
        frame_raw_index=np.asarray([row["raw_frame"] for row in frame_stats], dtype=np.int32),
        frame_source_index=np.asarray([row["source_frame"] for row in frame_stats], dtype=np.int32),
        frame_valid_pixels=np.asarray([row["valid_pixels"] for row in frame_stats], dtype=np.int32),
        frame_sampled_pixels=np.asarray([row["sampled_pixels"] for row in frame_stats], dtype=np.int32),
    )

    print(json.dumps({
        "sequence": seq,
        "existing_six_align_scale": saved_scale,
        "optimized_scale": huber_scale,
        "frames_used": len(frame_stats),
        "samples_used": int(x_all.size),
        "summary": str(summary_path),
        "metrics": metrics.get("weighted_huber_optimized", {}),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
