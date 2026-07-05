#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


THIS_FILE = Path(__file__).resolve()
CRISP_ROOT = THIS_FILE.parents[2]
FAR_ROOT = CRISP_ROOT.parent
GMR_ROOT = FAR_ROOT / "GMR"


def _resolve_holosoma_retargeting_root() -> Path:
    candidates: list[Path] = []

    if os.environ.get("HOLOSOMA_RETARGETING_ROOT"):
        candidates.append(Path(os.environ["HOLOSOMA_RETARGETING_ROOT"]).expanduser())

    if os.environ.get("HOLOSOMA_ROOT"):
        holosoma_root = Path(os.environ["HOLOSOMA_ROOT"]).expanduser()
        candidates.extend(
            [
                holosoma_root / "src" / "holosoma_retargeting" / "holosoma_retargeting",
                holosoma_root / "src" / "holosoma_retargeting",
            ]
        )

    candidates.extend(
        [
            CRISP_ROOT / "real2sim2real" / "src" / "holosoma_retargeting" / "holosoma_retargeting",
            CRISP_ROOT / "real2sim2real" / "src" / "holosoma_retargeting",
            FAR_ROOT / "holosoma" / "src" / "holosoma_retargeting" / "holosoma_retargeting",
            FAR_ROOT / "holosoma" / "src" / "holosoma_retargeting",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "examples" / "robot_retarget.py").is_file() and (candidate / "models").is_dir():
            return candidate

    checked = "\n  - ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "Could not resolve Holosoma retargeting root. Checked:\n"
        f"  - {checked}\n"
        "Set HOLOSOMA_RETARGETING_ROOT or initialize the real2sim2real submodule."
    )


def _default_holosoma_conda_exe() -> str:
    if os.environ.get("HOLOSOMA_CONDA_EXE"):
        return os.environ["HOLOSOMA_CONDA_EXE"]
    holosoma_conda = Path.home() / ".holosoma_deps" / "miniconda3" / "bin" / "conda"
    if holosoma_conda.exists():
        return str(holosoma_conda)
    return os.environ.get("CONDA_EXE", "conda")


HOLOSOMA_RT_ROOT = _resolve_holosoma_retargeting_root()
HOLOSOMA_PYTHONPATH_ROOT = HOLOSOMA_RT_ROOT.parent

if str(GMR_ROOT) not in sys.path:
    sys.path.insert(0, str(GMR_ROOT))

DEFAULT_POST_SCENE_ROOT = CRISP_ROOT / "results" / "output" / "post_scene"
DEFAULT_HMR_INIT_ROOT = CRISP_ROOT / "results" / "init" / "hmr"
DEFAULT_OUTPUT_ROOT = CRISP_ROOT / "results" / "output" / "retargeting"
DEFAULT_GMR_BODY_MODELS = GMR_ROOT / "assets" / "body_models"
DEFAULT_CRISP_SMPLX_MODELS = CRISP_ROOT / "prep" / "data" / "smplx" / "models" / "smplx"

HOLOSOMA_ROBOT_URDFS = {
    "g1": HOLOSOMA_RT_ROOT / "models" / "g1" / "g1_29dof.urdf",
    "t1": HOLOSOMA_RT_ROOT / "models" / "t1" / "t1_23dof.urdf",
}

COMMON_TO_GMR_ROBOT = {
    "g1": "unitree_g1",
    "t1": "booster_t1",
}

def _log(msg: str) -> None:
    print(msg, flush=True)


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
) -> None:
    rendered = shlex.join(str(part) for part in cmd)
    if cwd is not None:
        _log(f"[cmd] (cwd={cwd}) {rendered}")
    else:
        _log(f"[cmd] {rendered}")
    run_env = None
    if env:
        run_env = os.environ.copy()
        run_env.update(env)
        if "PYTHONPATH" in env:
            _log(f"[env] PYTHONPATH={env['PYTHONPATH']}")
    if dry_run:
        return
    subprocess.run([str(part) for part in cmd], cwd=str(cwd) if cwd else None, check=True, env=run_env)


def _ensure_file(path: Path, desc: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing {desc}: {path}")
    return path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _scalar_float(value: object, default: float = 0.0) -> float:
    arr = np.asarray(value)
    if arr.size == 0:
        return float(default)
    return float(arr.reshape(-1)[0])


def _decode_names(raw_names: object | None) -> list[str]:
    if raw_names is None:
        return []
    out: list[str] = []
    for item in np.asarray(raw_names).reshape(-1).tolist():
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def _resample_linear_to_num_frames(values: np.ndarray, target_num_frames: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError(f"Expected at least 2D array for resampling, got {arr.shape}")
    if target_num_frames <= 0:
        raise ValueError(f"target_num_frames must be positive, got {target_num_frames}")
    if arr.shape[0] == target_num_frames:
        return arr.astype(np.float32)
    if arr.shape[0] == 1:
        return np.repeat(arr.astype(np.float32), target_num_frames, axis=0)

    src_t = np.linspace(0.0, 1.0, arr.shape[0], dtype=np.float64)
    dst_t = np.linspace(0.0, 1.0, target_num_frames, dtype=np.float64)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty((target_num_frames, flat.shape[1]), dtype=np.float32)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, flat[:, col]).astype(np.float32)
    return out.reshape((target_num_frames,) + arr.shape[1:]).astype(np.float32)


def _load_joint_frames_npz(
    gmr_input: Path,
    *,
    smplx_root: Path,
    tgt_fps: int,
    load_smplx_file,
    get_smplx_data_offline_fast,
) -> tuple[list[dict[str, list[np.ndarray]]], float, float, np.ndarray, list[str]]:
    with np.load(str(gmr_input), allow_pickle=True) as data:
        if "body_positions_world" not in data or "body_names" not in data:
            raise KeyError(
                f"joint_frames_npz input requires body_positions_world and body_names: {gmr_input}"
            )
        body_positions = np.asarray(data["body_positions_world"], dtype=np.float32)
        body_names = _decode_names(data["body_names"])
        source_smplx_raw = str(np.asarray(data["source_smplx_npz"]).reshape(-1)[0]).strip() if "source_smplx_npz" in data else ""
        source_smplx_npz = Path(source_smplx_raw).resolve() if source_smplx_raw else None
        actual_human_height = (
            _scalar_float(data["actual_human_height_m"])
            if "actual_human_height_m" in data
            else float("nan")
        )
        mocap_frame_rate = _scalar_float(data["mocap_frame_rate"], float(tgt_fps)) if "mocap_frame_rate" in data else float(tgt_fps)

    if body_positions.ndim != 3 or body_positions.shape[-1] != 3:
        raise ValueError(f"Expected body_positions_world with shape (T,B,3), got {body_positions.shape}")
    if len(body_names) != body_positions.shape[1]:
        raise ValueError(
            f"body_names length mismatch for {gmr_input}: {len(body_names)} vs {body_positions.shape[1]}"
        )

    source_frames = None
    smplx_output = None
    body_model = None
    if source_smplx_npz is not None and source_smplx_npz.exists():
        smplx_data, body_model, smplx_output, source_human_height = load_smplx_file(str(source_smplx_npz), smplx_root)
        source_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=tgt_fps,
        )
        if not np.isfinite(actual_human_height) or actual_human_height <= 1e-8:
            actual_human_height = float(source_human_height)
        body_positions = _resample_linear_to_num_frames(body_positions, len(source_frames))
    else:
        aligned_fps = float(mocap_frame_rate)
        if not np.isfinite(actual_human_height) or actual_human_height <= 1e-8:
            z_extent = float(np.nanmax(body_positions[..., 2]) - np.nanmin(body_positions[..., 2]))
            actual_human_height = z_extent if 1.2 <= z_extent <= 2.4 else 1.7
        source_frames = [{} for _ in range(body_positions.shape[0])]

    merged_frames: list[dict[str, list[np.ndarray]]] = []
    identity_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    for frame_idx, source_frame in enumerate(source_frames):
        merged_frame: dict[str, list[np.ndarray]] = {}
        for body_name, payload in source_frame.items():
            pos = np.asarray(payload[0], dtype=np.float32).copy()
            quat = np.asarray(payload[1], dtype=np.float32).copy()
            merged_frame[str(body_name)] = [pos, quat]
        for body_idx, body_name in enumerate(body_names):
            quat = merged_frame[body_name][1] if body_name in merged_frame else identity_quat.copy()
            merged_frame[body_name] = [np.asarray(body_positions[frame_idx, body_idx], dtype=np.float32), quat]
        merged_frames.append(merged_frame)

    if smplx_output is not None:
        del smplx_output
    if body_model is not None:
        del body_model
    return merged_frames, float(aligned_fps), float(actual_human_height), body_positions, body_names


def _resolve_holosoma_input(
    seq_name: str,
    hmr_type: str,
    post_scene_root: Path,
    output_root: Path,
) -> tuple[Path, Path]:
    hmr_dir = post_scene_root / seq_name / hmr_type / "hmr"
    direct_npz = hmr_dir / f"{seq_name}.npz"
    if direct_npz.exists():
        return hmr_dir, direct_npz

    fallback_npz = hmr_dir / "hps_track_smplx.npz"
    if fallback_npz.exists():
        staging_dir = _ensure_dir(output_root / "_inputs" / "holosoma" / seq_name / hmr_type)
        staged_npz = staging_dir / f"{seq_name}.npz"
        if staged_npz.exists() or staged_npz.is_symlink():
            staged_npz.unlink()
        staged_npz.symlink_to(fallback_npz)
        return staging_dir, staged_npz

    raise FileNotFoundError(
        f"Could not find holosoma-ready SMPL-X joints for sequence '{seq_name}'. "
        f"Checked: {direct_npz} and {fallback_npz}"
    )


def _resolve_gmr_input(
    seq_name: str,
    hmr_init_root: Path,
    explicit_input_file: Path | None,
    input_format: str,
) -> Path:
    if explicit_input_file is not None:
        return _ensure_file(explicit_input_file, f"GMR input file ({input_format})")
    if input_format == "gvhmr":
        return _ensure_file(hmr_init_root / seq_name / "hmr4d_results.pt", "GMR GVHMR prediction")
    raise FileNotFoundError(
        f"GMR input format '{input_format}' requires --gmr-input-file. "
        f"No default path is defined for sequence '{seq_name}'."
    )


def _ensure_gmr_body_models(
    gmr_body_models_root: Path,
    crisp_smplx_models_root: Path,
    *,
    dry_run: bool = False,
) -> Path:
    smplx_dir = gmr_body_models_root / "smplx"
    needed = ("SMPLX_NEUTRAL.pkl", "SMPLX_MALE.pkl", "SMPLX_FEMALE.pkl")

    if all((smplx_dir / name).exists() for name in needed):
        return smplx_dir

    if not crisp_smplx_models_root.exists():
        raise FileNotFoundError(
            "GMR needs SMPL-X body models, but none were found under "
            f"{gmr_body_models_root} or {crisp_smplx_models_root}"
        )

    if dry_run:
        _log(f"[dry-run] Would link {smplx_dir} -> {crisp_smplx_models_root}")
        return smplx_dir

    _ensure_dir(gmr_body_models_root)
    if smplx_dir.exists() or smplx_dir.is_symlink():
        if smplx_dir.is_symlink() or smplx_dir.is_file():
            smplx_dir.unlink()
        else:
            raise RuntimeError(
                f"Refusing to replace existing directory at {smplx_dir}. "
                "Move it aside or populate it with SMPL-X models directly."
            )
    smplx_dir.symlink_to(crisp_smplx_models_root)
    return smplx_dir


def _resolve_holosoma_robot(robot: str) -> tuple[str, Path]:
    if robot not in HOLOSOMA_ROBOT_URDFS:
        supported = ", ".join(sorted(HOLOSOMA_ROBOT_URDFS))
        raise ValueError(f"Holosoma backend only supports: {supported}. Got: {robot}")
    urdf = _ensure_file(HOLOSOMA_ROBOT_URDFS[robot], "holosoma robot URDF")
    return robot, urdf


def _resolve_gmr_robot(robot: str, explicit_gmr_robot: str | None) -> str:
    if explicit_gmr_robot:
        return explicit_gmr_robot
    return COMMON_TO_GMR_ROBOT.get(robot, robot)


def _run_holosoma(args: argparse.Namespace) -> Path:
    robot_id, robot_urdf = _resolve_holosoma_robot(args.robot)
    data_path, motion_npz = _resolve_holosoma_input(args.seq_name, args.hmr_type, args.post_scene_root, args.output_root)
    out_dir = _ensure_dir(args.output_root / "holosoma" / args.seq_name / robot_id)
    out_npz = out_dir / f"{args.seq_name}.npz"

    holosoma_env = os.environ.copy()
    holosoma_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(HOLOSOMA_PYTHONPATH_ROOT),
            *([holosoma_env["PYTHONPATH"]] if holosoma_env.get("PYTHONPATH") else []),
        ]
    )

    _log(f"[holosoma] root:   {HOLOSOMA_RT_ROOT}")
    _log(f"[holosoma] conda:  {args.holosoma_conda_exe}")
    _log(f"[holosoma] env:    {args.holosoma_env}")
    _log(f"[holosoma] input:  {motion_npz}")
    _log(f"[holosoma] output: {out_npz}")

    cmd = [
        args.holosoma_conda_exe,
        "run",
        "-n",
        args.holosoma_env,
        "python",
        str(HOLOSOMA_RT_ROOT / "examples" / "robot_retarget.py"),
        "--robot",
        robot_id,
        "--task-type",
        "robot_only",
        "--task-name",
        args.seq_name,
        "--data_format",
        "smplx",
        "--data_path",
        str(data_path),
        "--save_dir",
        str(out_dir),
        "--robot-config.robot-urdf-file",
        str(robot_urdf),
    ]
    _run(cmd, cwd=HOLOSOMA_RT_ROOT, dry_run=args.dry_run, env=holosoma_env)

    if not args.dry_run:
        _ensure_file(out_npz, "holosoma retargeted output")
    return out_npz


def _run_gmr(args: argparse.Namespace) -> Path:
    gmr_robot = _resolve_gmr_robot(args.robot, args.gmr_robot)
    gmr_input = _resolve_gmr_input(args.seq_name, args.hmr_init_root, args.gmr_input_file, args.gmr_input_format)
    _ensure_gmr_body_models(args.gmr_body_models_root, args.crisp_smplx_models_root, dry_run=args.dry_run)

    out_dir = _ensure_dir(args.output_root / "gmr" / args.seq_name / gmr_robot)
    raw_pkl = out_dir / f"{args.seq_name}_{gmr_robot}.pkl"
    qpos_npz = out_dir / f"{args.seq_name}_{gmr_robot}_qpos.npz"

    _log(f"[gmr] input:       {gmr_input}")
    _log(f"[gmr] input_format:{args.gmr_input_format}")
    _log(f"[gmr] raw output:  {raw_pkl}")
    _log(f"[gmr] qpos output: {qpos_npz}")

    cmd = [
        "conda",
        "run",
        "-n",
        args.gmr_env,
        "python",
        str(THIS_FILE),
        "--_internal-gmr-run",
        "--_gmr-input",
        str(gmr_input),
        "--_gmr-input-format",
        str(args.gmr_input_format),
        "--_gmr-raw-pkl",
        str(raw_pkl),
        "--_gmr-qpos-npz",
        str(qpos_npz),
        "--_gmr-robot-id",
        gmr_robot,
        "--_gmr-tgt-fps",
        str(args.gmr_tgt_fps),
    ]
    _run(cmd, cwd=CRISP_ROOT, dry_run=args.dry_run)

    if not args.dry_run:
        _ensure_file(raw_pkl, "GMR raw pickle output")
        _ensure_file(qpos_npz, "GMR normalized qpos output")
    return qpos_npz


def _run_gmr_internal(args: argparse.Namespace) -> None:
    gmr_input = _ensure_file(args._gmr_input, "internal GMR input")
    raw_pkl = args._gmr_raw_pkl
    qpos_npz = args._gmr_qpos_npz
    gmr_robot = args._gmr_robot_id
    tgt_fps = args._gmr_tgt_fps
    input_format = str(args._gmr_input_format or "gvhmr").strip().lower()

    import torch

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.smpl import (
        get_gvhmr_data_offline_fast,
        get_smplx_data_offline_fast,
        load_gvhmr_pred_file,
        load_smplx_file,
    )

    smplx_root = _ensure_file(DEFAULT_GMR_BODY_MODELS / "smplx" / "SMPLX_NEUTRAL.pkl", "GMR SMPL-X model").parent.parent

    smplx_output = None
    body_model = None
    alignment_metadata: dict[str, np.ndarray | np.generic | str] = {}
    joint_frame_positions = None
    joint_frame_names: list[str] | None = None
    if input_format == "gvhmr":
        smplx_data, body_model, smplx_output, actual_human_height = load_gvhmr_pred_file(str(gmr_input), smplx_root)
        smplx_frames, aligned_fps = get_gvhmr_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=tgt_fps,
        )
    elif input_format == "smplx_npz":
        smplx_data, body_model, smplx_output, actual_human_height = load_smplx_file(str(gmr_input), smplx_root)
        smplx_frames, aligned_fps = get_smplx_data_offline_fast(
            smplx_data,
            body_model,
            smplx_output,
            tgt_fps=tgt_fps,
        )
    elif input_format == "joint_frames_npz":
        smplx_frames, aligned_fps, actual_human_height, joint_frame_positions, joint_frame_names = _load_joint_frames_npz(
            gmr_input,
            smplx_root=smplx_root,
            tgt_fps=tgt_fps,
            load_smplx_file=load_smplx_file,
            get_smplx_data_offline_fast=get_smplx_data_offline_fast,
        )
    else:
        raise ValueError(f"Unsupported GMR input format: {input_format}")

    retargeter = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=gmr_robot,
    )

    qpos_list: list[np.ndarray] = []
    num_frames = len(smplx_frames)
    for idx, frame in enumerate(smplx_frames):
        qpos = np.asarray(retargeter.retarget(frame), dtype=np.float32)
        qpos_list.append(qpos)
        if idx == 0 or (idx + 1) == num_frames or (idx + 1) % 50 == 0:
            _log(f"[gmr-internal] processed {idx + 1}/{num_frames} frames")

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    if input_format == "joint_frames_npz" and joint_frame_positions is not None and joint_frame_names is not None:
        alignment_metadata = {
            "root_alignment_mode": np.array("none_direct_gmr_from_hmr_joints"),
        }
    root_pos = qpos_arr[:, :3]
    root_rot_wxyz = qpos_arr[:, 3:7]
    dof_pos = qpos_arr[:, 7:]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]

    _ensure_dir(raw_pkl.parent)
    with raw_pkl.open("wb") as f:
        pickle.dump(
            {
                "fps": aligned_fps,
                "root_pos": root_pos,
                "root_rot": root_rot_xyzw,
                "dof_pos": dof_pos,
                "local_body_pos": None,
                "link_body_list": None,
            },
            f,
        )

    np.savez_compressed(
        qpos_npz,
        qpos=qpos_arr,
        fps=np.float32(aligned_fps),
        backend="gmr",
        robot=gmr_robot,
        actual_human_height_m=np.float32(actual_human_height),
        src_human="smplx" if input_format != "joint_frames_npz" else "smplx_joint_frames",
        gmr_input_file=str(gmr_input),
        gmr_input_format=input_format,
        raw_pickle=str(raw_pkl),
        **alignment_metadata,
    )

    _log(f"[gmr-internal] saved raw pickle: {raw_pkl}")
    _log(f"[gmr-internal] saved qpos npz:  {qpos_npz}")

    # Avoid torch shutdown warnings keeping references alive longer than needed.
    if smplx_output is not None:
        del smplx_output
    if body_model is not None:
        del body_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run retargeting from CRISP-Real2Sim outputs using either holosoma or GMR. "
            "Holosoma consumes post-scene SMPL-X joints, while GMR consumes GVHMR predictions."
        )
    )
    parser.add_argument("--backend", choices=("holosoma", "gmr"), help="Retargeting backend to run.")
    parser.add_argument("--seq-name", help="Sequence name under CRISP-Real2Sim results.")
    parser.add_argument("--hmr-type", default="gv", help="Subfolder under post_scene/<seq>/ (default: gv).")
    parser.add_argument(
        "--robot",
        default="g1",
        help=(
            "Robot identifier. "
            "For holosoma, supported values are g1 and t1. "
            "For GMR, g1/t1 are mapped to unitree_g1/booster_t1 unless --gmr-robot is set."
        ),
    )
    parser.add_argument("--gmr-robot", default=None, help="Explicit GMR robot id, e.g. fourier_gr3.")
    parser.add_argument(
        "--holosoma-env",
        default=os.environ.get("HOLOSOMA_ENV", "hsretargeting"),
        help="Conda environment for holosoma retargeting.",
    )
    parser.add_argument(
        "--holosoma-conda-exe",
        default=_default_holosoma_conda_exe(),
        help="Conda executable used for holosoma retargeting.",
    )
    parser.add_argument("--gmr-env", default="gmr", help="Conda environment for GMR retargeting.")
    parser.add_argument("--gmr-tgt-fps", type=int, default=30, help="Target FPS for offline GMR retargeting.")
    parser.add_argument("--gmr-input-file", type=Path, default=None, help="Optional explicit GMR input file.")
    parser.add_argument(
        "--gmr-input-format",
        default="gvhmr",
        choices=("gvhmr", "smplx_npz", "joint_frames_npz"),
        help="Input format for GMR. Default uses GVHMR prediction pt.",
    )
    parser.add_argument("--post-scene-root", type=Path, default=DEFAULT_POST_SCENE_ROOT)
    parser.add_argument("--hmr-init-root", type=Path, default=DEFAULT_HMR_INIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gmr-body-models-root", type=Path, default=DEFAULT_GMR_BODY_MODELS)
    parser.add_argument("--crisp-smplx-models-root", type=Path, default=DEFAULT_CRISP_SMPLX_MODELS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and inferred paths without running.")

    parser.add_argument("--_internal-gmr-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-input-format", help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-raw-pkl", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-qpos-npz", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-robot-id", help=argparse.SUPPRESS)
    parser.add_argument("--_gmr-tgt-fps", type=int, default=30, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args._internal_gmr_run:
        required = {
            "_gmr_input": args._gmr_input,
            "_gmr_raw_pkl": args._gmr_raw_pkl,
            "_gmr_qpos_npz": args._gmr_qpos_npz,
            "_gmr_robot_id": args._gmr_robot_id,
        }
        missing = [key for key, value in required.items() if value in (None, "")]
        if missing:
            raise SystemExit(f"Missing internal GMR args: {', '.join(missing)}")
        _run_gmr_internal(args)
        return

    if not args.backend or not args.seq_name:
        parser.error("--backend and --seq-name are required")

    args.post_scene_root = args.post_scene_root.resolve()
    args.hmr_init_root = args.hmr_init_root.resolve()
    args.output_root = args.output_root.resolve()
    args.gmr_body_models_root = args.gmr_body_models_root.resolve()
    args.crisp_smplx_models_root = args.crisp_smplx_models_root.resolve()
    args.gmr_input_file = args.gmr_input_file.resolve() if args.gmr_input_file is not None else None

    if args.backend == "holosoma":
        output = _run_holosoma(args)
    else:
        output = _run_gmr(args)

    _log(f"[done] backend={args.backend} seq={args.seq_name} output={output}")


if __name__ == "__main__":
    main()
