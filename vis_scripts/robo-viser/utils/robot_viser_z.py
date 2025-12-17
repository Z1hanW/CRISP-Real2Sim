import os
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

from .viser_visualizer import ViserHelper


def _parse_floats(s: str) -> List[float]:
    return [float(x) for x in s.split()] if s is not None else []


def _quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    # Both as xyzw
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w], dtype=np.float32)


def _quat_rotate_vec_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    # Rotate v by quaternion q (xyzw)
    # v' = q * (v,0) * q^{-1}
    x, y, z, w = q
    # Convert to rotation matrix directly for simplicity
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return (R @ v.reshape(3, 1)).reshape(3)


class RobotMjcfViser:
    """Visualize a MuJoCo MJCF humanoid by parsing geoms and updating per body.

    Limitations:
    - Supports box/capsule/sphere geoms with 'size' attribute.
    - Capsule is approximated as a cylinder.
    - Local geom pose via 'pos' and optional 'quat'.
    """

    def __init__(
        self,
        viser: ViserHelper,
        mjcf_path: str,
        body_names: Optional[List[str]],
        *,
        prefix: str = "/robot",
    ):
        self._viser = viser
        self._mjcf_path = mjcf_path
        self._body_names = body_names
        prefix_clean = prefix.rstrip("/") or "/robot"
        if not prefix_clean.startswith("/"):
            prefix_clean = "/" + prefix_clean
        self._prefix = prefix_clean
        self._geom_specs: Dict[int, List[Tuple[str, np.ndarray, np.ndarray]]] = {}
        # Each entry: body_index -> list of (mesh_name, local_pos(3), local_quat_xyzw(4))

        self._load_geoms()

    def _load_geoms(self):
        if not os.path.exists(self._mjcf_path):
            print(f"[RobotMjcfViser] MJCF not found: {self._mjcf_path}")
            return
        try:
            import trimesh
        except Exception as e:
            print(f"[RobotMjcfViser] trimesh not available: {e}")
            return

        tree = ET.parse(self._mjcf_path)
        root = tree.getroot()

        # Collect bodies in pre-order to match likely asset order
        # Collect bodies under worldbody only (skip actuators/assets etc.)
        bodies = []
        def visit(node):
            if node.tag == 'body':
                bodies.append(node)
            for ch in list(node):
                visit(ch)
        # Start from worldbody if present
        world = root.find('worldbody')
        visit(world if world is not None else root)

        # Name to index map if body_names provided; else fall back to enumeration order
        name_to_idx = {n: i for i, n in enumerate(self._body_names)} if self._body_names else None

        for enum_idx, b in enumerate(bodies):
            bname = b.attrib.get('name', f'body_{enum_idx}')
            if name_to_idx is not None:
                if bname not in name_to_idx:
                    # Skip bodies not present in the loaded asset ordering
                    continue
                body_idx = name_to_idx[bname]
            else:
                body_idx = enum_idx
            for g in b.findall('geom'):
                gtype = g.attrib.get('type', 'capsule')
                size = _parse_floats(g.attrib.get('size'))
                lpos = np.array(_parse_floats(g.attrib.get('pos')), dtype=np.float32) if g.attrib.get('pos') else np.zeros(3, dtype=np.float32)
                # MuJoCo MJCF stores quat as w x y z; convert to xyzw for internal use
                if g.attrib.get('quat'):
                    q_wxyz = np.array(_parse_floats(g.attrib.get('quat')), dtype=np.float32)
                    if q_wxyz.shape[0] == 4:
                        lquat = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]], dtype=np.float32)
                    else:
                        lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                else:
                    lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                fromto = _parse_floats(g.attrib.get('fromto')) if g.attrib.get('fromto') else []

                mesh = None
                if gtype == 'box' and len(size) == 3:
                    extents = 2 * np.array(size, dtype=np.float32)
                    mesh = trimesh.creation.box(extents=extents)
                elif gtype == 'sphere' and len(size) >= 1:
                    mesh = trimesh.creation.icosphere(subdivisions=1, radius=size[0])
                elif gtype == 'capsule':
                    # Prefer fromto if available; otherwise use size=[radius, half]
                    if len(fromto) == 6 and len(size) >= 1:
                        p1 = np.array(fromto[:3], dtype=np.float32)
                        p2 = np.array(fromto[3:], dtype=np.float32)
                        v = p2 - p1
                        length = float(np.linalg.norm(v))
                        if length < 1e-8:
                            continue
                        radius = float(size[0])
                        # If the segment is very short relative to radius, render as a sphere
                        if length < max(1e-3, 0.5 * radius):
                            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                            lpos = (p1 + p2) * 0.5
                            lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                        else:
                            mesh = trimesh.creation.cylinder(radius=radius, height=length, sections=24)
                            # Override local pose: center and rotation aligning z to v
                            z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                            dir = v / length
                            axis = np.cross(z, dir)
                            norm_axis = np.linalg.norm(axis)
                            if norm_axis < 1e-8:
                                # Aligned or opposite
                                if np.dot(z, dir) > 0:
                                    lquat = np.array([0, 0, 0, 1], dtype=np.float32)
                                else:
                                    # 180 deg about X
                                    lquat = np.array([1, 0, 0, 0], dtype=np.float32)
                            else:
                                axis = axis / norm_axis
                                angle = float(np.arccos(np.clip(np.dot(z, dir), -1.0, 1.0)))
                                s = np.sin(angle / 2.0)
                                lquat = np.array([axis[0] * s, axis[1] * s, axis[2] * s, np.cos(angle / 2.0)], dtype=np.float32)
                            lpos = (p1 + p2) * 0.5
                    elif len(size) >= 2:
                        radius, half = float(size[0]), float(size[1])
                        height = 2.0 * half
                        if height < max(1e-3, 0.5 * radius):
                            mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                        else:
                            mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
                    else:
                        continue
                else:
                    continue

                name = f"{self._prefix}/{body_idx}/{len(self._geom_specs.get(body_idx, []))}"
                self._viser.add_mesh_simple(name, mesh.vertices, mesh.faces, color=(0.6, 0.9, 0.6))
                self._geom_specs.setdefault(body_idx, []).append((name, lpos, lquat))

    def update(self, body_pos_xy: np.ndarray, body_quat_xyzw: np.ndarray, world_offset: Optional[np.ndarray] = None):
        if not self._geom_specs:
            return
        if world_offset is None:
            world_offset = np.zeros(3, dtype=np.float32)

        for body_idx, geoms in self._geom_specs.items():
            if body_idx >= body_pos_xy.shape[0]:
                continue
            p_b = body_pos_xy[body_idx].astype(np.float32)
            q_b = body_quat_xyzw[body_idx].astype(np.float32)
            for name, lpos, lquat in geoms:
                # Compose transforms: world_q = q_b * lquat
                q_w = _quat_mul_xyzw(q_b, lquat)
                # world_p = p_b + R(q_b) * lpos
                p_w = p_b + _quat_rotate_vec_xyzw(q_b, lpos)
                p_w = p_w + world_offset
                # Convert to wxyz for viser
                wxyz = np.array([q_w[3], q_w[0], q_w[1], q_w[2]], dtype=np.float32)
                self._viser.set_transform(name, p_w, wxyz)
