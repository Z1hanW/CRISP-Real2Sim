#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Optional, Tuple, List

import cv2
import numpy as np
import imageio.v3 as iio
from contextlib import suppress

# =========================
# Viser 轻量封装（兼容前文）
# =========================
class ViserHelper:
    """Lightweight wrapper around viser for meshes and point clouds."""
    _global_servers: Dict[int, "ViserHelper"] = {}

    def __init__(self, port: int = 8080):
        self.port = port
        self._server = None
        self._ok = False
        self._handles = {}
        self._init_server()

    # Expose server for advanced use if needed
    @property
    def server(self):
        return self._server

    def ok(self) -> bool:
        return self._ok

    def _init_server(self):
        try:
            import viser
        except Exception as e:  # pragma: no cover
            print(f"[Viser] Not available: {e}")
            return
        # Reuse or create new server for port
        prev = ViserHelper._global_servers.get(self.port, None)
        if prev is not None and prev._server is not None:
            try:
                prev._server.stop()
            except Exception:
                pass
            ViserHelper._global_servers.pop(self.port, None)
        self._server = viser.ViserServer(port=self.port)
        ViserHelper._global_servers[self.port] = self
        self._ok = True
        self._add_ground_plane()

    def _add_ground_plane(self):
        """Add a checkerboard grid so z=0 is visually grounded."""
        if not self._ok or self._server is None:
            return
        scene = getattr(self._server, "scene", None)
        if scene is None or not hasattr(scene, "add_grid"):
            return
        try:
            scene.add_grid(
                "/ground",
                width=30.0,
                height=30.0,
                position=(0.0, 0.0, 0.0),
                plane="xy",
            )
        except TypeError:
            scene.add_grid(
                "/ground",
                width=30.0,
                height=30.0,
                position=(0.0, 0.0, 0.0),
            )

    def close(self):
        """Stop the underlying Viser server and release cached handles."""
        if self._server is None:
            return
        with suppress(Exception):
            self._server.stop()
        if ViserHelper._global_servers.get(self.port) is self:
            ViserHelper._global_servers.pop(self.port, None)
        self._server = None
        self._handles.clear()
        self._ok = False

    def add_mesh_simple(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.7, 0.9),
        side: str = "double",
    ):
        if not self._ok:
            return
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices.astype(np.float32),
            faces.astype(np.int32),
            color=color,
            side=side,
        )
        self._handles[name] = handle

    def set_transform(self, name: str, position: np.ndarray, wxyz: np.ndarray):
        if not self._ok:
            return
        h = self._handles.get(name)
        if h is None:
            return
        h.position = position.astype(np.float32)
        h.wxyz = wxyz.astype(np.float32)

    def update_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        color: Optional[np.ndarray] = None,
        point_size: float = 0.02,
    ):
        if not self._ok:
            return
        points = points.astype(np.float32)
        if name not in self._handles:
            try:
                handle = self._server.scene.add_point_cloud(
                    name,
                    points=points,
                    colors=color if color is not None else None,
                    point_size=point_size,
                    precision="float32",
                )
            except TypeError:
                handle = self._server.scene.add_point_cloud(
                    name,
                    points=points,
                    colors=color if color is not None else None,
                    point_size=point_size,
                )
            self._handles[name] = handle
        else:
            h = self._handles[name]
            h.points = points
            if color is not None:
                h.colors = color
            h.point_size = point_size

    def update_line_segments(
        self,
        name: str,
        segments: np.ndarray,
        colors: Optional[np.ndarray] = None,
        line_width: float = 1.5,
    ):
        if not self._ok:
            return
        segments = segments.astype(np.float32)
        if name not in self._handles:
            handle = self._server.scene.add_line_segments(
                name,
                points=segments,
                colors=colors,
                line_width=line_width,
            )
            self._handles[name] = handle
        else:
            h = self._handles[name]
            h.points = segments
            if colors is not None:
                h.colors = colors
            h.line_width = line_width

    def set_camera(self, position: np.ndarray, lookat: np.ndarray):
        if not self._ok:
            return
        for _, client in self._server.get_clients().items():
            client.camera.position = position
            client.camera.look_at = lookat


# =========================
# 相机/渲染工具函数（模块级）
# =========================
def _ensure_Twc_is_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    if T.shape == (3, 4):
        Twc = np.eye(4, dtype=np.float64)
        Twc[:3, :4] = T
        return Twc
    elif T.shape == (4, 4):
        return T.astype(np.float64)
    else:
        raise ValueError(f"Unexpected Twc shape {T.shape}, expect (4,4) or (3,4).")

def _k_to_fov_y(K: np.ndarray, H: int) -> float:
    """垂直视场（弧度），兼容 three.js/viser。"""
    fy = float(K[1, 1])
    return float(2.0 * np.arctan2(H, 2.0 * fy))

def _twc_to_cam_pose(T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """cam->world (4x4) 转 (position, quaternion_wxyz) for viser."""
    import viser as _viser  # 延迟导入，避免没有安装时报错
    R_wc = np.asarray(T_wc[:3, :3], dtype=float)
    t_wc = np.asarray(T_wc[:3, 3], dtype=float)
    q_wxyz = _viser.transforms.SO3.from_matrix(R_wc).wxyz
    return t_wc.astype(float), np.asarray(q_wxyz, dtype=float).reshape(4,)

def _normalize_host_for_browser(host: str) -> str:
    """Chrome can't dial 0.0.0.0, so map it (and similar) to loopback."""
    if host in ("0.0.0.0", "::", "::0", "", None):
        return "127.0.0.1"
    return host


def _get_client_or_launch(server, timeout_sec: float = 20.0):
    """获得一个连到 Viser 的 client。优先复用；否则尝试用 playwright 打开 headless。"""
    host, port = server.get_host(), server.get_port()
    url = f"http://{_normalize_host_for_browser(host)}:{port}"

    clients = server.get_clients()
    if clients:
        return list(clients.values())[0], None

    launched = None
    with suppress(Exception):
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
            ],
        )
        page = browser.new_page(
            viewport={"width": 1280, "height": 800, "deviceScaleFactor": 1.0}
        )
        page.goto(url, wait_until="load")
        launched = (pw, browser, page)

    t0 = time.time()
    while True:
        clients = server.get_clients()
        if clients:
            return list(clients.values())[0], launched
        if time.time() - t0 > timeout_sec:
            if launched is not None:
                pw, browser, page = launched
                with suppress(Exception):
                    page.close()
                    browser.close()
                    pw.stop()
            raise TimeoutError(f"没有前端连接。请在浏览器打开 {url} ，或安装 playwright 后重试。")
        time.sleep(0.05)

def _close_launched(launched):
    if launched is None:
        return
    pw, browser, page = launched
    with suppress(Exception):
        page.close()
        browser.close()
        pw.stop()

def add_robot_once(server, faces: np.ndarray, init_vertices: np.ndarray):
    """创建一次 /robot 网格，并返回 handle；之后只更新 .vertices。"""
    return server.scene.add_mesh_simple(
        name="/robot",
        vertices=np.asarray(init_vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
        color=[255, 255, 255],
        flat_shading=False,
        wireframe=False,
    )

def build_cameras_from_data(
    data: Dict, img_focal: float
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
    """
    从 data 构造相机数组：
      - per_Twc:   List[(4,4)]  cam->world
      - per_K:     List[(3,3)]  fx=fy=img_focal, cx=W/2, cy=H/2
      - per_HW:    List[(H,W)]
    要求 data 包含:
      data["images"]: (T,H,W,3)
      data["cam_c2w"]: (T,4,4) 或 (T,3,4)
    """
    images = data["images"]
    cam_c2w = data["cam_c2w"]
    T_total = len(images)
    per_Twc, per_K, per_HW = [], [], []

    for i in range(T_total):
        img = images[i]
        H, W = int(img.shape[0]), int(img.shape[1])
        per_HW.append((H, W))

        Twc_i = _ensure_Twc_is_4x4(cam_c2w[i])
        per_Twc.append(Twc_i)

        fx = float(img_focal)
        fy = float(img_focal)
        cx = W * 0.5  # 中心
        cy = H * 0.5
        K = np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        per_K.append(K)

    return per_Twc, per_K, per_HW

def render_robot_image_and_video(
    server,
    pred_vert: np.ndarray,  # (T,V,3)
    faces: np.ndarray,
    per_Twc: List[np.ndarray],  # len T, (4,4)
    per_K: List[np.ndarray],  # len T, (3,3)
    per_HW: List[Tuple[int, int]],  # len T, (H,W)
    t_image: int,  # 导出单张的 timestep
    video_frames: List[int],  # 导出视频的帧索引
    out_dir: str,
    fps: int = 30,
):
    os.makedirs(out_dir, exist_ok=True)
    client, launched = _get_client_or_launch(server)

    # 一次性创建 /robot
    robot = add_robot_once(server, faces, pred_vert[0])

    # 单帧 PNG
    H, W = per_HW[t_image]
    pos, quat = _twc_to_cam_pose(per_Twc[t_image])
    fov = _k_to_fov_y(per_K[t_image], H)
    with client.atomic():
        robot.vertices = pred_vert[t_image].astype(np.float32)
        client.camera.position = pos
        client.camera.wxyz = quat
        client.camera.fov = float(fov)
    rgb = client.get_render(height=int(H), width=int(W), timeout=60.0)
    out_png = os.path.join(out_dir, f"robot_t{t_image:05d}.png")
    cv2.imwrite(out_png, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"[robot] wrote {out_png}")

    # 序列 MP4
    frames = []
    for t in video_frames:
        H, W = per_HW[t]
        pos, quat = _twc_to_cam_pose(per_Twc[t])
        fov = _k_to_fov_y(per_K[t], H)
        with client.atomic():
            robot.vertices = pred_vert[t].astype(np.float32)
            client.camera.position = pos
            client.camera.wxyz = quat
            client.camera.fov = float(fov)
        frames.append(client.get_render(height=int(H), width=int(W), timeout=60.0))
    if save_video and rendered_images:
        video_path = os.path.join(output_dir, "robot_render.mp4")
        H, W = rendered_images[0].shape[:2]
        # 常见 FourCC：'mp4v' 基本都能写；若要 H.264 可试 'avc1'（取决于你系统的 ffmpeg/gstreamer）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, float(camera_handler.fps), (W, H))
        if not writer.isOpened():
            raise RuntimeError("Failed to open VideoWriter. Try a different fourcc like 'avc1' or install codecs.")

        for rgb in rendered_images:
            # Viser 返回的是 RGB，需要转成 BGR 再写
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            # 确保尺寸一致（保险）
            if bgr.shape[1] != W or bgr.shape[0] != H:
                bgr = cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)
            writer.write(bgr)

        writer.release()
        print(f"[Renderer] Saved video: {video_path}")


    _close_launched(launched)


# =========================
# 最小调用示例（与你现有工程对接）
# =========================
def demo_call(
    data: Dict,
    img_focal: float,
    pred_vert: np.ndarray,  # (T,V,3)
    faces: np.ndarray,  # (F,3)
    tgt_name: str,
    port: int = 8080,
    t_image: int = 0,
    video_stride: int = 1,
):
    """
    直接调用：给 data/img_focal/pred_vert/faces/tgt_name 即可渲染。
    """
    # 1) 相机数组（cx, cy 取中心）
    per_Twc, per_K, per_HW = build_cameras_from_data(data, img_focal)
    T_total = len(pred_vert)
    video_frames = list(range(0, T_total, max(1, int(video_stride))))

    # 2) viser server
    vh = ViserHelper(port=port)
    if not vh.ok():
        raise RuntimeError("Viser not available. Please `pip install viser` first.")
    server = vh.server

    # 3) 输出目录
    out_dir = f"/data3/zihanwa3/_Robotics/_geo/differentiable-blocksworld/_rendered_views/robot/{tgt_name}"
    os.makedirs(out_dir, exist_ok=True)

    # 4) 渲染
    render_robot_image_and_video(
        server=server,
        pred_vert=pred_vert,
        faces=faces,
        per_Twc=per_Twc,
        per_K=per_K,
        per_HW=per_HW,
        t_image=int(t_image),
        video_frames=video_frames,
        out_dir=out_dir,
        fps=30,
    )
