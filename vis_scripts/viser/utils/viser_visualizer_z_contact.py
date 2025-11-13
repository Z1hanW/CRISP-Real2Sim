#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass

import cv2
import numpy as np
import imageio.v3 as iio
from contextlib import suppress

# =========================
# 数据容器
# =========================
@dataclass
class RobotData:
    """单个robot的数据"""
    name: str
    vertices: np.ndarray  # (T, V, 3)
    faces: np.ndarray     # (F, 3)
    color: Tuple[float, float, float] = (0.6, 0.7, 0.9)
    visible: bool = True

@dataclass
class SceneData:
    """场景数据（比如点云或mesh）"""
    name: str
    type: str  # "mesh" or "pointcloud"
    vertices: Optional[np.ndarray] = None  # for mesh
    faces: Optional[np.ndarray] = None     # for mesh
    points: Optional[np.ndarray] = None    # for pointcloud
    colors: Optional[np.ndarray] = None    # colors
    visible: bool = True

# =========================
# 增强版 Viser Helper
# =========================
class EnhancedViserHelper:
    """Enhanced wrapper with multi-object support and visibility control."""
    _global_servers: Dict[int, "EnhancedViserHelper"] = {}

    def __init__(self, port: int = 8080):
        self.port = port
        self._server = None
        self._ok = False
        self._handles = {}
        self._visibility_states = {}
        self._init_server()
        self._setup_ui()

    @property
    def server(self):
        return self._server

    def ok(self) -> bool:
        return self._ok

    def _init_server(self):
        try:
            import viser
        except Exception as e:
            print(f"[Viser] Not available: {e}")
            return
        
        # Reuse or create new server for port
        prev = EnhancedViserHelper._global_servers.get(self.port, None)
        if prev is not None and prev._server is not None:
            try:
                prev._server.stop()
            except Exception:
                pass
            EnhancedViserHelper._global_servers.pop(self.port, None)
        
        self._server = viser.ViserServer(port=self.port)
        EnhancedViserHelper._global_servers[self.port] = self
        self._ok = True
        print(f"[Viser] Server started at http://localhost:{self.port}")
        self._add_ground_plane()

    def _add_ground_plane(self):
        """Ensure a checkerboard ground is visible at z=0."""
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

    def _setup_ui(self):
        """设置UI控制面板"""
        if not self._ok:
            return
        
        # 创建一个可折叠的控制面板
        with self._server.gui.add_folder("Visibility Controls") as folder:
            self._visibility_folder = folder
    
    def add_visibility_control(self, name: str, default: bool = True):
        """为对象添加visibility控制"""
        if not self._ok:
            return
        
        # 创建checkbox
        checkbox = self._server.gui.add_checkbox(
            label=f"Show {name}",
            initial_value=default,
            folder=self._visibility_folder
        )
        
        # 设置回调
        @checkbox.on_update
        def _(_):
            self.set_visibility(name, checkbox.value)
        
        self._visibility_states[name] = checkbox

    def set_visibility(self, name: str, visible: bool):
        """设置对象可见性"""
        if name in self._handles:
            self._handles[name].visible = visible

    def add_mesh(
        self,
        name: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: Tuple[float, float, float] = (0.6, 0.7, 0.9),
        wireframe: bool = False,
        opacity: float = 1.0,
        side: str = "double",
        add_ui_control: bool = True,
    ):
        """添加mesh并可选创建UI控制"""
        if not self._ok:
            return
        
        handle = self._server.scene.add_mesh_simple(
            name,
            vertices.astype(np.float32),
            faces.astype(np.int32),
            color=color,
            wireframe=wireframe,
            opacity=opacity,
            side=side,
        )
        self._handles[name] = handle
        
        if add_ui_control:
            self.add_visibility_control(name, default=True)
        
        return handle

    def add_point_cloud(
        self,
        name: str,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        point_size: float = 0.02,
        add_ui_control: bool = True,
    ):
        """添加点云并可选创建UI控制"""
        if not self._ok:
            return
        
        points = points.astype(np.float32)
        handle = self._server.scene.add_point_cloud(
            name,
            points=points,
            colors=colors.astype(np.uint8) if colors is not None else None,
            point_size=point_size,
        )
        self._handles[name] = handle
        
        if add_ui_control:
            self.add_visibility_control(name, default=True)
        
        return handle

    def update_mesh_vertices(self, name: str, vertices: np.ndarray):
        """更新mesh顶点"""
        if name in self._handles:
            self._handles[name].vertices = vertices.astype(np.float32)

    def update_point_cloud(self, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """更新点云"""
        if name in self._handles:
            self._handles[name].points = points.astype(np.float32)
            if colors is not None:
                self._handles[name].colors = colors.astype(np.uint8)

    def set_transform(self, name: str, position: np.ndarray, wxyz: np.ndarray):
        """设置变换"""
        if name in self._handles:
            self._handles[name].position = position.astype(np.float32)
            self._handles[name].wxyz = wxyz.astype(np.float32)

    def add_frame_slider(self, max_frames: int, callback):
        """添加帧控制滑块"""
        if not self._ok:
            return
        
        slider = self._server.gui.add_slider(
            label="Frame",
            min=0,
            max=max_frames - 1,
            step=1,
            initial_value=0,
        )
        
        @slider.on_update
        def _(_):
            callback(slider.value)
        
        return slider

# =========================
# 多机器人场景渲染器
# =========================
class MultiRobotSceneRenderer:
    """管理多个robot和scene的渲染"""
    
    def __init__(self, port: int = 8080):
        self.viser = EnhancedViserHelper(port=port)
        self.robots: Dict[str, RobotData] = {}
        self.scenes: Dict[str, SceneData] = {}
        self.current_frame = 0
        self.max_frames = 0
        
    def add_robot(self, robot: RobotData):
        """添加机器人"""
        if not self.viser.ok():
            return
        
        self.robots[robot.name] = robot
        self.max_frames = max(self.max_frames, len(robot.vertices))
        
        # 添加到场景
        self.viser.add_mesh(
            name=f"/robot_{robot.name}",
            vertices=robot.vertices[0],
            faces=robot.faces,
            color=robot.color,
            wireframe=False,
            opacity=1.0,
            add_ui_control=True,
        )
        
    def add_scene(self, scene: SceneData):
        """添加场景元素"""
        if not self.viser.ok():
            return
        
        self.scenes[scene.name] = scene
        
        if scene.type == "mesh" and scene.vertices is not None and scene.faces is not None:
            self.viser.add_mesh(
                name=f"/scene_{scene.name}",
                vertices=scene.vertices,
                faces=scene.faces,
                color=(0.8, 0.8, 0.8),
                wireframe=False,
                opacity=0.8,
                add_ui_control=True,
            )
        elif scene.type == "pointcloud" and scene.points is not None:
            self.viser.add_point_cloud(
                name=f"/scene_{scene.name}",
                points=scene.points,
                colors=scene.colors,
                point_size=0.01,
                add_ui_control=True,
            )
    
    def update_frame(self, frame_idx: int):
        """更新到指定帧"""
        self.current_frame = frame_idx
        
        # 更新所有robot的mesh
        for name, robot in self.robots.items():
            if frame_idx < len(robot.vertices):
                self.viser.update_mesh_vertices(
                    f"/robot_{name}", 
                    robot.vertices[frame_idx]
                )
    
    def setup_animation_controls(self):
        """设置动画控制"""
        if self.max_frames > 1:
            self.viser.add_frame_slider(self.max_frames, self.update_frame)
            
            # 添加播放控制
            play_button = self.viser.server.gui.add_button("Play/Pause")
            self.playing = False
            
            @play_button.on_click
            def _(_):
                self.playing = not self.playing
                if self.playing:
                    self._play_animation()
    
    def _play_animation(self):
        """播放动画"""
        import threading
        
        def animate():
            while self.playing:
                self.current_frame = (self.current_frame + 1) % self.max_frames
                self.update_frame(self.current_frame)
                time.sleep(1.0 / 30.0)  # 30 FPS
        
        thread = threading.Thread(target=animate)
        thread.daemon = True
        thread.start()
    
    def render_and_save(
        self,
        cameras: List[Dict],  # 相机参数列表
        output_dir: str,
        render_frames: Optional[List[int]] = None,
    ):
        """渲染并保存图像/视频"""
        os.makedirs(output_dir, exist_ok=True)
        
        if render_frames is None:
            render_frames = list(range(self.max_frames))
        
        client, launched = _get_client_or_launch(self.viser.server)
        
        rendered_images = []
        for frame_idx in render_frames:
            self.update_frame(frame_idx)
            
            if frame_idx < len(cameras):
                cam = cameras[frame_idx]
                pos, quat = _twc_to_cam_pose(cam['Twc'])
                fov = _k_to_fov_y(cam['K'], cam['H'])
                
                with client.atomic():
                    client.camera.position = pos
                    client.camera.wxyz = quat
                    client.camera.fov = float(fov)
                
                rgb = client.get_render(height=cam['H'], width=cam['W'])
                rendered_images.append(rgb)
                
                # 保存单帧
                if frame_idx == 0:
                    cv2.imwrite(
                        os.path.join(output_dir, f"frame_{frame_idx:05d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    )
        
        # 保存视频
        if len(rendered_images) > 1:
            video_path = os.path.join(output_dir, "render.mp4")
            iio.imwrite(video_path, np.array(rendered_images), fps=30)
            print(f"Saved video: {video_path}")
        
        _close_launched(launched)

# =========================
# 辅助函数（从原代码继承）
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
    fy = float(K[1, 1])
    return float(2.0 * np.arctan2(H, 2.0 * fy))

def _twc_to_cam_pose(T_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import viser as _viser
    R_wc = np.asarray(T_wc[:3, :3], dtype=float)
    t_wc = np.asarray(T_wc[:3, 3], dtype=float)
    q_wxyz = _viser.transforms.SO3.from_matrix(R_wc).wxyz
    return t_wc.astype(float), np.asarray(q_wxyz, dtype=float).reshape(4,)

def _get_client_or_launch(server, timeout_sec: float = 20.0):
    host, port = server.get_host(), server.get_port()
    url = f"http://{host}:{port}"
    
    clients = server.get_clients()
    if clients:
        return list(clients.values())[0], None
    
    launched = None
    with suppress(Exception):
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        browser = pw.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(url, wait_until="load")
        launched = (pw, browser, page)
    
    t0 = time.time()
    while True:
        clients = server.get_clients()
        if clients:
            return list(clients.values())[0], launched
        if time.time() - t0 > timeout_sec:
            if launched is not None:
                _close_launched(launched)
            raise TimeoutError(f"No client connected. Please open {url}")
        time.sleep(0.05)

def _close_launched(launched):
    if launched is None:
        return
    pw, browser, page = launched
    with suppress(Exception):
        page.close()
        browser.close()
        pw.stop()
