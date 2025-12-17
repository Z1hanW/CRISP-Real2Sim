import numpy as np
from typing import Dict, Optional, Tuple


class ViserHelper:
    """Lightweight wrapper around viser for meshes and point clouds.

    - Starts a single global server per port
    - Provides simple helpers to add meshes and update point clouds/lines
    - Fails gracefully if viser is not installed
    """

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
        """Add a checkerboard grid to visualize the z=0 plane."""
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

    def add_mesh_simple(self, name: str, vertices: np.ndarray, faces: np.ndarray, color: Tuple[float, float, float] = (0.6, 0.7, 0.9), side: str = "double"):
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

    def update_point_cloud(self, name: str, points: np.ndarray, color: Optional[np.ndarray] = None, point_size: float = 0.02):
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

    def update_line_segments(self, name: str, segments: np.ndarray, colors: Optional[np.ndarray] = None, line_width: float = 1.5):
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
# Export / Embed GUI for Viser
# =========================
from pathlib import Path
import io, zipfile, subprocess, tempfile, shutil, time, re

class ViserEmbedExporter:
    """
    Adds an 'Export / Embed' panel to a Viser server.

    Workflow:
      1) Click [🔴 Start recording]
      2) Play your animation (your loop is already updating scene objects)
      3) Click [⏹ Stop & save .viser]
      4) (Optional) Click [📦 Build client & ZIP] to get a self-contained bundle
    """
    def __init__(self, server, out_root: Path):
        import viser  # ensure available
        self.server = server
        self.out_root = Path(out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        (self.out_root / "recordings").mkdir(exist_ok=True)

        self._recording = False
        self._serializer = None
        self._frames = 0
        self._last_bytes = None
        self._last_filename = None
        self._initial_cam_qs = ""  # appended to iframe src

        # --- GUI ---
        # Folder to group controls
        try:
            folder_cm = server.gui.add_folder("Export / Embed (Viser)", expand_by_default=True)
        except Exception:
            # If add_folder context manager is not available, fallback to flat panel.
            folder_cm = None

        def _slug(s: str) -> str:
            s = s.strip()
            s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
            return s or "recording"

        def make_embed_html(viser_file_name: str, with_initial_cam: str) -> str:
            # iframe points to local client build with ?playbackPath=...
            # The client folder will be <bundle>/viser-client/
            # The .viser will be <bundle>/recordings/<name>.viser
            qs = with_initial_cam or ""
            if qs and not qs.startswith("&"):
                qs = "&" + qs
            return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Viser embed - {viser_file_name}</title>
<style>
  html,body{{height:100%;margin:0}}
  .container{{position:fixed;inset:0}}
  iframe{{width:100%;height:100%;border:0}}
</style>
</head>
<body>
<div class="container">
  <iframe src="./viser-client/index.html?playbackPath=./recordings/{viser_file_name}{qs}" allowfullscreen></iframe>
</div>
</body>
</html>"""

        # flat or in-folder helpers
        add_text = (folder_cm.add_text if folder_cm else server.gui.add_text)
        add_button = (folder_cm.add_button if folder_cm else server.gui.add_button)

        self._name_text = add_text("file prefix", "robot_run")
        self._status_text = add_text("status", "idle")
        self._status_text.disabled = True

        btn_start = add_button("🔴 Start recording")
        btn_stop  = add_button("⏹ Stop & save .viser")
        btn_cam   = add_button("📷 Set initial camera from current view")
        btn_zip   = add_button("📦 Build client & ZIP (embed bundle)")

        @btn_start.on_click
        def _(_evt: "viser.GuiEvent") -> None:
            if self._recording:
                self._status_text.value = "already recording..."
                return
            # start a fresh serializer; subsequent property changes are tracked
            self._serializer = self.server.get_scene_serializer()
            self._recording = True
            self._frames = 0
            self._last_bytes = None
            self._last_filename = None
            self._status_text.value = "recording..."

        @btn_stop.on_click
        def _(evt: "viser.GuiEvent") -> None:
            if not self._recording or self._serializer is None:
                self._status_text.value = "not recording"
                return
            # finalize
            data = self._serializer.serialize()
            self._last_bytes = data
            ts = time.strftime("%Y%m%d-%H%M%S")
            prefix = _slug(self._name_text.value)
            self._last_filename = f"{prefix}_{ts}.viser"
            # write to disk
            out_path = self.out_root / "recordings" / self._last_filename
            out_path.write_bytes(data)
            # trigger browser download if possible
            if evt.client is not None and hasattr(evt.client, "send_file_download"):
                evt.client.send_file_download(self._last_filename, data)
            # reset state
            self._recording = False
            self._serializer = None
            self._status_text.value = f"saved {self._last_filename} ({self._frames} frames)"

        @btn_cam.on_click
        def _(evt: "viser.GuiEvent") -> None:
            if evt.client is None:
                self._status_text.value = "no client"
                return
            pos = evt.client.camera.position
            look = evt.client.camera.look_at
            up = evt.client.camera.up
            def pack(v): return ",".join(f"{float(x):.3f}" for x in (v[0], v[1], v[2]))
            self._initial_cam_qs = (
                f"initialCameraPosition={pack(pos)}"
                f"&initialCameraLookAt={pack(look)}"
                f"&initialCameraUp={pack(up)}"
            )
            self._status_text.value = "captured initial camera"

        @btn_zip.on_click
        def _(evt: "viser.GuiEvent") -> None:
            """
            Build a one-click bundle:
                bundle/
                  ├─ embed.html
                  ├─ recordings/<name>.viser
                  └─ viser-client/ (built by `viser-build-client`)
            """
            if self._last_bytes is None or self._last_filename is None:
                self._status_text.value = "record a .viser first (Stop & save)"
                return

            prefix = _slug(self._name_text.value)
            try:
                tmp_root = Path(tempfile.mkdtemp(prefix="viser_embed_"))
                (tmp_root / "recordings").mkdir(exist_ok=True)

                # write .viser
                (tmp_root / "recordings" / self._last_filename).write_bytes(self._last_bytes)

                # build client (requires Viser CLI; see docs)
                client_dir = tmp_root / "viser-client"
                subprocess.run(
                    ["viser-build-client", "--output-dir", str(client_dir)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

                # write embed.html (uses initial camera if captured)
                html = make_embed_html(self._last_filename, self._initial_cam_qs)
                (tmp_root / "embed.html").write_text(html, encoding="utf-8")

                # zip everything
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for p in tmp_root.rglob("*"):
                        zf.write(p, p.relative_to(tmp_root).as_posix())
                zip_bytes = buf.getvalue()
                zip_name = f"{prefix}_viser_embed_bundle.zip"

                if evt.client is not None and hasattr(evt.client, "send_file_download"):
                    evt.client.send_file_download(zip_name, zip_bytes)

                # clean temp
                shutil.rmtree(tmp_root, ignore_errors=True)
                self._status_text.value = "bundle ready (downloaded)"
            except FileNotFoundError as e:
                self._status_text.value = "missing `viser-build-client` — see console"
                print("[ViserEmbedExporter] Could not run `viser-build-client`. Install/update `viser` and try again.")
            except subprocess.CalledProcessError as e:
                self._status_text.value = "client build failed — see console"
                print("[ViserEmbedExporter] build error:\n", e.stdout)

    @property
    def is_recording(self) -> bool:
        return self._recording

    def on_frame(self, dt: float):
        """Call this once per visualized frame while your app runs."""
        if self._recording and self._serializer is not None:
            # record *time* between frames; state changes are captured implicitly
            self._serializer.insert_sleep(float(dt))
            self._frames += 1
