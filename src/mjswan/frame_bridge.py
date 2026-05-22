
"""TCP/WebSocket compatibility bridge for the FFS MuJoCo camera client.

The browser runtime renders virtual stereo frames and streams them to this
module over WebSocket. The existing Fast-FoundationStereo camera_mujoco client
connects to the TCP side and sees the same wire protocol used by the old Python
MuJoCo frame server:

    <III little-endian header> + left JPEG + right JPEG + meta JSON

Newline-delimited JSON commands from FFS are forwarded back to the browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any

try:  # websockets is an optional runtime dependency for non-frame-server usage.
    import websockets
    from websockets.server import WebSocketServerProtocol
except Exception:  # pragma: no cover - import availability is environment-specific.
    websockets = None  # type: ignore[assignment]
    WebSocketServerProtocol = Any  # type: ignore[misc,assignment]


@dataclass(slots=True)
class StereoFrame:
    left: bytes
    right: bytes
    meta: bytes
    seq: int
    created_at: float


class MujocoFrameBridge:
    """Bridge browser-rendered stereo frames to the legacy FFS TCP protocol."""

    def __init__(
        self,
        *,
        tcp_host: str = "0.0.0.0",
        tcp_port: int = 9876,
        ws_host: str = "0.0.0.0",
        ws_port: int = 9877,
    ) -> None:
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.ws_host = ws_host
        self.ws_port = ws_port
        self._latest: StereoFrame | None = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._tcp_thread: threading.Thread | None = None
        self._ws_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_clients: set[WebSocketServerProtocol] = set()
        self._frames_since_log = 0
        self._last_frame_log_at = time.time()

    def start(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets is required for mjswan frame bridge")
        self._tcp_thread = threading.Thread(target=self._run_tcp_server, name="mjswan-frame-tcp", daemon=True)
        self._ws_thread = threading.Thread(target=self._run_ws_server, name="mjswan-frame-ws", daemon=True)
        self._tcp_thread.start()
        self._ws_thread.start()
        print(
            f"[mjswan-frame] TCP {self.tcp_host}:{self.tcp_port} <-> "
            f"WS {self.ws_host}:{self.ws_port}",
            flush=True,
        )

    def stop(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def _run_ws_server(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)

        async def main() -> None:
            async with websockets.serve(  # type: ignore[union-attr]
                self._handle_ws,
                self.ws_host,
                self.ws_port,
                max_size=32 * 1024 * 1024,
                ping_interval=20,
                ping_timeout=20,
            ):
                while not self._stop.is_set():
                    await asyncio.sleep(0.2)

        try:
            loop.run_until_complete(main())
        except RuntimeError as exc:
            if not self._stop.is_set():
                print(f"[mjswan-frame] websocket server stopped: {exc}")
        finally:
            loop.close()

    async def _handle_ws(self, ws: WebSocketServerProtocol) -> None:
        self._ws_clients.add(ws)
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    self._ingest_binary_frame(message)
                else:
                    self._ingest_text_frame(message)
        finally:
            self._ws_clients.discard(ws)

    def _ingest_text_frame(self, message: str) -> None:
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if data.get("type") == "stereo_frame":
            left = data.get("left")
            right = data.get("right")
            meta = data.get("meta") or {}
            if isinstance(left, str) and isinstance(right, str):
                import base64

                self._store_frame(
                    base64.b64decode(left),
                    base64.b64decode(right),
                    meta,
                    int(data.get("seq") or 0),
                )

    def _ingest_binary_frame(self, payload: bytes) -> None:
        if len(payload) < 4:
            return
        header_size = struct.unpack_from("<I", payload, 0)[0]
        header_end = 4 + header_size
        if header_size <= 0 or header_end > len(payload):
            return
        try:
            header = json.loads(payload[4:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        left_size = int(header.get("left_size") or 0)
        right_size = int(header.get("right_size") or 0)
        left_start = header_end
        left_end = left_start + left_size
        right_end = left_end + right_size
        if left_size <= 0 or right_size <= 0 or right_end > len(payload):
            return
        self._store_frame(
            payload[left_start:left_end],
            payload[left_end:right_end],
            header.get("meta") or {},
            int(header.get("seq") or 0),
        )

    def _store_frame(self, left: bytes, right: bytes, meta: dict[str, Any], seq: int) -> None:
        meta_bytes = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        frame = StereoFrame(left=left, right=right, meta=meta_bytes, seq=seq, created_at=time.time())
        with self._condition:
            self._latest = frame
            self._condition.notify_all()
        self._frames_since_log += 1
        now = time.time()
        elapsed = now - self._last_frame_log_at
        if elapsed >= 5.0:
            fps = self._frames_since_log / elapsed
            pose_source = meta.get("pose_source") or "unknown"
            print(
                f"[mjswan-frame] browser stereo frames: {fps:.1f} fps "
                f"seq={seq} source={pose_source} left={len(left) // 1024}KB right={len(right) // 1024}KB",
                flush=True,
            )
            self._frames_since_log = 0
            self._last_frame_log_at = now

    def _run_tcp_server(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.tcp_host, self.tcp_port))
            server.listen(1)
            server.settimeout(0.5)
            while not self._stop.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                print(f"[mjswan-frame] FFS connected from {addr[0]}:{addr[1]}")
                with conn:
                    conn.settimeout(2.0)
                    self._serve_tcp_client(conn)
                print("[mjswan-frame] FFS disconnected")

    def _serve_tcp_client(self, conn: socket.socket) -> None:
        command_buffer = b""
        last_seq = -1
        last_wait_log_at = 0.0
        while not self._stop.is_set():
            frame = self._wait_for_new_frame(last_seq, timeout=0.1)
            try:
                if frame is None:
                    now = time.time()
                    if now - last_wait_log_at >= 5.0:
                        if self._latest is None:
                            detail = "no browser frame received yet"
                        else:
                            detail = f"latest seq={self._latest.seq} age={now - self._latest.created_at:.1f}s"
                        print(f"[mjswan-frame] waiting for new stereo frame ({detail})", flush=True)
                        last_wait_log_at = now
                    command_buffer = self._drain_tcp_commands(conn, command_buffer)
                    continue
                last_seq = frame.seq
                packet = struct.pack("<III", len(frame.left), len(frame.right), len(frame.meta))
                conn.sendall(packet)
                conn.sendall(frame.left)
                conn.sendall(frame.right)
                conn.sendall(frame.meta)
                command_buffer = self._drain_tcp_commands(conn, command_buffer)
            except (ConnectionError, OSError):
                break

    def _wait_for_new_frame(self, last_seq: int, timeout: float) -> StereoFrame | None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop.is_set():
                if self._latest is not None and self._latest.seq != last_seq:
                    return self._latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
        return None

    def _drain_tcp_commands(self, conn: socket.socket, buffer: bytes) -> bytes:
        while True:
            readable, _, _ = select.select([conn], [], [], 0)
            if not readable:
                break
            try:
                chunk = conn.recv(4096)
            except (BlockingIOError, socket.timeout):
                break
            except OSError as exc:
                raise ConnectionError("TCP command socket error") from exc
            if not chunk:
                raise ConnectionError("TCP client disconnected")
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if line:
                    self._forward_command(line)
        return buffer

    def _forward_command(self, line: bytes) -> None:
        try:
            command = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return
        if not isinstance(command, dict):
            return
        print(f"[mjswan-frame] TCP command from FFS: {command}", flush=True)
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast_command(command), self._loop)

    async def _broadcast_command(self, command: dict[str, Any]) -> None:
        if not self._ws_clients:
            print("[mjswan-frame] no browser WS client for command", flush=True)
            return
        payload = json.dumps(command, separators=(",", ":"))
        print(f"[mjswan-frame] broadcasting command to {len(self._ws_clients)} browser client(s): {command}", flush=True)
        stale: list[WebSocketServerProtocol] = []
        for ws in list(self._ws_clients):
            try:
                await ws.send(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._ws_clients.discard(ws)


def maybe_start_frame_bridge() -> MujocoFrameBridge | None:
    port_value = os.environ.get("FRAME_PORT") or os.environ.get("MJSWAN_FRAME_PORT")
    if not port_value:
        return None
    tcp_host = os.environ.get("FRAME_HOST", "0.0.0.0")
    ws_host = os.environ.get("FRAME_WS_HOST", "0.0.0.0")
    tcp_port = int(port_value)
    ws_port = int(os.environ.get("FRAME_WS_PORT", os.environ.get("MJSWAN_FRAME_WS_PORT", tcp_port + 1)))
    bridge = MujocoFrameBridge(tcp_host=tcp_host, tcp_port=tcp_port, ws_host=ws_host, ws_port=ws_port)
    bridge.start()
    return bridge
