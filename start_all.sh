#!/usr/bin/env bash
#
# 一键启动遥操仿真全链路服务
#
# 服务链路:
#   mjswan (仿真+帧桥接)  →  Fast-FoundationStereo (深度估计)  →  StereoSpatial (点云展示+控制)
#
# 端口:
#   8080  - mjswan 仿真 Web UI
#   9876  - 帧桥接 TCP (FFS 连接)
#   9877  - 帧桥接 WebSocket (浏览器连接)
#   8091  - Splat 静态文件服务
#   8765  - FFS 感知 WebSocket
#   8190  - StereoSpatial 场景中继
#   5174  - StereoSpatial 前端 UI
#
# 用法:
#   ./start_all.sh              # 启动所有服务
#   ./start_all.sh --rebuild    # 重建 mjswan 前端后启动
#   ./start_all.sh --stop       # 停止所有后台服务
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAN_IP="${LAN_IP:-192.168.3.38}"

# 项目路径
MJSWAN_DIR="$SCRIPT_DIR"
FFS_DIR="${FFS_DIR:-$HOME/Fast-FoundationStereo}"
SPATIAL_DIR="${SPATIAL_DIR:-$HOME/StereoSpatial/SpatialCanvas}"
SPLAT_ROOT="${SPLAT_ROOT:-$HOME/CloudTwin_Splat/public}"

# 端口配置
MJSWAN_PORT="${PORT:-8080}"
FRAME_PORT="${FRAME_PORT:-9876}"
SPLAT_PORT="${SPLAT_PORT:-8091}"
FFS_PORT="${FFS_PORT:-8765}"
RELAY_PORT="${RELAY_PORT:-8190}"
UI_PORT="${UI_PORT:-5174}"

# conda 环境
FFS_CONDA_ENV="${FFS_CONDA_ENV:-ffs}"

# PID 文件目录
PID_DIR="$MJSWAN_DIR/.run_pids"

# ─── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log()  { echo -e "${GREEN}[start_all]${NC} $*"; }
warn() { echo -e "${YELLOW}[start_all]${NC} $*"; }
err()  { echo -e "${RED}[start_all]${NC} $*" >&2; }

# ─── 停止所有服务 ─────────────────────────────────────────────────────────────
stop_all() {
  log "停止所有服务..."
  if [[ -d "$PID_DIR" ]]; then
    for pidfile in "$PID_DIR"/*.pid; do
      [[ -f "$pidfile" ]] || continue
      pid=$(cat "$pidfile")
      name=$(basename "$pidfile" .pid)
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && log "已停止 $name (PID $pid)" || warn "无法停止 $name (PID $pid)"
        # 等待进程退出
        for _ in {1..10}; do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.2
        done
        # 强制杀掉
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null
      else
        log "$name (PID $pid) 已不存在"
      fi
      rm -f "$pidfile"
    done
    rmdir "$PID_DIR" 2>/dev/null || true
  else
    warn "没有找到运行中的服务"
  fi
  exit 0
}

if [[ "${1:-}" == "--stop" ]]; then
  stop_all
fi

# ─── 前置检查 ─────────────────────────────────────────────────────────────────
check_dir() {
  if [[ ! -d "$1" ]]; then
    err "目录不存在: $1"
    exit 1
  fi
}

check_dir "$MJSWAN_DIR"
check_dir "$FFS_DIR"
check_dir "$SPATIAL_DIR"

if ! command -v uv >/dev/null 2>&1; then
  err "uv 未安装"
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  err "node 未安装"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  err "conda 未安装"
  exit 1
fi

# ─── 准备 PID 目录 ────────────────────────────────────────────────────────────
mkdir -p "$PID_DIR"

save_pid() {
  echo "$2" > "$PID_DIR/$1.pid"
}

# ─── 启动函数 ─────────────────────────────────────────────────────────────────

start_splat_server() {
  if [[ ! -d "$SPLAT_ROOT" ]]; then
    warn "Splat 目录不存在: $SPLAT_ROOT，跳过 Splat 服务"
    return
  fi
  log "启动 Splat 服务 (端口 $SPLAT_PORT)..."
  cd "$MJSWAN_DIR"
  uv run python scripts/serve_splats.py \
    --root "$SPLAT_ROOT" --port "$SPLAT_PORT" \
    > /tmp/mjswan_splat.log 2>&1 &
  save_pid "splat_server" $!
  log "  Splat 服务 PID=$! → http://${LAN_IP}:${SPLAT_PORT}"
}

start_scene_relay() {
  log "启动 Scene Relay (端口 $RELAY_PORT)..."
  cd "$SPATIAL_DIR"
  node scene-relay.mjs "$RELAY_PORT" \
    > /tmp/mjswan_relay.log 2>&1 &
  save_pid "scene_relay" $!
  log "  Scene Relay PID=$! → ws://${LAN_IP}:${RELAY_PORT}"
}

start_spatial_ui() {
  log "启动 StereoSpatial UI (端口 $UI_PORT)..."
  cd "$SPATIAL_DIR"
  npx vite --host 0.0.0.0 --port "$UI_PORT" \
    > /tmp/mjswan_spatial_ui.log 2>&1 &
  save_pid "spatial_ui" $!
  log "  StereoSpatial UI PID=$! → http://${LAN_IP}:${UI_PORT}"
}

start_mjswan() {
  local rebuild_flag="${1:-}"
  log "启动 mjswan 仿真 (端口 $MJSWAN_PORT, 帧桥接 $FRAME_PORT)..."
  cd "$MJSWAN_DIR"
  PORT="$MJSWAN_PORT" FRAME_PORT="$FRAME_PORT" \
  MJSWAN_SPLAT_HOST="$LAN_IP" MJSWAN_SPLAT_PORT="$SPLAT_PORT" \
    bash start_demo.sh $rebuild_flag \
    > /tmp/mjswan_demo.log 2>&1 &
  save_pid "mjswan" $!
  log "  mjswan PID=$! → http://${LAN_IP}:${MJSWAN_PORT}"
}

start_ffs() {
  log "启动 Fast-FoundationStereo (端口 $FFS_PORT)..."
  cd "$FFS_DIR"

  # 使用 conda 环境
  eval "$(conda shell.bash hook)"
  conda activate "$FFS_CONDA_ENV"

  python scripts/run_perception_service.py \
    --camera mujoco \
    --mujoco-host 127.0.0.1 \
    --mujoco-port "$FRAME_PORT" \
    --detect-source left_ir \
    --transport websocket \
    --host 0.0.0.0 \
    --port "$FFS_PORT" \
    --relay-url "ws://127.0.0.1:${RELAY_PORT}" \
    --scene-refresh-interval 5 \
    --scene-max-depth 12 \
    > /tmp/mjswan_ffs.log 2>&1 &
  save_pid "ffs" $!
  log "  FFS PID=$! → ws://${LAN_IP}:${FFS_PORT}"
}

# ─── 主流程 ───────────────────────────────────────────────────────────────────

# 先停止旧的服务
if [[ -d "$PID_DIR" ]] && ls "$PID_DIR"/*.pid >/dev/null 2>&1; then
  warn "检测到旧服务，先停止..."
  stop_all_quiet() {
    for pidfile in "$PID_DIR"/*.pid; do
      [[ -f "$pidfile" ]] || continue
      pid=$(cat "$pidfile")
      kill "$pid" 2>/dev/null || true
    done
    rm -f "$PID_DIR"/*.pid
  }
  stop_all_quiet
  sleep 1
fi

REBUILD_FLAG=""
if [[ "${1:-}" == "--rebuild" ]]; then
  REBUILD_FLAG="--rebuild"
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        遥操仿真环境 - 全链路启动                           ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  mjswan 仿真:        http://${LAN_IP}:${MJSWAN_PORT}             ║${NC}"
echo -e "${CYAN}║  帧桥接 TCP:         ${LAN_IP}:${FRAME_PORT}                     ║${NC}"
echo -e "${CYAN}║  Splat 服务:         http://${LAN_IP}:${SPLAT_PORT}             ║${NC}"
echo -e "${CYAN}║  FFS 感知:           ws://${LAN_IP}:${FFS_PORT}               ║${NC}"
echo -e "${CYAN}║  Scene Relay:        ws://${LAN_IP}:${RELAY_PORT}               ║${NC}"
echo -e "${CYAN}║  StereoSpatial UI:   http://${LAN_IP}:${UI_PORT}             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# 按依赖顺序启动: relay → splat → mjswan → (等 mjswan ready) → ffs → ui
start_scene_relay
start_splat_server
sleep 1

start_mjswan "$REBUILD_FLAG"

# 等待 mjswan 帧桥接就绪
log "等待 mjswan 帧桥接端口 ${FRAME_PORT} 就绪..."
for i in {1..30}; do
  if ss -ltnp 2>/dev/null | grep -q ":${FRAME_PORT} " 2>/dev/null; then
    log "帧桥接已就绪"
    break
  fi
  if [[ $i -eq 30 ]]; then
    warn "等待超时，继续启动 FFS（可能需要手动确认连接）"
  fi
  sleep 2
done

start_ffs
sleep 1
start_spatial_ui

echo ""
log "所有服务已启动！"
log "日志文件:"
log "  mjswan:       /tmp/mjswan_demo.log"
log "  FFS:          /tmp/mjswan_ffs.log"
log "  Scene Relay:  /tmp/mjswan_relay.log"
log "  Splat:        /tmp/mjswan_splat.log"
log "  SpatialUI:    /tmp/mjswan_spatial_ui.log"
echo ""
log "停止所有服务: ./start_all.sh --stop"
log "查看状态: cat $PID_DIR/*.pid"
echo ""

# 等待任意子进程退出
wait
