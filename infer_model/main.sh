#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
FPS="${FPS:-30}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_STARTUP_SEC="${CAMERA_STARTUP_SEC:-5}"
NIGHT_LED="${NIGHT_LED:-1}"
NIGHT_LED_PIN="${NIGHT_LED_PIN:-17}"
NIGHT_LED_START_HOUR="${NIGHT_LED_START_HOUR:-18}"
NIGHT_LED_END_HOUR="${NIGHT_LED_END_HOUR:-7}"
NIGHT_LED_CHECK_SEC="${NIGHT_LED_CHECK_SEC:-30}"
NIGHT_LED_ACTIVE_LOW="${NIGHT_LED_ACTIVE_LOW:-0}"
STREAM_URL="http://${HOST}:${PORT}/stream.mjpg"
CAMERA_PID=""
NIGHT_LED_PID=""

cleanup() {
  if [[ -n "$NIGHT_LED_PID" ]] && kill -0 "$NIGHT_LED_PID" 2>/dev/null; then
    kill "$NIGHT_LED_PID" 2>/dev/null || true
    wait "$NIGHT_LED_PID" 2>/dev/null || true
  fi

  if [[ -n "$CAMERA_PID" ]] && kill -0 "$CAMERA_PID" 2>/dev/null; then
    kill "$CAMERA_PID" 2>/dev/null || true
    wait "$CAMERA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "Missing .venv. Create the Python 3.11 virtual environment first." >&2
  exit 1
fi

if ! pgrep -f "camera_server_picamera2.py.*--port ${PORT}" >/dev/null; then
  /usr/bin/python3 "$SCRIPT_DIR/camera_server_picamera2.py" \
    --camera-index "$CAMERA_INDEX" \
    --host "$HOST" \
    --port "$PORT" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --fps "$FPS" &
  CAMERA_PID="$!"
  sleep "$CAMERA_STARTUP_SEC"

  if ! kill -0 "$CAMERA_PID" 2>/dev/null; then
    echo "Camera server failed to start." >&2
    wait "$CAMERA_PID" 2>/dev/null || true
    exit 1
  fi
fi

if [[ "$NIGHT_LED" != "0" ]]; then
  night_led_args=(
    --pin "$NIGHT_LED_PIN"
    --start-hour "$NIGHT_LED_START_HOUR"
    --end-hour "$NIGHT_LED_END_HOUR"
    --check-sec "$NIGHT_LED_CHECK_SEC"
  )

  if [[ "$NIGHT_LED_ACTIVE_LOW" == "1" ]]; then
    night_led_args+=(--active-low)
  fi

  /usr/bin/python3 "$SCRIPT_DIR/night_led.py" "${night_led_args[@]}" &
  NIGHT_LED_PID="$!"
  sleep 0.5

  if ! kill -0 "$NIGHT_LED_PID" 2>/dev/null; then
    echo "Night LED helper failed to start." >&2
    wait "$NIGHT_LED_PID" 2>/dev/null || true
    NIGHT_LED_PID=""
  fi
fi

source ".venv/bin/activate"

python "$SCRIPT_DIR/run_inference.py" \
  --source mjpeg \
  --stream-url "$STREAM_URL" \
  --mirror \
  "$@"
