#!/usr/bin/env python3
"""Raspberry Pi drowsiness inference entrypoint.

Default temporary model names:
    models/eye_state_model.tflite
    models/mouth_state_model.tflite

The script can also run with --rule-only before trained models exist.
"""

from __future__ import annotations

import argparse
import collections
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np


Point = Tuple[float, float]

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
LEFT_EYE_CROP = (33, 133, 159, 145, 160, 144, 158, 153)
RIGHT_EYE_CROP = (362, 263, 386, 374, 385, 380, 387, 373)
MOUTH_CROP = (61, 291, 13, 14, 0, 17, 82, 312, 87, 317)
MOUTH_RATIO = (13, 14, 82, 87, 312, 317, 61, 291)


class PiCameraSource:
    def __init__(self, width: int, height: int, fps: int) -> None:
        from picamera2 import Picamera2

        self._picam2 = Picamera2()
        config = self._picam2.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self._picam2.configure(config)
        self._picam2.set_controls({"FrameRate": fps})
        self._picam2.start()
        time.sleep(1.0)

    def read(self) -> Tuple[bool, np.ndarray]:
        return True, self._picam2.capture_array()

    def release(self) -> None:
        self._picam2.stop()


class OpenCVCameraSource:
    def __init__(self, camera_id: int, width: int, height: int, fps: int) -> None:
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open OpenCV camera id {camera_id}")

    def read(self) -> Tuple[bool, np.ndarray]:
        ok, bgr = self._cap.read()
        if not ok:
            return False, bgr
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self._cap.release()


class MjpegStreamSource:
    def __init__(self, stream_url: str) -> None:
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open MJPEG stream: {stream_url}")

    def read(self) -> Tuple[bool, np.ndarray]:
        ok, bgr = self._cap.read()
        if not ok:
            return False, bgr
        return True, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        self._cap.release()


class TFLiteClassifier:
    def __init__(
        self,
        model_path: Path,
        positive_index: int,
        normalization: str,
        num_threads: int,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model file: {model_path}")

        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                import tensorflow as tf
            except ImportError as exc:
                raise RuntimeError(
                    "Install tflite-runtime or tensorflow to load .tflite models."
                ) from exc
            Interpreter = tf.lite.Interpreter

        self._interpreter = Interpreter(
            model_path=str(model_path),
            num_threads=num_threads,
        )
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]
        self._positive_index = positive_index
        self._normalization = normalization

        _, height, width, channels = self._input["shape"]
        if channels != 3:
            raise ValueError(f"Expected RGB input with 3 channels, got {channels}")
        self.input_size = (int(width), int(height))

    def predict_positive(self, rgb_crop: np.ndarray) -> float:
        resized = cv2.resize(rgb_crop, self.input_size, interpolation=cv2.INTER_AREA)
        input_tensor = self._prepare_input(resized)
        self._interpreter.set_tensor(self._input["index"], input_tensor)
        self._interpreter.invoke()
        output = self._interpreter.get_tensor(self._output["index"])[0]
        output = self._dequantize_if_needed(output, self._output)

        if output.size == 1:
            return float(output.reshape(-1)[0])

        scores = output.reshape(-1).astype(np.float32)
        if not np.isclose(scores.sum(), 1.0, atol=0.05):
            scores = softmax(scores)
        return float(scores[self._positive_index])

    def _prepare_input(self, rgb: np.ndarray) -> np.ndarray:
        dtype = self._input["dtype"]
        data = self._normalize(rgb)
        if np.issubdtype(dtype, np.integer):
            scale, zero_point = self._input.get("quantization", (0.0, 0))
            if scale and scale > 0:
                data = (data / scale) + zero_point
            info = np.iinfo(dtype)
            data = np.clip(np.round(data), info.min, info.max)
            return np.expand_dims(data.astype(dtype), axis=0)

        return np.expand_dims(data, axis=0).astype(dtype)

    def _normalize(self, rgb: np.ndarray) -> np.ndarray:
        data = rgb.astype(np.float32)
        if self._normalization == "zero_one":
            return data / 255.0
        if self._normalization == "minus_one_one":
            return (data / 127.5) - 1.0
        if self._normalization == "raw":
            return data
        raise ValueError(f"Unknown normalization: {self._normalization}")

    @staticmethod
    def _dequantize_if_needed(output: np.ndarray, detail: dict) -> np.ndarray:
        if not np.issubdtype(output.dtype, np.integer):
            return output.astype(np.float32)
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if not scale or scale <= 0:
            return output.astype(np.float32)
        return (output.astype(np.float32) - zero_point) * scale


class ServoAlert:
    def __init__(
        self,
        pin: Optional[int],
        min_pulse_width: float,
        max_pulse_width: float,
        center_angle: float,
        sweep_angle: float,
        step_sec: float,
    ) -> None:
        self._servo = None
        self._active = False
        self._index = 0
        self._next_move_at = 0.0
        self._center_angle = center_angle
        self._step_sec = step_sec
        self._sequence = [center_angle, sweep_angle, center_angle, -sweep_angle]
        if pin is None:
            return

        from gpiozero import AngularServo

        self._servo = AngularServo(
            pin,
            min_pulse_width=min_pulse_width,
            max_pulse_width=max_pulse_width,
        )
        self._servo.angle = self._center_angle

    def set_active(self, active: bool, now: float) -> None:
        if self._servo is None:
            return

        if active:
            if not self._active:
                self._active = True
                self._index = 0
                self._next_move_at = 0.0
            if now >= self._next_move_at:
                self._servo.angle = self._sequence[self._index]
                self._index = (self._index + 1) % len(self._sequence)
                self._next_move_at = now + self._step_sec
            return

        if self._active:
            self._active = False
            self._index = 0
            self._next_move_at = 0.0
            self._servo.angle = self._center_angle

    def close(self) -> None:
        if self._servo is None:
            return
        self._servo.angle = self._center_angle
        time.sleep(0.2)
        self._servo.detach()


@dataclass
class SignalTimers:
    eye_started_at: Optional[float] = None
    yawn_started_at: Optional[float] = None
    head_started_at: Optional[float] = None

    def update(
        self,
        now: float,
        eye_closed: bool,
        yawning: bool,
        head_down: bool,
    ) -> Tuple[float, float, float]:
        self.eye_started_at = next_start(self.eye_started_at, eye_closed, now)
        self.yawn_started_at = next_start(self.yawn_started_at, yawning, now)
        self.head_started_at = next_start(self.head_started_at, head_down, now)
        return (
            held_for(self.eye_started_at, now),
            held_for(self.yawn_started_at, now),
            held_for(self.head_started_at, now),
        )


def next_start(started_at: Optional[float], active: bool, now: float) -> Optional[float]:
    if active:
        return started_at if started_at is not None else now
    return None


def held_for(started_at: Optional[float], now: float) -> float:
    return 0.0 if started_at is None else now - started_at


def make_camera(args: argparse.Namespace):
    if args.source == "picamera2":
        return PiCameraSource(args.width, args.height, args.fps)
    if args.source == "opencv":
        return OpenCVCameraSource(args.camera_id, args.width, args.height, args.fps)
    if args.source == "mjpeg":
        return MjpegStreamSource(args.stream_url)
    try:
        return PiCameraSource(args.width, args.height, args.fps)
    except Exception as exc:
        print(f"Picamera2 unavailable ({exc}); falling back to OpenCV camera.")
        return OpenCVCameraSource(args.camera_id, args.width, args.height, args.fps)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def landmark_point(landmarks: Sequence, index: int, width: int, height: int) -> Point:
    landmark = landmarks[index]
    return landmark.x * width, landmark.y * height


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def crop_from_landmarks(
    rgb: np.ndarray,
    landmarks: Sequence,
    indices: Sequence[int],
    margin: float,
) -> Optional[np.ndarray]:
    height, width = rgb.shape[:2]
    points = np.array(
        [landmark_point(landmarks, index, width, height) for index in indices],
        dtype=np.float32,
    )
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    box_w = x_max - x_min
    box_h = y_max - y_min
    pad = max(box_w, box_h) * margin

    x1 = max(0, int(x_min - pad))
    y1 = max(0, int(y_min - pad))
    x2 = min(width, int(x_max + pad))
    y2 = min(height, int(y_max + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return rgb[y1:y2, x1:x2]


def eye_aspect_ratio(landmarks: Sequence, indices: Sequence[int], width: int, height: int) -> float:
    p1, p2, p3, p4, p5, p6 = [
        landmark_point(landmarks, index, width, height) for index in indices
    ]
    horizontal = distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * horizontal)


def mouth_aspect_ratio(landmarks: Sequence, width: int, height: int) -> float:
    upper_inner = landmark_point(landmarks, MOUTH_RATIO[0], width, height)
    lower_inner = landmark_point(landmarks, MOUTH_RATIO[1], width, height)
    upper_left = landmark_point(landmarks, MOUTH_RATIO[2], width, height)
    lower_left = landmark_point(landmarks, MOUTH_RATIO[3], width, height)
    upper_right = landmark_point(landmarks, MOUTH_RATIO[4], width, height)
    lower_right = landmark_point(landmarks, MOUTH_RATIO[5], width, height)
    left_corner = landmark_point(landmarks, MOUTH_RATIO[6], width, height)
    right_corner = landmark_point(landmarks, MOUTH_RATIO[7], width, height)

    horizontal = distance(left_corner, right_corner)
    if horizontal == 0:
        return 0.0
    vertical = (
        distance(upper_inner, lower_inner)
        + distance(upper_left, lower_left)
        + distance(upper_right, lower_right)
    ) / 3.0
    return vertical / horizontal


def head_pose_degrees(landmarks: Sequence, width: int, height: int) -> Tuple[float, float, float]:
    image_points = np.array(
        [
            landmark_point(landmarks, 1, width, height),
            landmark_point(landmarks, 152, width, height),
            landmark_point(landmarks, 33, width, height),
            landmark_point(landmarks, 263, width, height),
            landmark_point(landmarks, 61, width, height),
            landmark_point(landmarks, 291, width, height),
        ],
        dtype=np.float64,
    )
    model_points = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )
    focal_length = float(width)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ok, rotation_vec, translation_vec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    projection_mat = np.hstack((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_mat)
    pitch, yaw, roll = euler_angles.flatten()
    return float(pitch), float(yaw), float(roll)


def is_head_down(delta_pitch: float, args: argparse.Namespace) -> bool:
    if args.head_sign == "positive":
        return delta_pitch >= args.head_drop_deg
    if args.head_sign == "negative":
        return delta_pitch <= -args.head_drop_deg
    return abs(delta_pitch) >= args.head_drop_deg


def put_line(frame: np.ndarray, text: str, y: int, color: Tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )


def parse_args() -> argparse.Namespace:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Raspberry Pi drowsiness inference")
    parser.add_argument("--source", choices=("auto", "picamera2", "opencv", "mjpeg"), default="auto")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--stream-url", default="http://127.0.0.1:8000/stream.mjpg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument("--no-display", action="store_true")

    parser.add_argument("--rule-only", action="store_true")
    parser.add_argument("--eye-model", type=Path, default=base_dir / "models" / "eye_state_model.tflite")
    parser.add_argument("--mouth-model", type=Path, default=base_dir / "models" / "mouth_state_model.tflite")
    parser.add_argument("--eye-closed-index", type=int, default=1)
    parser.add_argument("--mouth-yawn-index", type=int, default=1)
    parser.add_argument("--eye-threshold", type=float, default=0.8)
    parser.add_argument("--mouth-threshold", type=float, default=0.8)
    parser.add_argument(
        "--input-normalization",
        choices=("zero_one", "minus_one_one", "raw"),
        default="zero_one",
    )
    parser.add_argument("--num-threads", type=int, default=2)

    parser.add_argument("--rule-ear-threshold", type=float, default=0.21)
    parser.add_argument("--rule-mar-threshold", type=float, default=0.28)
    parser.add_argument("--head-drop-deg", type=float, default=15.0)
    parser.add_argument("--head-sign", choices=("either", "positive", "negative"), default="either")
    parser.add_argument("--calibration-sec", type=float, default=2.0)
    parser.add_argument("--enable-head", dest="disable_head", action="store_false")
    parser.add_argument("--disable-head", dest="disable_head", action="store_true")
    parser.set_defaults(disable_head=True)

    parser.add_argument("--eye-sec", type=float, default=2.0)
    parser.add_argument("--yawn-sec", type=float, default=3.0)
    parser.add_argument("--head-sec", type=float, default=1.5)
    parser.add_argument("--window-sec", type=float, default=5.0)
    parser.add_argument("--perclos-threshold", type=float, default=0.45)

    parser.add_argument("--servo-pin", type=int, default=None)
    parser.add_argument("--servo-center-angle", type=float, default=0.0)
    parser.add_argument("--servo-sweep-angle", type=float, default=90.0)
    parser.add_argument("--servo-step-sec", type=float, default=1.0)
    parser.add_argument("--servo-min-pulse-width", type=float, default=0.0005)
    parser.add_argument("--servo-max-pulse-width", type=float, default=0.0025)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    eye_model = None
    mouth_model = None
    if not args.rule_only:
        eye_model = TFLiteClassifier(
            args.eye_model,
            args.eye_closed_index,
            args.input_normalization,
            args.num_threads,
        )
        mouth_model = TFLiteClassifier(
            args.mouth_model,
            args.mouth_yawn_index,
            args.input_normalization,
            args.num_threads,
        )

    camera = make_camera(args)
    servo = ServoAlert(
        args.servo_pin,
        args.servo_min_pulse_width,
        args.servo_max_pulse_width,
        args.servo_center_angle,
        args.servo_sweep_angle,
        args.servo_step_sec,
    )

    timers = SignalTimers()
    history: Deque[Tuple[float, bool, bool, bool]] = collections.deque()
    calibration_started_at: Optional[float] = None
    calibration_pitches = []
    baseline_pitch: Optional[float] = None
    fps = 0.0
    previous_frame_at = time.monotonic()
    previous_log_at = 0.0

    mp_face_mesh = mp.solutions.face_mesh

    try:
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            while True:
                ok, rgb = camera.read()
                if not ok:
                    print("Camera frame read failed.")
                    break

                if args.mirror:
                    rgb = cv2.flip(rgb, 1)

                now = time.monotonic()
                dt = max(now - previous_frame_at, 1e-6)
                previous_frame_at = now
                fps = (0.9 * fps) + (0.1 * (1.0 / dt)) if fps else 1.0 / dt

                height, width = rgb.shape[:2]
                results = face_mesh.process(rgb)
                face_found = bool(results.multi_face_landmarks)

                eye_prob = 0.0
                mouth_prob = 0.0
                ear = 0.0
                mar = 0.0
                pitch = 0.0
                delta_pitch = 0.0
                eye_closed = False
                yawning = False
                head_down = False

                if face_found:
                    landmarks = results.multi_face_landmarks[0].landmark

                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, width, height)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, width, height)
                    ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(landmarks, width, height)
                    if not args.disable_head:
                        pitch, _, _ = head_pose_degrees(landmarks, width, height)

                        if baseline_pitch is None:
                            if calibration_started_at is None:
                                calibration_started_at = now
                                calibration_pitches.clear()
                            if now - calibration_started_at <= args.calibration_sec:
                                calibration_pitches.append(pitch)
                            elif calibration_pitches:
                                baseline_pitch = statistics.median(calibration_pitches)
                        else:
                            delta_pitch = pitch - baseline_pitch
                            head_down = is_head_down(delta_pitch, args)

                    if args.rule_only:
                        eye_closed = ear < args.rule_ear_threshold
                        yawning = mar > args.rule_mar_threshold
                    else:
                        left_eye = crop_from_landmarks(rgb, landmarks, LEFT_EYE_CROP, margin=0.9)
                        right_eye = crop_from_landmarks(rgb, landmarks, RIGHT_EYE_CROP, margin=0.9)
                        mouth = crop_from_landmarks(rgb, landmarks, MOUTH_CROP, margin=0.8)
                        if left_eye is not None and right_eye is not None:
                            left_prob = eye_model.predict_positive(left_eye)
                            right_prob = eye_model.predict_positive(right_eye)
                            eye_prob = max(left_prob, right_prob)
                            eye_closed = eye_prob >= args.eye_threshold
                        if mouth is not None:
                            mouth_prob = mouth_model.predict_positive(mouth)
                            yawning = mouth_prob >= args.mouth_threshold

                eye_hold, yawn_hold, head_hold = timers.update(
                    now,
                    eye_closed,
                    yawning,
                    head_down,
                )

                history.append((now, eye_closed, yawning, head_down))
                while history and now - history[0][0] > args.window_sec:
                    history.popleft()

                eye_ratio = (
                    sum(1 for _, eye, _, _ in history if eye) / len(history)
                    if history
                    else 0.0
                )

                drowsy = (
                    eye_hold >= args.eye_sec
                    or yawn_hold >= args.yawn_sec
                    or head_hold >= args.head_sec
                    or eye_ratio >= args.perclos_threshold
                )

                status = "DROWSY" if drowsy else "AWAKE"
                if not face_found:
                    status = "NO FACE"
                elif not args.disable_head and baseline_pitch is None:
                    status = "CALIBRATING"

                servo.set_active(status == "DROWSY", now)

                if args.no_display:
                    if now - previous_log_at >= 1.0:
                        previous_log_at = now
                        print(
                            f"{status} fps={fps:.1f} eye={eye_prob:.2f} "
                            f"mouth={mouth_prob:.2f} ear={ear:.3f} mar={mar:.3f} "
                            f"pitch_delta={delta_pitch:.1f}"
                        )
                    continue

                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                color = (0, 0, 255) if status == "DROWSY" else (0, 180, 0)
                if status in ("NO FACE", "CALIBRATING"):
                    color = (0, 180, 255)

                put_line(frame, f"Status: {status}", 30, color)
                put_line(frame, f"FPS: {fps:4.1f}", 60, (255, 255, 255))
                put_line(
                    frame,
                    f"Eye prob: {eye_prob:.2f} closed={eye_closed} hold={eye_hold:.1f}s EAR={ear:.3f}",
                    90,
                    (255, 255, 255),
                )
                put_line(
                    frame,
                    f"Mouth prob: {mouth_prob:.2f} yawn={yawning} hold={yawn_hold:.1f}s MAR={mar:.3f}",
                    120,
                    (255, 255, 255),
                )
                put_line(
                    frame,
                    f"Pitch delta: {delta_pitch:.1f} head={head_down} hold={head_hold:.1f}s",
                    150,
                    (255, 255, 255),
                )
                put_line(
                    frame,
                    f"PERCLOS-ish {args.window_sec:.0f}s: {eye_ratio:.2f}",
                    180,
                    (255, 255, 255),
                )

                cv2.imshow("Drowsiness inference", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
    finally:
        servo.close()
        camera.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
