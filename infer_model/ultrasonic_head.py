#!/usr/bin/env python3
"""HC-SR04 based head-drop helper for Raspberry Pi.

The detector measures the distance between an ultrasonic sensor and the user's
face/head. It calibrates a normal seated baseline, then reports head_down when
the distance grows by a configured amount for a sustained period.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass(frozen=True)
class UltrasonicSnapshot:
    distance_cm: Optional[float] = None
    baseline_cm: Optional[float] = None
    delta_cm: Optional[float] = None
    head_down: bool = False
    hold_sec: float = 0.0
    ready: bool = False
    error: Optional[str] = None


class UltrasonicHeadDetector:
    """Continuously measure HC-SR04 distance in a background thread."""

    def __init__(
        self,
        trigger_pin: int = 23,
        echo_pin: int = 24,
        baseline_sec: float = 3.0,
        threshold_cm: float = 12.0,
        hold_sec: float = 1.0,
        min_distance_cm: float = 5.0,
        max_distance_cm: float = 120.0,
        sample_interval_sec: float = 0.08,
        echo_timeout_sec: float = 0.03,
        smoothing_window: int = 5,
    ) -> None:
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self.baseline_sec = baseline_sec
        self.threshold_cm = threshold_cm
        self.required_hold_sec = hold_sec
        self.min_distance_cm = min_distance_cm
        self.max_distance_cm = max_distance_cm
        self.sample_interval_sec = sample_interval_sec
        self.echo_timeout_sec = echo_timeout_sec
        self.smoothing_window = smoothing_window

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._snapshot = UltrasonicSnapshot(error="not started")
        self._gpio = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="ultrasonic-head",
            daemon=True,
        )
        self._thread.start()

    def snapshot(self) -> UltrasonicSnapshot:
        with self._lock:
            return self._snapshot

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._cleanup_gpio()

    def _run(self) -> None:
        try:
            import RPi.GPIO as GPIO
        except Exception as exc:
            self._set_snapshot(UltrasonicSnapshot(error=f"GPIO import failed: {exc}"))
            return

        self._gpio = GPIO
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trigger_pin, GPIO.OUT)
            GPIO.setup(self.echo_pin, GPIO.IN)
            GPIO.output(self.trigger_pin, False)
            time.sleep(0.05)
        except Exception as exc:
            self._set_snapshot(UltrasonicSnapshot(error=f"GPIO setup failed: {exc}"))
            self._cleanup_gpio()
            return

        distances: Deque[float] = collections.deque(maxlen=self.smoothing_window)
        baseline_values = []
        baseline_started_at = time.monotonic()
        baseline_cm: Optional[float] = None
        head_candidate_started_at: Optional[float] = None

        while not self._stop_event.is_set():
            now = time.monotonic()
            measured_cm = self._measure_distance()
            filtered_cm = None

            if measured_cm is not None and self.min_distance_cm <= measured_cm <= self.max_distance_cm:
                distances.append(measured_cm)
                filtered_cm = statistics.median(distances)

            if baseline_cm is None:
                if filtered_cm is not None:
                    baseline_values.append(filtered_cm)

                if now - baseline_started_at >= self.baseline_sec:
                    if baseline_values:
                        baseline_cm = statistics.median(baseline_values)
                    else:
                        baseline_started_at = now

                self._set_snapshot(
                    UltrasonicSnapshot(
                        distance_cm=filtered_cm,
                        baseline_cm=baseline_cm,
                        ready=baseline_cm is not None,
                    )
                )
                time.sleep(self.sample_interval_sec)
                continue

            delta_cm = None if filtered_cm is None else filtered_cm - baseline_cm
            candidate = delta_cm is not None and delta_cm >= self.threshold_cm

            if candidate:
                if head_candidate_started_at is None:
                    head_candidate_started_at = now
            else:
                head_candidate_started_at = None

            current_hold_sec = (
                0.0
                if head_candidate_started_at is None
                else now - head_candidate_started_at
            )
            head_down = current_hold_sec >= self.required_hold_sec

            self._set_snapshot(
                UltrasonicSnapshot(
                    distance_cm=filtered_cm,
                    baseline_cm=baseline_cm,
                    delta_cm=delta_cm,
                    head_down=head_down,
                    hold_sec=current_hold_sec,
                    ready=True,
                )
            )
            time.sleep(self.sample_interval_sec)

        self._cleanup_gpio()

    def _measure_distance(self) -> Optional[float]:
        GPIO = self._gpio
        if GPIO is None:
            return None

        try:
            GPIO.output(self.trigger_pin, True)
            time.sleep(0.00001)
            GPIO.output(self.trigger_pin, False)

            wait_started_at = time.monotonic()
            while GPIO.input(self.echo_pin) == 0:
                if time.monotonic() - wait_started_at > self.echo_timeout_sec:
                    return None

            pulse_started_at = time.monotonic()
            while GPIO.input(self.echo_pin) == 1:
                if time.monotonic() - pulse_started_at > self.echo_timeout_sec:
                    return None

            pulse_sec = time.monotonic() - pulse_started_at
            return (pulse_sec * 34300.0) / 2.0
        except Exception as exc:
            self._set_snapshot(UltrasonicSnapshot(error=f"measurement failed: {exc}"))
            return None

    def _set_snapshot(self, snapshot: UltrasonicSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def _cleanup_gpio(self) -> None:
        GPIO = self._gpio
        if GPIO is None:
            return
        try:
            GPIO.cleanup([self.trigger_pin, self.echo_pin])
        except Exception:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        finally:
            self._gpio = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test HC-SR04 head-drop detector")
    parser.add_argument("--trigger-pin", type=int, default=23)
    parser.add_argument("--echo-pin", type=int, default=24)
    parser.add_argument("--baseline-sec", type=float, default=3.0)
    parser.add_argument("--threshold-cm", type=float, default=12.0)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = UltrasonicHeadDetector(
        trigger_pin=args.trigger_pin,
        echo_pin=args.echo_pin,
        baseline_sec=args.baseline_sec,
        threshold_cm=args.threshold_cm,
        hold_sec=args.hold_sec,
    )
    detector.start()
    try:
        while True:
            snapshot = detector.snapshot()
            print(
                "distance={distance} baseline={baseline} delta={delta} "
                "head_down={head_down} hold={hold:.2f}s ready={ready} error={error}".format(
                    distance=_fmt(snapshot.distance_cm),
                    baseline=_fmt(snapshot.baseline_cm),
                    delta=_fmt(snapshot.delta_cm),
                    head_down=snapshot.head_down,
                    hold=snapshot.hold_sec,
                    ready=snapshot.ready,
                    error=snapshot.error,
                )
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping ultrasonic head detector.")
    finally:
        detector.stop()


def _fmt(value: Optional[float]) -> str:
    return "None" if value is None else f"{value:.1f}cm"


if __name__ == "__main__":
    main()
