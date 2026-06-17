#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import time
from datetime import datetime

from gpiozero import LED


def is_night_hour(hour: int, start_hour: int, end_hour: int) -> bool:
    if start_hour == end_hour:
        return True
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control an LED during night hours")
    parser.add_argument("--pin", type=int, default=17, help="BCM GPIO pin number")
    parser.add_argument("--start-hour", type=int, default=18, choices=range(24))
    parser.add_argument("--end-hour", type=int, default=7, choices=range(24))
    parser.add_argument("--check-sec", type=float, default=30.0)
    parser.add_argument("--active-low", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    running = True

    def stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    led = LED(args.pin, active_high=not args.active_low)
    last_state: bool | None = None

    try:
        while running:
            now = datetime.now()
            night = is_night_hour(now.hour, args.start_hour, args.end_hour)
            if night != last_state:
                if night:
                    led.on()
                    print(f"Night LED on: GPIO {args.pin}", flush=True)
                else:
                    led.off()
                    print(f"Night LED off: GPIO {args.pin}", flush=True)
                last_state = night
            deadline = time.monotonic() + max(args.check_sec, 0.1)
            while running and time.monotonic() < deadline:
                time.sleep(min(0.5, deadline - time.monotonic()))
    finally:
        led.off()
        led.close()


if __name__ == "__main__":
    main()
