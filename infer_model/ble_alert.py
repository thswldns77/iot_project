#!/usr/bin/env python3
"""BLE drowsiness status broadcaster for Raspberry Pi.

The Raspberry Pi acts as a BLE peripheral named DrowsyPi. Android connects as a
BLE central/client, subscribes to the status characteristic, and receives a
single-byte value whenever the drowsiness status changes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional


DEFAULT_DEVICE_NAME = "DrowsyPi"
DEFAULT_SERVICE_UUID = "0000d001-0000-1000-8000-00805f9b34fb"
DEFAULT_CHARACTERISTIC_UUID = "0000d002-0000-1000-8000-00805f9b34fb"

STATUS_VALUES = {
    "AWAKE": 0,
    "DROWSY": 1,
    "NO FACE": 2,
    "CALIBRATING": 2,
}


@dataclass(frozen=True)
class BleStatus:
    name: str
    value: int


class BleAlert:
    """Small synchronous wrapper around an async Bless BLE GATT server."""

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        service_uuid: str = DEFAULT_SERVICE_UUID,
        characteristic_uuid: str = DEFAULT_CHARACTERISTIC_UUID,
        startup_timeout_sec: float = 10.0,
    ) -> None:
        self.device_name = device_name
        self.service_uuid = service_uuid
        self.characteristic_uuid = characteristic_uuid
        self.startup_timeout_sec = startup_timeout_sec

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._stop_event: Optional[asyncio.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._startup_error: Optional[BaseException] = None
        self._last_status: Optional[BleStatus] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start BLE advertising in a background event-loop thread."""
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="ble-alert",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=self.startup_timeout_sec):
            raise RuntimeError("BLE server startup timed out.")
        if self._startup_error is not None:
            raise RuntimeError(
                "BLE server failed to start. Check bluetooth service and install "
                "bless/dbus-next in the Python 3.11 virtual environment."
            ) from self._startup_error

    def set_status(self, status: str) -> None:
        """Send a status update if it changed since the previous call."""
        normalized = status.upper()
        if normalized not in STATUS_VALUES:
            normalized = "NO FACE"
        next_status = BleStatus(normalized, STATUS_VALUES[normalized])

        with self._lock:
            if self._last_status == next_status:
                return
            self._last_status = next_status

        if self._loop is None or self._server is None:
            return

        future = asyncio.run_coroutine_threadsafe(
            self._publish_status(next_status),
            self._loop,
        )
        future.add_done_callback(self._log_publish_error)

    def stop(self) -> None:
        """Stop advertising and close the background event-loop thread."""
        if self._loop is None or self._stop_event is None:
            return

        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._stopped.wait(timeout=5.0)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run_event_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            logging.exception("BLE alert server crashed")
        finally:
            self._server = None
            self._stopped.set()
            loop.close()

    async def _serve(self) -> None:
        try:
            from bless import BlessServer
            from bless import GATTAttributePermissions, GATTCharacteristicProperties
        except ImportError as exc:
            self._startup_error = exc
            self._ready.set()
            return

        server = BlessServer(name=self.device_name)
        server.read_request_func = self._read_request
        server.write_request_func = self._write_request

        await server.add_new_service(self.service_uuid)
        await server.add_new_characteristic(
            self.service_uuid,
            self.characteristic_uuid,
            GATTCharacteristicProperties.read | GATTCharacteristicProperties.notify,
            bytearray([STATUS_VALUES["AWAKE"]]),
            GATTAttributePermissions.readable,
        )
        await server.start()

        self._server = server
        self._stop_event = asyncio.Event()
        self._ready.set()
        logging.info(
            "BLE alert advertising as %s service=%s characteristic=%s",
            self.device_name,
            self.service_uuid,
            self.characteristic_uuid,
        )

        try:
            await self._stop_event.wait()
        finally:
            await server.stop()
            logging.info("BLE alert stopped")

    async def _publish_status(self, status: BleStatus) -> None:
        if self._server is None:
            return

        characteristic = self._server.get_characteristic(self.characteristic_uuid)
        if characteristic is None:
            return

        characteristic.value = bytearray([status.value])
        updated = self._server.update_value(self.service_uuid, self.characteristic_uuid)
        if updated:
            logging.info("BLE status sent: %s=%d", status.name, status.value)
        else:
            logging.warning("BLE status update was not accepted: %s", status.name)

    def _read_request(self, characteristic) -> bytearray:
        return characteristic.value

    def _write_request(self, characteristic, value) -> None:
        # The status characteristic is read/notify only. Keep the callback for
        # backends that expect a write handler to exist.
        return None

    @staticmethod
    def _log_publish_error(future) -> None:
        try:
            future.result()
        except Exception:
            logging.exception("BLE status publish failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test DrowsyPi BLE alert server")
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--service-uuid", default=DEFAULT_SERVICE_UUID)
    parser.add_argument("--characteristic-uuid", default=DEFAULT_CHARACTERISTIC_UUID)
    parser.add_argument("--interval-sec", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ble = BleAlert(
        device_name=args.device_name,
        service_uuid=args.service_uuid,
        characteristic_uuid=args.characteristic_uuid,
    )
    ble.start()

    statuses = ("AWAKE", "DROWSY", "NO FACE")
    try:
        index = 0
        while True:
            status = statuses[index % len(statuses)]
            ble.set_status(status)
            print(f"BLE test status: {status}")
            index += 1
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("Stopping BLE test server.")
    finally:
        ble.stop()


if __name__ == "__main__":
    main()
