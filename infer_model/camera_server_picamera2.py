#!/usr/bin/env python3
"""Serve Raspberry Pi Picamera2 frames as a local MJPEG stream.

Run this script with the system Python that can import Picamera2. The inference
script can then run in a separate Python 3.11 virtual environment and read
http://127.0.0.1:8000/stream.mjpg with OpenCV.
"""

from __future__ import annotations

import argparse
import io
import logging
import socketserver
from http import server
from threading import Condition

from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput


class StreamingOutput(io.BufferedIOBase):
    """Picamera2 writes JPEG bytes here, and HTTP clients wait for new frames."""

    def __init__(self) -> None:
        self.frame: bytes | None = None
        self.condition = Condition()

    def write(self, buf: bytes) -> int:
        with self.condition:
            self.frame = bytes(buf)
            self.condition.notify_all()
        return len(buf)


class StreamingHandler(server.BaseHTTPRequestHandler):
    output: StreamingOutput

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(301)
            self.send_header("Location", "/stream.mjpg")
            self.end_headers()
            return

        if self.path != "/stream.mjpg":
            self.send_error(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
        self.end_headers()

        try:
            while True:
                with self.output.condition:
                    self.output.condition.wait()
                    frame = self.output.frame
                if frame is None:
                    continue

                self.wfile.write(b"--FRAME\r\n")
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.end_headers()
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
        except Exception as exc:
            logging.info("Removed streaming client %s: %s", self.client_address, exc)

    def log_message(self, format: str, *args) -> None:
        logging.info("%s - %s", self.address_string(), format % args)


class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve Picamera2 as MJPEG")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--quality", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = StreamingOutput()
    StreamingHandler.output = output

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (args.width, args.height)},
        controls={"FrameRate": args.fps},
    )
    picam2.configure(config)

    address = (args.host, args.port)
    httpd = StreamingServer(address, StreamingHandler)
    stream_url = f"http://{args.host}:{args.port}/stream.mjpg"
    logging.info("Starting Picamera2 MJPEG server at %s", stream_url)

    recording = False
    try:
        picam2.start_recording(
            JpegEncoder(q=args.quality),
            FileOutput(output),
        )
        recording = True
        logging.info("Picamera2 is recording; waiting for MJPEG clients")
        httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception:
        logging.exception("Picamera2 MJPEG server failed")
        raise
    finally:
        logging.info("Stopping Picamera2 MJPEG server")
        if recording:
            picam2.stop_recording()
        httpd.server_close()


if __name__ == "__main__":
    main()
