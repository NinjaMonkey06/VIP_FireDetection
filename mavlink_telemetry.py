"""
mavlink_telemetry.py — Log a drone's position and attitude from its flight controller
=====================================================================================
Flight controllers such as the Pixhawk (PX4 or ArduPilot firmware) report the drone's
state as MAVLink messages. This listens to them on the computer that records the
video and writes the timed telemetry CSV that process_video.py reads
(--timed_telemetry_csv), on the same clock as the video's frame times, so no clock has
to be matched afterwards.

It is used through record_camera.py:

    python record_camera.py --source 0 --mavlink /dev/ttyACM0 --output_dir flight_01

--mavlink is a pymavlink connection string:
    /dev/ttyACM0            the flight controller's USB port, plugged into the Pi
    /dev/serial0,57600      a telemetry port wired to the Pi's serial pins, with its baud rate
    udpin:0.0.0.0:14550     MAVLink forwarded over the network (also a simulator)

One row is written per ATTITUDE message, with the latest position before it:
    time_s                  seconds since the first video frame
    latitude, longitude     from GLOBAL_POSITION_INT
    altitude_agl_m          height above the take-off (home) point: use
                            process_video.py --csv_altitude_reference takeoff
    gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg
                            the DRONE's attitude (yaw clockwise from north, pitch nose
                            up, roll right side down). The columns keep the names the
                            pipeline reads; for a camera fixed to the drone, give
                            process_video.py --camera_mount_deg so it turns them into
                            the camera's direction.
    timestamp               this computer's clock, UTC
    altitude_msl_m, position_age_s, gps_fix_type, satellites
                            for checking the data: how old the position was when the
                            attitude arrived, and the GPS fix (3 = 3D, 6 = RTK fixed)

Needs pymavlink (pip install pymavlink). Tested against MAVLink messages sent over the
network by pymavlink itself, not yet against a real flight controller.
"""

import csv
import math
import threading
import time
from datetime import datetime, timezone

from geoloc.metadata import TIMED_CSV_COLUMNS

COLUMNS = (*TIMED_CSV_COLUMNS, "timestamp", "altitude_msl_m", "position_age_s", "gps_fix_type", "satellites")
MESSAGES = ("ATTITUDE", "GLOBAL_POSITION_INT", "GPS_RAW_INT")
MESSAGE_IDS = {"ATTITUDE": 30, "GLOBAL_POSITION_INT": 33, "GPS_RAW_INT": 24}
SET_MESSAGE_INTERVAL = 511      # MAV_CMD_SET_MESSAGE_INTERVAL


def connect(connection, rate_hz=20.0, heartbeat_timeout_s=10.0):
    """Opens a MAVLink connection, waits for the flight controller's heartbeat and asks
    it to send attitude and position rate_hz times a second."""
    try:
        from pymavlink import mavutil
    except ImportError:
        raise OSError("--mavlink needs the pymavlink library (pip install pymavlink)")
    device, _, baud = connection.partition(",")
    try:
        link = mavutil.mavlink_connection(device, baud=int(baud)) if baud else mavutil.mavlink_connection(device)
    except Exception as e:
        raise OSError(f"cannot open MAVLink connection {connection!r}: {e}")
    if link.wait_heartbeat(timeout=heartbeat_timeout_s) is None:
        raise OSError(f"no heartbeat from a flight controller on {connection!r} within {heartbeat_timeout_s:g} s: "
                      f"check the cable, the port name and the baud rate")
    interval_us = 1e6 / rate_hz
    for name in ("ATTITUDE", "GLOBAL_POSITION_INT"):
        link.mav.command_long_send(link.target_system, link.target_component, SET_MESSAGE_INTERVAL, 0,
                                   MESSAGE_IDS[name], interval_us, 0, 0, 0, 0, 0)
    link.mav.command_long_send(link.target_system, link.target_component, SET_MESSAGE_INTERVAL, 0,
                               MESSAGE_IDS["GPS_RAW_INT"], 1e6, 0, 0, 0, 0, 0)
    return link


class TelemetryLogger:
    """Writes telemetry rows from a MAVLink link in a background thread. Rows start once
    start(origin) is called with the clock reading of the first video frame, so time_s
    counts from that frame."""

    def __init__(self, link, csv_path, clock=time.monotonic, wall_clock=time.time):
        self.link, self.csv_path, self.clock, self.wall_clock = link, csv_path, clock, wall_clock
        self.origin = None
        self.rows = 0
        self.rows_without_position = 0
        self.first_time_s = self.last_time_s = None
        self._position = self._position_at = None
        self._gps = (None, None)
        self._stop = threading.Event()
        self._file = open(csv_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=COLUMNS)
        self._writer.writeheader()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def start(self, origin):
        self.origin = origin

    def handle(self, msg, now, utc):
        """One received message. Keeps the latest position and GPS state; an attitude
        message becomes a row."""
        kind = msg.get_type()
        if kind == "GLOBAL_POSITION_INT":
            self._position, self._position_at = msg, now
        elif kind == "GPS_RAW_INT":
            self._gps = (msg.fix_type, msg.satellites_visible)
        elif kind == "ATTITUDE" and self.origin is not None:
            if self._position is None:
                self.rows_without_position += 1      # attitude without a position cannot be geolocated
                return
            p = self._position
            time_s = now - self.origin
            self._writer.writerow({
                "time_s": f"{time_s:.4f}",
                "latitude": p.lat / 1e7, "longitude": p.lon / 1e7,
                "altitude_agl_m": round(p.relative_alt / 1000.0, 3),
                "gimbal_yaw_deg": round(math.degrees(msg.yaw), 3),
                "gimbal_pitch_deg": round(math.degrees(msg.pitch), 3),
                "gimbal_roll_deg": round(math.degrees(msg.roll), 3),
                "timestamp": datetime.fromtimestamp(utc, timezone.utc).isoformat(timespec="milliseconds"),
                "altitude_msl_m": round(p.alt / 1000.0, 3),
                "position_age_s": round(now - self._position_at, 3),
                "gps_fix_type": "" if self._gps[0] is None else self._gps[0],
                "satellites": "" if self._gps[1] is None else self._gps[1]})
            self._file.flush()
            self.rows += 1
            self.last_time_s = time_s
            if self.first_time_s is None:
                self.first_time_s = time_s

    def _run(self):
        while not self._stop.is_set():
            try:
                msg = self.link.recv_match(type=list(MESSAGES), blocking=True, timeout=0.5)
            except Exception:           # a link that drops out must not stop the video recording
                time.sleep(0.1)
                continue
            if msg is not None:
                self.handle(msg, self.clock(), self.wall_clock())

    def stop(self):
        """Stops the thread and closes the file. Returns a summary for recording.json."""
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._file.close()
        span = (self.last_time_s - self.first_time_s) if self.rows > 1 else 0.0
        return {"file": "telemetry.csv", "samples": self.rows,
                "rate_hz": round((self.rows - 1) / span, 2) if span > 0 else None,
                "first_time_s": None if self.first_time_s is None else round(self.first_time_s, 3),
                "last_time_s": None if self.last_time_s is None else round(self.last_time_s, 3),
                "attitude_without_position": self.rows_without_position}
