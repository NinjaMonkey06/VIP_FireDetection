"""Tests for mavlink_telemetry.py and the fixed-camera path through the pipeline. Stand-ins
replace the flight controller and the pymavlink library, so no drone is needed."""

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest

import numpy as np

import mavlink_telemetry
import record_camera
from geoloc.projection import camera_angles_from_vehicle, gimbal_to_ned
from tests.test_process_video import CAMERA, frame
from tests.test_record_camera import FakeCamera

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Message:
    def __init__(self, kind, **fields):
        self._kind = kind
        self.__dict__.update(fields)

    def get_type(self):
        return self._kind


def position(lat=34.15, lon=-117.85, relative_alt_m=100.0, alt_m=350.0):
    return Message("GLOBAL_POSITION_INT", lat=round(lat * 1e7), lon=round(lon * 1e7),
                   relative_alt=round(relative_alt_m * 1000), alt=round(alt_m * 1000))


def attitude(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    return Message("ATTITUDE", yaw=math.radians(yaw_deg), pitch=math.radians(pitch_deg), roll=math.radians(roll_deg))


class FakeLink:
    """A MAVLink link that delivers a list of messages, then nothing."""

    def __init__(self, messages=()):
        self.messages = list(messages)
        self.commands = []
        self.target_system, self.target_component = 1, 1
        self.mav = types.SimpleNamespace(command_long_send=lambda *args: self.commands.append(args))

    def recv_match(self, type=None, blocking=True, timeout=None):
        if self.messages:
            return self.messages.pop(0)
        time.sleep(0.01)
        return None

    def wait_heartbeat(self, timeout=None):
        return object()


def read_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCameraAnglesFromVehicle(unittest.TestCase):
    def test_camera_looking_straight_down_follows_the_drone(self):
        # level drone heading 30°: the camera looks straight down, top of the picture towards 30°
        np.testing.assert_allclose(camera_angles_from_vehicle(30.0, 0.0, 0.0), (30.0, -90.0, 0.0), atol=1e-9)
        # nose up 10°: a camera fixed to the belly now looks 10° ahead of straight down
        np.testing.assert_allclose(camera_angles_from_vehicle(0.0, 10.0, 0.0), (0.0, -80.0, 0.0), atol=1e-9)
        # right side down 10°: it looks 10° to the left (west of a north-facing drone)
        yaw, pitch, _ = camera_angles_from_vehicle(0.0, 0.0, 10.0)
        np.testing.assert_allclose((yaw, pitch), (-90.0, -80.0), atol=1e-9)

    def test_forward_looking_mount(self):
        np.testing.assert_allclose(camera_angles_from_vehicle(120.0, 0.0, 0.0, 0.0, -30.0, 0.0),
                                   (120.0, -30.0, 0.0), atol=1e-9)

    def test_result_is_the_two_rotations_combined(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            vehicle = rng.uniform([-180, -30, -30], [180, 30, 30])
            mount = rng.uniform([-180, -90, -20], [180, 0, 20])
            if rng.random() < 0.5:
                mount = np.array([rng.uniform(-180, 180), -90.0, 0.0])      # straight down
            if rng.random() < 0.2:
                vehicle[1:] = 0.0                                            # level: the degenerate case
            out = camera_angles_from_vehicle(*vehicle, *mount)
            np.testing.assert_allclose(gimbal_to_ned(*out), gimbal_to_ned(*vehicle) @ gimbal_to_ned(*mount),
                                       atol=1e-9)


class TestTelemetryLogger(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "telemetry.csv")

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_row_per_attitude_with_the_latest_position_on_the_video_clock(self):
        logger = mavlink_telemetry.TelemetryLogger(FakeLink(), self.path)
        utc = 1794787200.0                                       # 2026-11-16T00:00:00Z
        logger.handle(attitude(10.0), 99.0, utc)                 # before the first frame: not written
        logger.start(origin=100.0)
        logger.handle(attitude(11.0), 100.2, utc)                # no position yet: counted, not written
        logger.handle(Message("GPS_RAW_INT", fix_type=3, satellites_visible=14), 100.3, utc)
        logger.handle(position(lon=-117.85), 100.4, utc)
        logger.handle(attitude(30.0, 2.0, -1.5), 100.5, utc + 0.5)
        logger.handle(position(lon=-117.84, relative_alt_m=80.0), 100.9, utc)
        logger.handle(attitude(40.0), 101.0, utc + 1.0)
        summary = logger.stop()
        rows = read_rows(self.path)
        self.assertEqual(list(rows[0])[:7], list(mavlink_telemetry.TIMED_CSV_COLUMNS))
        self.assertEqual([float(r["time_s"]) for r in rows], [0.5, 1.0])
        first = rows[0]
        self.assertEqual((float(first["latitude"]), float(first["longitude"]), float(first["altitude_agl_m"])),
                         (34.15, -117.85, 100.0))
        self.assertEqual((float(first["gimbal_yaw_deg"]), float(first["gimbal_pitch_deg"]),
                          float(first["gimbal_roll_deg"])), (30.0, 2.0, -1.5))
        self.assertEqual(first["timestamp"], "2026-11-16T00:00:00.500+00:00")
        self.assertEqual((float(first["position_age_s"]), first["gps_fix_type"], first["satellites"]), (0.1, "3", "14"))
        self.assertEqual((float(rows[1]["longitude"]), float(rows[1]["altitude_agl_m"])), (-117.84, 80.0))
        self.assertEqual(summary, {"file": "telemetry.csv", "samples": 2, "rate_hz": 2.0, "first_time_s": 0.5,
                                   "last_time_s": 1.0, "attitude_without_position": 1})

    def test_background_thread_reads_the_link(self):
        link = FakeLink()
        logger = mavlink_telemetry.TelemetryLogger(link, self.path)
        logger.start(origin=time.monotonic() - 1.0)
        link.messages.extend([position(), attitude(5.0), attitude(6.0), attitude(7.0)])
        deadline = time.time() + 5.0
        while link.messages and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.05)
        summary = logger.stop()
        self.assertEqual(summary["samples"], 3)
        self.assertEqual([float(r["gimbal_yaw_deg"]) for r in read_rows(self.path)], [5.0, 6.0, 7.0])

    def test_a_link_that_fails_does_not_stop_the_logger(self):
        class Broken(FakeLink):
            def recv_match(self, **kw):
                raise OSError("cable unplugged")
        logger = mavlink_telemetry.TelemetryLogger(Broken(), self.path)
        logger.start(origin=0.0)
        time.sleep(0.05)
        self.assertEqual(logger.stop()["samples"], 0)


class TestConnect(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("pymavlink")
        self.opened = []

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("pymavlink", None)
        else:
            sys.modules["pymavlink"] = self._saved

    def install(self, link):
        def mavlink_connection(device, **kw):
            self.opened.append((device, kw))
            return link
        module = types.ModuleType("pymavlink")
        module.mavutil = types.SimpleNamespace(mavlink_connection=mavlink_connection)
        sys.modules["pymavlink"] = module

    def test_waits_for_the_heartbeat_and_asks_for_the_message_rates(self):
        link = FakeLink()
        self.install(link)
        self.assertIs(mavlink_telemetry.connect("/dev/serial0,57600", rate_hz=20.0), link)
        self.assertEqual(self.opened, [("/dev/serial0", {"baud": 57600})])
        asked = {c[4]: c[5] for c in link.commands}             # message id -> interval in microseconds
        self.assertEqual(asked, {30: 50000.0, 33: 50000.0, 24: 1e6})
        self.assertTrue(all(c[2] == 511 for c in link.commands))

    def test_no_heartbeat_or_no_library_is_a_clear_error(self):
        silent = FakeLink()
        silent.wait_heartbeat = lambda timeout=None: None
        self.install(silent)
        with self.assertRaisesRegex(OSError, "no heartbeat"):
            mavlink_telemetry.connect("/dev/ttyACM0", heartbeat_timeout_s=0.1)
        self.assertEqual(self.opened, [("/dev/ttyACM0", {})])
        sys.modules["pymavlink"] = None
        with self.assertRaisesRegex(OSError, "pip install pymavlink"):
            mavlink_telemetry.connect("/dev/ttyACM0")


class ScriptedFlight(FakeCamera):
    """A camera whose every frame comes with the flight controller's messages of that moment:
    the drone flies east at 0.001° per frame, level, heading 30°, 100 m up."""

    def __init__(self, frames, ticks):
        super().__init__(frames)
        self.ticks, self.logger, self.k = list(ticks), None, 0

    def read(self):
        now = self.ticks[self.k] if self.k < len(self.ticks) else None
        if now is not None and self.frames:
            self.logger.handle(position(lon=-117.85 + 0.001 * self.k), now - 0.02, 0.0)
            self.logger.handle(attitude(30.0), now - 0.01, 0.0)
        self.k += 1
        return super().read()


class TestRecordingWithTelemetryThroughThePipeline(unittest.TestCase):
    def test_video_and_telemetry_share_one_clock_and_the_fixed_camera_is_geolocated(self):
        with tempfile.TemporaryDirectory() as d:
            ticks = [100.0, 101.0, 102.0, 103.0]
            camera = ScriptedFlight([frame(s) for s in (0, 20, 40, 60)], ticks)
            clock = iter(ticks + [999.0])
            out = os.path.join(d, "flight")
            os.makedirs(out)
            logger = mavlink_telemetry.TelemetryLogger(FakeLink(), os.path.join(out, "telemetry.csv"))
            camera.logger = logger
            summary = record_camera.record(camera, out, codec="ffv1", clock=lambda: next(clock), telemetry=logger,
                                           max_frames=4)
            self.assertEqual(summary["frames"], 4)
            self.assertEqual(summary["telemetry"]["samples"], 3)       # none before the first frame
            rows = read_rows(os.path.join(out, "telemetry.csv"))
            np.testing.assert_allclose([float(r["time_s"]) for r in rows], [0.99, 1.99, 2.99], atol=1e-6)

            cam_json = os.path.join(d, "camera.json")
            with open(cam_json, "w") as f:
                json.dump(CAMERA, f)

            def process(name, *extra):
                result = os.path.join(d, name)
                proc = subprocess.run(
                    [sys.executable, os.path.join(REPO, "process_video.py"), "--video", os.path.join(out, "video.avi"),
                     "--frame_times_csv", os.path.join(out, "frame_times.csv"),
                     "--timed_telemetry_csv", os.path.join(out, "telemetry.csv"), "--camera_config", cam_json,
                     "--csv_altitude_reference", "takeoff", "--output_dir", result, *extra],
                    capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(os.path.join(result, "geolocated_results.json")) as f:
                    return json.load(f)["results"]

            mounted = process("mounted", "--camera_mount_deg", "0", "-90", "0")
            drone = mounted[2]["geolocation"]["drone"]
            # frame 2 is at 2.0 s, just after the sample at 1.99 s (drone at -117.848)
            self.assertAlmostEqual(drone["lon"], -117.848, places=4)
            self.assertEqual((drone["gimbal_yaw_deg"], drone["gimbal_pitch_deg"], drone["gimbal_roll_deg"]),
                             (30.0, -90.0, 0.0))
            self.assertEqual(drone["altitude_reference"], "takeoff")
            self.assertEqual(mounted[2]["fire_regions"][0]["geolocation_status"], "ok")
            # the blob is at the image centre in frame 2... 40 px right of it: on the ground, to the
            # right of a drone heading 30°, i.e. towards the south-east
            c = mounted[2]["fire_regions"][0]["gps_centroid"]
            self.assertGreater(c["lon"], drone["lon"])
            self.assertLess(c["lat"], drone["lat"])

            # without the mount the level attitude is taken as the camera pointing at the horizon: wrong place
            level = process("level")
            self.assertEqual(level[2]["geolocation"]["drone"]["gimbal_pitch_deg"], 0.0)
            self.assertNotEqual(level[2]["fire_regions"][0]["geolocation_status"], "ok")


if __name__ == "__main__":
    unittest.main()
