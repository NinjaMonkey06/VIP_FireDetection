"""Tests for record_camera.py. A video file stands in for the camera (OpenCV opens both the same
way), and fake clocks stand in for a camera that delivers frames unevenly."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

import record_camera
from tests.test_process_video import CAMERA, frame

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_video(path, frames, fps=10.0):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"FFV1"), fps, (640, 512))
    if not writer.isOpened():
        raise unittest.SkipTest("FFV1 video writer not available")
    for f in frames:
        writer.write(f)
    writer.release()


class FakeCamera:
    """A live camera: no frame count, and every third read fails once."""

    def __init__(self, frames, fail_every=0):
        self.frames, self.fail_every, self.reads = list(frames), fail_every, 0

    def get(self, prop):
        return 0.0

    def read(self):
        self.reads += 1
        if self.fail_every and self.reads % self.fail_every == 0:
            return False, None
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)


class RecordTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.frames = [frame(s) for s in (0, 20, 40, 60, 80, 100)]
        self.source = os.path.join(self.dir, "source.avi")
        write_video(self.source, self.frames)

    def tearDown(self):
        self._tmp.cleanup()

    def read_times(self, out):
        with open(os.path.join(out, "frame_times.csv"), newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def read_video(self, path):
        cap, frames = cv2.VideoCapture(path), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        return frames


class TestRecord(RecordTest):
    def test_lossless_recording_keeps_every_frame_and_gives_each_a_time(self):
        out = os.path.join(self.dir, "rec")
        cap = record_camera.open_source(self.source)
        summary = record_camera.record(cap, out, codec="ffv1")
        cap.release()
        recorded = self.read_video(os.path.join(out, "video.avi"))
        self.assertEqual(len(recorded), 6)
        for a, b in zip(recorded, self.frames):
            np.testing.assert_array_equal(a, b)
        rows = self.read_times(out)
        self.assertEqual([int(r["frame_index"]) for r in rows], [0, 1, 2, 3, 4, 5])
        self.assertEqual(float(rows[0]["time_s"]), 0.0)
        times = [float(r["time_s"]) for r in rows]
        self.assertEqual(times, sorted(times))
        self.assertEqual((summary["frames"], summary["width"], summary["height"]), (6, 640, 512))
        self.assertEqual(summary["stopped_by"], "source ended")
        with open(os.path.join(out, "recording.json")) as f:
            self.assertEqual(json.load(f), summary)

    def test_times_are_the_real_capture_times_not_the_nominal_rate(self):
        # The camera says 30 fps but delivers a frame every 0.1 s, then stalls for 2 s.
        ticks = iter([100.0, 100.1, 100.2, 102.2, 102.3, 102.4, 999.0])
        out = os.path.join(self.dir, "uneven")
        summary = record_camera.record(FakeCamera(self.frames), out, codec="ffv1", fps=30.0,
                                       clock=lambda: next(ticks), wall_clock=lambda: 1794787200.0, max_frames=6)
        rows = self.read_times(out)
        np.testing.assert_allclose([float(r["time_s"]) for r in rows], [0.0, 0.1, 0.2, 2.2, 2.3, 2.4], atol=1e-6)
        self.assertEqual(rows[0]["utc_time"], "2026-11-16T00:00:00.000+00:00")
        self.assertEqual(summary["nominal_fps"], 30.0)
        self.assertAlmostEqual(summary["measured_fps"], 5 / 2.4, places=3)
        self.assertEqual(summary["start_utc"], "2026-11-16T00:00:00.000+00:00")
        self.assertEqual(summary["end_utc"], "2026-11-16T00:00:02.400+00:00")

    def test_a_missed_frame_is_skipped_and_the_rows_still_match_the_video(self):
        out = os.path.join(self.dir, "missed")
        summary = record_camera.record(FakeCamera(self.frames, fail_every=3), out, codec="ffv1", max_frames=6)
        self.assertEqual(summary["frames"], 6)
        self.assertEqual(len(self.read_times(out)), len(self.read_video(os.path.join(out, "video.avi"))))

    def test_stops_after_the_duration_or_when_the_camera_dies(self):
        ticks = iter([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
        out = os.path.join(self.dir, "timed")
        summary = record_camera.record(FakeCamera(self.frames), out, codec="ffv1", duration_s=1.0,
                                       clock=lambda: next(ticks))
        self.assertEqual((summary["frames"], summary["stopped_by"]), (3, "duration 1 s reached"))

        out = os.path.join(self.dir, "dead")
        summary = record_camera.record(FakeCamera(self.frames[:2]), out, codec="ffv1")
        self.assertEqual(summary["frames"], 2)
        self.assertIn("stopped delivering", summary["stopped_by"])
        self.assertEqual(len(self.read_video(os.path.join(out, "video.avi"))), 2)     # the file is still complete

    def test_camera_that_cannot_be_opened_or_gives_nothing_is_an_error(self):
        with self.assertRaises(OSError):
            record_camera.open_source(os.path.join(self.dir, "no_such_camera.avi"))
        with self.assertRaisesRegex(OSError, "no frames"):
            record_camera.record(FakeCamera([]), os.path.join(self.dir, "empty"), codec="ffv1")


class FakePicamera2:
    """Stands in for picamera2.Picamera2 on a computer that is not a Raspberry Pi."""
    made = []

    def __init__(self):
        self.calls, self.frames = [], [frame(s) for s in (0, 20, 40)]
        FakePicamera2.made.append(self)

    def create_video_configuration(self, main, controls):
        return {"main": main, "controls": controls}

    def configure(self, config):
        self.calls.append(("configure", config))

    def start(self):
        self.calls.append("start")

    def capture_array(self):
        if not self.frames:
            raise RuntimeError("camera unplugged")
        return self.frames.pop(0)

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


class TestRaspberryPiCamera(RecordTest):
    def setUp(self):
        super().setUp()
        FakePicamera2.made.clear()
        self._saved = sys.modules.get("picamera2")
        sys.modules["picamera2"] = type(sys)("picamera2")
        sys.modules["picamera2"].Picamera2 = FakePicamera2

    def tearDown(self):
        if self._saved is None:
            del sys.modules["picamera2"]
        else:
            sys.modules["picamera2"] = self._saved
        super().tearDown()

    def test_pi_camera_is_configured_for_opencv_and_recorded_like_any_camera(self):
        cap = record_camera.open_source("picamera", 640, 512, 15)
        camera = FakePicamera2.made[0]
        self.assertEqual(camera.calls[0], ("configure", {"main": {"format": "RGB888", "size": (640, 512)},
                                                         "controls": {"FrameRate": 15}}))
        out = os.path.join(self.dir, "pi")
        summary = record_camera.record(cap, out, codec="ffv1", max_frames=3)
        cap.release()
        self.assertEqual((summary["frames"], summary["nominal_fps"]), (3, 15.0))
        recorded = self.read_video(os.path.join(out, "video.avi"))
        np.testing.assert_array_equal(recorded[2], frame(40))
        self.assertEqual(camera.calls[-2:], ["stop", "close"])

    def test_without_the_library_the_error_says_what_to_install(self):
        sys.modules["picamera2"] = None            # makes "import picamera2" fail, as off a Pi
        with self.assertRaisesRegex(OSError, "python3-picamera2"):
            record_camera.open_source("picamera")


class TestCommandLineAndPipeline(RecordTest):
    def run_tool(self, script, *args):
        return subprocess.run([sys.executable, os.path.join(REPO, script), *args], capture_output=True, text=True,
                              encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})

    def test_recording_goes_through_the_pipeline_with_its_real_frame_times(self):
        out = os.path.join(self.dir, "rec")
        proc = self.run_tool("record_camera.py", "--source", self.source, "--output_dir", out, "--codec", "ffv1")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("6 frames", proc.stdout)

        # Real capture times: one frame per second, although the file says 10 frames per second.
        times = os.path.join(self.dir, "frame_times.csv")
        with open(times, "w", newline="") as f:
            f.write("frame_index,time_s,utc_time\n" + "".join(f"{i},{float(i)},\n" for i in range(5)))    # none for frame 5
        camera = os.path.join(self.dir, "camera.json")
        with open(camera, "w") as f:
            json.dump(CAMERA, f)
        telemetry = os.path.join(self.dir, "telemetry.csv")       # the drone flies east at 0.001 deg per second
        with open(telemetry, "w") as f:
            f.write("time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                    "0,34.15,-117.850,100,0,-90,0\n10,34.15,-117.840,100,0,-90,0\n")
        video = os.path.join(out, "video.avi")
        common = ("--video", video, "--timed_telemetry_csv", telemetry, "--camera_config", camera,
                  "--telemetry_max_gap_s", "5")

        with_times = os.path.join(self.dir, "with_times")
        proc = self.run_tool("process_video.py", *common, "--frame_times_csv", times, "--output_dir", with_times)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(with_times, "geolocated_results.json")) as f:
            data = json.load(f)
        self.assertEqual(data["summary"]["frame_times"], "frame_times_csv")
        self.assertEqual([r["video_time_s"] for r in data["results"]], [0.0, 1.0, 2.0, 3.0, 4.0, None])
        self.assertAlmostEqual(data["results"][4]["geolocation"]["drone"]["lon"], -117.846)      # 4 s into the flight
        self.assertEqual(data["results"][5]["geolocation"]["status"], "missing_metadata")
        self.assertIn("frame_times.csv", data["results"][5]["geolocation"]["note"])

        # Without them the video's own clock (0.1 s per frame) puts frame 4 at 0.4 s: the wrong position.
        without = os.path.join(self.dir, "without")
        proc = self.run_tool("process_video.py", *common, "--output_dir", without)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(without, "geolocated_results.json")) as f:
            data = json.load(f)
        self.assertAlmostEqual(data["results"][4]["geolocation"]["drone"]["lon"], -117.8496)

    def test_bad_frame_times_file_is_refused(self):
        bad = os.path.join(self.dir, "bad.csv")
        with open(bad, "w") as f:
            f.write("frame,seconds\n0,0\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            record_camera.load_frame_times(bad)


if __name__ == "__main__":
    unittest.main()
