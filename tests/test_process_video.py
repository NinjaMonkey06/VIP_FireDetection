"""End-to-end tests for process_video.py: a synthetic lossless video with telemetry matched
by frame number and by time."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

import localize

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMERA = {"name": "synthetic 640x512", "calibration_width": 640, "calibration_height": 512,
          "fov_deg": 90.0, "fov_type": "horizontal"}
FPS = 2.0     # frame n is at n / 2 s


def frame(shift):
    """A dark gradient with one bright blob, moved `shift` px to the right."""
    img = np.tile(np.linspace(20, 60, 640, dtype=np.uint8), (512, 1))
    img[220:300, 250 + shift:330 + shift] = 250
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def run(*args):
    return subprocess.run([sys.executable, os.path.join(REPO, "process_video.py"), *args],
                          capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})


class ProcessVideoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        cls.frames = [frame(s) for s in (0, 20, 40, 60)]
        cls.video = os.path.join(d, "clip.avi")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (640, 512))
        if not writer.isOpened():
            raise unittest.SkipTest("FFV1 video writer not available")
        for f in cls.frames:
            writer.write(f)
        writer.release()
        cls.camera = os.path.join(d, "camera.json")
        with open(cls.camera, "w") as f:
            json.dump(CAMERA, f)
        cls.by_frame = os.path.join(d, "by_frame.csv")        # no row for frame 00002
        with open(cls.by_frame, "w") as f:
            f.write("frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                    "00000,34.15,-117.85,100,0,-90,0\n00001,34.15,-117.85,100,0,-90,0\n"
                    "00003,34.15,-117.85,100,0,-90,0\n")
        cls.timed = os.path.join(d, "timed.csv")              # samples at 0, 1.0 and 1.5 s; frame 1 is at 0.5 s
        with open(cls.timed, "w") as f:
            f.write("time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg,timestamp\n"
                    "0.0,34.15,-117.85,100,0,-90,0,2026-11-12T10:00:00\n"
                    "1.0,34.15,-117.84,100,0,-90,0,2026-11-12T10:00:01\n"
                    "1.5,34.15,-117.84,100,0,-90,0,2026-11-12T10:00:01.5\n")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def results(self, out):
        with open(os.path.join(out, "geolocated_results.json")) as f:
            return json.load(f)


class TestByFrameNumber(ProcessVideoTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = os.path.join(cls._tmp.name, "by_frame")
        cls.proc = run("--video", cls.video, "--telemetry_csv", cls.by_frame, "--camera_config", cls.camera,
                       "--output_dir", cls.out, "--annotate", "--tables_dir", os.path.join(cls.out, "tables"))

    def test_runs_every_frame(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        data = self.results(self.out)
        self.assertEqual([r["frame_id"] for r in data["results"]], ["00000", "00001", "00002", "00003"])
        self.assertEqual([r["video_time_s"] for r in data["results"]], [0.0, 0.5, 1.0, 1.5])
        self.assertEqual(data["summary"]["telemetry_matching"], "frame_number")
        for r in data["results"]:      # no detector in front: a frame is fire only if a region was found
            self.assertEqual(r["fire_detected"], bool(r["fire_regions"]))

    def test_video_frame_is_processed_like_the_same_still_image(self):
        path = os.path.join(self._tmp.name, "still.png")
        Image.fromarray(cv2.cvtColor(self.frames[1], cv2.COLOR_BGR2RGB)).save(path)
        gray, size = localize.load_thermal_gray(path)
        mask = localize.otsu_fire_mask(gray, 5, 0.85)[0]
        _, _, regions = localize.regions_from_mask(mask, size, 15, 200, 50000, 0.01, 2.0)
        video_regions = self.results(self.out)["results"][1]["fire_regions"]
        self.assertEqual(len(regions), 1)
        self.assertEqual([g["pixel_256"] for g in video_regions], [g["pixel_256"] for g in regions])
        self.assertEqual([g["fire_pixels_256"] for g in video_regions], [g["fire_pixels_256"] for g in regions])

    def test_frame_without_telemetry_is_reported_and_the_run_continues(self):
        data = self.results(self.out)
        self.assertEqual(data["results"][2]["geolocation"]["status"], "missing_metadata")
        self.assertEqual(data["results"][3]["geolocation"]["status"], "ok")
        self.assertEqual(data["summary"]["geolocation"]["frames_missing_metadata"], 1)

    def test_outputs(self):
        data = self.results(self.out)
        with open(os.path.join(self.out, "frames.jsonl"), encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([l["frame_id"] for l in lines], ["00000", "00001", "00002", "00003"])
        self.assertEqual(lines[1]["fire_regions"], data["results"][1]["fire_regions"])

        with open(os.path.join(self.out, "border_points.csv"), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        expected = sum(len(g["gps_polygon"] or g["normalized"]) for r in data["results"] for g in r["fire_regions"])
        self.assertEqual(len(rows), expected)
        self.assertEqual(rows[0]["frame_id"], "00000")
        self.assertEqual(rows[0]["point_type"], "fire_edge")
        self.assertEqual({r["point_type"] for r in rows if r["frame_id"] == "00002"}, {"missing_metadata"})

        with open(os.path.join(self.out, "fire_borders.geojson")) as f:
            geo = json.load(f)
        self.assertEqual(len(geo["features"]), 3)             # the frame without telemetry has no border
        self.assertEqual(geo["features"][1]["properties"]["video_time_s"], 0.5)

        self.assertEqual(sorted(os.listdir(os.path.join(self.out, "tables"))),
                         ["00000_contour_01.csv", "00001_contour_01.csv", "00002_contour_01.csv", "00003_contour_01.csv"])
        cap = cv2.VideoCapture(os.path.join(self.out, "annotated.mp4"))
        n = 0
        while cap.read()[0]:
            n += 1
        self.assertEqual(n, 4)


class TestByTime(ProcessVideoTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.out = os.path.join(cls._tmp.name, "by_time")
        cls.proc = run("--video", cls.video, "--timed_telemetry_csv", cls.timed, "--camera_config", cls.camera,
                       "--output_dir", cls.out, "--telemetry_max_gap_s", "0.6")
        cls.out_step = os.path.join(cls._tmp.name, "by_time_step")
        cls.proc_step = run("--video", cls.video, "--timed_telemetry_csv", cls.timed, "--camera_config", cls.camera,
                            "--output_dir", cls.out_step, "--frame_step", "2")

    def test_pose_is_interpolated_to_the_frame_time(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        data = self.results(self.out)
        self.assertEqual(data["summary"]["telemetry_matching"], "time")
        drones = [r["geolocation"]["drone"] for r in data["results"]]
        self.assertAlmostEqual(drones[0]["lon"], -117.85)
        self.assertAlmostEqual(drones[1]["lon"], -117.845)          # 0.5 s: halfway between the 0 s and 1 s samples
        self.assertAlmostEqual(drones[2]["lon"], -117.84)
        self.assertEqual(data["results"][1]["geolocation"]["timestamp"], "2026-11-12T10:00:00.500000")
        self.assertEqual(data["summary"]["geolocation"]["frames_missing_metadata"], 0)
        # the blob moves right in the image = east on the ground, on top of the drone's own eastward move
        east = [np.mean([v["lon"] for v in r["fire_regions"][0]["gps_polygon"]]) for r in data["results"]]
        self.assertTrue(all(b > a for a, b in zip(east, east[1:])))

    def test_frame_step_skips_frames_but_keeps_their_times(self):
        self.assertEqual(self.proc_step.returncode, 0, self.proc_step.stderr)
        data = self.results(self.out_step)
        self.assertEqual([r["frame_id"] for r in data["results"]], ["00000", "00002"])
        self.assertEqual([r["video_time_s"] for r in data["results"]], [0.0, 1.0])
        self.assertEqual((data["summary"]["frames_read"], data["summary"]["frames_processed"]), (4, 2))
        self.assertAlmostEqual(data["results"][1]["geolocation"]["drone"]["lon"], -117.84)

    def test_frame_beyond_the_telemetry_gets_no_pose(self):
        with tempfile.TemporaryDirectory() as d:
            short = os.path.join(d, "short.csv")
            with open(short, "w") as f:
                f.write("time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                        "0.0,34.15,-117.85,100,0,-90,0\n0.5,34.15,-117.85,100,0,-90,0\n")
            out = os.path.join(d, "out")
            proc = run("--video", self.video, "--timed_telemetry_csv", short, "--camera_config", self.camera,
                       "--output_dir", out, "--telemetry_max_gap_s", "0.2")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            statuses = [r["geolocation"]["status"] for r in self.results(out)["results"]]
            self.assertEqual(statuses, ["ok", "ok", "missing_metadata", "missing_metadata"])


class TestBrightBackgroundGate(unittest.TestCase):
    """With auto gain a fire scene has a dark background; a scene without fire does not, but Otsu
    still splits it into a brighter and a darker part."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        warm = np.tile(np.linspace(110, 130, 640, dtype=np.uint8), (512, 1))    # no fire: mid-grey ground
        warm[220:300, 250:330] = 220                                              # and a sun-warmed patch
        cls.video = os.path.join(d, "clip.avi")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (640, 512))
        if not writer.isOpened():
            raise unittest.SkipTest("FFV1 video writer not available")
        writer.write(frame(0))
        writer.write(cv2.cvtColor(warm, cv2.COLOR_GRAY2BGR))
        writer.release()
        cls.camera = os.path.join(d, "camera.json")
        with open(cls.camera, "w") as f:
            json.dump(CAMERA, f)
        cls.telemetry = os.path.join(d, "telemetry.csv")
        with open(cls.telemetry, "w") as f:
            f.write("frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                    "00000,34.15,-117.85,100,0,-90,0\n00001,34.15,-117.85,100,0,-90,0\n")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def process(self, name, *extra):
        out = os.path.join(self._tmp.name, name)
        proc = run("--video", self.video, "--telemetry_csv", self.telemetry, "--camera_config", self.camera,
                   "--output_dir", out, *extra)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(os.path.join(out, "geolocated_results.json")) as f:
            return json.load(f)

    def test_off_by_default_and_the_level_is_always_reported(self):
        data = self.process("plain")
        fire, warm = data["results"]
        self.assertLess(fire["background_gray"], 60)
        self.assertGreater(warm["background_gray"], 100)
        self.assertTrue(warm["fire_detected"])                 # the false alarm the gate is for
        self.assertEqual(data["summary"]["frames_skipped_bright_background"], 0)

    def test_bright_frame_is_not_localized_and_the_fire_frame_is_untouched(self):
        plain, gated = self.process("plain2"), self.process("gated", "--max_background_gray", "60")
        fire, warm = gated["results"]
        self.assertEqual(fire["fire_regions"], plain["results"][0]["fire_regions"])
        self.assertFalse(warm["localization_run"])
        self.assertFalse(warm["fire_detected"])
        self.assertEqual(warm["fire_regions"], [])
        self.assertIn("background", warm["localization_note"])
        self.assertNotIn("geolocation", warm)
        self.assertEqual(gated["summary"]["frames_skipped_bright_background"], 1)
        self.assertEqual(gated["summary"]["frames_with_regions"], 1)


class TestHotspotMode(unittest.TestCase):
    def test_small_warm_target_needs_the_hotspot_mode(self):
        with tempfile.TemporaryDirectory() as d:
            noise = cv2.GaussianBlur(np.random.default_rng(0).normal(0, 1, (512, 640)).astype(np.float32), (0, 0), 4)
            ground = np.clip(60 + 6 * noise / noise.std(), 0, 255).astype(np.uint8)
            ground[290:305, 390:410] = 180                              # a 20 x 15 px target
            video = os.path.join(d, "clip.avi")
            writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"FFV1"), FPS, (640, 512))
            if not writer.isOpened():
                raise unittest.SkipTest("FFV1 video writer not available")
            writer.write(cv2.cvtColor(ground, cv2.COLOR_GRAY2BGR))
            writer.release()
            camera = os.path.join(d, "camera.json")
            with open(camera, "w") as f:
                json.dump(CAMERA, f)
            telemetry = os.path.join(d, "telemetry.csv")
            with open(telemetry, "w") as f:
                f.write("frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                        "00000,34.15,-117.85,100,0,-90,0\n")
            found = {}
            for mode in ("otsu", "hotspot"):
                out = os.path.join(d, mode)
                proc = run("--video", video, "--telemetry_csv", telemetry, "--camera_config", camera,
                           "--output_dir", out, "--threshold_mode", mode)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(os.path.join(out, "geolocated_results.json")) as f:
                    found[mode] = json.load(f)["results"][0]
            self.assertEqual(found["otsu"]["fire_regions"], [])
            self.assertFalse(found["otsu"]["fire_detected"])
            hot = found["hotspot"]
            self.assertEqual((hot["threshold_mode"], len(hot["fire_regions"]), hot["fire_detected"]), ("hotspot", 1, True))
            self.assertEqual(hot["fire_regions"][0]["geolocation_status"], "ok")
            # 80 px right and 41 px below the centre at 0.3125 m per pixel: 25 m east, 13 m south of the drone
            c, drone = hot["fire_regions"][0]["gps_centroid"], hot["geolocation"]["drone"]
            from geoloc.projection import latlon_to_offset
            north, east = latlon_to_offset(drone["lat"], drone["lon"], c["lat"], c["lon"])
            self.assertAlmostEqual(float(east), 25.0, delta=1.0)
            self.assertAlmostEqual(float(north), -13.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
