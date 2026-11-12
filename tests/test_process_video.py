"""End-to-end test for process_video.py: a synthetic lossless video + telemetry CSV."""

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


def frame(shift):
    """A dark gradient with one bright blob, moved `shift` px to the right."""
    img = np.tile(np.linspace(20, 60, 640, dtype=np.uint8), (512, 1))
    img[220:300, 250 + shift:330 + shift] = 250
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class TestProcessVideo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        d = cls._tmp.name
        cls.frames = [frame(s) for s in (0, 20, 40)]
        cls.video = os.path.join(d, "clip.avi")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"FFV1"), 2.0, (640, 512))
        if not writer.isOpened():
            raise unittest.SkipTest("FFV1 video writer not available")
        for f in cls.frames:
            writer.write(f)
        writer.release()
        with open(os.path.join(d, "camera.json"), "w") as f:
            json.dump(CAMERA, f)
        with open(os.path.join(d, "telemetry.csv"), "w") as f:     # no row for frame 00002
            f.write("frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg\n"
                    "00000,34.15,-117.85,100,0,-90,0\n00001,34.15,-117.85,100,0,-90,0\n")
        cls.out = os.path.join(d, "out")
        cls.proc = subprocess.run(
            [sys.executable, os.path.join(REPO, "process_video.py"), "--video", cls.video,
             "--telemetry_csv", os.path.join(d, "telemetry.csv"), "--camera_config", os.path.join(d, "camera.json"),
             "--output_dir", cls.out, "--annotate"],
            capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        with open(os.path.join(cls.out, "geolocated_results.json")) as f:
            cls.data = json.load(f)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_runs_every_frame(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr)
        self.assertEqual([r["frame_id"] for r in self.data["results"]], ["00000", "00001", "00002"])
        self.assertEqual(self.data["summary"]["frames"], 3)

    def test_video_frame_is_processed_like_the_same_still_image(self):
        path = os.path.join(self._tmp.name, "still.png")
        Image.fromarray(cv2.cvtColor(self.frames[1], cv2.COLOR_BGR2RGB)).save(path)
        gray, size = localize.load_thermal_gray(path)
        mask = localize.otsu_fire_mask(gray, 5, 0.85)[0]
        _, _, regions = localize.regions_from_mask(mask, size, 15, 200, 50000, 0.01)
        video_regions = self.data["results"][1]["fire_regions"]
        self.assertEqual(len(regions), 1)
        self.assertEqual([g["pixel_256"] for g in video_regions], [g["pixel_256"] for g in regions])
        self.assertEqual([g["fire_pixels_256"] for g in video_regions], [g["fire_pixels_256"] for g in regions])

    def test_geolocated_with_its_own_telemetry_row(self):
        r = self.data["results"][1]
        self.assertEqual(r["geolocation"]["status"], "ok")
        self.assertEqual(r["fire_regions"][0]["geolocation_status"], "ok")
        # the blob moved right in the image; at nadir with yaw 0 that is east on the ground
        east = [np.mean([v["lon"] for v in res["fire_regions"][0]["gps_polygon"]]) for res in self.data["results"][:2]]
        self.assertGreater(east[1], east[0])

    def test_frame_without_telemetry_is_reported_and_the_run_continues(self):
        r = self.data["results"][2]
        self.assertEqual(r["geolocation"]["status"], "missing_metadata")
        self.assertEqual(self.data["summary"]["geolocation"]["frames_missing_metadata"], 1)

    def test_tables_and_annotated_video(self):
        tables = sorted(os.listdir(os.path.join(self.out, "contour_tables")))
        self.assertEqual(tables, ["00000_contour_01.csv", "00001_contour_01.csv", "00002_contour_01.csv"])
        cap = cv2.VideoCapture(os.path.join(self.out, "annotated.mp4"))
        n = 0
        while cap.read()[0]:
            n += 1
        self.assertEqual(n, 3)


if __name__ == "__main__":
    unittest.main()
