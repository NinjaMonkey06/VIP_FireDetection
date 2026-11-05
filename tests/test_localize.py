"""Tests for localize.py's fire-pixel output (the hot pixels each merged region covers)."""

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


def blobs_mask():
    """256x256 mask: a cluster of three spots close together, and one spot far away."""
    m = np.zeros((256, 256), np.uint8)
    for x, y in ((40, 40), (52, 44), (46, 54)):
        cv2.circle(m, (x, y), 3, 255, -1)
    cv2.rectangle(m, (180, 180), (195, 190), 255, -1)
    return m


class TestFirePixels(unittest.TestCase):
    def test_runs_reconstruct_the_mask(self):
        m = blobs_mask()
        rebuilt = np.zeros_like(m)
        for row, first, last in localize.mask_runs(m):
            rebuilt[row, first:last + 1] = 255
        np.testing.assert_array_equal(rebuilt, m)

    def test_each_region_gets_only_its_own_hot_pixels(self):
        m = blobs_mask()
        dilated = localize.dilate_mask_for_merging(m, 15)
        contours = localize.extract_contours(dilated, 200, 50000, 0.01)
        self.assertEqual(len(contours), 2)          # the three close spots merged into one region
        runs = localize.fire_pixel_runs(m, dilated, contours)
        counts = sorted(sum(r[2] - r[1] + 1 for r in rr) for rr in runs)
        cluster = cv2.countNonZero(m[:100, :100])
        rect = cv2.countNonZero(m[150:, 150:])
        self.assertEqual(counts, sorted([cluster, rect]))
        # the outline (dilated) is much larger than the hot pixels it covers
        self.assertLess(cluster, 0.5 * max(cv2.contourArea(c) for c in contours))

    def test_encoded_region_carries_fire_pixels(self):
        m = blobs_mask()
        dilated = localize.dilate_mask_for_merging(m, 15)
        contours = localize.extract_contours(dilated, 200, 50000, 0.01)
        regions = localize.encode_contours(contours, (640, 512), localize.fire_pixel_runs(m, dilated, contours))
        self.assertEqual(sum(r["fire_area_px"] for r in regions), cv2.countNonZero(m))
        self.assertIn("fire_pixels_256", regions[0])


def background(hottest_c=None):
    """640x512 temperatures: a 10-30 °C gradient, optionally with one warm patch."""
    t = np.tile(np.linspace(10.0, 30.0, 640, dtype=np.float32), (512, 1))
    if hottest_c is not None:
        t[200:240, 300:340] = hottest_c
    return t


def regions_of(mask):
    return localize.extract_contours(localize.dilate_mask_for_merging(mask, 15), 200, 50000, 0.01)


class TestAbsoluteThreshold(unittest.TestCase):
    def test_nothing_hot_enough_gives_no_regions_where_otsu_still_finds_some(self):
        t = background(hottest_c=79.0)
        mask, seeds = localize.absolute_fire_mask(t, 80.0, 60.0, 10)
        self.assertEqual((cv2.countNonZero(mask), seeds, len(regions_of(mask))), (0, 0, 0))

        gray = cv2.resize((t * 255.0 / t.max()).astype(np.uint8), (256, 256))
        self.assertGreaterEqual(len(regions_of(localize.otsu_fire_mask(gray, 5, 0.85)[0])), 1)

    def test_hot_blob_is_one_region_of_the_right_size(self):
        t = background()
        t[200:251, 300:364] = 300.0                       # 64 x 51 px: 10% x ~10% of the frame
        mask, seeds = localize.absolute_fire_mask(t, 80.0, 60.0, 10)
        self.assertEqual(seeds, 64 * 51)
        self.assertEqual(len(regions_of(mask)), 1)
        self.assertAlmostEqual(cv2.countNonZero(mask), 64 * 51 * (256 / 640) * (256 / 512), delta=60)

    def test_lower_temperature_grows_the_region_only_if_connected(self):
        t = background()
        t[200:240, 300:340] = 300.0                       # flame core
        t[190:250, 290:350][t[190:250, 290:350] < 55.0] = 55.0   # warm ring around it
        t[400:440, 100:140] = 55.0                        # just as warm, but not connected to a seed
        at_60 = cv2.countNonZero(localize.absolute_fire_mask(t, 80.0, 60.0, 10)[0])
        at_50 = localize.absolute_fire_mask(t, 80.0, 50.0, 10)[0]
        self.assertGreater(cv2.countNonZero(at_50), at_60)
        self.assertEqual(cv2.countNonZero(at_50[195:225, 35:60]), 0)     # the unconnected patch stays out

    def test_region_needs_enough_seed_pixels(self):
        t = background()
        t[100, 100:109] = 300.0                           # 9 seeds
        self.assertEqual(localize.absolute_fire_mask(t, 80.0, 60.0, 10)[1], 0)
        t[100, 109] = 300.0                               # 10 seeds
        self.assertEqual(localize.absolute_fire_mask(t, 80.0, 60.0, 10)[1], 10)


class TestAbsoluteModeCommandLine(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.thermal = os.path.join(self._tmp.name, "thermal")
        self.temps = os.path.join(self._tmp.name, "temps")
        self.out = os.path.join(self._tmp.name, "out.json")
        os.makedirs(self.thermal)
        os.makedirs(self.temps)
        hot = background()
        hot[200:260, 300:380] = 300.0
        for name, t in (("hot", hot), ("cool", background())):
            Image.fromarray(np.uint8(np.clip(t, 0, 255))).save(os.path.join(self.thermal, name + ".JPG"))
            Image.fromarray(t, mode="F").save(os.path.join(self.temps, name + ".TIFF"))

    def tearDown(self):
        self._tmp.cleanup()

    def run_localize(self, *extra):
        return subprocess.run([sys.executable, os.path.join(REPO, "localize.py"), "--thermal_dir", self.thermal,
                               "--output_json", self.out, *extra], capture_output=True, text=True, encoding="utf-8",
                              env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})

    def test_runs_without_a_detector_and_localization_decides(self):
        proc = self.run_localize("--threshold_mode", "absolute", "--temperature_dir", self.temps)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        with open(self.out) as f:
            data = json.load(f)
        by_id = {r["frame_id"]: r for r in data["results"]}
        self.assertTrue(by_id["hot"]["fire_detected"])
        self.assertEqual(len(by_id["hot"]["fire_regions"]), 1)
        self.assertFalse(by_id["cool"]["fire_detected"])
        self.assertEqual(by_id["cool"]["fire_regions"], [])
        self.assertEqual((data["summary"]["fire_detected"], data["summary"]["clear_frames"]), (1, 1))
        self.assertEqual(data["summary"]["parameters"]["threshold_mode"], "absolute")

    def test_otsu_mode_still_needs_a_detector(self):
        proc = self.run_localize()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("--inference_json", proc.stderr)

    def test_missing_temperature_image_is_an_error_not_no_fire(self):
        os.remove(os.path.join(self.temps, "cool.TIFF"))
        proc = self.run_localize("--threshold_mode", "absolute", "--temperature_dir", self.temps)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("no temperature image", proc.stderr)

    def test_low_temperature_cannot_exceed_seed(self):
        proc = self.run_localize("--threshold_mode", "absolute", "--temperature_dir", self.temps,
                                 "--seed_temp_c", "50")
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
