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


def outline_error_px(contour, mask):
    """Largest distance from a pixel on the mask's true border to the simplified outline."""
    raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(abs(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
               for c in raw for x, y in c.reshape(-1, 2))


class TestOutline(unittest.TestCase):
    def large_and_small(self):
        m = np.zeros((256, 256), np.uint8)
        cv2.ellipse(m, (128, 110), (100, 60), 20, 0, 360, 255, -1)       # perimeter ~520 px: 1% is ~5 px
        small = np.zeros((256, 256), np.uint8)
        cv2.circle(small, (60, 60), 12, 255, -1)                          # perimeter ~80 px: 1% is under 1 px
        return m, small

    def test_cap_keeps_a_large_outline_close_to_its_border(self):
        large, _ = self.large_and_small()
        plain = localize.extract_contours(large, 200, 50000, 0.01)[0]
        capped = localize.extract_contours(large, 200, 50000, 0.01, max_epsilon_px=2.0)[0]
        self.assertGreater(outline_error_px(plain, large), 3.0)
        self.assertLessEqual(outline_error_px(capped, large), 2.0 + 1e-6)
        self.assertGreater(len(capped), len(plain))

    def test_cap_leaves_small_regions_unchanged_and_zero_means_no_cap(self):
        large, small = self.large_and_small()
        np.testing.assert_array_equal(localize.extract_contours(small, 200, 50000, 0.01)[0],
                                      localize.extract_contours(small, 200, 50000, 0.01, max_epsilon_px=2.0)[0])
        np.testing.assert_array_equal(localize.extract_contours(large, 200, 50000, 0.01)[0],
                                      localize.extract_contours(large, 200, 50000, 0.01, max_epsilon_px=0)[0])

    def test_normalized_points_are_pixel_centres_on_the_fire_pixel_grid(self):
        # A rectangle of pixels, not dilated: its outline runs through the centres of its border
        # pixels, so outline and fire-pixel cells must share one centre.
        m = np.zeros((256, 256), np.uint8)
        m[100:140, 60:120] = 255                       # columns 60..119, rows 100..139
        _, contours, regions = localize.regions_from_mask(m, (640, 512), 0, 200, 50000, 0.01)
        reg = regions[0]
        xs = [p["x"] * 256 for p in reg["normalized"]]
        ys = [p["y"] * 256 for p in reg["normalized"]]
        self.assertAlmostEqual(min(xs), 60.5, places=3)
        self.assertAlmostEqual(max(xs), 119.5, places=3)
        self.assertAlmostEqual((min(xs) + max(xs)) / 2, (60 + 120) / 2, places=3)      # cells cover [60, 120)
        self.assertAlmostEqual((min(ys) + max(ys)) / 2, (100 + 140) / 2, places=3)
        self.assertEqual(sorted({p["x"] for p in reg["pixel_256"]}), [60, 119])         # indices stay as they were


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


def warm_scene_with_fire():
    """256x256: 85% warm ground (120), a dark strip (20), one small fire (250). Plain
    Otsu splits dark vs everything else, so merging yields one region too big to keep."""
    g = np.full((256, 256), 120, np.uint8)
    g[:, :38] = 20
    g[100:116, 150:166] = 250
    return g


class TestSecondLevelOtsu(unittest.TestCase):
    def test_oversized_region_is_rethresholded_to_the_fire(self):
        g = warm_scene_with_fire()
        plain = localize.otsu_fire_mask(g, 5, 0.85)[0]
        self.assertEqual(len(regions_of(plain)), 0)                  # dropped by the size filter
        mask, _, applied, rounds = localize.second_level_otsu_mask(g, 5, 0.85, 15, 50000)
        self.assertGreaterEqual(rounds, 1)
        regions = regions_of(mask)
        self.assertEqual(len(regions), 1)
        x, y, w, h = cv2.boundingRect(regions[0])
        self.assertTrue(x < 158 < x + w and y < 108 < y + h)         # around the fire
        self.assertEqual(cv2.countNonZero(mask[:, :38]), 0)

    def test_frames_without_the_problem_are_unchanged(self):
        g = np.tile(np.linspace(10, 60, 256).astype(np.uint8), (256, 1))
        g[100:120, 100:140] = 250
        plain, otsu, applied = localize.otsu_fire_mask(g, 5, 0.85)
        mask, otsu2, applied2, rounds = localize.second_level_otsu_mask(g, 5, 0.85, 15, 50000)
        self.assertEqual(rounds, 0)
        self.assertTrue(np.array_equal(mask, plain))
        self.assertEqual((otsu, applied), (otsu2, applied2))


class TestSecondLevelCommandLine(unittest.TestCase):
    def test_on_by_default_and_can_be_switched_off(self):
        with tempfile.TemporaryDirectory() as d:
            thermal = os.path.join(d, "thermal")
            os.makedirs(thermal)
            Image.fromarray(cv2.resize(warm_scene_with_fire(), (640, 512), interpolation=cv2.INTER_NEAREST)).save(
                os.path.join(thermal, "f.png"))
            with open(os.path.join(d, "inf.json"), "w") as f:
                json.dump({"summary": {}, "results": [{"frame_id": "f", "filename": "f.png", "fire_detected": True,
                                                        "classifier_prediction": "YN"}]}, f)
            found = []
            for extra in ([], ["--no_second_level_otsu"]):
                out = os.path.join(d, "out.json")
                proc = subprocess.run([sys.executable, os.path.join(REPO, "localize.py"), "--inference_json",
                                       os.path.join(d, "inf.json"), "--thermal_dir", thermal, "--output_json", out,
                                       *extra], capture_output=True, text=True, encoding="utf-8",
                                      env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(out) as f:
                    data = json.load(f)
                found.append(len(data["results"][0]["fire_regions"]))
                self.assertEqual(data["summary"]["parameters"]["second_level_otsu"], not extra)
            self.assertEqual(found, [1, 0])


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


def textured_ground(seed=0, level=60, spread=6):
    """256x256: ground with gentle texture and nothing hot."""
    noise = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, (256, 256)).astype(np.float32), (0, 0), 2)
    return np.clip(level + spread * noise / noise.std(), 0, 255).astype(np.uint8)


class TestHotspotThreshold(unittest.TestCase):
    """A small hot area in a large scene: too few pixels for Otsu to treat as a group."""

    def scene_with_small_target(self):
        g = textured_ground()
        g[120:126, 150:158] = 180          # 8 x 6 px: 0.07% of the picture
        return g

    def test_small_target_is_outlined_where_otsu_gives_nothing(self):
        g = self.scene_with_small_target()
        self.assertEqual(len(regions_of(localize.otsu_fire_mask(g, 5, 0.85)[0])), 0)
        self.assertEqual(len(regions_of(localize.second_level_otsu_mask(g, 5, 0.85, 15, 50000)[0])), 0)
        mask, applied = localize.hotspot_fire_mask(g, 5)
        regions = regions_of(mask)
        self.assertEqual(len(regions), 1)
        x, y, w, h = cv2.boundingRect(regions[0])
        self.assertTrue(x < 154 < x + w and y < 123 < y + h)
        self.assertGreaterEqual(applied, 60 + 25)

    def test_ground_alone_gives_no_region(self):
        for seed in range(5):
            mask, _ = localize.hotspot_fire_mask(textured_ground(seed, spread=15), 5)
            self.assertEqual(len(regions_of(mask)), 0)

    def test_threshold_follows_the_spread_of_the_picture(self):
        quiet = localize.hotspot_fire_mask(textured_ground(spread=2), 5)[1]
        busy = localize.hotspot_fire_mask(textured_ground(spread=15), 5)[1]
        self.assertAlmostEqual(quiet, 60 + 25, delta=2)            # the minimum contrast decides
        self.assertGreater(busy, quiet + 20)                       # six times a wider spread decides
        self.assertLessEqual(localize.hotspot_fire_mask(np.full((256, 256), 250, np.uint8), 5)[1], 254)


if __name__ == "__main__":
    unittest.main()
