"""Tests for detect_by_temperature.py using synthetic float temperature TIFFs."""

import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from detect_by_temperature import build_results, index_temperature_files, temperature_decision


def write_temperatures(path, hot_pixels=0, hot_c=300.0, background_c=15.0):
    t = np.full((64, 80), background_c, np.float32)
    t.flat[:hot_pixels] = hot_c
    Image.fromarray(t, mode="F").save(path)


class TestTemperatureDecision(unittest.TestCase):
    def test_needs_enough_pixels_above_the_threshold(self):
        t = np.full((10, 10), 20.0, np.float32)
        t.flat[:9] = 81.0
        self.assertEqual(temperature_decision(t, 80.0, 10), (False, 9, 81.0))
        t.flat[9] = 81.0
        self.assertTrue(temperature_decision(t, 80.0, 10)[0])

    def test_threshold_is_strict(self):
        t = np.full((10, 10), 80.0, np.float32)
        self.assertEqual(temperature_decision(t, 80.0, 1)[:2], (False, 0))


class TestBuildResults(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = self._tmp.name
        write_temperatures(os.path.join(d, "hot.TIFF"), hot_pixels=50)
        write_temperatures(os.path.join(d, "cool.TIFF"))
        self.files = index_temperature_files(d)

    def tearDown(self):
        self._tmp.cleanup()

    def test_temperature_only(self):
        r = {x["filename"]: x for x in build_results(["hot.JPG", "cool.JPG"], self.files, 80.0, 10)}
        self.assertTrue(r["hot.JPG"]["fire_detected"])
        self.assertEqual(r["hot.JPG"]["temperature"], {"fire": True, "hot_pixels": 50, "max_temp_c": 300.0})
        self.assertFalse(r["cool.JPG"]["fire_detected"])
        for x in r.values():     # fields localize.py reads
            for key in ("frame_id", "filename", "fire_detected", "classifier_prediction"):
                self.assertIn(key, x)

    def test_either_classifier_or_temperature_means_fire(self):
        clf = {"hot.JPG":  {"frame_id": "hot", "filename": "hot.JPG", "classifier_prediction": "NN",
                            "fire_detected": False, "probabilities": {"NN": 0.98, "YN": 0.01, "YY": 0.01}},
               "cool.JPG": {"frame_id": "cool", "filename": "cool.JPG", "classifier_prediction": "YY",
                            "fire_detected": True, "probabilities": {"NN": 0.0, "YN": 0.0, "YY": 1.0}}}
        r = {x["filename"]: x for x in build_results(["hot.JPG", "cool.JPG"], self.files, 80.0, 10, clf)}
        self.assertTrue(r["hot.JPG"]["fire_detected"])
        self.assertEqual(r["hot.JPG"]["detected_by"], ["temperature"])
        self.assertEqual(r["hot.JPG"]["classifier_prediction"], "NN")          # classifier output kept
        self.assertTrue(r["cool.JPG"]["fire_detected"])
        self.assertEqual(r["cool.JPG"]["detected_by"], ["classifier"])
        self.assertFalse(clf["hot.JPG"]["fire_detected"])                     # input not modified

    def test_missing_temperature_image_is_an_error_not_no_fire(self):
        with self.assertRaisesRegex(ValueError, "no temperature image"):
            build_results(["hot.JPG", "other.JPG"], self.files, 80.0, 10)

    def test_8bit_image_is_rejected(self):
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(os.path.join(self._tmp.name, "palette.TIFF"))
        with self.assertRaisesRegex(ValueError, "not temperatures"):
            build_results(["palette.JPG"], index_temperature_files(self._tmp.name), 80.0, 10)


if __name__ == "__main__":
    unittest.main()
