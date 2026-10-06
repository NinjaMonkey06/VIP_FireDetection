"""End-to-end tests for geolocate.py: synthetic localization JSON + sidecar CSV."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import geolocate
from geoloc.camera import camera_config_from_dict
from geoloc.metadata import SidecarCsvLoader
from geoloc.projection import latlon_to_offset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_JSON = os.path.join(REPO, "Sample Localization Output", "localization_results.json")
LAT0, LON0 = 34.15, -117.85

# 640x512 thermal camera with 90° horizontal FOV → fx = fy = 320, 100 m AGL → 0.3125 m/px
CAMERA_DICT = {"name": "synthetic 640x512", "calibration_width": 640, "calibration_height": 512,
               "fov_deg": 90.0, "fov_type": "horizontal"}
CAMERA = camera_config_from_dict(CAMERA_DICT)
CSV_HEADER = "frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg,timestamp\n"


def region(norm_pts):
    return {
        "pixel_256":  [{"x": int(round(x * 256)), "y": int(round(y * 256))} for x, y in norm_pts],
        "normalized": [{"x": x, "y": y} for x, y in norm_pts],
        "area_px": 0.0, "n_points": len(norm_pts),
        "original_wh": {"width": 640, "height": 512},
        "gps_polygon": None, "fire_front_arc": None,
    }


SQUARE = [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)]          # centered
EDGE = [(0.0, 0.3), (0.2, 0.3), (0.2, 0.5), (0.0, 0.5)]            # touches left edge
TOP = [(0.4, 0.0), (0.6, 0.0), (0.6, 0.3), (0.4, 0.3)]             # top rows


def frame(fid, regions, run=True, pred="YN"):
    return {"frame_id": fid, "filename": fid + ".jpg", "classifier_prediction": pred,
            "fire_detected": run, "localization_run": run,
            "fire_regions": [region(r) for r in regions]}


def localization_data():
    return {"summary": {"parameters": {"merge_kernel_size": 15}},
            "results": [frame("nadir", [SQUARE, EDGE]),
                        frame("yawed", [SQUARE]),
                        frame("no_meta", [SQUARE]),
                        frame("oblique", [TOP, SQUARE]),
                        {"frame_id": "clear", "filename": "clear.jpg", "fire_detected": False,
                         "localization_run": False}]}


CSV_ROWS = (f"nadir,{LAT0},{LON0},100,0,-90,0,2026-10-06T12:00:00\n"
            f"yawed,{LAT0},{LON0},100,90,-90,0,\n"
            f"oblique,{LAT0},{LON0},100,0,-10,0,\n")


class GeolocateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self._tmp.name, "meta.csv")
        with open(self.csv, "w", encoding="utf-8") as f:
            f.write(CSV_HEADER + CSV_ROWS)
        self.data = localization_data()
        self.counts = geolocate.geolocate_results(self.data["results"], CAMERA,
                                                  SidecarCsvLoader(self.csv), merge_kernel_size=15)
        self.by_id = {r["frame_id"]: r for r in self.data["results"]}

    def tearDown(self):
        self._tmp.cleanup()

    def offsets(self, reg):
        n, e = latlon_to_offset(LAT0, LON0, [p["lat"] for p in reg["gps_polygon"]],
                                [p["lon"] for p in reg["gps_polygon"]])
        return np.column_stack((n, e))


class TestGeolocateResults(GeolocateTest):
    def test_counts(self):
        self.assertEqual(self.counts, {"frames_geolocated": 3, "frames_missing_metadata": 1,
                                       "regions_geolocated": 4, "regions_crossing_horizon": 1})

    def test_centered_square_uses_original_resolution(self):
        # normalized 0.4..0.6 of 640x512 → ±64 px x ±51.2 px → ±20 m E x ±16 m N
        reg = self.by_id["nadir"]["fire_regions"][0]
        self.assertEqual(reg["geolocation_status"], "ok")
        self.assertAlmostEqual(reg["ground_area_m2"], 40.0 * 32.0, places=1)
        self.assertAlmostEqual(reg["gps_centroid"]["lat"], LAT0, places=8)
        self.assertAlmostEqual(reg["gps_centroid"]["lon"], LON0, places=8)
        np.testing.assert_allclose(self.offsets(reg),
                                   [[16, -20], [16, 20], [-16, 20], [-16, -20]], atol=0.01)
        self.assertFalse(reg["touches_frame_edge"])
        self.assertIsNone(reg["fire_front_arc"])

    def test_pixel_256_would_give_a_different_answer(self):
        # Treating pixel_256 as original pixels: a 51x51 px square about (128, 128) → wrong place and size.
        reg = self.by_id["nadir"]["fire_regions"][0]
        wrong = copy.deepcopy(reg)
        wrong["normalized"] = [{"x": p["x"] / 640, "y": p["y"] / 512} for p in reg["pixel_256"]]
        geolocate.geolocate_region(wrong, CAMERA, SidecarCsvLoader(self.csv).load("nadir"))
        self.assertGreater(abs(wrong["ground_area_m2"] - reg["ground_area_m2"]), 500)
        self.assertNotAlmostEqual(wrong["gps_centroid"]["lat"], reg["gps_centroid"]["lat"], places=5)

    def test_yaw_rotates_polygon(self):
        # Yaw 90: image up = east, image right = south.
        np.testing.assert_allclose(self.offsets(self.by_id["yawed"]["fire_regions"][0]),
                                   [[20, 16], [-20, 16], [-20, -16], [20, -16]], atol=0.01)

    def test_edge_flag(self):
        self.assertTrue(self.by_id["nadir"]["fire_regions"][1]["touches_frame_edge"])

    def test_missing_metadata_leaves_polygon_null(self):
        r = self.by_id["no_meta"]
        self.assertEqual(r["geolocation"]["status"], "missing_metadata")
        self.assertEqual(r["fire_regions"][0]["geolocation_status"], "missing_metadata")
        self.assertIsNone(r["fire_regions"][0]["gps_polygon"])

    def test_oblique_frame(self):
        r = self.by_id["oblique"]
        top, center = r["fire_regions"]
        self.assertEqual(top["geolocation_status"], "crosses_horizon")
        self.assertIsNone(top["gps_polygon"])
        self.assertEqual(center["geolocation_status"], "ok")
        # center pixel at pitch -10° lands 100 / tan(10°) = 567 m north
        n, e = latlon_to_offset(LAT0, LON0, center["gps_centroid"]["lat"], center["gps_centroid"]["lon"])
        self.assertGreater(float(n), 400)
        self.assertTrue(any("oblique" in w for w in r["geolocation"]["warnings"]))

    def test_frame_level_fields(self):
        g = self.by_id["nadir"]["geolocation"]
        self.assertAlmostEqual(g["gsd_center_m"], 0.3125, places=4)
        self.assertAlmostEqual(g["dilation_buffer_m"], 7 * 2.5 * 0.3125, places=2)   # 5.47 m
        self.assertEqual(g["timestamp"], "2026-10-06T12:00:00")
        self.assertEqual(g["warnings"], [])

    def test_clear_frame_untouched(self):
        self.assertNotIn("geolocation", self.by_id["clear"])

    def test_takeoff_altitude_warning(self):
        data = localization_data()
        geolocate.geolocate_results(data["results"], CAMERA, SidecarCsvLoader(self.csv, "takeoff"))
        self.assertTrue(any("takeoff" in w for w in data["results"][0]["geolocation"]["warnings"]))


class TestGeoJson(GeolocateTest):
    def test_lon_lat_order_and_closed_rings(self):
        gj = geolocate.to_geojson(self.data["results"])
        self.assertEqual(len(gj["features"]), 4)
        for feat in gj["features"]:
            ring = feat["geometry"]["coordinates"][0]
            self.assertEqual(ring[0], ring[-1])
            lon, lat = ring[0]
            self.assertLess(abs(lon - LON0), 0.1)
            self.assertLess(abs(lat - LAT0), 0.1)


class TestSampleOutputSchema(unittest.TestCase):
    """Runs on the repo's real localize.py output with synthetic nadir metadata.
    The sample frames are 958x760 (not FLAME-2's native 640x512), so the camera
    here is a synthetic one calibrated at 958x760 with fx = 479 (90° HFOV)."""

    CAMERA_958 = camera_config_from_dict({"name": "synthetic 958x760", "calibration_width": 958,
                                          "calibration_height": 760, "fx": 479.0, "fy": 479.0})

    def test_640x512_config_rejects_sample_frames(self):
        with open(SAMPLE_JSON) as f:
            data = json.load(f)
        reg = next(g for r in data["results"] for g in r.get("fire_regions") or [])
        with self.assertRaises(ValueError):
            CAMERA.intrinsics_for(reg["original_wh"]["width"], reg["original_wh"]["height"])

    def test_every_sample_region_geolocates(self):
        with open(SAMPLE_JSON) as f:
            data = json.load(f)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "meta.csv")
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(CSV_HEADER + "".join(f"{r['frame_id']},{LAT0},{LON0},120,30,-90,0,\n"
                                             for r in data["results"]))
            counts = geolocate.geolocate_results(data["results"], self.CAMERA_958, SidecarCsvLoader(csv_path),
                                                  merge_kernel_size=15)
        n_regions = sum(len(r.get("fire_regions") or []) for r in data["results"] if r.get("localization_run"))
        self.assertGreater(n_regions, 0)
        self.assertEqual(counts["regions_geolocated"], n_regions)
        # Nadir footprint at 120 m is 240 m x 190.4 m, so every vertex lies within its half-diagonal.
        for r in data["results"]:
            for reg in r.get("fire_regions") or []:
                n, e = latlon_to_offset(LAT0, LON0, [p["lat"] for p in reg["gps_polygon"]],
                                        [p["lon"] for p in reg["gps_polygon"]])
                self.assertLessEqual(np.hypot(n, e).max(), np.hypot(120, 95.2) + 1e-6)


class TestCli(unittest.TestCase):
    def test_cli_json_and_geojson(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {name: os.path.join(tmp, name) for name in ("loc.json", "cam.json", "meta.csv", "out.json")}
            with open(paths["loc.json"], "w") as f:
                json.dump(localization_data(), f)
            with open(paths["cam.json"], "w") as f:
                json.dump(CAMERA_DICT, f)
            with open(paths["meta.csv"], "w", encoding="utf-8") as f:
                f.write(CSV_HEADER + CSV_ROWS)
            for fmt, ext in (("json", ".json"), ("geojson", ".geojson")):
                with self.subTest(fmt=fmt):
                    proc = subprocess.run(
                        [sys.executable, os.path.join(REPO, "geolocate.py"),
                         "--localization_json", paths["loc.json"], "--camera_config", paths["cam.json"],
                         "--metadata_csv", paths["meta.csv"], "--output_json", paths["out.json"],
                         "--output_format", fmt],
                        capture_output=True, text=True, encoding="utf-8",
                        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                    self.assertEqual(proc.returncode, 0, proc.stderr)
                    with open(os.path.splitext(paths["out.json"])[0] + ext) as f:
                        out = json.load(f)
                    if fmt == "json":
                        self.assertEqual(out["summary"]["geolocation"]["regions_geolocated"], 4)
                    else:
                        self.assertEqual(len(out["features"]), 4)


if __name__ == "__main__":
    unittest.main()
