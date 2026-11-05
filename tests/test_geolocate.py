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
from geoloc.metadata import FrameMetadata, SidecarCsvLoader
from geoloc.projection import ReliabilityLimits, latlon_to_offset

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_JSON = os.path.join(REPO, "results", "flame2_sample", "localization_results.json")
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


class TestContourTables(GeolocateTest):
    def regions(self, status):
        return [(r["frame_id"], n, g) for r in self.data["results"]
                for n, g in enumerate(r.get("fire_regions") or [], start=1)
                if g.get("geolocation_status") == status]

    def test_every_border_point_gets_a_row_in_polygon_order(self):
        reg = self.by_id["nadir"]["fire_regions"][0]
        rows = geolocate.contour_table_rows("nadir", 1, reg)
        self.assertEqual([r["point"] for r in rows], [1, 2, 3, 4])
        self.assertEqual([(r["lat"], r["lon"]) for r in rows], [(v["lat"], v["lon"]) for v in reg["gps_polygon"]])
        self.assertEqual({r["point_type"] for r in rows}, {"fire_edge"})
        self.assertEqual((rows[0]["frame_id"], rows[0]["contour"]), ("nadir", 1))

    def test_points_on_the_image_border_are_labelled(self):
        rows = geolocate.contour_table_rows("nadir", 2, self.by_id["nadir"]["fire_regions"][1])
        self.assertEqual([r["point_type"] for r in rows], ["image_border", "fire_edge", "fire_edge", "image_border"])

    def test_clipped_contour_labels_points_made_by_clipping(self):
        frame_id, n, reg = self.regions("clipped_to_reliable")[0]
        types = [r["point_type"] for r in geolocate.contour_table_rows(frame_id, n, reg)]
        self.assertIn("clipped", types)
        self.assertEqual(types.count("clipped"), sum(bool(v.get("clip_vertex")) for v in reg["gps_polygon"]))

    def test_contour_without_coordinates_is_reported_not_skipped(self):
        for status in ("unreliable", "missing_metadata"):
            frame_id, n, reg = self.regions(status)[0]
            rows = geolocate.contour_table_rows(frame_id, n, reg)
            self.assertEqual(len(rows), len(reg["normalized"]))
            self.assertEqual({r["point_type"] for r in rows}, {status})
            self.assertEqual({r["lat"] for r in rows}, {""})

    def test_one_file_per_contour(self):
        tables = os.path.join(self._tmp.name, "tables")
        written = geolocate.write_contour_tables(self.data["results"], tables)
        self.assertEqual(written, 6)        # 2 nadir + 1 yawed + 1 no_meta + 2 oblique; clear frame none
        with open(os.path.join(tables, "nadir_contour_02.csv"), encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines[0], ",".join(geolocate.CONTOUR_TABLE_COLUMNS))
        self.assertEqual(len(lines), 5)
        self.assertEqual(len(os.listdir(tables)), 6)


class TestGeolocateResults(GeolocateTest):
    def test_fire_area_from_hot_pixels(self):
        data = [frame("nadir", [SQUARE])]
        data[0]["fire_regions"][0]["fire_pixels_256"] = [[120, 110, 119], [121, 110, 119]]   # 20 cells
        geolocate.geolocate_results(data, CAMERA, SidecarCsvLoader(self.csv))
        reg = data[0]["fire_regions"][0]
        cell_m2 = (640 / 256 * 0.3125) * (512 / 256 * 0.3125)       # 0.3125 m/px at 100 m
        self.assertAlmostEqual(reg["fire_area_m2"], 20 * cell_m2, places=2)
        self.assertEqual(reg["fire_area_excluded_fraction"], 0.0)
        self.assertLess(reg["fire_area_m2"], reg["ground_area_m2"])

    def test_fire_area_absent_without_hot_pixels(self):
        reg = self.by_id["nadir"]["fire_regions"][0]
        self.assertIsNone(reg["fire_area_m2"])

    def test_counts(self):
        self.assertEqual(self.counts, {"frames_geolocated": 3, "frames_missing_metadata": 1,
                                       "regions_geolocated": 3, "regions_clipped_to_reliable": 1,
                                       "regions_unreliable": 1,
                                       "frames_by_altitude_reference": {"agl": 3}})

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
        # Pitch -10°, fx = 320, 100 m AGL → reliable disk radius 100 / tan(10°) = 567 m.
        # TOP (rows 0..154) is entirely above the horizon → unreliable.
        # SQUARE straddles the 10° line (the image center row): its top edge is ~1° below
        # the horizon (~6 km away), its bottom edge ~19° (~290 m) → clipped to the near part.
        r = self.by_id["oblique"]
        top, center = r["fire_regions"]
        self.assertEqual(top["geolocation_status"], "unreliable")
        self.assertIsNone(top["gps_polygon"])
        self.assertIsNone(top["gps_centroid"])
        self.assertEqual(top["unreliable_vertices"], {"above_horizon": 4})
        self.assertEqual(top["removed_area_fraction"], 1.0)

        self.assertEqual(center["geolocation_status"], "clipped_to_reliable")
        self.assertTrue(center["clipped"])
        self.assertEqual(center["unreliable_vertices"], {"below_min_depression": 2})
        self.assertGreater(center["removed_area_fraction"], 0.4)
        self.assertLess(center["removed_area_fraction"], 0.6)
        self.assertGreater(center["ground_area_m2"], 0.0)
        self.assertIsNotNone(center["gps_centroid"])
        for v in center["gps_polygon"]:
            self.assertGreaterEqual(v["depression_deg"], 9.99)
            self.assertLessEqual(v["ground_range_m"], 567.2)
        # The two reliable original vertices are kept; everything else came from clipping.
        self.assertEqual(sum(not v.get("clip_vertex", False) for v in center["gps_polygon"]), 2)
        self.assertTrue(any("oblique" in w for w in r["geolocation"]["warnings"]))

    def test_oblique_limits_are_configurable(self):
        loader = SidecarCsvLoader(self.csv)

        def run(limits):
            f = frame("oblique", [SQUARE])
            geolocate.geolocate_results([f], CAMERA, loader, limits=limits)
            return f["fire_regions"][0]

        off = run(ReliabilityLimits(None, None))
        self.assertEqual(off["geolocation_status"], "ok")             # old behaviour: ~6 km polygon
        self.assertGreater(max(v["ground_range_m"] for v in off["gps_polygon"]), 5000)

        capped = run(ReliabilityLimits(None, 400.0))
        self.assertEqual(capped["geolocation_status"], "clipped_to_reliable")
        self.assertLessEqual(max(v["ground_range_m"] for v in capped["gps_polygon"]), 400.0)
        self.assertEqual(capped["unreliable_vertices"], {"beyond_max_range": 2})

        # The whole square is farther than 250 m → nothing reliable.
        self.assertEqual(run(ReliabilityLimits(None, 250.0))["geolocation_status"], "unreliable")

    def test_center_above_horizon_still_writes_strict_json(self):
        class LevelCamera:   # gimbal pointing at the horizon
            def load(self, frame_id, image_path=None):
                return FrameMetadata(LAT0, LON0, 100.0, 0.0, 0.0, 0.0, None, "agl", "stub")
        data = [frame("level", [SQUARE])]
        geolocate.geolocate_results(data, CAMERA, LevelCamera(), merge_kernel_size=15)
        self.assertIsNone(data[0]["geolocation"]["gsd_center_m"])
        self.assertIsNone(data[0]["geolocation"]["dilation_buffer_m"])
        json.dumps(data, allow_nan=False)   # raises if any NaN leaked into the output

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


class TestImageSizeSource(GeolocateTest):
    """The camera size check uses original_wh from the localization JSON (the size of
    the thermal image localize.py read), never the 256x256 space or any overlay size."""

    def project_with_wh(self, w, h):
        reg = region(SQUARE)
        reg["original_wh"] = {"width": w, "height": h}
        geolocate.geolocate_region(reg, CAMERA, SidecarCsvLoader(self.csv).load("nadir"))
        return reg

    def test_plain_resize_of_calibrated_frame_gives_same_gps(self):
        a, b = self.project_with_wh(640, 512), self.project_with_wh(1280, 1024)
        self.assertEqual(a["gps_polygon"], b["gps_polygon"])
        self.assertEqual(a["ground_area_m2"], b["ground_area_m2"])

    def test_real_mismatch_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "958x760"):
            self.project_with_wh(958, 760)


class TestGeoJson(GeolocateTest):
    def test_lon_lat_order_and_closed_rings(self):
        gj = geolocate.to_geojson(self.data["results"])
        self.assertEqual(len(gj["features"]), 4)   # 3 ok + 1 clipped; the unreliable one is left out
        self.assertEqual(sorted(f["properties"]["clipped"] for f in gj["features"]),
                         [False, False, False, True])
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
                        self.assertEqual(out["summary"]["geolocation"]["regions_geolocated"], 3)
                        self.assertEqual(out["summary"]["geolocation"]["min_depression_deg"], 10.0)
                    else:
                        self.assertEqual(len(out["features"]), 4)


if __name__ == "__main__":
    unittest.main()
