"""Tests for geoloc/camera.py and the shipped camera configs."""

import math
import os
import unittest

import numpy as np

from geoloc.camera import camera_config_from_dict, focal_from_fov, load_camera_config
from geoloc.projection import CameraPose, pixels_to_ground

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "camera_configs")
BASE = {"name": "test", "calibration_width": 1000, "calibration_height": 750}


def cfg(**kw):
    return camera_config_from_dict({**BASE, **kw})


class TestFovConversion(unittest.TestCase):
    def test_diagonal_fov_of_hfov_90_on_4_3(self):
        dfov = math.degrees(2 * math.atan(1.25))          # 102.68°
        self.assertAlmostEqual(dfov, 102.68, places=2)
        self.assertAlmostEqual(focal_from_fov(dfov, "diagonal", 1000, 750), 500.0, places=9)

    def test_fov_types_agree_for_the_same_lens(self):
        h = focal_from_fov(90.0, "horizontal", 1000, 750)
        v = focal_from_fov(math.degrees(2 * math.atan(375 / h)), "vertical", 1000, 750)
        d = focal_from_fov(math.degrees(2 * math.atan(625 / h)), "diagonal", 1000, 750)
        self.assertAlmostEqual(v, h, places=9)
        self.assertAlmostEqual(d, h, places=9)

    def test_fov_type_required(self):
        with self.assertRaises(ValueError):
            cfg(fov_deg=90.0)
        with self.assertRaises(ValueError):
            cfg(fov_deg=90.0, fov_type="horizonal")


class TestIntrinsicsSpec(unittest.TestCase):
    def test_both_or_neither_rejected(self):
        with self.assertRaises(ValueError):
            cfg()
        with self.assertRaises(ValueError):
            cfg(fov_deg=90.0, fov_type="horizontal", fx=500.0, fy=500.0)

    def test_explicit_intrinsics_scale_with_resolution(self):
        c = cfg(fx=800.0, fy=810.0, cx=505.0, cy=370.0, dist_coeffs=[-0.1, 0.01, 0, 0, 0])
        k = c.intrinsics_for(500, 375)
        self.assertEqual((k.fx, k.fy, k.cx, k.cy), (400.0, 405.0, 252.5, 185.0))
        self.assertEqual(k.dist_coeffs, (-0.1, 0.01, 0.0, 0.0, 0.0))

    def test_principal_point_defaults_to_center(self):
        k = cfg(fov_deg=90.0, fov_type="horizontal").intrinsics_for(1000, 750)
        self.assertEqual((k.cx, k.cy), (500.0, 375.0))

    def test_aspect_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            cfg(fov_deg=90.0, fov_type="horizontal").intrinsics_for(1600, 900)

    def test_resize_rounding_allowed_but_not_small_aspect_changes(self):
        c = camera_config_from_dict({"name": "flame", "calibration_width": 640, "calibration_height": 512,
                                     "fov_deg": 45.0, "fov_type": "horizontal"})
        c.intrinsics_for(958, 766)                        # 958 * 0.8 = 766.4 → rounding only
        with self.assertRaises(ValueError):
            c.intrinsics_for(958, 760)                    # repo sample frames: 0.84% off → not a resize

    def test_crop_shifts_principal_point_then_scales(self):
        c = cfg(fx=500.0, fy=500.0, crop={"x": 100, "y": 125, "width": 800, "height": 500})
        k = c.intrinsics_for(1600, 1000)                  # crop upscaled 2x
        self.assertEqual((k.fx, k.cx, k.cy), (1000.0, 800.0, 500.0))

    def test_crop_outside_frame_rejected(self):
        with self.assertRaises(ValueError):
            cfg(fx=500.0, fy=500.0, crop={"x": 300, "y": 0, "width": 800, "height": 750})


class TestShippedConfigs(unittest.TestCase):
    def test_dji_sample_is_diagonal_fov(self):
        c = load_camera_config(os.path.join(CONFIG_DIR, "dji_mini4pro.json"))
        k = c.intrinsics_for(8064, 6048)
        self.assertAlmostEqual(2 * math.degrees(math.atan(k.width / 2 / k.fx)), 69.73, delta=0.01)
        self.assertEqual(c.altitude_reference, "takeoff")
        self.assertIn("gimbal_yaw_deg", c.metadata_fields)

    def test_dji_sample_same_ground_point_at_any_resolution(self):
        c = load_camera_config(os.path.join(CONFIG_DIR, "dji_mini4pro.json"))
        pose = CameraPose(100.0, 25.0)
        norm = np.array([[0.7, 0.2]])
        a = pixels_to_ground(norm * [8064, 6048], c.intrinsics_for(8064, 6048), pose)
        b = pixels_to_ground(norm * [4032, 3024], c.intrinsics_for(4032, 3024), pose)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_template_refuses_to_load_unfilled(self):
        with self.assertRaises(ValueError):
            load_camera_config(os.path.join(CONFIG_DIR, "template.json"))


if __name__ == "__main__":
    unittest.main()
