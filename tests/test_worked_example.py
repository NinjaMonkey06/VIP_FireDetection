"""Verifies every number in docs/geolocation_worked_example.md against the library."""

import math
import unittest

import numpy as np

from geoloc.projection import (
    CAM_TO_GIMBAL, VERTEX_OK, CameraPose, Intrinsics, depression_deg, gimbal_to_ned,
    ground_to_pixels, offset_to_latlon, pixels_to_ground, pixels_to_normalized,
    pixels_to_rays_ned, project_contour,
)

INTR = Intrinsics(640, 512, 640.0, 640.0, 320.0, 256.0)
POSE = CameraPose(100.0, 30.0, -60.0, 0.0)
PIXEL = np.array([[480.0, 160.0]])
LAT0, LON0 = 34.0, -118.0


class TestWorkedExample(unittest.TestCase):
    def test_step1_normalized(self):
        np.testing.assert_allclose(pixels_to_normalized(PIXEL, INTR)[0], [0.25, -0.15], atol=1e-12)

    def test_step2_gimbal_frame(self):
        np.testing.assert_allclose(CAM_TO_GIMBAL @ [0.25, -0.15, 1.0], [1.0, 0.25, -0.15], atol=1e-12)

    def test_step4_pitch(self):
        after_pitch = gimbal_to_ned(0.0, -60.0, 0.0) @ [1.0, 0.25, -0.15]
        np.testing.assert_allclose(after_pitch, [0.6299038, 0.25, 0.7910254], atol=1e-7)
        axis = gimbal_to_ned(0.0, -60.0, 0.0) @ [1.0, 0.0, 0.0]
        np.testing.assert_allclose(axis, [0.5, 0.0, 0.8660254], atol=1e-7)

    def test_step5_yaw(self):
        np.testing.assert_allclose(pixels_to_rays_ned(PIXEL, INTR, POSE)[0],
                                   [0.4205127, 0.5314583, 0.7910254], atol=1e-7)

    def test_step6_ground(self):
        north, east = pixels_to_ground(PIXEL, INTR, POSE)[0]
        self.assertAlmostEqual(100.0 / 0.7910254, 126.4182, places=4)
        self.assertAlmostEqual(north, 53.160, places=3)
        self.assertAlmostEqual(east, 67.186, places=3)
        self.assertAlmostEqual(math.hypot(north, east), 85.674, places=3)
        ray = pixels_to_rays_ned(PIXEL, INTR, POSE)[0]
        self.assertAlmostEqual(100.0 / ray[2] * np.linalg.norm(ray), 131.68, places=2)

    def test_step7_reliability(self):
        ray = pixels_to_rays_ned(PIXEL, INTR, POSE)
        self.assertAlmostEqual(math.hypot(ray[0, 0], ray[0, 1]), 0.6777011, places=7)
        dep = depression_deg(ray)[0]
        self.assertAlmostEqual(dep, 49.41, places=2)
        self.assertAlmostEqual(100.0 * math.radians(1.0) / math.sin(math.radians(dep)) ** 2, 3.0, places=1)
        self.assertEqual(project_contour(PIXEL, INTR, POSE, LAT0, LON0).vertex_status, [VERTEX_OK])

    def test_step8_latlon(self):
        north, east = pixels_to_ground(PIXEL, INTR, POSE)[0]
        self.assertAlmostEqual(math.cos(math.radians(LAT0)), 0.829038, places=6)
        lat, lon = offset_to_latlon(LAT0, LON0, north, east)
        self.assertAlmostEqual(float(lat), 34.00047755, places=8)
        self.assertAlmostEqual(float(lon), -117.99927200, places=8)

    def test_sanity_checks(self):
        np.testing.assert_allclose(pixels_to_ground(np.array([[320.0, 256.0]]), INTR, POSE)[0],
                                   [50.000, 28.868], atol=1e-3)
        ne = pixels_to_ground(PIXEL, INTR, POSE)
        np.testing.assert_allclose(ground_to_pixels(ne, INTR, POSE)[0], [480.0, 160.0], atol=1e-9)
        nadir = pixels_to_ground(PIXEL, INTR, CameraPose(100.0, 30.0, -90.0, 0.0))[0]
        np.testing.assert_allclose(nadir, [0.490, 29.151], atol=1e-3)

    def test_sign_conventions(self):
        nadir = lambda yaw: CameraPose(100.0, yaw, -90.0, 0.0)
        above_center = np.array([[320.0, 128.0]])
        # Yaw 0: image top points north. Yaw +90 (clockwise): image top points east.
        np.testing.assert_allclose(pixels_to_ground(above_center, INTR, nadir(0.0))[0], [20.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(pixels_to_ground(above_center, INTR, nadir(90.0))[0], [0.0, 20.0], atol=1e-9)
        # Pitch: positive tilts up, -90 is straight down.
        self.assertGreater((gimbal_to_ned(0.0, 10.0, 0.0) @ [1, 0, 0])[2], -1.0)
        self.assertLess((gimbal_to_ned(0.0, 10.0, 0.0) @ [1, 0, 0])[2], 0.0)       # up = negative down
        np.testing.assert_allclose(gimbal_to_ned(0.0, -90.0, 0.0) @ [1, 0, 0], [0, 0, 1], atol=1e-12)
        # Roll: positive puts the right side down.
        self.assertGreater((gimbal_to_ned(0.0, 0.0, 10.0) @ [0, 1, 0])[2], 0.0)


if __name__ == "__main__":
    unittest.main()
