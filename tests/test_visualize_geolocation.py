"""Tests for the overlay geometry in visualize_geolocation.py: what is drawn must be what
geolocate.py would compute for those pixels."""

import unittest

import numpy as np

from geoloc.projection import CameraPose, Intrinsics, depression_deg, pixels_to_ground, pixels_to_rays_ned
from visualize_geolocation import ground_ring_px, horizon_px

INTR = Intrinsics(640, 512, 697.3, 697.3, 334.1, 266.4)
OBLIQUE = CameraPose(80.0, 30.0, -20.0, 2.0)


class TestOverlayGeometry(unittest.TestCase):
    def test_ring_pixels_project_back_to_the_ring(self):
        for radius in (100.0, 250.0):
            px = ground_ring_px(radius, INTR, OBLIQUE)
            px = px[~np.isnan(px[:, 0])]
            self.assertGreater(len(px), 10)
            ne = pixels_to_ground(px, INTR, OBLIQUE)
            np.testing.assert_allclose(np.hypot(ne[:, 0], ne[:, 1]), radius, rtol=1e-9)

    def test_ring_behind_camera_is_dropped(self):
        px = ground_ring_px(100.0, INTR, OBLIQUE)
        self.assertTrue(np.isnan(px[:, 0]).any())   # the half of the ring behind the drone

    def test_horizon_pixels_have_zero_depression(self):
        px = horizon_px(INTR, OBLIQUE)
        np.testing.assert_allclose(depression_deg(pixels_to_rays_ned(px, INTR, OBLIQUE)), 0.0, atol=1e-9)

    def test_nadir_horizon_is_outside_the_frame(self):
        px = horizon_px(INTR, CameraPose(80.0, 0.0, -90.0, 0.0))
        self.assertTrue(px is None or not ((px[:, 1] >= 0) & (px[:, 1] <= INTR.height)).any())


if __name__ == "__main__":
    unittest.main()
