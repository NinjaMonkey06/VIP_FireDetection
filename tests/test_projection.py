"""
Synthetic tests for geoloc/projection.py.

Fixture camera: 1000x800 px, 90° horizontal FOV → fx = fy = 500, principal
point at the center. Drone at 100 m AGL unless stated.
"""

import math
import unittest

import numpy as np

from geoloc.projection import (
    ABOVE_HORIZON, BELOW_MIN_DEPRESSION, BEYOND_MAX_RANGE, EARTH_RADIUS_M, VERTEX_OK,
    CameraPose, Intrinsics, ReliabilityLimits, camera_to_ned, clip_polygon_disk, polygon_area,
    ground_sample_distance_m, ground_to_pixels, latlon_to_offset,
    normalized_to_pixels, offset_to_latlon, pixels_to_ground,
    pixels_to_normalized, polygon_area_centroid, project_contour,
)
from tests import test_citrus_oracle as citrus

INTR = Intrinsics(width=1000, height=800, fx=500.0, fy=500.0, cx=500.0, cy=400.0)
CENTER = (500.0, 400.0)
ALT = 100.0
LAT0, LON0 = 34.15, -117.85
YAWS = [0, 45, 90, 180, 270, -30]


def ground(px, yaw=0.0, pitch=-90.0, roll=0.0, alt=ALT, intr=INTR):
    return pixels_to_ground(np.array([px]), intr, CameraPose(alt, yaw, pitch, roll))[0]


class TestCenterPixel(unittest.TestCase):
    def test_center_maps_to_drone_position(self):
        for yaw in YAWS:
            for alt in (10.0, 100.0, 400.0):
                with self.subTest(yaw=yaw, alt=alt):
                    n, e = ground(CENTER, yaw=yaw, alt=alt)
                    self.assertAlmostEqual(n, 0.0, places=9)
                    self.assertAlmostEqual(e, 0.0, places=9)

    def test_center_maps_to_drone_latlon(self):
        pose = CameraPose(ALT, 37.0)
        res = project_contour(np.array([CENTER]), INTR, pose, LAT0, LON0)
        self.assertAlmostEqual(res.latlon[0, 0], LAT0, places=12)
        self.assertAlmostEqual(res.latlon[0, 1], LON0, places=12)


class TestCardinalDirections(unittest.TestCase):
    """250 px from center at fx=500, 100 m AGL → 50 m on the ground."""

    def check(self, px, yaw, north, east):
        n, e = ground(px, yaw=yaw)
        self.assertAlmostEqual(n, north, places=6)
        self.assertAlmostEqual(e, east, places=6)

    def test_pixel_above_center(self):
        d = 50.0 / math.sqrt(2)
        cases = {0: (50, 0), 90: (0, 50), 180: (-50, 0), 270: (0, -50),
                 45: (d, d), -30: (50 * math.cos(math.radians(30)), -25.0)}
        for yaw, (n, e) in cases.items():
            with self.subTest(yaw=yaw):
                self.check((500, 150), yaw, n, e)

    def test_pixel_right_of_center(self):
        cases = {0: (0, 50), 90: (-50, 0), 180: (0, -50), 270: (50, 0)}
        for yaw, (n, e) in cases.items():
            with self.subTest(yaw=yaw):
                self.check((750, 400), yaw, n, e)


class TestScaling(unittest.TestCase):
    def test_offset_proportional_to_altitude(self):
        for yaw in YAWS:
            with self.subTest(yaw=yaw):
                a = ground((620, 310), yaw=yaw, alt=80.0)
                b = ground((620, 310), yaw=yaw, alt=160.0)
                np.testing.assert_allclose(b, 2 * a, rtol=1e-12)

    def test_resolution_independent(self):
        big = Intrinsics(2000, 1600, 1000.0, 1000.0, 1000.0, 800.0)
        for yaw in YAWS:
            with self.subTest(yaw=yaw):
                np.testing.assert_allclose(ground((620, 310), yaw=yaw),
                                           ground((1240, 620), yaw=yaw, intr=big), atol=1e-9)

    def test_ground_sample_distance_nadir(self):
        self.assertAlmostEqual(ground_sample_distance_m(INTR, CameraPose(ALT, 0.0)), 0.2, places=9)

    def test_non_positive_altitude_rejected(self):
        with self.assertRaises(ValueError):
            ground(CENTER, alt=0.0)


class TestLatLon(unittest.TestCase):
    def test_100m_north_at_equator(self):
        lat, lon = offset_to_latlon(0.0, 0.0, 100.0, 0.0)
        self.assertAlmostEqual(float(lat), 0.000898315, places=9)
        self.assertEqual(float(lon), 0.0)

    def test_longitude_scale_doubles_at_60_degrees(self):
        _, lon_eq = offset_to_latlon(0.0, 10.0, 0.0, 100.0)
        _, lon_60 = offset_to_latlon(60.0, 10.0, 0.0, 100.0)
        self.assertAlmostEqual(float(lon_60 - 10.0) / float(lon_eq - 10.0), 2.0, places=9)

    def test_round_trip_sub_millimetre(self):
        rng = np.random.default_rng(0)
        north, east = rng.uniform(-2000, 2000, 50), rng.uniform(-2000, 2000, 50)
        lat, lon = offset_to_latlon(LAT0, LON0, north, east)
        n2, e2 = latlon_to_offset(LAT0, LON0, lat, lon)
        np.testing.assert_allclose(n2, north, atol=1e-3)
        np.testing.assert_allclose(e2, east, atol=1e-3)

    def test_earth_radius_matches_reference(self):
        self.assertEqual(EARTH_RADIUS_M, 6378137.0)


class TestAgainstCitrusReference(unittest.TestCase):
    """Same fixture fed to the verbatim reference functions (FOV 90° horizontal)."""

    def citrus(self, u, v, yaw):
        img = np.zeros((INTR.height, INTR.width), dtype=np.uint8)
        _, cx, cy, _, _, W_r, H_r = citrus.rotate_image_and_box(
            img, u, v, 0, 0, citrus.get_rotation_for_north(yaw))
        east, north = citrus.pixel_to_meters(cx, cy, W_r, H_r, ALT, 90.0)
        return north, east

    def test_identical_at_yaw_zero(self):
        for px in [(500, 150), (750, 400), (123, 456), (999, 1), (0, 799)]:
            with self.subTest(px=px):
                np.testing.assert_allclose(ground(px, yaw=0.0), self.citrus(*px, 0.0), atol=1e-9)

    def test_deliberately_differs_at_yaw_90(self):
        # Reference fx comes from the rotated canvas width (800 instead of 1000),
        # so its distances are 1000/800 = 1.25x ours; direction agrees.
        ours = ground((500, 150), yaw=90.0)
        ref = np.array(self.citrus(500, 150, 90.0))
        np.testing.assert_allclose(ours, [0.0, 50.0], atol=1e-9)
        np.testing.assert_allclose(ref, [0.0, 62.5], atol=1e-6)


class TestOblique(unittest.TestCase):
    def test_center_ray_at_pitch(self):
        cases = {-45.0: 100.0, -60.0: 100.0 / math.tan(math.radians(60)), -30.0: 100.0 * math.sqrt(3)}
        for pitch, dist in cases.items():
            with self.subTest(pitch=pitch):
                np.testing.assert_allclose(ground(CENTER, pitch=pitch), [dist, 0.0], atol=1e-9)

    def test_center_ray_follows_yaw(self):
        np.testing.assert_allclose(ground(CENTER, yaw=90.0, pitch=-45.0), [0.0, 100.0], atol=1e-9)
        np.testing.assert_allclose(ground(CENTER, yaw=180.0, pitch=-45.0), [-100.0, 0.0], atol=1e-9)

    def test_image_top_is_farther_than_bottom_when_tilted(self):
        top, bottom = ground((500, 300), pitch=-45.0), ground((500, 500), pitch=-45.0)
        self.assertGreater(top[0], 100.0)
        self.assertLess(bottom[0], 100.0)

    def test_horizon_and_above_return_nan(self):
        self.assertTrue(np.isnan(ground(CENTER, pitch=0.0)).all())
        # Vertical half-FOV is atan(400/500) = 38.7°, so at pitch -10° the top row is above the horizon.
        self.assertTrue(np.isnan(ground((500, 0), pitch=-10.0)).all())
        self.assertFalse(np.isnan(ground((500, 799), pitch=-10.0)).any())

    def test_roll_does_not_move_center_ray(self):
        np.testing.assert_allclose(ground(CENTER, pitch=-50.0, roll=15.0),
                                   ground(CENTER, pitch=-50.0), atol=1e-9)

    def test_round_trip_through_ground_random_poses(self):
        rng = np.random.default_rng(1)
        px = np.column_stack((rng.uniform(0, 1000, 40), rng.uniform(0, 800, 40)))
        for _ in range(20):
            pose = CameraPose(rng.uniform(20, 300), rng.uniform(-180, 180),
                              rng.uniform(-90, -50), rng.uniform(-10, 10))
            with self.subTest(pose=pose):
                ne = pixels_to_ground(px, INTR, pose)
                self.assertFalse(np.isnan(ne).any())
                np.testing.assert_allclose(ground_to_pixels(ne, INTR, pose), px, atol=1e-6)


class TestDistortion(unittest.TestCase):
    DIST = Intrinsics(1000, 800, 500.0, 500.0, 500.0, 400.0, (-0.2, 0.05, 0.001, -0.0005, 0.0))

    def test_zero_coefficients_change_nothing(self):
        zero = Intrinsics(1000, 800, 500.0, 500.0, 500.0, 400.0, (0.0, 0.0, 0.0, 0.0, 0.0))
        px = np.array([[10.0, 20.0], [640.0, 512.0], [999.0, 799.0]])
        np.testing.assert_array_equal(pixels_to_normalized(px, zero), pixels_to_normalized(px, INTR))

    def test_distort_then_undistort_recovers_pixel(self):
        rng = np.random.default_rng(2)
        xy = rng.uniform(-0.8, 0.8, (50, 2))
        px = normalized_to_pixels(xy, self.DIST)
        np.testing.assert_allclose(pixels_to_normalized(px, self.DIST), xy, atol=1e-9)

    def test_distortion_moves_off_center_points_only(self):
        np.testing.assert_allclose(ground(CENTER, intr=self.DIST), [0.0, 0.0], atol=1e-9)
        self.assertGreater(np.abs(ground((900, 700), intr=self.DIST) - ground((900, 700))).max(), 1.0)

    def test_ground_round_trip_with_distortion(self):
        pose = CameraPose(120.0, 33.0, -70.0, 2.0)
        px = np.array([[100.0, 100.0], [500.0, 400.0], [900.0, 700.0]])
        ne = pixels_to_ground(px, self.DIST, pose)
        np.testing.assert_allclose(ground_to_pixels(ne, self.DIST, pose), px, atol=1e-6)


class TestPolygons(unittest.TestCase):
    SQUARE = np.array([[450, 350], [550, 350], [550, 450], [450, 450]], dtype=float)

    def test_centered_square_area_and_centroid(self):
        for yaw in YAWS:
            with self.subTest(yaw=yaw):
                res = project_contour(self.SQUARE, INTR, CameraPose(ALT, yaw), LAT0, LON0)
                self.assertAlmostEqual(res.area_m2, 400.0, places=6)     # 20 m x 20 m
                self.assertAlmostEqual(res.centroid_latlon[0], LAT0, places=12)
                self.assertAlmostEqual(res.centroid_latlon[1], LON0, places=12)
                self.assertEqual(res.n_unreliable, 0)

    def test_area_centroid_not_vertex_mean(self):
        # Rectangle N 10..30, E 0..20 plus an extra collinear vertex on one edge:
        # the vertex mean shifts, the area centroid must not.
        ne = np.array([[10, 0], [10, 10], [10, 20], [30, 20], [30, 0]], dtype=float)
        area, (cn, ce) = polygon_area_centroid(ne)
        self.assertAlmostEqual(area, 400.0)
        self.assertAlmostEqual(cn, 20.0)
        self.assertAlmostEqual(ce, 10.0)
        self.assertAlmostEqual(polygon_area_centroid(ne[::-1])[0], 400.0)   # winding-independent

    def test_degenerate_polygon_falls_back_to_vertex_mean(self):
        area, (cn, ce) = polygon_area_centroid(np.array([[0, 0], [10, 10], [20, 20]], dtype=float))
        self.assertEqual(area, 0.0)
        self.assertAlmostEqual(cn, 10.0)
        self.assertAlmostEqual(ce, 10.0)

    def test_polygon_crossing_horizon_is_clipped(self):
        res = project_contour(np.array([[400, 0], [600, 0], [600, 600], [400, 600]], dtype=float),
                              INTR, CameraPose(ALT, 0.0, -10.0), LAT0, LON0)
        self.assertEqual(res.vertex_status, [ABOVE_HORIZON, ABOVE_HORIZON, VERTEX_OK, VERTEX_OK])
        self.assertEqual(res.n_unreliable, 2)
        self.assertTrue(np.isnan(res.latlon[:2]).all())
        self.assertFalse(np.isnan(res.latlon[2:]).any())       # reliable vertices still projected
        self.assertTrue(res.clipped)
        self.assertGreater(res.area_m2, 0.0)
        radius = 100.0 / math.tan(math.radians(10.0))
        self.assertLessEqual(np.hypot(*res.polygon_north_east.T).max(), radius + 1e-6)


class TestReliability(unittest.TestCase):
    """Vertices near/above the horizon are marked unreliable instead of projected."""
    # Pitch -10°: a pixel 500·tan(8°) = 70.27 px above center looks 2° below the horizon.
    NEAR_HORIZON = np.array([[500.0, 400.0 - 500.0 * math.tan(math.radians(8.0))]])
    POSE = CameraPose(ALT, 0.0, -10.0)

    def project(self, pts, pose=POSE, limits=ReliabilityLimits()):
        return project_contour(np.asarray(pts, dtype=float), INTR, pose, LAT0, LON0, limits=limits)

    def test_center_ray_depression_equals_minus_pitch(self):
        for pitch in (-90.0, -60.0, -30.0, -5.0):
            with self.subTest(pitch=pitch):
                res = self.project([CENTER], CameraPose(ALT, 25.0, pitch))
                self.assertAlmostEqual(res.depression_deg[0], -pitch, places=9)

    def test_default_rejects_shallow_rays(self):
        self.assertEqual(ReliabilityLimits().min_depression_deg, 10.0)
        self.assertEqual(self.project([CENTER], CameraPose(ALT, 0.0, -30.0)).vertex_status, [VERTEX_OK])
        self.assertEqual(self.project([CENTER], CameraPose(ALT, 0.0, -5.0)).vertex_status,
                         [BELOW_MIN_DEPRESSION])

    def test_near_horizon_vertex_is_not_projected(self):
        res = self.project(self.NEAR_HORIZON)
        self.assertAlmostEqual(res.depression_deg[0], 2.0, places=9)
        self.assertEqual(res.vertex_status, [BELOW_MIN_DEPRESSION])
        self.assertTrue(np.isnan(res.latlon).all())
        # The raw hit is still reported for diagnostics: 100 / tan(2°) = 2863.6 m away.
        self.assertAlmostEqual(res.ground_range_m[0], 100.0 / math.tan(math.radians(2.0)), places=6)

    def test_limits_disabled_projects_far_away(self):
        res = self.project(self.NEAR_HORIZON, limits=ReliabilityLimits(None, None))
        self.assertEqual(res.vertex_status, [VERTEX_OK])
        n, _ = latlon_to_offset(LAT0, LON0, res.latlon[0, 0], res.latlon[0, 1])
        self.assertAlmostEqual(float(n), 2863.6, delta=0.1)

    def test_max_range(self):
        limits = ReliabilityLimits(None, 1000.0)
        self.assertEqual(self.project(self.NEAR_HORIZON, limits=limits).vertex_status, [BEYOND_MAX_RANGE])
        # Center ray at pitch -30° lands 173 m away: within range.
        self.assertEqual(self.project([CENTER], CameraPose(ALT, 0.0, -30.0), limits).vertex_status,
                         [VERTEX_OK])

    def test_above_horizon_flagged_even_with_limits_disabled(self):
        res = self.project([[500.0, 0.0]], limits=ReliabilityLimits(None, None))
        self.assertEqual(res.vertex_status, [ABOVE_HORIZON])
        self.assertTrue(np.isnan(res.ground_range_m).all())
        self.assertLess(res.depression_deg[0], 0.0)

    def test_one_shallow_vertex_clips_instead_of_dropping(self):
        pts = np.vstack(([[450, 450], [550, 450], [550, 550]], self.NEAR_HORIZON))
        res = self.project(pts)
        self.assertEqual(res.vertex_status, [VERTEX_OK] * 3 + [BELOW_MIN_DEPRESSION])
        self.assertTrue(res.clipped)
        self.assertIsNotNone(res.centroid_latlon)
        # The three reliable vertices survive unchanged; the rest of the outline is new.
        self.assertEqual(res.polygon_clip_vertex.count(False), 3)
        dep = np.degrees(np.arctan2(ALT, np.hypot(*res.polygon_north_east.T)))
        self.assertGreaterEqual(dep.min(), 10.0 - 1e-6)


def half_disk_image_polygon(intr, pose, width_m, top_v):
    """Image polygon whose below-horizon part is exactly the ground strip
    {forward >= 0, |sideways| <= width_m} relative to the drone, for a camera that
    sees both its nadir point and the horizon. Its sides are the images of the
    strip's two long edges, continued past their vanishing point into the sky."""
    psi = math.radians(pose.yaw_deg)
    fwd, side = np.array([math.cos(psi), math.sin(psi)]), np.array([-math.sin(psi), math.cos(psi)])
    bottom = ground_to_pixels(np.array([-width_m * side, width_m * side]), intr, pose)
    d_cam = camera_to_ned(pose).T @ np.array([fwd[0], fwd[1], 0.0])
    vp = np.array([intr.fx * d_cam[0] / d_cam[2] + intr.cx, intr.fy * d_cam[1] / d_cam[2] + intr.cy])
    top = [b + (vp - b) * (top_v - b[1]) / (vp[1] - b[1]) for b in bottom]
    return np.array([bottom[0], bottom[1], top[1], top[0]]), fwd


class TestClipping(unittest.TestCase):
    """Regions with unreliable vertices are clipped to their reliable part
    (on flat ground: a disk of radius h / tan(min_depression) around the drone)."""

    def test_fully_reliable_region_is_unchanged(self):
        sq = np.array([[450, 350], [550, 350], [550, 450], [450, 450]], dtype=float)
        res = project_contour(sq, INTR, CameraPose(ALT, 20.0), LAT0, LON0)
        self.assertFalse(res.clipped)
        self.assertEqual(res.removed_area_fraction, 0.0)
        np.testing.assert_array_equal(res.polygon_north_east, res.north_east)
        np.testing.assert_array_equal(res.polygon_latlon, res.latlon)
        self.assertEqual(res.polygon_clip_vertex, [False] * 4)
        self.assertAlmostEqual(res.area_m2, 400.0, places=6)

    def test_half_reliable_nadir_region_gives_half_disk(self):
        # Nadir, 1 m = 5 px. Limit chosen so the reliable disk has radius 30 m.
        # Ground rectangle N 0..60, E -60..60 -> clipped part is the half disk N >= 0.
        limits = ReliabilityLimits(min_depression_deg=math.degrees(math.atan2(100.0, 30.0)))
        rect = np.array([[200, 400], [800, 400], [800, 100], [200, 100]], dtype=float)
        res = project_contour(rect, INTR, CameraPose(ALT, 0.0), LAT0, LON0, limits=limits)
        self.assertEqual(res.vertex_status, [BELOW_MIN_DEPRESSION] * 4)   # every vertex is too far,
        self.assertTrue(res.clipped)                                       # but part is reliable
        half_disk = math.pi * 30.0 ** 2 / 2                                # 1413.7 m²
        self.assertAlmostEqual(res.area_m2 / half_disk, 1.0, delta=1e-3)
        n, e = latlon_to_offset(LAT0, LON0, *res.centroid_latlon)
        self.assertAlmostEqual(float(n), 4 * 30.0 / (3 * math.pi), delta=0.01)   # 12.73 m
        self.assertAlmostEqual(float(e), 0.0, delta=0.01)
        # Nadir view: image area is proportional to ground area, so removed = 1 - 1413.7 / 7200.
        self.assertAlmostEqual(res.removed_area_fraction, 1 - half_disk / 7200.0, delta=1e-3)
        self.assertTrue(all(res.polygon_clip_vertex))

    def test_half_above_horizon_oblique_region_gives_half_disk(self):
        # Wide camera (vertical FOV 127°) at pitch -45° sees its own nadir point and the sky.
        # The polygon runs from the ground line through the nadir point up past the horizon.
        wide = Intrinsics(1000, 800, 200.0, 200.0, 500.0, 400.0)
        radius = 100.0 / math.tan(math.radians(20.0))                      # 274.7 m
        limits = ReliabilityLimits(min_depression_deg=20.0)
        for yaw in (0.0, 30.0, -120.0):
            with self.subTest(yaw=yaw):
                pose = CameraPose(ALT, yaw, -45.0)
                poly, fwd = half_disk_image_polygon(wide, pose, 600.0, top_v=50.0)
                res = project_contour(poly, wide, pose, LAT0, LON0, limits=limits)
                self.assertEqual(res.vertex_status,
                                 [BELOW_MIN_DEPRESSION] * 2 + [ABOVE_HORIZON] * 2)
                self.assertTrue(res.clipped)
                self.assertAlmostEqual(res.area_m2 / (math.pi * radius ** 2 / 2), 1.0, delta=1e-3)
                n, e = latlon_to_offset(LAT0, LON0, *res.centroid_latlon)
                c = 4 * radius / (3 * math.pi)                             # 116.6 m along the yaw
                np.testing.assert_allclose([float(n), float(e)], c * fwd, atol=0.2)
                self.assertGreater(res.removed_area_fraction, 0.0)
                self.assertLess(res.removed_area_fraction, 1.0)

    def test_region_with_no_reliable_part_is_unreliable(self):
        pose = CameraPose(ALT, 0.0, -10.0)
        above = np.array([[400, 20], [600, 20], [600, 100], [400, 100]], dtype=float)
        # Rows 330..340 are about 2-2.5° below the horizon: below the 10° minimum.
        shallow = np.array([[450, 330], [550, 330], [550, 340], [450, 340]], dtype=float)
        for name, pts in (("above_horizon", above), ("too_shallow", shallow)):
            with self.subTest(name):
                res = project_contour(pts, INTR, pose, LAT0, LON0)
                self.assertEqual(res.n_unreliable, 4)
                self.assertIsNone(res.polygon_north_east)
                self.assertIsNone(res.area_m2)
                self.assertIsNone(res.centroid_latlon)
                self.assertEqual(res.removed_area_fraction, 1.0)

    def test_no_limits_and_crossing_horizon_cannot_be_bounded(self):
        res = project_contour(np.array([[400, 0], [600, 0], [600, 600], [400, 600]], dtype=float),
                              INTR, CameraPose(ALT, 0.0, -10.0), LAT0, LON0,
                              limits=ReliabilityLimits(None, None))
        self.assertIsNone(res.area_m2)

    def test_clip_polygon_disk_square_inside_and_around(self):
        sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        self.assertAlmostEqual(polygon_area(clip_polygon_disk(sq, 10.0)), 4.0, places=9)
        self.assertAlmostEqual(polygon_area(clip_polygon_disk(sq * 100, 10.0)) / (math.pi * 100), 1.0,
                               delta=1e-4)
        self.assertEqual(len(clip_polygon_disk(sq + 50, 10.0)), 0)


class TestTerrainInterface(unittest.TestCase):
    def test_custom_terrain_is_used(self):
        class RaisedGround:   # ground 20 m higher than the take-off reference
            def intersect(self, camera_height_m, rays_ned):
                rays = np.asarray(rays_ned)
                return rays[:, :2] * ((camera_height_m - 20.0) / rays[:, 2])[:, None]
        ne = pixels_to_ground(np.array([[500.0, 150.0]]), INTR, CameraPose(ALT, 0.0), RaisedGround())
        np.testing.assert_allclose(ne[0], [40.0, 0.0], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
