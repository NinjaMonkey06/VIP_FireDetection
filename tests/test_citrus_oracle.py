"""
test_citrus_oracle.py — Characterization tests for the reference geotagging code
=================================================================================
The functions in the "Reference code" section below are copied verbatim from
Geotagging_System.ipynb in https://github.com/afif1230/citrus-uav-pipeline
(commit 06066f2), Copyright (c) 2026 afif1230, MIT License — see
THIRD_PARTY_LICENSES/citrus-uav-pipeline.txt.

These tests document two scale issues in that reference, measured against an
independent nadir pinhole forward model:

    1. Rotated-canvas fx: after rotating the image north-up, fx is derived from
       the enlarged canvas width instead of the original image width, so
       ground distances are scaled by W / new_W (depends on yaw).

    2. FOV type: DJI publishes the Mini 4 Pro FOV (82.1°) alongside a 24 mm
       format-equivalent focal length. Only a *diagonal* 82.1° is consistent
       with 24 mm (24.84 vs 19.87 mm); the reference treats it as horizontal, scaling ground
       distances by diag_px / W = 1.25 on a 4:3 sensor.

These tests PASS when the issues are present. Run directly for a numbers table:
    python -m tests.test_citrus_oracle
"""

import math
import unittest
from typing import Tuple

import cv2
import numpy as np

# ==================== Reference code (verbatim) ====================
DEG_TO_RAD = math.pi / 180.0
FOV = 82.1
EAST_CORRECTION = 0
NORTH_CORRECTION = 0


def get_rotation_for_north(gimbal_yaw: float) -> float:
    return -gimbal_yaw


def rotate_image_and_box(image: np.ndarray, cx: float, cy: float, w: float, h: float, angle: float):
    """Rotate both image and YOLO box together. Handles arbitrary angles."""
    orig_H, orig_W = image.shape[:2]

    if abs(angle) < 0.01:  # Effectively 0
        return image, cx, cy, w, h, orig_W, orig_H

    # For arbitrary angles - just rotate by the given angle
    center = (orig_W // 2, orig_H // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_W = int(orig_H * sin + orig_W * cos)
    new_H = int(orig_H * cos + orig_W * sin)

    M[0, 2] += new_W / 2 - center[0]
    M[1, 2] += new_H / 2 - center[1]

    rotated = cv2.warpAffine(image, M, (new_W, new_H))

    # Rotate the bounding box center
    box_center = np.array([cx, cy, 1])
    new_center = M @ box_center
    new_cx = new_center[0]
    new_cy = new_center[1]

    # Adjust box dimensions
    angle_rad = math.radians(-angle)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = w * cos_a + h * sin_a
    new_h = w * sin_a + h * cos_a

    return rotated, new_cx, new_cy, new_w, new_h, new_W, new_H


def pixel_to_meters(u_px: float, v_px: float,
                   W: int, H: int,
                   altitude: float, fovx: float,
                   apply_correction: bool = True) -> Tuple[float, float]:
    fx = (W / 2.0) / math.tan(0.5 * fovx * DEG_TO_RAD)
    fy = fx

    cx = W / 2.0
    cy = H / 2.0

    dx_px = u_px - cx
    dy_px = v_px - cy

    east_m = (dx_px / fx) * altitude
    north_m = -(dy_px / fy) * altitude

    if apply_correction:
        east_m += EAST_CORRECTION
        north_m += NORTH_CORRECTION

    return east_m, north_m

# ==================== Independent ground truth ====================
# DJI Mini 4 Pro published specs: 82.1° FOV, 24 mm format-equivalent, 8064x6048.
W, H = 8064, 6048
ALT = 100.0
FULL_FRAME_DIAG_MM = math.hypot(36.0, 24.0)       # 43.27 mm
DJI_EQUIV_FOCAL_MM = 24.0


def true_nadir_pixel(east_m, north_m, yaw_deg, fx, W, H, alt):
    """Where a ground point appears in an UNROTATED nadir image.
    Image right = (cos yaw) east - (sin yaw) north; image up = yaw direction."""
    psi = math.radians(yaw_deg)
    right = east_m * math.cos(psi) - north_m * math.sin(psi)
    fwd = east_m * math.sin(psi) + north_m * math.cos(psi)
    return W / 2.0 + fx * right / alt, H / 2.0 - fx * fwd / alt


def citrus_project(u, v, yaw_deg, W, H, alt):
    """Run the reference path: rotate north-up, then pixel_to_meters."""
    img = np.zeros((H, W), dtype=np.uint8)
    _, cx_r, cy_r, _, _, W_r, H_r = rotate_image_and_box(
        img, u, v, 0, 0, get_rotation_for_north(yaw_deg))
    east, north = pixel_to_meters(cx_r, cy_r, W_r, H_r, alt, FOV)
    return east, north, W_r


def fx_if_horizontal(fov_deg, W):
    return (W / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def fx_if_diagonal(fov_deg, W, H):
    return (math.hypot(W, H) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def equiv_focal_mm(fov_deg, fov_type):
    """35 mm-equivalent focal length implied by a FOV on a 4:3 frame whose
    diagonal matches the full-frame diagonal (DJI's convention)."""
    span = FULL_FRAME_DIAG_MM if fov_type == "diagonal" else FULL_FRAME_DIAG_MM * 0.8
    return (span / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


YAWS = [0, 30, 45, 90, 135, 180, 270]
TARGET_N = 30.0   # ground target 30 m north of the drone


class TestRotatedCanvasFx(unittest.TestCase):
    """Issue 1 — isolate it by using the reference's own FOV interpretation
    (82.1° horizontal) for the ground truth too."""

    def test_direction_is_correct_but_distance_scales_with_canvas(self):
        fx = fx_if_horizontal(FOV, W)
        for yaw in YAWS:
            with self.subTest(yaw=yaw):
                u, v = true_nadir_pixel(0.0, TARGET_N, yaw, fx, W, H, ALT)
                east, north, W_r = citrus_project(u, v, yaw, W, H, ALT)
                dist = math.hypot(east, north)
                bearing = math.degrees(math.atan2(east, north)) % 360
                # Direction is right: the -yaw rotation sign is correct.
                self.assertAlmostEqual(min(bearing, 360 - bearing), 0.0, places=3)
                # Magnitude is scaled by W / new_W.
                self.assertAlmostEqual(dist, TARGET_N * W / W_r, places=3)

    def test_concrete_values(self):
        fx = fx_if_horizontal(FOV, W)
        expected = {0: 30.000, 45: 24.246, 90: 40.000, 180: 30.000}
        for yaw, exp in expected.items():
            with self.subTest(yaw=yaw):
                u, v = true_nadir_pixel(0.0, TARGET_N, yaw, fx, W, H, ALT)
                east, north, _ = citrus_project(u, v, yaw, W, H, ALT)
                self.assertAlmostEqual(math.hypot(east, north), exp, places=2)


class TestFovType(unittest.TestCase):
    """Issue 2 — which FOV interpretation matches DJI's 24 mm equivalent?"""

    def test_only_diagonal_matches_24mm_equivalent(self):
        diag = equiv_focal_mm(FOV, "diagonal")
        horiz = equiv_focal_mm(FOV, "horizontal")
        self.assertAlmostEqual(diag, 24.84, delta=0.01)
        self.assertAlmostEqual(horiz, 19.87, delta=0.01)
        self.assertLess(abs(diag - DJI_EQUIV_FOCAL_MM) / DJI_EQUIV_FOCAL_MM, 0.05)
        self.assertGreater(abs(horiz - DJI_EQUIV_FOCAL_MM) / DJI_EQUIV_FOCAL_MM, 0.15)

    def test_distance_inflation_at_yaw_zero(self):
        # Truth: 82.1° is diagonal. Yaw 0 → no rotation, so issue 1 is absent.
        fx_true = fx_if_diagonal(FOV, W, H)
        u, v = true_nadir_pixel(0.0, TARGET_N, 0, fx_true, W, H, ALT)
        east, north, _ = citrus_project(u, v, 0, W, H, ALT)
        self.assertAlmostEqual(north, TARGET_N * 1.25, places=6)   # 37.5 m
        self.assertAlmostEqual(east, 0.0, places=6)
        self.assertAlmostEqual(2 * math.degrees(math.atan(W / 2 / fx_true)), 69.73, delta=0.01)


class TestCombined(unittest.TestCase):
    """Both issues together: reported distance / true distance = diag_px / new_W."""

    def test_combined_ratio(self):
        fx_true = fx_if_diagonal(FOV, W, H)
        for yaw in YAWS:
            with self.subTest(yaw=yaw):
                u, v = true_nadir_pixel(0.0, TARGET_N, yaw, fx_true, W, H, ALT)
                east, north, W_r = citrus_project(u, v, yaw, W, H, ALT)
                self.assertAlmostEqual(math.hypot(east, north) / TARGET_N,
                                       math.hypot(W, H) / W_r, places=4)


def report():
    print(f"Camera: {W}x{H}, FOV {FOV}°, altitude {ALT} m, target {TARGET_N} m north\n")
    print("FOV type check (35 mm-equivalent focal implied by 82.1°; DJI publishes 24 mm):")
    print(f"  as horizontal: {equiv_focal_mm(FOV, 'horizontal'):.2f} mm")
    print(f"  as diagonal:   {equiv_focal_mm(FOV, 'diagonal'):.2f} mm")
    print(f"  true HFOV if diagonal: {2 * math.degrees(math.atan(W / 2 / fx_if_diagonal(FOV, W, H))):.2f}°\n")
    fx_h = fx_if_horizontal(FOV, W)
    fx_d = fx_if_diagonal(FOV, W, H)
    print(f"{'yaw':>5} {'canvas W':>9} | {'issue 1 only':>13} {'err':>7} | {'both issues':>12} {'err':>7}")
    for yaw in YAWS:
        u, v = true_nadir_pixel(0.0, TARGET_N, yaw, fx_h, W, H, ALT)
        e1, n1, W_r = citrus_project(u, v, yaw, W, H, ALT)
        u, v = true_nadir_pixel(0.0, TARGET_N, yaw, fx_d, W, H, ALT)
        e2, n2, _ = citrus_project(u, v, yaw, W, H, ALT)
        d1, d2 = math.hypot(e1, n1), math.hypot(e2, n2)
        print(f"{yaw:>5} {W_r:>9} | {d1:>11.2f} m {100 * (d1 / TARGET_N - 1):>+6.1f}% |"
              f" {d2:>10.2f} m {100 * (d2 / TARGET_N - 1):>+6.1f}%")


if __name__ == "__main__":
    report()
    print()
    unittest.main(verbosity=2)
