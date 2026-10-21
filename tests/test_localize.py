"""Tests for localize.py's fire-pixel output (the hot pixels each merged region covers)."""

import unittest

import cv2
import numpy as np

import localize


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


if __name__ == "__main__":
    unittest.main()
