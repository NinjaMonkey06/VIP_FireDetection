"""Tests for geoloc.metadata.TimedTelemetryLoader: telemetry samples matched to a frame time."""

import os
import tempfile
import unittest
from datetime import datetime

from geoloc.metadata import MetadataError, MissingMetadataError, TimedTelemetryLoader, wrap_degrees

HEADER = "time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg,timestamp\n"


class TimedTelemetryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def loader(self, rows, header=HEADER, **kw):
        path = os.path.join(self._tmp.name, "telemetry.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + rows)
        return TimedTelemetryLoader(path, **kw)

    def test_interpolates_between_samples(self):
        m = self.loader("0,42.0,-121.0,60,10,-20,0,2026-11-12T10:00:00\n"
                        "2,42.002,-121.004,64,30,-24,2,2026-11-12T10:00:02\n").at(0.5)
        self.assertAlmostEqual(m.latitude_deg, 42.0005, places=9)
        self.assertAlmostEqual(m.longitude_deg, -121.001, places=9)
        self.assertAlmostEqual(m.altitude_agl_m, 61.0)
        self.assertAlmostEqual(m.gimbal_yaw_deg, 15.0)
        self.assertAlmostEqual(m.gimbal_pitch_deg, -21.0)
        self.assertAlmostEqual(m.gimbal_roll_deg, 0.5)
        self.assertEqual(m.timestamp, datetime(2026, 11, 12, 10, 0, 0, 500000))
        self.assertEqual(m.altitude_reference, "agl")
        self.assertIn("@0.500s", m.source)

    def test_yaw_crosses_180_the_short_way(self):
        m = self.loader("0,42,-121,60,170,-20,0,\n1,42,-121,60,-170,-20,0,\n").at(0.5)
        self.assertAlmostEqual(abs(m.gimbal_yaw_deg), 180.0)
        m = self.loader("0,42,-121,60,170,-20,0,\n1,42,-121,60,-170,-20,0,\n").at(0.25)
        self.assertAlmostEqual(m.gimbal_yaw_deg, 175.0)
        self.assertIsNone(m.timestamp)

    def test_holds_the_end_samples_within_the_gap_and_not_beyond(self):
        ld = self.loader("1,42,-121,60,10,-20,0,\n2,42,-121,70,20,-20,0,\n", max_gap_s=0.5)
        self.assertEqual(ld.at(0.6).altitude_agl_m, 60.0)
        self.assertEqual(ld.at(2.4).altitude_agl_m, 70.0)
        for t in (0.4, 2.6):
            with self.assertRaises(MissingMetadataError):
                ld.at(t)

    def test_does_not_bridge_a_hole_in_the_log(self):
        ld = self.loader("0,42,-121,60,10,-20,0,\n10,42.01,-121,80,100,-20,0,\n", max_gap_s=1.0)
        near_start, near_end = ld.at(0.9), ld.at(9.1)      # held, not pulled towards the far side
        self.assertEqual((near_start.latitude_deg, near_start.altitude_agl_m, near_start.gimbal_yaw_deg),
                         (42.0, 60.0, 10.0))
        self.assertEqual((near_end.latitude_deg, near_end.altitude_agl_m, near_end.gimbal_yaw_deg),
                         (42.01, 80.0, 100.0))
        with self.assertRaises(MissingMetadataError):
            ld.at(5.0)

    def test_a_gap_no_frame_would_be_rejected_in_is_interpolated(self):
        ld = self.loader("0,42,-121,60,10,-20,0,\n2,42,-121,80,10,-20,0,\n", max_gap_s=1.0)
        self.assertAlmostEqual(ld.at(0.5).altitude_agl_m, 65.0)

    def test_unsorted_rows_are_sorted_and_duplicates_rejected(self):
        ld = self.loader("2,42,-121,80,10,-20,0,\n0,42,-121,60,10,-20,0,\n")
        self.assertAlmostEqual(ld.at(1.0).altitude_agl_m, 70.0)
        with self.assertRaisesRegex(MetadataError, "duplicate"):
            self.loader("1,42,-121,60,10,-20,0,\n1,42,-121,61,10,-20,0,\n")

    def test_bad_files_are_rejected(self):
        with self.assertRaisesRegex(MetadataError, "missing columns"):
            self.loader("0,42,-121,60,10,-20\n", header="time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg\n")
        with self.assertRaisesRegex(MetadataError, "empty 'altitude_agl_m'"):
            self.loader("0,42,-121,,10,-20,0,\n")
        with self.assertRaisesRegex(MetadataError, "not a number"):
            self.loader("0,42,-121,sixty,10,-20,0,\n")
        with self.assertRaisesRegex(MetadataError, "no telemetry samples"):
            self.loader("")

    def test_wrap_degrees(self):
        self.assertEqual(wrap_degrees(190.0), -170.0)
        self.assertEqual(wrap_degrees(-190.0), 170.0)
        self.assertEqual(wrap_degrees(180.0), 180.0)
        self.assertEqual(wrap_degrees(540.0), 180.0)
        self.assertEqual(wrap_degrees(27.6), 27.6)


if __name__ == "__main__":
    unittest.main()
