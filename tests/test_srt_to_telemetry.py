"""Tests for srt_to_telemetry.py on a synthetic DJI-style subtitle track."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

from srt_to_telemetry import DEFAULT_KEYS, parse_srt, telemetry_rows

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SRT = """1
00:00:00,000 --> 00:00:00,033
<font size="28">SrtCnt : 1, DiffTime : 33ms
2022-10-27 15:40:11.000
[iso : 100] [shutter : 1/1000.0] [latitude: 0.000000] [longitude: 0.000000] [rel_alt: 0.000 abs_alt: 1507.670] [gb_yaw: -110.4 gb_pitch: -18.9 gb_roll: 0.0]</font>

2
00:00:00,033 --> 00:00:00,066
<font size="28">SrtCnt : 2, DiffTime : 33ms
2022-10-27 15:40:11.033
[iso : 100] [latitude: 42.866843] [longitude: -121.134371] [rel_alt: 60.919 abs_alt: 1568.589] [gb_yaw: -110.4 gb_pitch: -18.9 gb_roll: 0.0]</font>

3
00:00:00,066 --> 00:00:00,099
<font size="28">SrtCnt : 3, DiffTime : 33ms
2022-10-27 15:40:11.066
[iso : 100] [latitude: 42.866842] [longitude: -121.134371] [rel_alt: 60.942 abs_alt: 1568.612] [gb_yaw: -114.0 gb_pitch: -17.5 gb_roll: 0.0]</font>
"""


class TestParse(unittest.TestCase):
    def test_entries_times_keys_and_timestamps(self):
        entries = parse_srt(SRT)
        self.assertEqual([e[0] for e in entries], [1, 2, 3])
        self.assertEqual([e[1] for e in entries], [0.0, 0.033, 0.066])
        self.assertEqual(entries[1][2]["rel_alt"], 60.919)
        self.assertEqual(entries[1][2]["abs_alt"], 1568.589)       # second pair in the same bracket
        self.assertEqual(entries[2][2]["gb_yaw"], -114.0)
        self.assertEqual(entries[1][3], "2022-10-27T15:40:11.033")
        self.assertNotIn("SrtCnt", entries[0][2])                  # only bracketed keys

    def test_rows_skip_entries_without_a_fix_and_apply_mapping(self):
        rows, no_fix = telemetry_rows(parse_srt(SRT), DEFAULT_KEYS, {})
        self.assertEqual(no_fix, 1)
        self.assertEqual([r["time_s"] for r in rows], [0.033, 0.066])
        self.assertEqual((rows[0]["latitude"], rows[0]["altitude_agl_m"], rows[0]["gimbal_yaw_deg"]),
                         (42.866843, 60.919, -110.4))
        self.assertEqual(rows[0]["timestamp"], "2022-10-27T15:40:11.033")
        rows, _ = telemetry_rows(parse_srt(SRT), {**DEFAULT_KEYS, "altitude_agl_m": "abs_alt"},
                                 {"gimbal_pitch_deg": -90.0})
        self.assertEqual((rows[0]["altitude_agl_m"], rows[0]["gimbal_pitch_deg"]), (1568.589, -90.0))

    def test_missing_key_names_what_is_present(self):
        with self.assertRaisesRegex(KeyError, "gb_tilt.*keys present"):
            telemetry_rows(parse_srt(SRT), {**DEFAULT_KEYS, "gimbal_pitch_deg": "gb_tilt"}, {})


class TestCommandLine(unittest.TestCase):
    def test_writes_a_timed_telemetry_csv(self):
        with tempfile.TemporaryDirectory() as d:
            srt = os.path.join(d, "flight.SRT")
            with open(srt, "w", encoding="utf-8") as f:
                f.write(SRT)
            out = os.path.join(d, "telemetry.csv")
            proc = subprocess.run([sys.executable, os.path.join(REPO, "srt_to_telemetry.py"), "--srt", srt,
                                   "--output_csv", out, "--const", "gimbal_roll_deg=0"],
                                  capture_output=True, text=True, encoding="utf-8",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as f:
                lines = f.read().splitlines()
            self.assertEqual(lines[0], "time_s,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,"
                                       "gimbal_roll_deg,timestamp,srt_index")
            self.assertEqual(len(lines), 3)
            self.assertTrue(lines[1].startswith("0.033,42.866843,-121.134371,60.919,-110.4,-18.9,0.0,"))

            keys = subprocess.run([sys.executable, os.path.join(REPO, "srt_to_telemetry.py"), "--srt", srt,
                                   "--show_keys"], capture_output=True, text=True, encoding="utf-8",
                                  env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            self.assertEqual(keys.returncode, 0, keys.stderr)
            self.assertIn("gb_pitch", keys.stdout)


if __name__ == "__main__":
    unittest.main()
