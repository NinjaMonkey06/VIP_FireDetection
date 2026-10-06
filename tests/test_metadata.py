"""Tests for geoloc/metadata.py using synthetic JPEGs and CSVs written to a temp dir."""

import os
import struct
import tempfile
import unittest
from datetime import datetime

from PIL import Image

from geoloc.camera import camera_config_from_dict
from geoloc.metadata import (
    EXIF_IFD, GPS_IFD, ExifXmpLoader, MetadataError, MissingMetadataError,
    SidecarCsvLoader, dms_to_decimal,
)
from geoloc.projection import CameraPose

DJI_FIELDS = {
    "altitude_agl_m":   "xmp:drone-dji:RelativeAltitude",
    "gimbal_yaw_deg":   "xmp:drone-dji:GimbalYawDegree",
    "gimbal_pitch_deg": "xmp:drone-dji:GimbalPitchDegree",
    "gimbal_roll_deg":  "xmp:drone-dji:GimbalRollDegree",
    "timestamp":        "exif:DateTimeOriginal",
}


def camera(fields=None, altitude_reference="takeoff"):
    return camera_config_from_dict({
        "name": "test cam", "calibration_width": 64, "calibration_height": 48,
        "fov_deg": 80.0, "fov_type": "diagonal", "altitude_reference": altitude_reference,
        "metadata_fields": DJI_FIELDS if fields is None else fields,
    })


def xmp_packet(attrs: dict, elements: dict = None) -> bytes:
    a = " ".join(f'{k}="{v}"' for k, v in attrs.items())
    e = "".join(f"<{k}>{v}</{k}>" for k, v in (elements or {}).items())
    return (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF><rdf:Description '
            f'xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" {a}>{e}'
            f'</rdf:Description></rdf:RDF></x:xmpmeta>').encode()


def write_jpeg(path, gps=True, xmp=None, datetime_original="2026:10:06 12:34:56"):
    exif = Image.Exif()
    if gps:   # 34° 9' 18" S, 117° 51' 0" W
        exif[GPS_IFD] = {1: "S", 2: (34.0, 9.0, 18.0), 3: "W", 4: (117.0, 51.0, 0.0)}
    if datetime_original:
        exif[EXIF_IFD] = {36867: datetime_original}
    Image.new("L", (64, 48)).save(path, "JPEG", exif=exif)
    if xmp is not None:   # insert an APP1 XMP segment right after SOI
        payload = b"http://ns.adobe.com/xap/1.0/\x00" + xmp
        with open(path, "rb") as f:
            data = f.read()
        with open(path, "wb") as f:
            f.write(data[:2] + b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload + data[2:])


DJI_XMP = {"drone-dji:RelativeAltitude": "+100.20", "drone-dji:GimbalYawDegree": "-45.30",
           "drone-dji:GimbalPitchDegree": "-89.90", "drone-dji:GimbalRollDegree": "+0.00"}


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name):
        return os.path.join(self.tmp, name)


class TestExifXmpLoader(TempDirTest):
    def test_reads_gps_xmp_and_timestamp(self):
        p = self.path("dji.jpg")
        write_jpeg(p, xmp=xmp_packet(DJI_XMP))
        m = ExifXmpLoader(camera()).load("dji", p)
        self.assertAlmostEqual(m.latitude_deg, -(34 + 9 / 60 + 18 / 3600), places=9)
        self.assertAlmostEqual(m.longitude_deg, -(117 + 51 / 60), places=9)
        self.assertEqual((m.altitude_agl_m, m.gimbal_yaw_deg, m.gimbal_pitch_deg, m.gimbal_roll_deg),
                         (100.2, -45.3, -89.9, 0.0))
        self.assertEqual(m.timestamp, datetime(2026, 10, 6, 12, 34, 56))
        self.assertEqual(m.altitude_reference, "takeoff")
        self.assertEqual(m.pose(), CameraPose(100.2, -45.3, -89.9, 0.0))

    def test_xmp_element_syntax(self):
        p = self.path("elem.jpg")
        attrs = dict(DJI_XMP)
        yaw = attrs.pop("drone-dji:GimbalYawDegree")
        write_jpeg(p, xmp=xmp_packet(attrs, {"drone-dji:GimbalYawDegree": yaw}))
        self.assertEqual(ExifXmpLoader(camera()).load("elem", p).gimbal_yaw_deg, -45.3)

    def test_const_fields(self):
        p = self.path("const.jpg")
        write_jpeg(p, xmp=xmp_packet({"drone-dji:RelativeAltitude": "80"}))
        fields = {"altitude_agl_m": "xmp:drone-dji:RelativeAltitude", "gimbal_yaw_deg": "const:90",
                  "gimbal_pitch_deg": "const:-90", "gimbal_roll_deg": "const:0"}
        m = ExifXmpLoader(camera(fields)).load("const", p)
        self.assertEqual((m.altitude_agl_m, m.gimbal_yaw_deg, m.gimbal_pitch_deg), (80.0, 90.0, -90.0))
        self.assertIsNone(m.timestamp)

    def test_missing_field_raises_instead_of_defaulting(self):
        p = self.path("noyaw.jpg")
        attrs = {k: v for k, v in DJI_XMP.items() if k != "drone-dji:GimbalYawDegree"}
        write_jpeg(p, xmp=xmp_packet(attrs))
        with self.assertRaisesRegex(MissingMetadataError, "gimbal_yaw_deg"):
            ExifXmpLoader(camera()).load("noyaw", p)

    def test_missing_altitude_raises(self):
        p = self.path("noalt.jpg")
        write_jpeg(p, xmp=xmp_packet({k: v for k, v in DJI_XMP.items() if "Altitude" not in k}))
        with self.assertRaisesRegex(MissingMetadataError, "altitude_agl_m"):
            ExifXmpLoader(camera()).load("noalt", p)

    def test_missing_gps_raises(self):
        p = self.path("nogps.jpg")
        write_jpeg(p, gps=False, xmp=xmp_packet(DJI_XMP))
        with self.assertRaises(MissingMetadataError):
            ExifXmpLoader(camera()).load("nogps", p)

    def test_no_xmp_at_all_raises(self):
        p = self.path("plain.jpg")
        write_jpeg(p)
        with self.assertRaises(MissingMetadataError):
            ExifXmpLoader(camera()).load("plain", p)

    def test_config_must_map_required_fields(self):
        with self.assertRaisesRegex(MetadataError, "gimbal_roll_deg"):
            ExifXmpLoader(camera({k: v for k, v in DJI_FIELDS.items() if k != "gimbal_roll_deg"}))

    def test_config_rejects_unknown_source(self):
        with self.assertRaises(MetadataError):
            ExifXmpLoader(camera({**DJI_FIELDS, "gimbal_yaw_deg": "drone-dji:GimbalYawDegree"}))

    def test_dms_with_tuple_rationals(self):
        self.assertAlmostEqual(dms_to_decimal(((34, 1), (9, 1), (1800, 100)), b"N"), 34.155, places=9)


class TestSidecarCsvLoader(TempDirTest):
    HEADER = "frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg,gimbal_roll_deg,timestamp\n"

    def write(self, body, header=HEADER):
        p = self.path("meta.csv")
        with open(p, "w", encoding="utf-8") as f:
            f.write(header + body)
        return p

    def test_reads_rows(self):
        p = self.write("f1,34.15,-117.85,80,90,-90,0,2026-10-06T12:00:00\nf2,34.16,-117.84,82.5,-10,-60,1.5,\n")
        loader = SidecarCsvLoader(p)
        m1, m2 = loader.load("f1"), loader.load("f2")
        self.assertEqual((m1.latitude_deg, m1.longitude_deg, m1.altitude_agl_m), (34.15, -117.85, 80.0))
        self.assertEqual(m1.timestamp, datetime(2026, 10, 6, 12, 0, 0))
        self.assertEqual(m2.pose(), CameraPose(82.5, -10.0, -60.0, 1.5))
        self.assertIsNone(m2.timestamp)
        self.assertEqual(m1.altitude_reference, "agl")

    def test_unknown_frame_raises(self):
        with self.assertRaises(MissingMetadataError):
            SidecarCsvLoader(self.write("f1,34.15,-117.85,80,90,-90,0,\n")).load("f9")

    def test_empty_cell_raises(self):
        with self.assertRaisesRegex(MissingMetadataError, "gimbal_roll_deg"):
            SidecarCsvLoader(self.write("f1,34.15,-117.85,80,90,-90,,\n")).load("f1")

    def test_missing_column_rejected(self):
        with self.assertRaises(MetadataError):
            SidecarCsvLoader(self.write("f1,34.15,-117.85,80,90,-90\n",
                                        header="frame_id,latitude,longitude,altitude_agl_m,gimbal_yaw_deg,gimbal_pitch_deg\n"))

    def test_duplicate_frame_rejected(self):
        with self.assertRaises(MetadataError):
            SidecarCsvLoader(self.write("f1,34.15,-117.85,80,90,-90,0,\nf1,34.15,-117.85,80,90,-90,0,\n"))

    def test_invalid_values_rejected(self):
        loader = SidecarCsvLoader(self.write("bad_lat,95,-117.85,80,90,-90,0,\nbad_alt,34,-117,0,90,-90,0,\n"))
        for fid in ("bad_lat", "bad_alt"):
            with self.subTest(fid=fid), self.assertRaises(MetadataError):
                loader.load(fid)


if __name__ == "__main__":
    unittest.main()
