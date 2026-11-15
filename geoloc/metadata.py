"""
metadata.py — Per-frame metadata behind a single loader interface
==================================================================
Every loader returns a FrameMetadata (drone position, altitude, absolute gimbal
angles, timestamp) for a frame, so the source can be swapped — EXIF/XMP today,
a telemetry log later — without touching the geolocation code.

Loaders never fill in guessed values: a missing field raises MissingMetadataError.

EXIF GPS parsing (ratio_to_float, dms_to_decimal) adapted from
https://github.com/afif1230/citrus-uav-pipeline (Geotagging_System.ipynb,
commit 06066f2), Copyright (c) 2026 afif1230, MIT License — see
THIRD_PARTY_LICENSES/citrus-uav-pipeline.txt.
"""

import bisect
import csv
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, Protocol

from PIL import ExifTags, Image

from geoloc.camera import CameraConfig
from geoloc.projection import CameraPose

GPS_IFD = 0x8825
EXIF_IFD = 0x8769
REQUIRED_FIELDS = ("altitude_agl_m", "gimbal_yaw_deg", "gimbal_pitch_deg", "gimbal_roll_deg")
LASER_FIELDS = ("laser_distance_m", "laser_drone_altitude_m", "laser_target_altitude_m")


class MetadataError(ValueError):
    pass


class MissingMetadataError(MetadataError):
    pass


@dataclass(frozen=True)
class FrameMetadata:
    latitude_deg: float
    longitude_deg: float
    altitude_agl_m: float
    gimbal_yaw_deg: float           # absolute, clockwise from north
    gimbal_pitch_deg: float         # -90 = straight down
    gimbal_roll_deg: float
    timestamp: Optional[datetime]
    altitude_reference: str         # "agl", "takeoff", "msl" or "laser_target" — what altitude_agl_m was read from
    source: str
    altitude_note: Optional[str] = None   # e.g. why a laser height was not used

    def __post_init__(self):
        if not -90.0 <= self.latitude_deg <= 90.0:
            raise MetadataError(f"{self.source}: latitude {self.latitude_deg} out of range")
        if not -180.0 <= self.longitude_deg <= 180.0:
            raise MetadataError(f"{self.source}: longitude {self.longitude_deg} out of range")
        if self.altitude_agl_m <= 0.0:
            raise MetadataError(f"{self.source}: altitude {self.altitude_agl_m} m must be positive")

    def pose(self) -> CameraPose:
        return CameraPose(self.altitude_agl_m, self.gimbal_yaw_deg,
                          self.gimbal_pitch_deg, self.gimbal_roll_deg)


class MetadataLoader(Protocol):
    def load(self, frame_id: str, image_path: Optional[str] = None) -> FrameMetadata:
        """Metadata for one frame; raises MissingMetadataError if anything required is absent."""
        ...


# ==================== EXIF / XMP ====================
def ratio_to_float(v) -> float:
    if hasattr(v, "numerator") and hasattr(v, "denominator"):
        return float(v.numerator) / float(v.denominator)
    if isinstance(v, tuple) and len(v) == 2:
        return float(v[0]) / float(v[1])
    return float(v)


def dms_to_decimal(dms, ref) -> float:
    d, m, s = (ratio_to_float(x) for x in dms[:3])
    dec = d + m / 60.0 + s / 3600.0
    if ref in (b"S", "S", b"W", "W"):
        dec = -dec
    return dec


_XMP_PACKET = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.DOTALL)
_EXIF_NAME_TO_TAG = {name: tag for tag, name in ExifTags.TAGS.items()}


def read_xmp(path: str) -> Dict[str, str]:
    """All `prefix:Name` values from the image's XMP packet, written either as
    attributes (prefix:Name="v") or as elements (<prefix:Name>v</prefix:Name>)."""
    with open(path, "rb") as f:
        match = _XMP_PACKET.search(f.read())
    if not match:
        return {}
    xmp = match.group(0).decode("utf-8", errors="replace")
    values = dict(re.findall(r'([\w.-]+:[\w.-]+)="([^"]*)"', xmp))
    values.update(re.findall(r"<([\w.-]+:[\w.-]+)>([^<]*)</\1>", xmp))
    return values


class ExifXmpLoader:
    """Reads GPS from the EXIF GPS block and everything else from the fields the
    camera config names: 'xmp:<prefix:Name>', 'exif:<TagName>' or 'const:<number>'."""

    def __init__(self, camera: CameraConfig):
        self.camera = camera
        self.fields = camera.metadata_fields
        for name in REQUIRED_FIELDS:
            if name not in self.fields:
                raise MetadataError(f"{camera.name}: metadata_fields has no entry for '{name}' "
                                    f"(use 'const:<value>' if the camera does not record it)")
        for name, spec in self.fields.items():
            if spec.split(":", 1)[0] not in ("xmp", "exif", "const"):
                raise MetadataError(f"{camera.name}: metadata_fields['{name}'] = {spec!r} must start "
                                    f"with 'xmp:', 'exif:' or 'const:'")

    def load(self, frame_id: str, image_path: Optional[str] = None) -> FrameMetadata:
        if image_path is None:
            raise MissingMetadataError(f"{frame_id}: ExifXmpLoader needs the image path")
        with Image.open(image_path) as im:
            exif = im.getexif()
        xmp = read_xmp(image_path)

        def raw(name: str):
            kind, key = self.fields[name].split(":", 1)
            if kind == "const":
                return key
            if kind == "xmp":
                return xmp.get(key)
            tag = _EXIF_NAME_TO_TAG.get(key)
            return None if tag is None else exif.get(tag, exif.get_ifd(EXIF_IFD).get(tag))

        def number(name: str) -> float:
            value = raw(name)
            if value is None or str(value).strip() == "":
                raise MissingMetadataError(f"{frame_id}: '{name}' not found ({self.fields[name]})")
            try:
                return ratio_to_float(value) if not isinstance(value, str) else float(value)
            except (TypeError, ValueError, ZeroDivisionError):
                raise MetadataError(f"{frame_id}: '{name}' = {value!r} is not a number ({self.fields[name]})")

        if "latitude" in self.fields and "longitude" in self.fields:
            lat, lon = number("latitude"), number("longitude")
        else:
            gps = exif.get_ifd(GPS_IFD)
            if not all(gps.get(k) for k in (1, 2, 3, 4)):
                raise MissingMetadataError(f"{frame_id}: no EXIF GPS latitude/longitude in {image_path}")
            lat, lon = dms_to_decimal(gps[2], gps[1]), dms_to_decimal(gps[4], gps[3])

        timestamp = None
        if "timestamp" in self.fields and raw("timestamp"):
            try:
                timestamp = datetime.strptime(str(raw("timestamp")).strip(), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                raise MetadataError(f"{frame_id}: timestamp {raw('timestamp')!r} is not 'YYYY:MM:DD HH:MM:SS'")

        altitude, reference, note = self.altitude(number)
        return FrameMetadata(
            latitude_deg=lat, longitude_deg=lon,
            altitude_agl_m=altitude,
            gimbal_yaw_deg=number("gimbal_yaw_deg"),
            gimbal_pitch_deg=number("gimbal_pitch_deg"),
            gimbal_roll_deg=number("gimbal_roll_deg"),
            timestamp=timestamp,
            altitude_reference=reference,
            source=f"exif:{image_path}",
            altitude_note=note,
        )

    def altitude(self, number):
        """(altitude_m, altitude_reference, note) for one frame; `number(name)` reads a field."""
        return number("altitude_agl_m"), self.camera.altitude_reference, None


class LaserHeightLoader(ExifXmpLoader):
    """Like ExifXmpLoader, but the height comes from the laser rangefinder when it has a
    reading: drone altitude minus the altitude of the laser's hit point. Both come
    from the same altimeter, so the vertical datum cancels. This is the height above
    whatever the laser hit at the image centre. Elsewhere in the frame the ground is
    still assumed flat at that height.

    The laser measures the first surface it hits: over trees or smoke that is above
    the ground, and the height comes out too low. On FLAME 3 (Sycan Marsh) most
    readings hit canopy or smoke 3–60 m above the ground (checked against SRTM).

    Falls back to the altitude field, with its own reference, when there is no reading
    (laser distance 0 or fields absent), or when the laser ground point lies more than
    max_terrain_offset_m above or below the altitude field's reference. The laser can
    report a hit on smoke a few metres away, or on a distant ridge. The reason for a
    fallback is in FrameMetadata.altitude_note.

    A dataset convenience: the final drone may have no rangefinder, and a terrain model
    (behind the Terrain interface) is the general fix."""

    def __init__(self, camera: CameraConfig, max_terrain_offset_m: float = 50.0):
        super().__init__(camera)
        missing = [name for name in LASER_FIELDS if name not in self.fields]
        if missing:
            raise MetadataError(f"{camera.name}: metadata_fields has no entry for {missing}, "
                                f"needed for laser heights")
        self.max_terrain_offset_m = max_terrain_offset_m

    def altitude(self, number):
        fallback, reference, _ = super().altitude(number)
        try:
            distance, drone, target = (number(name) for name in LASER_FIELDS)
        except MissingMetadataError:
            return fallback, reference, "no laser fields; used the altitude field"
        if distance <= 0.0:
            return fallback, reference, "no laser reading; used the altitude field"
        height = drone - target
        offset = fallback - height    # laser ground point relative to the altitude reference
        if height <= 0.0 or abs(offset) > self.max_terrain_offset_m:
            return fallback, reference, (f"laser height {height:.1f} m rejected: ground point "
                                         f"{offset:+.1f} m from '{reference}' exceeds "
                                         f"±{self.max_terrain_offset_m:g} m; used the altitude field")
        return height, "laser_target", None


# ==================== Sidecar CSV ====================
CSV_COLUMNS = ("frame_id", "latitude", "longitude", "altitude_agl_m",
               "gimbal_yaw_deg", "gimbal_pitch_deg", "gimbal_roll_deg")


class SidecarCsvLoader:
    """One row per frame, keyed by frame_id. Required columns: CSV_COLUMNS;
    optional 'timestamp' column in ISO 8601."""

    def __init__(self, csv_path: str, altitude_reference: str = "agl"):
        self.csv_path = csv_path
        self.altitude_reference = altitude_reference
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in CSV_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise MetadataError(f"{csv_path}: missing columns {missing}")
            self.rows = {}
            for row in reader:
                fid = row["frame_id"].strip()
                if fid in self.rows:
                    raise MetadataError(f"{csv_path}: duplicate frame_id {fid!r}")
                self.rows[fid] = row

    def load(self, frame_id: str, image_path: Optional[str] = None) -> FrameMetadata:
        row = self.rows.get(frame_id)
        if row is None:
            raise MissingMetadataError(f"{frame_id}: no row in {self.csv_path}")

        def number(col: str) -> float:
            value = (row.get(col) or "").strip()
            if not value:
                raise MissingMetadataError(f"{frame_id}: empty '{col}' in {self.csv_path}")
            try:
                return float(value)
            except ValueError:
                raise MetadataError(f"{frame_id}: '{col}' = {value!r} is not a number in {self.csv_path}")

        ts = (row.get("timestamp") or "").strip()
        try:
            timestamp = datetime.fromisoformat(ts) if ts else None
        except ValueError:
            raise MetadataError(f"{frame_id}: timestamp {ts!r} is not ISO 8601 in {self.csv_path}")
        return FrameMetadata(
            latitude_deg=number("latitude"), longitude_deg=number("longitude"),
            altitude_agl_m=number("altitude_agl_m"),
            gimbal_yaw_deg=number("gimbal_yaw_deg"),
            gimbal_pitch_deg=number("gimbal_pitch_deg"),
            gimbal_roll_deg=number("gimbal_roll_deg"),
            timestamp=timestamp,
            altitude_reference=self.altitude_reference,
            source=f"csv:{self.csv_path}",
        )


# ==================== Timed telemetry (video) ====================
TIMED_CSV_COLUMNS = ("time_s", "latitude", "longitude", "altitude_agl_m",
                     "gimbal_yaw_deg", "gimbal_pitch_deg", "gimbal_roll_deg")


def wrap_degrees(angle: float) -> float:
    """Any angle → (-180, 180]; an angle already in range is returned unchanged."""
    if -180.0 < angle <= 180.0:
        return angle
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


class TimedTelemetryLoader:
    """Telemetry sampled at its own rate (a flight log, a DJI subtitle track), matched
    to a video frame by time. One row per sample with time_s = seconds from the start
    of the video (TIMED_CSV_COLUMNS; optional 'timestamp' in ISO 8601, extra columns
    are ignored). at(time_s) interpolates linearly between the two samples around that
    time, yaw along the shorter arc. Before the first or after the last sample it holds
    that sample for up to max_gap_s. A hole in the log (two samples more than
    2 · max_gap_s apart) is never interpolated across: a frame within max_gap_s of
    either side holds that sample, as at the ends of the log, and a frame deeper in
    the hole raises MissingMetadataError."""

    def __init__(self, csv_path: str, altitude_reference: str = "agl", max_gap_s: float = 1.0):
        self.csv_path = csv_path
        self.altitude_reference = altitude_reference
        self.max_gap_s = max_gap_s
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            missing = [c for c in TIMED_CSV_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                raise MetadataError(f"{csv_path}: missing columns {missing}")
            rows = list(reader)
        if not rows:
            raise MetadataError(f"{csv_path}: no telemetry samples")

        values = {c: [] for c in TIMED_CSV_COLUMNS}
        stamps = []
        for line, row in enumerate(rows, start=2):
            for c in TIMED_CSV_COLUMNS:
                cell = (row.get(c) or "").strip()
                if not cell:
                    raise MetadataError(f"{csv_path} line {line}: empty '{c}'")
                try:
                    values[c].append(float(cell))
                except ValueError:
                    raise MetadataError(f"{csv_path} line {line}: '{c}' = {cell!r} is not a number")
            ts = (row.get("timestamp") or "").strip()
            stamps.append(datetime.fromisoformat(ts) if ts else None)

        order = sorted(range(len(rows)), key=lambda k: values["time_s"][k])
        self.t = [values["time_s"][k] for k in order]
        if any(b <= a for a, b in zip(self.t, self.t[1:])):
            raise MetadataError(f"{csv_path}: duplicate time_s values")
        self.samples = {c: [values[c][k] for k in order] for c in TIMED_CSV_COLUMNS if c != "time_s"}
        self.stamps = [stamps[k] for k in order]
        yaw = self.samples["gimbal_yaw_deg"]          # unwrapped, so interpolation crosses ±180 the short way
        unwrapped = [yaw[0]]
        for a, b in zip(yaw, yaw[1:]):
            unwrapped.append(unwrapped[-1] + wrap_degrees(b - a))
        self.samples["gimbal_yaw_deg"] = unwrapped

    def at(self, time_s: float) -> FrameMetadata:
        t = self.t
        if time_s < t[0] - self.max_gap_s or time_s > t[-1] + self.max_gap_s:
            raise MissingMetadataError(f"{time_s:.3f} s: outside the telemetry's {t[0]:.3f}-{t[-1]:.3f} s "
                                       f"(±{self.max_gap_s:g} s) in {self.csv_path}")
        i = bisect.bisect_left(t, time_s)
        if i == 0 or i == len(t):                        # before the first / after the last sample: hold it
            i0 = i1 = 0 if i == 0 else len(t) - 1
            w = 0.0
        else:
            i0, i1 = i - 1, i
            if min(time_s - t[i0], t[i1] - time_s) > self.max_gap_s:
                raise MissingMetadataError(f"{time_s:.3f} s: nearest telemetry sample is more than "
                                           f"{self.max_gap_s:g} s away (samples at {t[i0]:.3f} s and {t[i1]:.3f} s)")
            if t[i1] - t[i0] > 2.0 * self.max_gap_s:     # a hole: hold the nearer sample
                i0 = i1 = i0 if time_s - t[i0] <= t[i1] - time_s else i1
                w = 0.0
            else:
                w = (time_s - t[i0]) / (t[i1] - t[i0])
        v = {c: s[i0] + w * (s[i1] - s[i0]) for c, s in self.samples.items()}
        stamp = self.stamps[i0]
        if stamp is not None:
            stamp = stamp + timedelta(seconds=time_s - t[i0])
        return FrameMetadata(
            latitude_deg=v["latitude"], longitude_deg=v["longitude"],
            altitude_agl_m=v["altitude_agl_m"],
            gimbal_yaw_deg=wrap_degrees(v["gimbal_yaw_deg"]),
            gimbal_pitch_deg=v["gimbal_pitch_deg"],
            gimbal_roll_deg=v["gimbal_roll_deg"],
            timestamp=stamp,
            altitude_reference=self.altitude_reference,
            source=f"csv:{self.csv_path}@{time_s:.3f}s",
        )
