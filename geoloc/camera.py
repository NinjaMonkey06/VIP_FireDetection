"""
camera.py — Camera-specific configuration (intrinsics, distortion, metadata fields)
===================================================================================
Every camera-specific value lives in a JSON config so the geolocation code stays
camera-agnostic. See camera_configs/template.json for all fields.

Intrinsics are given either explicitly (fx, fy, cx, cy at the calibration
resolution) or as an FOV with an explicit type — horizontal, vertical or
diagonal — because manufacturers usually publish diagonal FOV and treating it
as horizontal inflates ground distances (see tests/test_citrus_oracle.py).

Keys starting with "_" are treated as comments and ignored.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from geoloc.projection import Intrinsics

FOV_TYPES = ("horizontal", "vertical", "diagonal")
RESIZE_ROUNDING_PX = 1.0   # a pure resize can only change the aspect ratio by rounding


@dataclass(frozen=True)
class Crop:
    """Region of the calibrated frame that the processed image actually covers."""
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class CameraConfig:
    name: str
    calibration_width: int
    calibration_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Tuple[float, ...] = ()
    crop: Optional[Crop] = None
    altitude_reference: str = "agl"          # what the altitude metadata field is relative to
    metadata_fields: Dict[str, str] = field(default_factory=dict)

    def intrinsics_for(self, width: int, height: int) -> Intrinsics:
        """Intrinsics for an image of the given size: apply the crop (if any),
        then scale to the actual resolution. Distortion coefficients act on
        normalized coordinates, so they are unchanged."""
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        base_w, base_h = self.calibration_width, self.calibration_height
        if self.crop is not None:
            cx, cy = cx - self.crop.x, cy - self.crop.y
            base_w, base_h = self.crop.width, self.crop.height

        if abs(height - width * base_h / base_w) > RESIZE_ROUNDING_PX:
            raise ValueError(
                f"{self.name}: image {width}x{height} does not match the aspect ratio of "
                f"the {'cropped' if self.crop else 'calibrated'} frame {base_w}x{base_h}, so it is "
                f"not a plain resize. If this image is a crop of the sensor (e.g. 16:9 video from a 4:3 "
                f"sensor, or a crop made while aligning thermal to RGB), add a 'crop' entry to the camera config.")

        sx, sy = width / base_w, height / base_h
        return Intrinsics(width, height, fx * sx, fy * sy, cx * sx, cy * sy, self.dist_coeffs)


def focal_from_fov(fov_deg: float, fov_type: str, width: int, height: int) -> float:
    """Focal length in pixels (square pixels) from an FOV of a given type."""
    if fov_type not in FOV_TYPES:
        raise ValueError(f"fov_type must be one of {FOV_TYPES}, got {fov_type!r}")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    span = {"horizontal": width, "vertical": height, "diagonal": math.hypot(width, height)}[fov_type]
    return (span / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def camera_config_from_dict(d: dict) -> CameraConfig:
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    name = d.get("name", "unnamed camera")
    try:
        W, H = int(d["calibration_width"]), int(d["calibration_height"])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{name}: calibration_width and calibration_height must be set to integers")

    explicit = [d.get(k) for k in ("fx", "fy")]
    has_explicit = any(v is not None for v in explicit)
    has_fov = d.get("fov_deg") is not None
    if has_explicit == has_fov:
        raise ValueError(f"{name}: give either fx/fy (explicit intrinsics) or fov_deg + fov_type, not both or neither")

    if has_explicit:
        if None in explicit:
            raise ValueError(f"{name}: explicit intrinsics need both fx and fy")
        fx, fy = float(d["fx"]), float(d["fy"])
    else:
        if d.get("fov_type") is None:
            raise ValueError(f"{name}: fov_type is required with fov_deg ({'/'.join(FOV_TYPES)})")
        fx = fy = focal_from_fov(float(d["fov_deg"]), d["fov_type"], W, H)

    cx = float(d["cx"]) if d.get("cx") is not None else W / 2.0
    cy = float(d["cy"]) if d.get("cy") is not None else H / 2.0

    crop = None
    if d.get("crop") is not None:
        c = d["crop"]
        crop = Crop(int(c["x"]), int(c["y"]), int(c["width"]), int(c["height"]))
        if crop.x < 0 or crop.y < 0 or crop.x + crop.width > W or crop.y + crop.height > H:
            raise ValueError(f"{name}: crop {crop} lies outside the {W}x{H} calibrated frame")

    return CameraConfig(
        name=name, calibration_width=W, calibration_height=H,
        fx=fx, fy=fy, cx=cx, cy=cy,
        dist_coeffs=tuple(float(c) for c in d.get("dist_coeffs") or ()),
        crop=crop,
        altitude_reference=d.get("altitude_reference", "agl"),
        metadata_fields=dict(d.get("metadata_fields") or {}),
    )


def load_camera_config(path: str) -> CameraConfig:
    with open(path, encoding="utf-8") as f:
        return camera_config_from_dict(json.load(f))
