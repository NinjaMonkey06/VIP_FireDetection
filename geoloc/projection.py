"""
projection.py — Pixel → ground → GPS geometry (pure math, no I/O, no rendering)
================================================================================
Back-projects image pixels to flat ground using a pinhole camera model and the
gimbal attitude, then converts local north/east meters to lat/lon.

Portions adapted from https://github.com/afif1230/citrus-uav-pipeline
(Geotagging_System.ipynb, commit 06066f2), Copyright (c) 2026 afif1230,
MIT License — see THIRD_PARTY_LICENSES/citrus-uav-pipeline.txt.

Differences from the reference:
    - The image is never rotated. Yaw/pitch/roll rotate the viewing ray, so fx
      does not depend on a rotated canvas size (see tests/test_citrus_oracle.py).
    - Intrinsics are explicit (fx, fy, cx, cy, distortion) instead of fx being
      derived from an FOV of unspecified type.
    - General ray casting: any gimbal pitch, not only straight down. At
      pitch = -90° and roll = 0 it reduces exactly to the reference nadir math.
    - Ground intersection is behind a Terrain interface (flat ground for now).

Frames:
    camera  x right, y down, z along the optical axis (OpenCV convention)
    gimbal  x forward, y right, z down
    NED     x north, y east, z down; yaw clockwise from north, pitch negative
            looking down (-90° = nadir), roll positive right-side down.
            Angles are absolute (earth frame), as DJI reports gimbal angles.
"""

import math
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

EARTH_RADIUS_M = 6378137.0   # WGS84 equatorial radius, spherical approximation (as in reference)

# camera (right, down, optical) → gimbal (forward, right, down)
CAM_TO_GIMBAL = np.array([[0.0, 0.0, 1.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0]])


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for one specific image resolution."""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: Tuple[float, ...] = ()   # OpenCV order: k1, k2, p1, p2[, k3...]

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]])

    @property
    def has_distortion(self) -> bool:
        return any(c != 0.0 for c in self.dist_coeffs)


@dataclass(frozen=True)
class CameraPose:
    """Camera height above the ground directly below it, and absolute gimbal angles."""
    altitude_agl_m: float
    yaw_deg: float
    pitch_deg: float = -90.0
    roll_deg: float = 0.0


# ==================== Rotations ====================
def gimbal_to_ned(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Rotation matrix taking gimbal-frame vectors to NED: Rz(yaw) · Ry(pitch) · Rx(roll)."""
    y, p, r = (math.radians(a) for a in (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def camera_to_ned(pose: CameraPose) -> np.ndarray:
    return gimbal_to_ned(pose.yaw_deg, pose.pitch_deg, pose.roll_deg) @ CAM_TO_GIMBAL


# ==================== Pixels ↔ normalized camera coordinates ====================
def pixels_to_normalized(points_px: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """(N, 2) pixels → (N, 2) undistorted normalized coords ((u-cx)/fx, (v-cy)/fy)."""
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if not intr.has_distortion:
        return np.column_stack(((pts[:, 0] - intr.cx) / intr.fx,
                                (pts[:, 1] - intr.cy) / intr.fy))
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    out = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), intr.matrix(),
                                  np.asarray(intr.dist_coeffs, dtype=np.float64),
                                  None, None, criteria)
    return out.reshape(-1, 2)


def normalized_to_pixels(xy: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """(N, 2) undistorted normalized coords → (N, 2) pixels, applying distortion."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if not intr.has_distortion:
        return np.column_stack((xy[:, 0] * intr.fx + intr.cx,
                                xy[:, 1] * intr.fy + intr.cy))
    obj = np.column_stack((xy, np.ones(len(xy))))
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), intr.matrix(),
                               np.asarray(intr.dist_coeffs, dtype=np.float64))
    return out.reshape(-1, 2)


def pixels_to_rays_ned(points_px: np.ndarray, intr: Intrinsics, pose: CameraPose) -> np.ndarray:
    """(N, 2) pixels → (N, 3) unnormalized NED ray directions."""
    xy = pixels_to_normalized(points_px, intr)
    rays_cam = np.column_stack((xy, np.ones(len(xy))))
    return rays_cam @ camera_to_ned(pose).T


# ==================== Terrain ====================
class Terrain(Protocol):
    def intersect(self, camera_height_m: float, rays_ned: np.ndarray) -> np.ndarray:
        """Ground hit of each ray as (N, 2) [north_m, east_m] relative to the point
        directly below the camera; NaN rows where the ray never reaches the ground."""
        ...


class FlatTerrain:
    """Horizontal ground plane camera_height_m below the camera."""

    def intersect(self, camera_height_m: float, rays_ned: np.ndarray) -> np.ndarray:
        rays = np.asarray(rays_ned, dtype=np.float64).reshape(-1, 3)
        out = np.full((len(rays), 2), np.nan)
        hits = rays[:, 2] > 1e-12                      # ray must point downward
        t = camera_height_m / rays[hits, 2]
        out[hits] = rays[hits, :2] * t[:, None]
        return out


FLAT_TERRAIN = FlatTerrain()


# ==================== Pixels ↔ ground offsets ====================
def pixels_to_ground(points_px: np.ndarray, intr: Intrinsics, pose: CameraPose,
                     terrain: Terrain = FLAT_TERRAIN) -> np.ndarray:
    """(N, 2) pixels → (N, 2) [north_m, east_m] from the drone's ground position.
    NaN rows for pixels at or above the horizon."""
    if pose.altitude_agl_m <= 0:
        raise ValueError(f"altitude_agl_m must be positive, got {pose.altitude_agl_m}")
    return terrain.intersect(pose.altitude_agl_m, pixels_to_rays_ned(points_px, intr, pose))


def ground_to_pixels(north_east: np.ndarray, intr: Intrinsics, pose: CameraPose) -> np.ndarray:
    """Inverse of pixels_to_ground on flat terrain. (N, 2) [north_m, east_m] → (N, 2)
    pixels; NaN rows for points behind the camera."""
    ne = np.asarray(north_east, dtype=np.float64).reshape(-1, 2)
    rays_ned = np.column_stack((ne, np.full(len(ne), pose.altitude_agl_m)))
    rays_cam = rays_ned @ camera_to_ned(pose)          # R^T applied row-wise
    out = np.full((len(ne), 2), np.nan)
    front = rays_cam[:, 2] > 1e-12
    xy = rays_cam[front, :2] / rays_cam[front, 2:3]
    out[front] = normalized_to_pixels(xy, intr)
    return out


def ground_sample_distance_m(intr: Intrinsics, pose: CameraPose,
                             terrain: Terrain = FLAT_TERRAIN) -> float:
    """Ground distance spanned by one pixel step (horizontal) at the principal point."""
    ne = pixels_to_ground(np.array([[intr.cx, intr.cy], [intr.cx + 1.0, intr.cy]]),
                          intr, pose, terrain)
    return float(np.hypot(*(ne[1] - ne[0])))


# ==================== Local meters ↔ lat/lon ====================
def offset_to_latlon(lat0: float, lon0: float, north_m, east_m):
    """Local north/east meters around (lat0, lon0) → lat/lon (equirectangular)."""
    north_m = np.asarray(north_m, dtype=np.float64)
    east_m = np.asarray(east_m, dtype=np.float64)
    lat = lat0 + np.degrees(north_m / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat0))))
    return lat, lon


def latlon_to_offset(lat0: float, lon0: float, lat, lon):
    """Exact inverse of offset_to_latlon (uses the base latitude for the east scale)."""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    north = np.radians(lat - lat0) * EARTH_RADIUS_M
    east = np.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    return north, east


# ==================== Polygons ====================
def polygon_area_centroid(north_east: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    """Shoelace area (m², always positive) and area-weighted centroid (north, east).
    Falls back to the vertex mean for degenerate (zero-area) polygons."""
    ne = np.asarray(north_east, dtype=np.float64).reshape(-1, 2)
    n, e = ne[:, 0], ne[:, 1]
    n2, e2 = np.roll(n, -1), np.roll(e, -1)
    cross = e * n2 - e2 * n
    signed_area = 0.5 * cross.sum()
    if abs(signed_area) < 1e-9:
        return 0.0, (float(n.mean()), float(e.mean()))
    cn = ((n + n2) * cross).sum() / (6.0 * signed_area)
    ce = ((e + e2) * cross).sum() / (6.0 * signed_area)
    return float(abs(signed_area)), (float(cn), float(ce))


@dataclass
class ProjectedContour:
    latlon: np.ndarray                       # (N, 2) [lat, lon]; NaN rows for invalid vertices
    north_east: np.ndarray                   # (N, 2) meters from the drone's ground position
    n_invalid: int                           # vertices at/above the horizon
    area_m2: Optional[float]                 # None if any vertex is invalid
    centroid_latlon: Optional[Tuple[float, float]]


def project_contour(points_px: np.ndarray, intr: Intrinsics, pose: CameraPose,
                    lat0: float, lon0: float,
                    terrain: Terrain = FLAT_TERRAIN) -> ProjectedContour:
    """Project every contour vertex to the ground; centroid/area computed in ground
    meters (the image-space centroid is not the ground centroid for oblique views)."""
    ne = pixels_to_ground(points_px, intr, pose, terrain)
    invalid = np.isnan(ne).any(axis=1)
    lat, lon = offset_to_latlon(lat0, lon0, ne[:, 0], ne[:, 1])
    latlon = np.column_stack((lat, lon))
    if invalid.any() or len(ne) < 3:
        return ProjectedContour(latlon, ne, int(invalid.sum()), None, None)
    area, (cn, ce) = polygon_area_centroid(ne)
    clat, clon = offset_to_latlon(lat0, lon0, cn, ce)
    return ProjectedContour(latlon, ne, 0, area, (float(clat), float(clon)))
