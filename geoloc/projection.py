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
    - Vertices at, above or near the horizon are marked unreliable instead of
      being projected to huge, attitude-sensitive distances (ReliabilityLimits).

Frames:
    camera  x right, y down, z along the optical axis (OpenCV convention)
    gimbal  x forward, y right, z down
    NED     x north, y east, z down; yaw clockwise from north, pitch negative
            looking down (-90° = nadir), roll positive right-side down.
            Angles are absolute (earth frame), as DJI reports gimbal angles.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

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


# ==================== Reliability near the horizon ====================
VERTEX_OK = "ok"
ABOVE_HORIZON = "above_horizon"              # ray never reaches the ground
BELOW_MIN_DEPRESSION = "below_min_depression"
BEYOND_MAX_RANGE = "beyond_max_range"


@dataclass(frozen=True)
class ReliabilityLimits:
    """When a vertex counts as too far / too shallow to project.

    A ray depressed θ below the horizon from height h lands h / tan θ away, and an
    attitude error dθ moves it by h · dθ / sin²θ. At h = 100 m that is 58 m per
    degree of error at θ = 10°, 230 m at 5° and 5.7 km at 1°, so shallow rays are
    marked unreliable instead of projected. None disables a limit."""
    min_depression_deg: Optional[float] = 10.0
    max_range_m: Optional[float] = None


def depression_deg(rays_ned: np.ndarray) -> np.ndarray:
    """Angle of each NED ray below the horizontal plane (negative above it)."""
    rays = np.asarray(rays_ned, dtype=np.float64).reshape(-1, 3)
    return np.degrees(np.arctan2(rays[:, 2], np.hypot(rays[:, 0], rays[:, 1])))


def reliable_radius_m(altitude_agl_m: float, limits: ReliabilityLimits) -> Optional[float]:
    """On flat ground, "depression >= min" and "range <= max" are both a disk around the
    drone's ground position. Returns the radius of their intersection (None if unbounded)."""
    radii = []
    if limits.min_depression_deg is not None and limits.min_depression_deg > 0:
        radii.append(altitude_agl_m / math.tan(math.radians(limits.min_depression_deg)))
    if limits.max_range_m is not None:
        radii.append(limits.max_range_m)
    return min(radii) if radii else None


# ==================== Polygon clipping ====================
DISK_SIDES = 360   # inscribed 360-gon: relative area error ~5e-5


def clip_polygon_halfplane(pts: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Sutherland–Hodgman: keep the part of the polygon where a·x + b·y + c >= 0."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    out = []
    for i in range(len(pts)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        fp, fq = a * p[0] + b * p[1] + c, a * q[0] + b * q[1] + c
        if fp >= 0:
            out.append(p)
        if (fp >= 0) != (fq >= 0):
            out.append(p + (q - p) * (fp / (fp - fq)))
    return np.array(out).reshape(-1, 2)


def clip_polygon_disk(pts: np.ndarray, radius: float, sides: int = DISK_SIDES) -> np.ndarray:
    """Clip a polygon (any winding, may be concave) to a disk centred on the origin,
    approximated by an inscribed regular polygon. A concave polygon whose clipped
    part is disconnected comes back joined by zero-area bridges, so its area and
    centroid stay correct."""
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    disk = radius * np.column_stack((np.cos(angles), np.sin(angles)))     # counter-clockwise
    out = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    for i in range(sides):
        if len(out) == 0:
            break
        p, q = disk[i], disk[(i + 1) % sides]
        a, b = -(q[1] - p[1]), q[0] - p[0]                                 # inside = left of p→q
        out = clip_polygon_halfplane(out, a, b, -(a * p[0] + b * p[1]))
    return out


def polygon_area(pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


# ==================== Contours ====================
@dataclass
class ProjectedContour:
    # Per ORIGINAL vertex (diagnostics):
    latlon: np.ndarray                       # (N, 2) [lat, lon]; NaN rows unless vertex_status is "ok"
    north_east: np.ndarray                   # (N, 2) raw ground hit; NaN only above the horizon
    ground_range_m: np.ndarray               # (N,) horizontal distance from the drone's ground position
    depression_deg: np.ndarray               # (N,) ray angle below the horizon
    vertex_status: List[str]                 # VERTEX_OK or the reason the vertex is unreliable
    # The reliable polygon (the original if every vertex is ok, else clipped); None if empty:
    polygon_north_east: Optional[np.ndarray]
    polygon_latlon: Optional[np.ndarray]
    polygon_clip_vertex: List[bool]          # True for vertices created by clipping
    clipped: bool
    removed_area_fraction: float             # of the original polygon, measured in image pixels
    area_m2: Optional[float]
    centroid_latlon: Optional[Tuple[float, float]]

    @property
    def n_unreliable(self) -> int:
        return sum(s != VERTEX_OK for s in self.vertex_status)


def classify_vertices(ground_range_m: np.ndarray, depression: np.ndarray,
                      limits: ReliabilityLimits) -> List[str]:
    status = []
    for rng, dep in zip(ground_range_m, depression):
        if np.isnan(rng):
            status.append(ABOVE_HORIZON)
        elif limits.min_depression_deg is not None and dep < limits.min_depression_deg:
            status.append(BELOW_MIN_DEPRESSION)
        elif limits.max_range_m is not None and rng > limits.max_range_m:
            status.append(BEYOND_MAX_RANGE)
        else:
            status.append(VERTEX_OK)
    return status


def clip_to_reliable(points_px: np.ndarray, intr: Intrinsics, pose: CameraPose,
                     radius_m: float) -> np.ndarray:
    """Ground polygon (N, 2) of the part of an image polygon whose rays reach flat
    ground within radius_m of the drone's ground position, i.e. whose depression and
    range are within limits. Empty array if there is no such part.

    1. In undistorted image coordinates the ray's down component is linear,
       g(x, y) = R[2] · (x, y, 1), so "below the horizon" is a half-plane. Every
       reliable ray has g >= sin(atan(h / radius)), so clipping at half that value
       drops everything at, above or grazing the horizon without touching the
       reliable part.
    2. The rest maps to finite ground points, and straight image edges stay straight
       on flat ground (homography), so the polygon is mapped vertex by vertex.
    3. Clip to the disk of reliable range."""
    h = pose.altitude_agl_m
    xy = pixels_to_normalized(points_px, intr)
    R = camera_to_ned(pose)
    eps = 0.5 * math.sin(math.atan2(h, radius_m))
    xy = clip_polygon_halfplane(xy, R[2, 0], R[2, 1], R[2, 2] - eps)
    if len(xy) < 3:
        return np.empty((0, 2))
    rays = np.column_stack((xy, np.ones(len(xy)))) @ R.T
    ground = FLAT_TERRAIN.intersect(h, rays)
    return clip_polygon_disk(ground, radius_m)


def project_contour(points_px: np.ndarray, intr: Intrinsics, pose: CameraPose,
                    lat0: float, lon0: float,
                    terrain: Terrain = FLAT_TERRAIN,
                    limits: ReliabilityLimits = ReliabilityLimits()) -> ProjectedContour:
    """Project a contour to the ground. If every vertex is reliable the polygon is used
    as is. Otherwise it is clipped to its reliable part (rays at least
    limits.min_depression_deg below the horizon and within limits.max_range_m), and
    centroid/area come from the clipped polygon. Centroid/area are computed in ground
    meters (the image-space centroid is not the ground centroid for oblique views)."""
    if pose.altitude_agl_m <= 0:
        raise ValueError(f"altitude_agl_m must be positive, got {pose.altitude_agl_m}")
    points_px = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    rays = pixels_to_rays_ned(points_px, intr, pose)
    ne = terrain.intersect(pose.altitude_agl_m, rays)
    ground_range = np.hypot(ne[:, 0], ne[:, 1])
    depression = depression_deg(rays)
    status = classify_vertices(ground_range, depression, limits)

    ok = np.array([s == VERTEX_OK for s in status], dtype=bool)
    lat, lon = offset_to_latlon(lat0, lon0, ne[:, 0], ne[:, 1])
    latlon = np.column_stack((lat, lon))
    latlon[~ok] = np.nan
    diagnostics = (latlon, ne, ground_range, depression, status)

    if ok.all():
        polygon, clip_vertex, clipped, removed = ne, [False] * len(ne), False, 0.0
    else:
        radius = reliable_radius_m(pose.altitude_agl_m, limits)
        if radius is None:      # no limits: the part below the horizon is unbounded
            polygon = np.empty((0, 2))
        else:
            if not isinstance(terrain, FlatTerrain):
                raise NotImplementedError("clipping to the reliable part assumes flat terrain")
            polygon = clip_to_reliable(points_px, intr, pose, radius)
        if len(polygon) < 3 or polygon_area(polygon) < 1e-6:
            return ProjectedContour(*diagnostics, None, None, [], True, 1.0, None, None)
        # Vertices matching an original reliable vertex are kept; the rest were made by clipping.
        originals = ne[ok]
        clip_vertex = [not (len(originals) > 0 and np.hypot(*(originals - v).T).min() < 1e-6)
                       for v in polygon]
        clipped = True
        # The removed part's ground area is unbounded or unreliable by definition, so the
        # removed fraction is measured in image pixels.
        kept_px = ground_to_pixels(polygon, intr, pose)
        removed = max(0.0, 1.0 - polygon_area(kept_px) / polygon_area(points_px))

    if len(polygon) < 3:
        return ProjectedContour(*diagnostics, None, None, [], clipped, removed, None, None)
    area, (cn, ce) = polygon_area_centroid(polygon)
    plat, plon = offset_to_latlon(lat0, lon0, polygon[:, 0], polygon[:, 1])
    clat, clon = offset_to_latlon(lat0, lon0, cn, ce)
    return ProjectedContour(*diagnostics, polygon, np.column_stack((plat, plon)), clip_vertex,
                            clipped, removed, area, (float(clat), float(clon)))


# ==================== Fire pixels ====================
def pixel_runs_ground_area(runs, cell_w: float, cell_h: float, intr: Intrinsics, pose: CameraPose,
                           terrain: Terrain = FLAT_TERRAIN,
                           limits: ReliabilityLimits = ReliabilityLimits()) -> Tuple[float, float]:
    """Ground area of fire pixels given as runs [row, first_col, last_col] on a grid
    whose cell (col, row) covers image pixels [col·cell_w, (col+1)·cell_w] x
    [row·cell_h, (row+1)·cell_h]. A run is a rectangle in the image and, on flat
    ground, straight lines stay straight, so the quadrilateral of its projected corners
    is its exact footprint.

    A run counts only if all four corners are reliable (same per-vertex rule as
    contours). Returns (area_m2 of the reliable runs, fraction of pixels excluded)."""
    runs = np.asarray(runs, dtype=np.float64).reshape(-1, 3)
    if len(runs) == 0:
        return 0.0, 0.0
    row, first, last = runs[:, 0], runs[:, 1], runs[:, 2] + 1.0
    corners = np.stack([np.column_stack((first, row)), np.column_stack((last, row)),
                        np.column_stack((last, row + 1.0)), np.column_stack((first, row + 1.0))], axis=1)
    rays = pixels_to_rays_ned((corners * [cell_w, cell_h]).reshape(-1, 2), intr, pose)
    ne = terrain.intersect(pose.altitude_agl_m, rays)
    status = classify_vertices(np.hypot(ne[:, 0], ne[:, 1]), depression_deg(rays), limits)
    ok = np.array([s == VERTEX_OK for s in status]).reshape(-1, 4).all(axis=1)

    quads = ne.reshape(-1, 4, 2)
    n, e = quads[..., 0], quads[..., 1]
    area = 0.5 * np.abs((e * np.roll(n, -1, axis=1) - np.roll(e, -1, axis=1) * n).sum(axis=1))
    pixels = runs[:, 2] - runs[:, 1] + 1.0
    return float(area[ok].sum()), float(pixels[~ok].sum() / pixels.sum())
