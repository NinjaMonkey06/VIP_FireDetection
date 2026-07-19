"""
geolocate.py — Phase 3: Fire Contour GPS Back-Projection + Fire Tracking
=========================================================================
Reads localization_results.json (produced by localize.py) and:

  1. Converts normalized image-space fire contour points to GPS world
     coordinates using drone telemetry and camera intrinsics.
  2. Tracks fire regions across the frame sequence and estimates a
     velocity vector (speed + heading) for each tracked fire, using
     GPS displacement of region centroids over time.
  3. Extracts a fire wavefront arc per region — using wind direction if
     provided, otherwise falling back to the fire's own estimated
     direction of movement from tracking (Option B in the original design).

Hardware assumptions (from project specification):
    Thermal camera : FLIR HADRON 640R+, 13.6mm focal length, 60Hz
    Altitude       : barometric + GPS fusion (from MAVLink v2)
    Gimbal         : real-time angle telemetry (pitch, roll, yaw) via MAVLink v2
    Telemetry bus  : MAVLink v2 over serial

Two telemetry sources are supported:
    --telemetry_log     : a real MAVLink v2 .tlog/.bin file (requires pymavlink)
    --simulate_telemetry : a synthetic constant-velocity flight path, for
                            development/testing before a real flight log exists.
                            Swap this for --telemetry_log once you have one —
                            no other pipeline code changes needed.

IMPORTANT — known gap (see README / report "Known Limitations"):
    infer.py / localize.py do not currently record a per-frame capture
    timestamp. Until that's added upstream, frame timestamps here are
    assigned sequentially from --fps (frame N -> t = N / fps), in the
    order frames appear in localization_results.json. This is fine for
    the simulated-telemetry path (which is generated on the same clock)
    and workable with a real .tlog if frames were captured at a known,
    constant rate — but it is an approximation. For rigorous real-flight
    geolocation, add a real capture timestamp to inference_results.json
    upstream and this script will use it automatically once wired in
    (see `assign_frame_timestamps`).

Output formats (--output_format):
    json    : structured GPS polygons + wavefront arcs + track/velocity data
    geojson : GeoJSON FeatureCollection (directly renderable in QGIS, Mapbox, etc.)
    csv     : flat CSV of GPS-projected contour points + per-region velocity

Usage:
    # With a real MAVLink log:
    python geolocate.py \\
        --localization_json localization_results.json \\
        --telemetry_log     flight_telemetry.tlog \\
        --fps 1.0 \\
        [--output_format    json|geojson|csv] \\
        [--output_file      geolocated_results.json] \\
        [--wind_direction_deg 270]

    # Without a flight log yet (simulated straight-line flight):
    python geolocate.py \\
        --localization_json localization_results.json \\
        --simulate_telemetry \\
        --sim_start_lat 34.1500 --sim_start_lon -117.8500 \\
        --sim_altitude_m 80 --sim_heading_deg 90 --sim_speed_mps 8 \\
        --fps 1.0

Dependencies:
    numpy (required)
    pymavlink (only required if using --telemetry_log with a real .tlog/.bin)
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import csv
import math
import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# SECTION 1 — CAMERA INTRINSICS
# =============================================================================
# FLIR HADRON 640R+ specifications (13.6mm lens):
#   Sensor resolution : 640 x 512 px
#   Pixel pitch       : 12 um
#   Focal length      : 13.6 mm
#
# Since the pipeline processes images at 256x256 (resized from 640x512),
# the intrinsics are scaled to match.

def get_camera_intrinsics(process_size: int = 256,
                           sensor_w_px: int = 640,
                           sensor_h_px: int = 512,
                           focal_length_mm: float = 13.6,
                           pixel_pitch_mm: float = 0.012) -> np.ndarray:
    """
    Returns the 3x3 camera intrinsic matrix K for the FLIR HADRON 640R+
    (13.6mm), scaled to the processing resolution used in the pipeline.
    """
    fx_native = focal_length_mm / pixel_pitch_mm   # ~1133.3 px
    fy_native = fx_native                            # square pixels
    cx_native = sensor_w_px / 2.0
    cy_native = sensor_h_px / 2.0

    scale_x = process_size / sensor_w_px
    scale_y = process_size / sensor_h_px

    fx = fx_native * scale_x
    fy = fy_native * scale_y
    cx = cx_native * scale_x
    cy = cy_native * scale_y

    return np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])


# =============================================================================
# SECTION 2 — MAVLINK TELEMETRY PARSING (real flight logs)
# =============================================================================
# MAVLink v2 messages relevant to geolocation:
#   GLOBAL_POSITION_INT : drone GPS lat, lon, alt (barometric + GPS fused alt)
#   MOUNT_ORIENTATION    : gimbal roll, pitch, yaw (camera pointing direction)

def parse_mavlink_log(tlog_path: str) -> dict:
    """
    Parses a MAVLink v2 .tlog/.bin log and returns a time-indexed dict of
    telemetry snapshots: {timestamp_s: {lat, lon, alt_m, gimbal_pitch,
    gimbal_roll, gimbal_yaw}}.

    Requires pymavlink: pip install pymavlink
    """
    try:
        from pymavlink import mavutil
    except ImportError as e:
        raise ImportError(
            "pymavlink is required to parse a real MAVLink log. "
            "Install with: pip install pymavlink"
        ) from e

    connection = mavutil.mavlink_connection(tlog_path)
    telemetry = {}

    while True:
        msg = connection.recv_match(blocking=False)
        if msg is None:
            break

        t = getattr(msg, "_timestamp", None)
        if t is None:
            continue

        mtype = msg.get_type()
        if mtype == "GLOBAL_POSITION_INT":
            snap = telemetry.setdefault(t, {})
            snap["lat"] = msg.lat / 1e7
            snap["lon"] = msg.lon / 1e7
            snap["alt_m"] = msg.relative_alt / 1000.0
        elif mtype == "MOUNT_ORIENTATION":
            snap = telemetry.setdefault(t, {})
            snap["gimbal_pitch"] = msg.pitch
            snap["gimbal_roll"] = msg.roll
            snap["gimbal_yaw"] = msg.yaw

    if not telemetry:
        raise ValueError(f"No usable GLOBAL_POSITION_INT/MOUNT_ORIENTATION messages found in {tlog_path}")

    return telemetry


def get_telemetry_at_frame(telemetry: dict, frame_timestamp: float) -> dict:
    """Finds the closest telemetry snapshot to the given frame timestamp."""
    timestamps = sorted(telemetry.keys())
    closest_t = min(timestamps, key=lambda t: abs(t - frame_timestamp))
    snap = telemetry[closest_t]
    required = ("lat", "lon", "alt_m", "gimbal_pitch", "gimbal_roll", "gimbal_yaw")
    missing = [k for k in required if k not in snap]
    if missing:
        raise ValueError(
            f"Telemetry snapshot at t={closest_t} is missing fields {missing}. "
            "GLOBAL_POSITION_INT and MOUNT_ORIENTATION messages may not be time-aligned "
            "closely enough in this log."
        )
    return snap


# =============================================================================
# SECTION 3 — SIMULATED TELEMETRY (for development before a real .tlog exists)
# =============================================================================
# Generates a synthetic constant-velocity, constant-altitude flight path with
# a nadir-mounted (straight-down), stabilized gimbal. One snapshot per frame,
# already time-aligned — no nearest-timestamp lookup needed.

def generate_simulated_telemetry(num_frames: int,
                                  fps: float,
                                  start_lat: float,
                                  start_lon: float,
                                  altitude_m: float = 80.0,
                                  heading_deg: float = 90.0,
                                  speed_mps: float = 8.0,
                                  gimbal_pitch_deg: float = -90.0,
                                  gimbal_roll_deg: float = 0.0,
                                  gimbal_yaw_deg: float = 0.0) -> list:
    """
    Returns a list of `num_frames` telemetry snapshots (one per frame, in
    order), simulating the drone flying a straight line at constant
    altitude/heading/speed with a stabilized, nadir-pointing gimbal.
    """
    snapshots = []
    for i in range(num_frames):
        t = i / fps
        dist_m = speed_mps * t
        lat, lon = local_meters_to_gps(
            north_m=dist_m * math.cos(math.radians(heading_deg)),
            east_m=dist_m * math.sin(math.radians(heading_deg)),
            ref_lat=start_lat, ref_lon=start_lon,
        )
        snapshots.append({
            "timestamp_s": t,
            "lat": lat,
            "lon": lon,
            "alt_m": altitude_m,
            "gimbal_pitch": gimbal_pitch_deg,
            "gimbal_roll": gimbal_roll_deg,
            "gimbal_yaw": gimbal_yaw_deg,
        })
    return snapshots


# =============================================================================
# SECTION 4 — ROTATION MATRIX FROM GIMBAL ANGLES
# =============================================================================
# World frame: NED (North, East, Down).
# Gimbal Euler angles (standard aerospace convention):
#   roll  : rotation about body X (forward), positive = right wing down
#   pitch : rotation about body Y (right), positive = nose up
#           (nadir camera => pitch ~= -90 deg, "looking down")
#   yaw   : rotation about body Z (down), positive = clockwise from North
#
# Camera-to-body axis mapping (fixed mount, not part of the gimbal angles):
#   Camera optical frame: x_c = image right, y_c = image down, z_c = boresight (forward)
#   This maps to the body frame as: body_forward = z_c, body_right = x_c, body_down = y_c
#   i.e. the camera boresight (center-pixel ray) is the body's forward axis,
#   which is what points straight down when the gimbal pitches to -90 deg.
#   (Verified: nadir pitch=-90, roll=0, yaw=0 sends the center-pixel ray to
#   pure NED [0,0,1] = straight down, right-of-center -> +East, below-center -> South.)

CAM_TO_BODY = np.array([
    [0.0, 0.0, 1.0],   # body_forward (X) = z_c
    [1.0, 0.0, 0.0],   # body_right   (Y) = x_c
    [0.0, 1.0, 0.0],   # body_down    (Z) = y_c
])


def euler_to_rotation_matrix(pitch_deg: float, roll_deg: float, yaw_deg: float) -> np.ndarray:
    """Converts gimbal Euler angles (deg) to a 3x3 rotation matrix R (world_from_body)."""
    p, r, y = map(math.radians, (pitch_deg, roll_deg, yaw_deg))

    R_roll = np.array([[1, 0, 0],
                        [0, math.cos(r), -math.sin(r)],
                        [0, math.sin(r), math.cos(r)]])

    R_pitch = np.array([[math.cos(p), 0, math.sin(p)],
                         [0, 1, 0],
                         [-math.sin(p), 0, math.cos(p)]])

    R_yaw = np.array([[math.cos(y), -math.sin(y), 0],
                       [math.sin(y), math.cos(y), 0],
                       [0, 0, 1]])

    return R_yaw @ R_pitch @ R_roll


# =============================================================================
# SECTION 5 — GEOMETRY HELPERS (spherical-earth approximation, per spec 7.3/7.4)
# =============================================================================

EARTH_RADIUS_M = 6_371_000.0


def haversine_distance_m(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in meters between two GPS points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing (degrees from North, clockwise) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def gps_to_local_meters(lat, lon, ref_lat, ref_lon) -> tuple:
    """Equirectangular approximation: GPS point -> (north_m, east_m) offset from a reference point."""
    north_m = math.radians(lat - ref_lat) * EARTH_RADIUS_M
    east_m = math.radians(lon - ref_lon) * EARTH_RADIUS_M * math.cos(math.radians(ref_lat))
    return north_m, east_m


def local_meters_to_gps(north_m, east_m, ref_lat, ref_lon) -> tuple:
    """Inverse of gps_to_local_meters: (north_m, east_m) offset -> (lat, lon)."""
    delta_lat = math.degrees(north_m / EARTH_RADIUS_M)
    delta_lon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(ref_lat))))
    return ref_lat + delta_lat, ref_lon + delta_lon


# =============================================================================
# SECTION 6 — PIXEL -> GPS BACK-PROJECTION
# =============================================================================
# Ground plane assumption: flat terrain at the drone's GPS altitude (relative
# altitude). Mountainous terrain requires DEM intersection — flagged as
# future work, unchanged from the original design.

PROCESS_SIZE = 256


def pixel_to_gps(u_norm: float, v_norm: float,
                  K: np.ndarray, R: np.ndarray,
                  drone_lat: float, drone_lon: float, altitude_m: float
                  ) -> Optional[tuple]:
    """
    Back-projects a single normalized image point (u_norm, v_norm) in [0,1]
    to a GPS coordinate, assuming flat ground at `altitude_m` below the drone.

    Returns (lat, lon) or None if the ray does not intersect the ground
    (e.g. points above the horizon — shouldn't happen for a near-nadir
    camera, but guarded against bad telemetry).
    """
    u_px = u_norm * PROCESS_SIZE
    v_px = v_norm * PROCESS_SIZE

    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u_px, v_px, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)

    ray_body = CAM_TO_BODY @ ray_cam
    ray_world = R @ ray_body   # [north_component, east_component, down_component]

    if ray_world[2] <= 1e-6:
        return None  # ray points at/above horizon — no ground intersection

    t = altitude_m / ray_world[2]
    north_offset_m = t * ray_world[0]
    east_offset_m = t * ray_world[1]

    return local_meters_to_gps(north_offset_m, east_offset_m, drone_lat, drone_lon)


# =============================================================================
# SECTION 7 — FIRE WAVEFRONT ARC EXTRACTION
# =============================================================================
# The fire propagation model needs the leading edge (wavefront) of the fire,
# not the full perimeter.
#
#   Option A — Wind-guided (preferred if wind data available): the wavefront
#   is the contour arc segment whose outward edge normal faces into the wind.
#
#   Option B — Movement-guided (fallback, no wind data): use the fire's own
#   estimated heading from cross-frame tracking (Section 8) as the direction
#   of spread. This is the "temporal frame differencing" fallback described
#   in the original design — tracking now supplies it directly.
#
#   If neither is available (e.g. a brand-new, untracked fire and no wind
#   input), the full polygon is returned.

def extract_fire_wavefront_arc(gps_polygon: list,
                                wind_direction_deg: Optional[float] = None,
                                movement_heading_deg: Optional[float] = None) -> list:
    """
    Extracts the fire wavefront arc from a GPS polygon.

    Args:
        gps_polygon: list of (lat, lon) tuples forming the fire boundary
        wind_direction_deg: wind direction in degrees from North (meteorological
            convention — direction wind is blowing TOWARD). Takes priority if given.
        movement_heading_deg: fire's own estimated heading of movement (degrees
            from North), from cross-frame tracking. Used if wind is unavailable.

    Returns:
        list of (lat, lon) tuples forming the wavefront arc (subset of the
        input polygon), or the full polygon if no direction is available.
    """
    direction_deg = wind_direction_deg if wind_direction_deg is not None else movement_heading_deg
    if direction_deg is None or len(gps_polygon) < 3:
        return list(gps_polygon)

    centroid_lat = sum(p[0] for p in gps_polygon) / len(gps_polygon)
    centroid_lon = sum(p[1] for p in gps_polygon) / len(gps_polygon)
    pts_m = [gps_to_local_meters(lat, lon, centroid_lat, centroid_lon) for lat, lon in gps_polygon]

    dir_rad = math.radians(direction_deg)
    dir_vec = np.array([math.cos(dir_rad), math.sin(dir_rad)])  # (north, east) components

    n = len(pts_m)
    wavefront_idx = set()
    for i in range(n):
        p1, p2 = pts_m[i], pts_m[(i + 1) % n]
        edge = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        norm = np.linalg.norm(edge)
        if norm < 1e-9:
            continue
        normal = np.array([edge[1], -edge[0]]) / norm  # perpendicular to edge
        # epsilon guards against floating-point noise (e.g. cos(90 deg) != exactly 0)
        # misclassifying a side edge (perpendicular to the direction) as forward-facing
        if np.dot(normal, dir_vec) > 1e-6:
            wavefront_idx.add(i)
            wavefront_idx.add((i + 1) % n)

    if len(wavefront_idx) < 2:
        return list(gps_polygon)  # direction didn't cleanly separate an arc — return full contour

    return [gps_polygon[i] for i in sorted(wavefront_idx)]


# =============================================================================
# SECTION 8 — FIRE TRACKING ACROSS FRAMES + VELOCITY ESTIMATION  (NEW)
# =============================================================================
# Matches fire regions to persistent tracks across the frame sequence and
# estimates each track's velocity vector (speed in m/s, heading in degrees
# from North) from the GPS displacement of its centroid over time.
#
# Matching strategy: greedy nearest-centroid matching, gated by a maximum
# distance per frame interval (--max_track_distance_m). This mirrors the
# project's "classical CV, explainable" design philosophy rather than
# reaching for a heavier tracker (Kalman/Hungarian) — reasonable for a
# handful of fire regions per frame, and easy to justify in the report.
#
# A track survives up to --max_coast_frames consecutive frames with no
# matching region (e.g. one bad detection) before being marked "lost".
# This does not delete history — `export_tracks()` still returns it.

def polygon_centroid(gps_polygon: list) -> tuple:
    """Simple vertex-mean centroid (consistent with the project's existing
    contour-centroid convention in localize.py's visualization)."""
    lat = sum(p[0] for p in gps_polygon) / len(gps_polygon)
    lon = sum(p[1] for p in gps_polygon) / len(gps_polygon)
    return lat, lon


@dataclass
class FireTrack:
    track_id: int
    positions: list = field(default_factory=list)   # list of position dicts, oldest -> newest
    status: str = "active"                            # "active" | "lost"
    frames_since_seen: int = 0

    @property
    def last_position(self):
        return self.positions[-1] if self.positions else None


class FireTracker:
    """
    Tracks fire regions across a sequence of frames using GPS centroid
    matching, and estimates a velocity vector per track.

    Call `update()` once per frame, IN TEMPORAL ORDER, even for frames with
    zero fire regions (pass an empty list) — this keeps coast/loss timing
    correct for tracks that momentarily have no matching detection.
    """

    def __init__(self,
                 max_match_distance_m: float = 40.0,
                 max_coast_frames: int = 1,
                 smoothing_window: int = 3):
        self.tracks = {}
        self._next_id = 1
        self.max_match_distance_m = max_match_distance_m
        self.max_coast_frames = max_coast_frames
        self.smoothing_window = smoothing_window

    def update(self, frame_id: str, timestamp_s: float, regions: list) -> list:
        """
        regions: list of dicts, each with 'centroid_gps': (lat, lon) and
                 optionally 'area_px'.

        Returns a list (parallel to `regions`) of dicts with:
            track_id, track_status ("new"|"continuing"),
            velocity_mps, velocity_smoothed_mps, heading_deg,
            track_age_s, track_length_frames
        """
        active_ids = [tid for tid, t in self.tracks.items() if t.status == "active"]

        candidates = []
        for tid in active_ids:
            last = self.tracks[tid].last_position
            if last is None:
                continue
            for ridx, region in enumerate(regions):
                lat, lon = region["centroid_gps"]
                d = haversine_distance_m(last["lat"], last["lon"], lat, lon)
                if d <= self.max_match_distance_m:
                    candidates.append((d, tid, ridx))
        candidates.sort(key=lambda x: x[0])

        assigned_track, assigned_region = set(), set()
        assignments = {}
        for d, tid, ridx in candidates:
            if tid in assigned_track or ridx in assigned_region:
                continue
            assignments[ridx] = tid
            assigned_track.add(tid)
            assigned_region.add(ridx)

        # Tracks with no match this frame: coast, or lose if past the limit
        for tid in active_ids:
            if tid not in assigned_track:
                self.tracks[tid].frames_since_seen += 1
                if self.tracks[tid].frames_since_seen > self.max_coast_frames:
                    self.tracks[tid].status = "lost"

        results = []
        for ridx, region in enumerate(regions):
            lat, lon = region["centroid_gps"]

            if ridx in assignments:
                tid = assignments[ridx]
                track = self.tracks[tid]
                prev = track.last_position
                track.frames_since_seen = 0

                dt = timestamp_s - prev["timestamp_s"]
                velocity_mps, heading_deg = None, None
                if dt > 0:
                    dist_m = haversine_distance_m(prev["lat"], prev["lon"], lat, lon)
                    velocity_mps = dist_m / dt
                    heading_deg = bearing_deg(prev["lat"], prev["lon"], lat, lon)

                track.positions.append({
                    "frame_id": frame_id, "timestamp_s": timestamp_s,
                    "lat": lat, "lon": lon, "area_px": region.get("area_px"),
                    "velocity_mps": velocity_mps, "heading_deg": heading_deg,
                })

                recent_v = [p["velocity_mps"] for p in track.positions[-self.smoothing_window:]
                            if p["velocity_mps"] is not None]
                smoothed = sum(recent_v) / len(recent_v) if recent_v else None
                status = "continuing"
            else:
                tid = self._next_id
                self._next_id += 1
                track = FireTrack(track_id=tid)
                self.tracks[tid] = track
                track.positions.append({
                    "frame_id": frame_id, "timestamp_s": timestamp_s,
                    "lat": lat, "lon": lon, "area_px": region.get("area_px"),
                    "velocity_mps": None, "heading_deg": None,
                })
                velocity_mps, heading_deg, smoothed = None, None, None
                status = "new"

            age_s = timestamp_s - track.positions[0]["timestamp_s"]
            results.append({
                "track_id": tid,
                "track_status": status,
                "velocity_mps": round(velocity_mps, 3) if velocity_mps is not None else None,
                "velocity_smoothed_mps": round(smoothed, 3) if smoothed is not None else None,
                "heading_deg": round(heading_deg, 1) if heading_deg is not None else None,
                "track_age_s": round(age_s, 2),
                "track_length_frames": len(track.positions),
            })
        return results

    def export_tracks(self) -> dict:
        """Full trajectory history for every track (active and lost)."""
        return {
            str(tid): {
                "track_id": tid,
                "status": t.status,
                "n_positions": len(t.positions),
                "positions": t.positions,
            }
            for tid, t in self.tracks.items()
        }


# =============================================================================
# SECTION 9 — FRAME TIMESTAMP ASSIGNMENT
# =============================================================================

def assign_frame_timestamps(results: list, fps: float) -> list:
    """
    Assigns a sequential timestamp to each frame based on its position in
    `results` and the assumed --fps. See module docstring: this is a stand-in
    until per-frame capture timestamps are recorded upstream in infer.py.
    """
    return [i / fps for i in range(len(results))]


# =============================================================================
# SECTION 10 — OUTPUT FORMATTERS
# =============================================================================

def to_geojson(results: list) -> dict:
    features = []
    for result in results:
        if not result.get("localization_run"):
            continue
        for i, region in enumerate(result.get("fire_regions", [])):
            if not region.get("gps_polygon"):
                continue
            ring = [[p["lon"], p["lat"]] for p in region["gps_polygon"]]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "frame_id": result["frame_id"],
                    "region_index": i,
                    "area_px": region.get("area_px"),
                    "n_points": region.get("n_points"),
                    "track_id": region.get("track_id"),
                    "track_status": region.get("track_status"),
                    "velocity_mps": region.get("velocity_mps"),
                    "velocity_smoothed_mps": region.get("velocity_smoothed_mps"),
                    "heading_deg": region.get("heading_deg"),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]}
            })
    return {"type": "FeatureCollection", "features": features}


def to_csv_rows(results: list) -> list:
    rows = []
    for result in results:
        if not result.get("localization_run"):
            continue
        for ri, region in enumerate(result.get("fire_regions", [])):
            if not region.get("gps_polygon"):
                continue
            for pi, pt in enumerate(region["gps_polygon"]):
                rows.append({
                    "frame_id": result["frame_id"], "region_index": ri, "point_index": pi,
                    "lat": pt["lat"], "lon": pt["lon"],
                    "area_px": region.get("area_px"),
                    "track_id": region.get("track_id"),
                    "track_status": region.get("track_status"),
                    "velocity_mps": region.get("velocity_mps"),
                    "velocity_smoothed_mps": region.get("velocity_smoothed_mps"),
                    "heading_deg": region.get("heading_deg"),
                })
    return rows


# =============================================================================
# SECTION 11 — SELF TEST (sanity-check the math, no I/O required)
# =============================================================================

def _self_test():
    print("[self-test] Nadir center-pixel projects straight down...")
    K = get_camera_intrinsics(256)
    R = euler_to_rotation_matrix(pitch_deg=-90, roll_deg=0, yaw_deg=0)
    gps = pixel_to_gps(0.5, 0.5, K, R, drone_lat=34.0, drone_lon=-118.0, altitude_m=100)
    d = haversine_distance_m(34.0, -118.0, gps[0], gps[1])
    assert d < 0.5, f"center pixel should land ~at nadir point, got {d:.2f}m off"
    print(f"    OK — offset = {d:.4f} m (expected ~0)")

    print("[self-test] Right-of-center pixel projects East of nadir...")
    gps_r = pixel_to_gps(0.9, 0.5, K, R, drone_lat=34.0, drone_lon=-118.0, altitude_m=100)
    assert gps_r[1] > -118.0, "expected positive (East) longitude offset"
    print(f"    OK — lon shifted East by {(gps_r[1] - (-118.0)):.6f} deg")

    print("[self-test] haversine/bearing round-trip via local_meters_to_gps...")
    lat2, lon2 = local_meters_to_gps(100, 50, 34.0, -118.0)
    d = haversine_distance_m(34.0, -118.0, lat2, lon2)
    assert abs(d - math.hypot(100, 50)) < 0.5
    print(f"    OK — distance = {d:.3f} m (expected {math.hypot(100,50):.3f} m)")

    print("[self-test] Tracker assigns velocity across two frames...")
    tracker = FireTracker(max_match_distance_m=100)
    r0 = tracker.update("f0", 0.0, [{"centroid_gps": (34.0, -118.0), "area_px": 100}])
    assert r0[0]["track_status"] == "new" and r0[0]["velocity_mps"] is None
    lat1, lon1 = local_meters_to_gps(20, 0, 34.0, -118.0)  # 20m North, 1s later
    r1 = tracker.update("f1", 1.0, [{"centroid_gps": (lat1, lon1), "area_px": 110}])
    assert r1[0]["track_status"] == "continuing"
    assert abs(r1[0]["velocity_mps"] - 20.0) < 0.1, r1[0]["velocity_mps"]
    assert abs(r1[0]["heading_deg"] - 0.0) < 1.0, r1[0]["heading_deg"]  # due North
    print(f"    OK — velocity={r1[0]['velocity_mps']} m/s heading={r1[0]['heading_deg']} deg")

    print("\n[self-test] All checks passed.")


# =============================================================================
# SECTION 12 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: GPS Back-Projection + Fire Tracking / Velocity")

    parser.add_argument("--localization_json", help="Path to localization_results.json from localize.py")
    parser.add_argument("--telemetry_log", default=None, help="Path to a real MAVLink v2 .tlog/.bin")
    parser.add_argument("--simulate_telemetry", action="store_true",
                         help="Use a synthetic straight-line flight instead of a real telemetry log")

    # Simulated flight parameters
    parser.add_argument("--sim_start_lat", type=float, default=34.1500)
    parser.add_argument("--sim_start_lon", type=float, default=-117.8500)
    parser.add_argument("--sim_altitude_m", type=float, default=80.0)
    parser.add_argument("--sim_heading_deg", type=float, default=90.0)
    parser.add_argument("--sim_speed_mps", type=float, default=8.0)
    parser.add_argument("--sim_gimbal_pitch_deg", type=float, default=-90.0)
    parser.add_argument("--sim_gimbal_roll_deg", type=float, default=0.0)
    parser.add_argument("--sim_gimbal_yaw_deg", type=float, default=0.0)

    parser.add_argument("--fps", type=float, default=1.0,
                         help="Assumed frame processing rate for sequential timestamp "
                              "assignment (see module docstring). Default: 1.0")

    parser.add_argument("--output_file", default="geolocated_results.json")
    parser.add_argument("--output_format", default="json", choices=["json", "geojson", "csv"])
    parser.add_argument("--wind_direction_deg", type=float, default=None,
                         help="Wind direction, degrees from North. If omitted, wavefront arc "
                              "extraction falls back to each fire's tracked movement heading.")

    # Tracking parameters
    parser.add_argument("--max_track_distance_m", type=float, default=40.0,
                         help="Max centroid displacement (m) between consecutive frames for a "
                              "region to be matched to an existing track. Default: 40.0")
    parser.add_argument("--max_coast_frames", type=int, default=1,
                         help="Frames a track may go unmatched before being marked lost. Default: 1")
    parser.add_argument("--track_smoothing_window", type=int, default=3,
                         help="Number of recent instantaneous velocities averaged into "
                              "velocity_smoothed_mps. Default: 3")

    parser.add_argument("--self_test", action="store_true",
                         help="Run internal math sanity checks and exit (no files needed)")

    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    if not args.localization_json:
        parser.error("--localization_json is required (unless using --self_test)")
    if not args.telemetry_log and not args.simulate_telemetry:
        parser.error("Provide either --telemetry_log <file> or --simulate_telemetry")

    with open(args.localization_json) as f:
        loc_data = json.load(f)
    results = loc_data["results"]
    print(f"[OK] Loaded {len(results)} frames from {args.localization_json}")

    timestamps = assign_frame_timestamps(results, args.fps)

    if args.simulate_telemetry:
        print(f"[OK] Using simulated telemetry: start=({args.sim_start_lat},{args.sim_start_lon}) "
              f"alt={args.sim_altitude_m}m heading={args.sim_heading_deg}deg speed={args.sim_speed_mps}m/s")
        telemetry_per_frame = generate_simulated_telemetry(
            num_frames=len(results), fps=args.fps,
            start_lat=args.sim_start_lat, start_lon=args.sim_start_lon,
            altitude_m=args.sim_altitude_m, heading_deg=args.sim_heading_deg,
            speed_mps=args.sim_speed_mps, gimbal_pitch_deg=args.sim_gimbal_pitch_deg,
            gimbal_roll_deg=args.sim_gimbal_roll_deg, gimbal_yaw_deg=args.sim_gimbal_yaw_deg,
        )
    else:
        print(f"[OK] Parsing MAVLink log: {args.telemetry_log}")
        telemetry_dict = parse_mavlink_log(args.telemetry_log)
        telemetry_per_frame = [get_telemetry_at_frame(telemetry_dict, t) for t in timestamps]

    K = get_camera_intrinsics(process_size=PROCESS_SIZE)
    tracker = FireTracker(
        max_match_distance_m=args.max_track_distance_m,
        max_coast_frames=args.max_coast_frames,
        smoothing_window=args.track_smoothing_window,
    )

    frames_geolocated = 0
    total_gps_points = 0

    for i, result in enumerate(results):
        result["timestamp_s"] = round(timestamps[i], 3)
        snap = telemetry_per_frame[i]

        regions = result.get("fire_regions") or []
        if not result.get("localization_run") or not regions:
            tracker.update(result["frame_id"], timestamps[i], [])  # keep coast timing correct
            continue

        R = euler_to_rotation_matrix(snap["gimbal_pitch"], snap["gimbal_roll"], snap["gimbal_yaw"])

        track_input = []
        for region in regions:
            gps_pts = []
            valid = True
            for pt in region["normalized"]:
                gps = pixel_to_gps(pt["x"], pt["y"], K, R, snap["lat"], snap["lon"], snap["alt_m"])
                if gps is None:
                    valid = False
                    break
                gps_pts.append(gps)

            if valid and gps_pts:
                region["gps_polygon"] = [{"lat": round(lat, 7), "lon": round(lon, 7)} for lat, lon in gps_pts]
                centroid = polygon_centroid(gps_pts)
                region["centroid_gps"] = {"lat": round(centroid[0], 7), "lon": round(centroid[1], 7)}
                total_gps_points += len(gps_pts)
            else:
                region["gps_polygon"] = None
                region["centroid_gps"] = None

            track_input.append({
                "centroid_gps": centroid if valid and gps_pts else (snap["lat"], snap["lon"]),
                "area_px": region.get("area_px"),
            })

        track_updates = tracker.update(result["frame_id"], timestamps[i], track_input)

        for region, tu in zip(regions, track_updates):
            region.update(tu)
            if region.get("gps_polygon"):
                gps_tuples = [(p["lat"], p["lon"]) for p in region["gps_polygon"]]
                arc = extract_fire_wavefront_arc(
                    gps_tuples,
                    wind_direction_deg=args.wind_direction_deg,
                    movement_heading_deg=tu["heading_deg"],
                )
                region["fire_front_arc"] = [{"lat": lat, "lon": lon} for lat, lon in arc]
            else:
                region["fire_front_arc"] = None

        result["telemetry_snapshot"] = {k: snap[k] for k in
                                         ("lat", "lon", "alt_m", "gimbal_pitch", "gimbal_roll", "gimbal_yaw")}
        frames_geolocated += 1

        n_new = sum(1 for tu in track_updates if tu["track_status"] == "new")
        n_cont = sum(1 for tu in track_updates if tu["track_status"] == "continuing")
        print(f"  {result['frame_id']:<28}  t={timestamps[i]:6.2f}s  "
              f"regions={len(regions)}  new_tracks={n_new}  continuing={n_cont}")

    tracks_export = tracker.export_tracks()
    n_active = sum(1 for t in tracks_export.values() if t["status"] == "active")

    summary = loc_data.get("summary", {})
    summary.update({
        "geolocation_run_on": frames_geolocated,
        "total_gps_points": total_gps_points,
        "telemetry_source": "simulated" if args.simulate_telemetry else args.telemetry_log,
        "fps_assumed": args.fps,
        "tracks_created": tracker._next_id - 1,
        "tracks_active_at_end": n_active,
        "tracking_parameters": {
            "max_track_distance_m": args.max_track_distance_m,
            "max_coast_frames": args.max_coast_frames,
            "track_smoothing_window": args.track_smoothing_window,
        },
    })

    output = {"summary": summary, "results": results, "tracks": tracks_export}

    base = os.path.splitext(args.output_file)[0]
    if args.output_format == "json":
        out_path = base + ".json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
    elif args.output_format == "geojson":
        out_path = base + ".geojson"
        with open(out_path, "w") as f:
            json.dump(to_geojson(results), f, indent=2)
    else:  # csv
        out_path = base + ".csv"
        rows = to_csv_rows(results)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        else:
            open(out_path, "w").close()

    print(f"\n[OK] Geolocated {frames_geolocated} frames, {total_gps_points} GPS points")
    print(f"[OK] Tracks created: {tracker._next_id - 1}  (active at end: {n_active})")
    print(f"[OK] Saved: {out_path} ({args.output_format})")


if __name__ == "__main__":
    main()
