"""
geolocate.py — Phase 3: Fire Contour GPS Back-Projection
=========================================================
Reads localization_results.json (produced by localize.py) and converts
normalized image-space fire contour points to GPS world coordinates using
drone telemetry and camera intrinsics.

The output is a set of GPS polygons representing fire region boundaries,
and simplified polyline arcs representing the fire wavefront — ready for
direct ingestion by the fire propagation model.

Hardware assumptions (from project specification):
    Thermal camera : FLIR HADRON 640R+, 13.6mm focal length, 60Hz
    Altitude       : barometric + GPS fusion (from MAVLink v2)
    Gimbal         : real-time angle telemetry (pitch, roll, yaw) via MAVLink v2
    Telemetry bus  : MAVLink v2 over serial

Output formats (--output_format):
    json    : structured GPS polygons + arc polylines per frame
    geojson : GeoJSON FeatureCollection (directly renderable in QGIS, Mapbox, etc.)
    csv     : flat CSV of GPS-projected contour points

Usage:
    python geolocate.py \\
        --localization_json localization_results.json \\
        --telemetry_log     flight_telemetry.tlog \\
        [--output_format    json|geojson|csv] \\
        [--output_file      geolocated_results.json] \\
        [--arc_only]

Dependencies (to install):
    pip install pymavlink pyproj numpy

Author note:
    This script is a fully specified stub. All logic is described in pseudocode
    comments. Implementation requires access to the drone's telemetry log or
    live MAVLink stream.
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import os
import json
import csv
import argparse
import numpy as np

# TODO: install and import pymavlink for telemetry parsing
# from pymavlink import mavutil

# TODO: install and import pyproj for GPS coordinate transforms
# from pyproj import Proj, Transformer


# =============================================================================
# SECTION 1 — CAMERA INTRINSICS
# =============================================================================
# FLIR HADRON 640R+ specifications (13.6mm lens):
#   Sensor resolution : 640 × 512 px
#   Pixel pitch       : 12 µm
#   Focal length      : 13.6 mm
#   Horizontal FoV    : ~50.4°  (derived: 2 × arctan(sensor_width / (2 × f)))
#   Vertical FoV      : ~40.6°
#
# Intrinsic matrix K (pinhole camera model):
#
#       [ fx   0   cx ]
#   K = [  0  fy   cy ]
#       [  0   0    1 ]
#
#   fx = fy = f_px = focal_length_mm / pixel_pitch_mm
#            = 13.6 / 0.012 = 1133.3 px
#   cx = 640 / 2 = 320
#   cy = 512 / 2 = 256
#
# Since our pipeline processes images at 256×256 (resized from 640×512),
# we must scale the intrinsics accordingly:
#   scale_x = 256 / 640 = 0.4
#   scale_y = 256 / 512 = 0.5
#   fx_scaled = fx × scale_x
#   fy_scaled = fy × scale_y
#   cx_scaled = cx × scale_x
#   cy_scaled = cy × scale_y


def get_camera_intrinsics(process_size: int = 256) -> np.ndarray:
    """
    Returns the camera intrinsic matrix K for the FLIR HADRON 640R+ (13.6mm),
    scaled to the processing resolution used in the pipeline.

    # PSEUDOCODE:
    #   sensor_w_px = 640
    #   sensor_h_px = 512
    #   focal_length_mm = 13.6
    #   pixel_pitch_mm  = 0.012
    #
    #   fx_native = focal_length_mm / pixel_pitch_mm       # = 1133.3 px
    #   fy_native = fx_native                               # square pixels
    #   cx_native = sensor_w_px / 2                        # = 320
    #   cy_native = sensor_h_px / 2                        # = 256
    #
    #   scale_x = process_size / sensor_w_px               # e.g. 256/640 = 0.4
    #   scale_y = process_size / sensor_h_px               # e.g. 256/512 = 0.5
    #
    #   fx = fx_native * scale_x
    #   fy = fy_native * scale_y
    #   cx = cx_native * scale_x
    #   cy = cy_native * scale_y
    #
    #   K = [[fx,  0, cx],
    #        [ 0, fy, cy],
    #        [ 0,  0,  1]]
    #   return K
    """
    # TODO: replace with computed values once verified against physical camera
    raise NotImplementedError("Implement using FLIR HADRON 640R+ spec sheet values")


# =============================================================================
# SECTION 2 — MAVLINK TELEMETRY PARSING
# =============================================================================
# MAVLink v2 over serial provides the following messages relevant to geolocation:
#
#   GLOBAL_POSITION_INT  : drone GPS lat, lon, alt (barometric + GPS fused alt)
#   ATTITUDE             : drone body roll, pitch, yaw
#   MOUNT_ORIENTATION    : gimbal roll, pitch, yaw (camera pointing direction)
#   GPS_RAW_INT          : raw GPS fix quality and satellite count
#
# For each frame, we need a telemetry snapshot at the frame's timestamp.
# Frame timestamps must be matched to telemetry messages by timestamp.
#
# MAVLink message timestamps are in Unix microseconds (time_boot_ms or time_usec).


def parse_mavlink_log(tlog_path: str) -> dict:
    """
    Parses a MAVLink v2 .tlog or .bin log file and returns a time-indexed
    dictionary of telemetry snapshots.

    Returns dict keyed by timestamp (seconds), each value containing:
        {
            "lat":          float,  # degrees
            "lon":          float,  # degrees
            "alt_m":        float,  # meters above ground (fused baro + GPS)
            "gimbal_pitch": float,  # degrees, negative = looking down
            "gimbal_roll":  float,  # degrees
            "gimbal_yaw":   float,  # degrees, relative to North
        }

    # PSEUDOCODE:
    #   connection = mavutil.mavlink_connection(tlog_path)
    #   telemetry  = {}
    #
    #   while True:
    #       msg = connection.recv_match(blocking=False)
    #       if msg is None: break
    #
    #       t = msg._timestamp
    #
    #       if msg.get_type() == 'GLOBAL_POSITION_INT':
    #           telemetry.setdefault(t, {})
    #           telemetry[t]['lat']    = msg.lat / 1e7   # MAVLink sends lat*1e7
    #           telemetry[t]['lon']    = msg.lon / 1e7
    #           telemetry[t]['alt_m']  = msg.relative_alt / 1000.0  # mm → m
    #
    #       if msg.get_type() == 'MOUNT_ORIENTATION':
    #           telemetry.setdefault(t, {})
    #           telemetry[t]['gimbal_pitch'] = msg.pitch
    #           telemetry[t]['gimbal_roll']  = msg.roll
    #           telemetry[t]['gimbal_yaw']   = msg.yaw
    #
    #   return telemetry
    """
    raise NotImplementedError("Requires pymavlink. Install: pip install pymavlink")


def get_telemetry_at_frame(telemetry: dict, frame_timestamp: float) -> dict:
    """
    Finds the closest telemetry snapshot to the given frame timestamp.

    # PSEUDOCODE:
    #   timestamps = sorted(telemetry.keys())
    #   closest_t  = min(timestamps, key=lambda t: abs(t - frame_timestamp))
    #   return telemetry[closest_t]
    """
    raise NotImplementedError


# =============================================================================
# SECTION 3 — ROTATION MATRIX FROM GIMBAL ANGLES
# =============================================================================
# The gimbal orientation is expressed as (pitch, roll, yaw) Euler angles.
# We need to convert these to a 3×3 rotation matrix R to rotate camera rays
# from image space into world (NED: North-East-Down) space.
#
# Convention:
#   pitch: rotation around Y axis (positive = nose up)
#   roll:  rotation around X axis (positive = right wing down)
#   yaw:   rotation around Z axis (positive = clockwise from North)
#
# R = R_yaw × R_pitch × R_roll
#
# Note: FLIR HADRON 640R+ is typically mounted nadir (pointing straight down).
# In this configuration, gimbal_pitch ≈ -90°, gimbal_roll ≈ 0°.
# If the gimbal stabilizes against drone motion, the drone's own attitude
# does NOT need to be added — the gimbal angles are already world-referenced.
# Confirm this with the drone's flight controller configuration.


def euler_to_rotation_matrix(pitch_deg: float,
                              roll_deg:  float,
                              yaw_deg:   float) -> np.ndarray:
    """
    Converts gimbal Euler angles to a 3×3 rotation matrix.

    # PSEUDOCODE:
    #   p = radians(pitch_deg)
    #   r = radians(roll_deg)
    #   y = radians(yaw_deg)
    #
    #   R_roll  = [[1,      0,       0     ],
    #              [0,  cos(r),  -sin(r)   ],
    #              [0,  sin(r),   cos(r)   ]]
    #
    #   R_pitch = [[ cos(p),  0,  sin(p)  ],
    #              [      0,  1,       0  ],
    #              [-sin(p),  0,  cos(p)  ]]
    #
    #   R_yaw   = [[cos(y), -sin(y),  0   ],
    #              [sin(y),  cos(y),  0   ],
    #              [     0,       0,  1   ]]
    #
    #   return R_yaw @ R_pitch @ R_roll
    """
    raise NotImplementedError


# =============================================================================
# SECTION 4 — PIXEL → GPS BACK-PROJECTION
# =============================================================================
# The back-projection maps each image pixel (u, v) to a GPS coordinate.
#
# Steps:
#   1. Convert (u, v) from normalized [0,1] to pixel_256 space
#   2. Apply inverse camera intrinsics to get a unit ray in camera space
#   3. Rotate ray into world (NED) space using gimbal rotation matrix R
#   4. Intersect ray with the ground plane at altitude h
#   5. Convert the ground-plane intersection offset (in meters) to GPS delta
#   6. Add GPS delta to drone's GPS position → absolute GPS coordinate
#
# Ground plane assumption:
#   This assumes flat terrain at the drone's GPS altitude.
#   For mountainous terrain, a DEM (Digital Elevation Model) intersection
#   is required — flagged as future work.
#
# Math detail for step 4:
#   Ray in world space: d = R × K_inv × [u, v, 1]^T  (unit direction vector)
#   Ground plane: z = 0 (in NED, z = -altitude, so z_world = altitude)
#   Scale t = altitude / d_z  where d_z is the downward component
#   Ground intersection: P = drone_pos + t × d
#
# Math detail for step 5:
#   Using small-angle approximation (valid for altitudes < ~5000m):
#   Δlat = (P_north / R_earth) × (180/π)
#   Δlon = (P_east  / (R_earth × cos(lat))) × (180/π)
#   R_earth ≈ 6,371,000 m


def pixel_to_gps(u_norm: float, v_norm: float,
                 K: np.ndarray,
                 R: np.ndarray,
                 drone_lat: float, drone_lon: float, altitude_m: float
                 ) -> tuple[float, float]:
    """
    Back-projects a single normalized image point (u_norm, v_norm) to GPS.

    Args:
        u_norm, v_norm : normalized [0,1] image coordinates (from localize.py output)
        K              : camera intrinsic matrix (scaled to process_size)
        R              : gimbal rotation matrix (world ← camera)
        drone_lat/lon  : drone GPS position in decimal degrees
        altitude_m     : drone altitude above ground in meters

    Returns:
        (lat, lon) in decimal degrees

    # PSEUDOCODE:
    #   PROCESS_SIZE = 256
    #   u_px = u_norm * PROCESS_SIZE
    #   v_px = v_norm * PROCESS_SIZE
    #
    #   # Step 1: pixel → normalized camera ray (homogeneous)
    #   ray_cam = inv(K) @ [u_px, v_px, 1]
    #   ray_cam = ray_cam / norm(ray_cam)   # unit vector
    #
    #   # Step 2: rotate ray into world (NED) space
    #   ray_world = R @ ray_cam
    #
    #   # Step 3: intersect with flat ground plane at altitude_m below drone
    #   #         In NED: down is positive Z; ground is at Z = altitude_m
    #   #         ray_world[2] is the downward component (must be > 0 for nadir camera)
    #   if ray_world[2] <= 0:
    #       return None  # ray points upward — no ground intersection
    #   t = altitude_m / ray_world[2]
    #
    #   # Step 4: 3D ground position offset in meters (North, East)
    #   north_offset_m = t * ray_world[0]
    #   east_offset_m  = t * ray_world[1]
    #
    #   # Step 5: offset → GPS delta
    #   R_earth = 6_371_000  # meters
    #   delta_lat = (north_offset_m / R_earth) * (180 / pi)
    #   delta_lon = (east_offset_m  / (R_earth * cos(radians(drone_lat)))) * (180 / pi)
    #
    #   gps_lat = drone_lat + delta_lat
    #   gps_lon = drone_lon + delta_lon
    #   return (gps_lat, gps_lon)
    """
    raise NotImplementedError


# =============================================================================
# SECTION 5 — FIRE WAVEFRONT ARC EXTRACTION
# =============================================================================
# The fire propagation model needs the leading edge (wavefront) of the fire,
# not the full perimeter. The wavefront is the arc of the fire boundary that
# faces the direction of spread.
#
# Two approaches:
#
# Option A — Wind-guided (preferred if wind data available):
#   The wavefront is the contour arc segment facing into the wind direction.
#   For each contour point, compute the outward normal. Points whose normal
#   aligns with the wind vector (dot product > 0) are on the wavefront side.
#
# Option B — Convex hull leading edge (fallback, no wind data):
#   Compute the convex hull of the fire region. The leading edge is the arc
#   furthest from the fire centroid, or in the direction of fire movement
#   estimated from temporal frame differencing.
#
# For now, we output the FULL contour polygon and flag the arc extraction
# as requiring wind direction input. The propagation model can select the
# relevant arc from the full polygon given wind data.


def extract_fire_wavefront_arc(gps_polygon: list[tuple],
                               wind_direction_deg: float = None
                               ) -> list[tuple]:
    """
    Extracts the fire wavefront arc from a GPS polygon.

    Args:
        gps_polygon        : list of (lat, lon) tuples forming the fire boundary
        wind_direction_deg : wind direction in degrees from North (meteorological).
                             If None, returns the full polygon (no arc extraction).

    Returns:
        list of (lat, lon) tuples forming the wavefront arc polyline

    # PSEUDOCODE (wind-guided, Option A):
    #   if wind_direction_deg is None:
    #       return gps_polygon   # full contour, let propagation model decide
    #
    #   # Convert polygon to local Cartesian (meters) for geometry operations
    #   centroid_lat = mean(lat for lat, lon in gps_polygon)
    #   centroid_lon = mean(lon for lat, lon in gps_polygon)
    #   pts_m = [gps_to_local_meters(lat, lon, centroid_lat, centroid_lon)
    #            for lat, lon in gps_polygon]
    #
    #   # Wind vector (unit, direction fire spreads INTO)
    #   wind_rad    = radians(wind_direction_deg)
    #   wind_vec    = [sin(wind_rad), cos(wind_rad)]   # East, North components
    #
    #   # For each polygon edge, compute outward normal and dot with wind vector
    #   wavefront_pts = []
    #   for i in range(len(pts_m)):
    #       p1 = pts_m[i]
    #       p2 = pts_m[(i + 1) % len(pts_m)]
    #       edge      = [p2[0]-p1[0], p2[1]-p1[1]]
    #       normal    = [edge[1], -edge[0]]   # perpendicular
    #       normal    = normalize(normal)
    #       if dot(normal, wind_vec) > 0:     # faces into wind = leading edge
    #           wavefront_pts.append(gps_polygon[i])
    #
    #   return wavefront_pts
    """
    # Temporary: return full polygon until wind data integration is complete
    return gps_polygon


# =============================================================================
# SECTION 6 — OUTPUT FORMATTERS
# =============================================================================

def to_geojson(results: list[dict]) -> dict:
    """
    Produces a GeoJSON FeatureCollection of GPS-projected fire polygons.
    Each fire region contour becomes a GeoJSON Polygon feature.
    If gps_polygon is None (geolocation not yet run), normalized coords are used
    as placeholders.

    # PSEUDOCODE:
    #   features = []
    #   for result in results where localization_run is True:
    #       for i, region in enumerate(result['fire_regions']):
    #           coords = region['gps_polygon'] if region['gps_polygon']
    #                    else [[p['x'], p['y']] for p in region['normalized']]
    #           close ring: if coords[0] != coords[-1]: coords.append(coords[0])
    #           feature = GeoJSON Feature with Polygon geometry
    #           feature.properties = frame_id, region_index, area_px, n_points
    #           features.append(feature)
    #   return GeoJSON FeatureCollection
    """
    raise NotImplementedError


def to_csv_rows(results: list[dict]) -> list[dict]:
    """
    Flat CSV: one row per GPS point.
    Columns: frame_id, region_index, point_index, lat, lon, x_norm, y_norm

    # PSEUDOCODE:
    #   rows = []
    #   for result in results where localization_run is True:
    #       for ri, region in enumerate(result['fire_regions']):
    #           pts = region['gps_polygon'] if available else region['normalized']
    #           for pi, pt in enumerate(pts):
    #               rows.append({frame_id, ri, pi, lat, lon, x_norm, y_norm})
    #   return rows
    """
    raise NotImplementedError


# =============================================================================
# SECTION 7 — MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: GPS Back-Projection of Fire Contours")

    parser.add_argument("--localization_json", required=True,
                        help="Path to localization_results.json from localize.py")
    parser.add_argument("--telemetry_log", required=True,
                        help="Path to MAVLink v2 telemetry log (.tlog or .bin)")
    parser.add_argument("--output_file", default="geolocated_results.json",
                        help="Output file path (default: geolocated_results.json)")
    parser.add_argument("--output_format", default="json",
                        choices=["json", "geojson", "csv"],
                        help="Output format: json | geojson | csv")
    parser.add_argument("--arc_only", action="store_true",
                        help="If set, output only the wavefront arc (requires wind data)")
    parser.add_argument("--wind_direction_deg", type=float, default=None,
                        help="Wind direction in degrees from North (meteorological convention). "
                             "Required for arc extraction. If omitted, full contour is output.")

    args = parser.parse_args()

    # ── PSEUDOCODE: Main pipeline ────────────────────────────────────────────
    #
    # 1. Load localization_results.json
    #       results = json.load(args.localization_json)
    #
    # 2. Parse MAVLink telemetry log
    #       telemetry = parse_mavlink_log(args.telemetry_log)
    #
    # 3. Build camera intrinsic matrix
    #       K = get_camera_intrinsics(process_size=256)
    #
    # 4. For each fire-positive frame:
    #
    #       a. Get telemetry at frame timestamp
    #              snap = get_telemetry_at_frame(telemetry, frame['timestamp'])
    #
    #       b. Build gimbal rotation matrix
    #              R = euler_to_rotation_matrix(snap['gimbal_pitch'],
    #                                           snap['gimbal_roll'],
    #                                           snap['gimbal_yaw'])
    #
    #       c. For each fire region contour:
    #              for each normalized point (u, v) in region['normalized']:
    #                  lat, lon = pixel_to_gps(u, v, K, R,
    #                                          snap['lat'], snap['lon'],
    #                                          snap['alt_m'])
    #              gps_polygon = [(lat, lon), ...]
    #              region['gps_polygon'] = gps_polygon
    #
    #       d. Extract fire wavefront arc
    #              arc = extract_fire_wavefront_arc(gps_polygon,
    #                                               args.wind_direction_deg)
    #              region['fire_front_arc'] = arc
    #
    # 5. Write output in chosen format (json / geojson / csv)
    #
    # 6. Print summary: frames processed, total GPS points, output path
    #

    raise NotImplementedError(
        "geolocate.py is a fully specified stub. "
        "Implement sections 1-7 above to activate the geolocation pipeline. "
        "All math and data formats are specified in the pseudocode comments."
    )


if __name__ == "__main__":
    main()
