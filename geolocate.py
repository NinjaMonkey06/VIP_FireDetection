"""
geolocate.py — Phase 3: Fire Contour Geolocation (pixel → GPS)
===============================================================
Reads localization_results.json produced by localize.py and back-projects every
fire contour vertex to GPS using per-frame drone metadata and a camera config.

Pipeline per region:
    normalized contour points × original_wh   (never pixel_256: the 256x256 resize is anisotropic)
        → undistort + pinhole ray (camera config)
        → rotate by absolute gimbal yaw/pitch/roll
        → intersect flat ground at altitude AGL
        → north/east meters → lat/lon
        → area-weighted centroid + ground area (computed in meters)

Fills per region: gps_polygon, gps_centroid, ground_area_m2, fire_area_m2,
touches_frame_edge, geolocation_status. fire_front_arc is left null (out of scope).
ground_area_m2 is the area of the outline, a fire zone (merged hot spots and the
gaps between them); fire_area_m2 is the ground area of the hot pixels themselves
(from localize.py's fire_pixels_256).

Oblique views: a ray above the horizon, less than --min_depression_deg below it
(default 10°), or landing farther than --max_range_m (default off) is unreliable.
A region with unreliable vertices is clipped to its reliable part (on flat ground,
a disk around the drone), and centroid/area come from the clipped polygon:
    ok                   every vertex reliable, polygon unchanged
    clipped_to_reliable  polygon clipped; removed_area_fraction says how much
                         (measured in image pixels: the removed ground area is unbounded)
    unreliable           no reliable part at all

Usage:
    python geolocate.py --localization_json localization_results.json \
                        --camera_config camera_configs/dji_mini4pro.json \
                        --metadata_csv frame_metadata.csv \
                        [--output_format geojson]

    # or read metadata from the images themselves (EXIF GPS + XMP gimbal/altitude):
    python geolocate.py --localization_json localization_results.json \
                        --camera_config camera_configs/dji_mini4pro.json \
                        --exif_image_dir custom_test/thermal

    # height from the laser rangefinder where it has a reading (else the altitude field):
    python geolocate.py ... --exif_image_dir <raw images> --altitude_source laser

Assumptions: flat ground (Terrain interface for a DEM later), GPS antenna at the
camera, metadata time-aligned with the frame, absolute gimbal angles.
"""

import argparse
import json
import os

import numpy as np

from geoloc.camera import load_camera_config
from geoloc.metadata import ExifXmpLoader, LaserHeightLoader, MetadataError, SidecarCsvLoader
from geoloc.projection import (FLAT_TERRAIN, ReliabilityLimits, ground_sample_distance_m,
                               pixel_runs_ground_area, project_contour)

PROCESS_SIZE = 256   # must match localize.py (pixel_256 coordinate space)


def touches_frame_edge(region) -> bool:
    edge = PROCESS_SIZE - 1
    return any(p["x"] <= 0 or p["y"] <= 0 or p["x"] >= edge or p["y"] >= edge
               for p in region["pixel_256"])


def finite_or_none(x, digits):
    return round(float(x), digits) if np.isfinite(x) else None


def encode_vertex(lat, lon, north_east, is_clip_vertex, altitude_m):
    ground_range = float(np.hypot(*north_east))
    vertex = {"lat": round(float(lat), 8), "lon": round(float(lon), 8),
              "ground_range_m": round(ground_range, 1),
              "depression_deg": round(float(np.degrees(np.arctan2(altitude_m, ground_range))), 2)}
    if is_clip_vertex:
        vertex["clip_vertex"] = True    # created by clipping, not a fire-contour vertex
    return vertex


def geolocate_region(region, camera, meta, terrain=FLAT_TERRAIN,
                     limits=ReliabilityLimits()) -> str:
    """Fills the region's GPS fields in place; returns its geolocation_status:
    'ok', 'clipped_to_reliable' or 'unreliable' (no reliable part)."""
    W, H = region["original_wh"]["width"], region["original_wh"]["height"]
    intr = camera.intrinsics_for(W, H)
    pts = np.array([[p["x"] * W, p["y"] * H] for p in region["normalized"]])
    pc = project_contour(pts, intr, meta.pose(), meta.latitude_deg, meta.longitude_deg,
                         terrain, limits)

    if pc.polygon_north_east is None:
        status = "unreliable"
    else:
        status = "clipped_to_reliable" if pc.clipped else "ok"
    reasons = {}
    for s in pc.vertex_status:
        if s != "ok":
            reasons[s] = reasons.get(s, 0) + 1

    # The outline is a fire zone (hot spots merged by dilation, plus the gaps between
    # them); the burning area comes from the hot pixels themselves.
    fire_area, fire_excluded = None, None
    if region.get("fire_pixels_256"):
        fire_area, fire_excluded = pixel_runs_ground_area(
            region["fire_pixels_256"], W / PROCESS_SIZE, H / PROCESS_SIZE, intr, meta.pose(),
            terrain, limits)
        if fire_excluded >= 1.0:
            fire_area = None

    region.update({
        "gps_polygon":    None if pc.polygon_north_east is None else
                          [encode_vertex(lat, lon, ne, clip, meta.altitude_agl_m)
                           for (lat, lon), ne, clip in zip(pc.polygon_latlon, pc.polygon_north_east,
                                                           pc.polygon_clip_vertex)],
        "gps_centroid":   None if pc.centroid_latlon is None else
                          {"lat": round(pc.centroid_latlon[0], 8), "lon": round(pc.centroid_latlon[1], 8)},
        "ground_area_m2": None if pc.area_m2 is None else round(pc.area_m2, 2),
        "fire_area_m2":   None if fire_area is None else round(fire_area, 2),
        "fire_area_excluded_fraction": None if fire_excluded is None else round(fire_excluded, 4),
        "clipped":        pc.clipped,
        "removed_area_fraction": round(pc.removed_area_fraction, 4),
        "unreliable_vertices": reasons,     # original contour vertices, by reason
        "touches_frame_edge": touches_frame_edge(region),
        "geolocation_status": status,
    })
    return status


def geolocate_results(results, camera, loader, image_dir=None, merge_kernel_size=0,
                      terrain=FLAT_TERRAIN, limits=ReliabilityLimits()) -> dict:
    """Geolocates every localized frame in place. Returns counts for the summary."""
    counts = {"frames_geolocated": 0, "frames_missing_metadata": 0, "regions_geolocated": 0,
              "regions_clipped_to_reliable": 0, "regions_unreliable": 0,
              "frames_by_altitude_reference": {}}

    for result in results:
        regions = result.get("fire_regions") or []
        if not result.get("localization_run") or not regions:
            continue

        image_path = os.path.join(image_dir, result["filename"]) if image_dir else None
        try:
            meta = loader.load(result["frame_id"], image_path)
        except MetadataError as e:
            counts["frames_missing_metadata"] += 1
            result["geolocation"] = {"status": "missing_metadata", "note": str(e)}
            for region in regions:
                region["geolocation_status"] = "missing_metadata"
            print(f"  [!] {result['frame_id']}: {e}")
            continue

        W, H = regions[0]["original_wh"]["width"], regions[0]["original_wh"]["height"]
        gsd = ground_sample_distance_m(camera.intrinsics_for(W, H), meta.pose(), terrain)
        warnings = []
        if meta.altitude_reference != "agl":
            warnings.append(f"altitude is relative to '{meta.altitude_reference}', used as AGL "
                            f"(assumes ground level with that reference)")
        if meta.altitude_note:
            warnings.append(meta.altitude_note)
        if abs(meta.gimbal_pitch_deg + 90.0) > 5.0:
            warnings.append(f"oblique view (pitch {meta.gimbal_pitch_deg}°): flat-ground errors grow off-nadir")

        result["geolocation"] = {
            "status": "ok",
            "camera": camera.name,
            "metadata_source": meta.source,
            "timestamp": meta.timestamp.isoformat() if meta.timestamp else None,
            "drone": {"lat": meta.latitude_deg, "lon": meta.longitude_deg,
                      "altitude_agl_m": meta.altitude_agl_m,
                      "altitude_reference": meta.altitude_reference,
                      "altitude_note": meta.altitude_note,
                      "gimbal_yaw_deg": meta.gimbal_yaw_deg,
                      "gimbal_pitch_deg": meta.gimbal_pitch_deg,
                      "gimbal_roll_deg": meta.gimbal_roll_deg},
            # None when the image center itself does not reach the ground.
            "gsd_center_m": finite_or_none(gsd, 4),
            # localize.py dilates the mask by merge_kernel_size px at 256 scale, pushing
            # polygons outward by about half the kernel. Approximate, at the image center.
            "dilation_buffer_m": finite_or_none((merge_kernel_size // 2) * (W / PROCESS_SIZE) * gsd, 2),
            "warnings": warnings,
        }

        statuses = [geolocate_region(region, camera, meta, terrain, limits) for region in regions]
        counts["frames_geolocated"] += 1
        by_ref = counts["frames_by_altitude_reference"]
        by_ref[meta.altitude_reference] = by_ref.get(meta.altitude_reference, 0) + 1
        counts["regions_geolocated"] += statuses.count("ok")
        counts["regions_clipped_to_reliable"] += statuses.count("clipped_to_reliable")
        counts["regions_unreliable"] += statuses.count("unreliable")
        print(f"  [✓] {result['frame_id']:<32} regions={len(regions)}  ok={statuses.count('ok')}"
              f"  clipped={statuses.count('clipped_to_reliable')}"
              f"  unreliable={statuses.count('unreliable')}  gsd={gsd:.3f} m/px")

    return counts


def to_geojson(results):
    """One Polygon feature per geolocated region (ok or clipped_to_reliable).
    GeoJSON order is [lon, lat]."""
    features = []
    for result in results:
        for i, region in enumerate(result.get("fire_regions") or []):
            if region.get("geolocation_status") not in ("ok", "clipped_to_reliable"):
                continue
            ring = [[p["lon"], p["lat"]] for p in region["gps_polygon"]]
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "frame_id":              result["frame_id"],
                    "region_index":          i,
                    "classifier_prediction": result.get("classifier_prediction"),
                    "centroid_lat":          region["gps_centroid"]["lat"],
                    "centroid_lon":          region["gps_centroid"]["lon"],
                    "ground_area_m2":        region["ground_area_m2"],
                    "fire_area_m2":          region.get("fire_area_m2"),
                    "clipped":               region["clipped"],
                    "removed_area_fraction": region["removed_area_fraction"],
                    "touches_frame_edge":    region["touches_frame_edge"],
                    "timestamp":             result["geolocation"]["timestamp"],
                    "dilation_buffer_m":     result["geolocation"]["dilation_buffer_m"],
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            })
    return {"type": "FeatureCollection", "features": features}


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Fire Contour Geolocation")
    parser.add_argument("--localization_json", required=True)
    parser.add_argument("--camera_config",     required=True,
                        help="Camera JSON (see camera_configs/template.json)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--metadata_csv",      help="Sidecar CSV, one row per frame_id")
    source.add_argument("--exif_image_dir",    help="Folder of the original images to read EXIF/XMP from")
    parser.add_argument("--altitude_source",   default="metadata", choices=["metadata", "laser"],
                        help="With --exif_image_dir: 'metadata' uses the altitude field; 'laser' uses "
                             "drone altitude minus the laser rangefinder's ground point where there "
                             "is a reading (default: metadata)")
    parser.add_argument("--laser_max_terrain_offset_m", type=float, default=50.0,
                        help="Reject a laser height whose ground point is farther than this above or "
                             "below the altitude field's reference (default: 50)")
    parser.add_argument("--csv_altitude_reference",default="agl", choices=["agl", "takeoff", "msl"],
                        help="What the CSV altitude column is relative to (default: agl)")
    parser.add_argument("--min_depression_deg", type=float, default=10.0,
                        help="Vertices whose ray is less than this far below the horizon are marked "
                             "unreliable (default: 10). Set a negative value to disable.")
    parser.add_argument("--max_range_m",       type=float, default=None,
                        help="Vertices farther than this from the drone's ground position are "
                             "marked unreliable (default: off)")
    parser.add_argument("--output_json",       default="geolocated_results.json")
    parser.add_argument("--output_format",     default="json", choices=["json", "geojson"])
    args = parser.parse_args()

    camera = load_camera_config(args.camera_config)
    if args.metadata_csv:
        if args.altitude_source != "metadata":
            parser.error("--altitude_source laser needs --exif_image_dir")
        loader = SidecarCsvLoader(args.metadata_csv, args.csv_altitude_reference)
    elif args.altitude_source == "laser":
        loader = LaserHeightLoader(camera, args.laser_max_terrain_offset_m)
    else:
        loader = ExifXmpLoader(camera)

    with open(args.localization_json) as f:
        data = json.load(f)
    results = data["results"]
    summary = data.get("summary", {})
    merge_kernel_size = summary.get("parameters", {}).get("merge_kernel_size", 0)

    print(f"[✓] Camera: {camera.name}   Metadata: {args.metadata_csv or args.exif_image_dir}\n")
    limits = ReliabilityLimits(args.min_depression_deg if args.min_depression_deg >= 0 else None,
                               args.max_range_m)
    counts = geolocate_results(results, camera, loader, args.exif_image_dir, merge_kernel_size,
                               limits=limits)
    summary["geolocation"] = {**counts, "camera": camera.name, "terrain": "flat",
                              "altitude_source": args.altitude_source,
                              "min_depression_deg": limits.min_depression_deg,
                              "max_range_m": limits.max_range_m}

    base = os.path.splitext(args.output_json)[0]
    if args.output_format == "json":
        out_path = base + ".json"
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "results": results}, f, indent=2, allow_nan=False)
    else:
        out_path = base + ".geojson"
        with open(out_path, "w") as f:
            json.dump(to_geojson(results), f, indent=2, allow_nan=False)

    print(f"\n[✓] Frames geolocated={counts['frames_geolocated']}  missing metadata={counts['frames_missing_metadata']}"
          f"  regions ok={counts['regions_geolocated']}  clipped={counts['regions_clipped_to_reliable']}"
          f"  unreliable={counts['regions_unreliable']}")
    print(f"[✓] Saved: {out_path}  ({args.output_format})")


if __name__ == "__main__":
    main()
