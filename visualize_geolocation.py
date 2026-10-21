"""
visualize_geolocation.py — Draw geolocate.py results on the thermal frames
==========================================================================
Reads geolocated_results.json produced by geolocate.py and, for every
geolocated frame, draws on the original thermal image:
    - each fire region, numbered: the geolocated part filled, any part clipped
      away as unreliable hatched
    - ground-distance rings around the drone's ground position (default
      100 / 200 / 400 m), projected into the image
    - the reliability limit of the run (--min_depression_deg / --max_range_m of
      geolocate.py) and the horizon
A side panel lists each region's GPS centroid, distance, zone area (the outline)
and burning area (the hot pixels), when localize.py provided them.

Rendering only. The geometry comes from geoloc/ with the same camera config
and drone pose geolocate.py used (read back from the JSON), so what is drawn
is what was computed.

Usage:
    python visualize_geolocation.py --geolocated_json geolocated_results.json \
                                    --camera_config camera_configs/<your_camera>.json \
                                    --thermal_dir custom_test/thermal \
                                    [--vis_dir geolocation_vis] [--rings_m 100,200,400]
"""

import argparse
import json
import math
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image

from geoloc.camera import load_camera_config
from geoloc.projection import (FLAT_TERRAIN, CameraPose, ReliabilityLimits, camera_to_ned,
                               depression_deg, ground_to_pixels, latlon_to_offset,
                               normalized_to_pixels, pixels_to_rays_ned, reliable_radius_m)

EDGE_MARGIN_PX = 50     # overlay points farther than this outside the frame are dropped
HATCH_PX = 6            # stripe width of the "not geolocated" hatching
OUTLINE = [patheffects.withStroke(linewidth=3, foreground="black")]


# ==================== Geometry (pixels of ground features) ====================
def ground_ring_px(radius_m, intr, pose, n=720):
    """(n, 2) pixels of the ground circle of radius_m around the drone's ground
    position; NaN where it is behind the camera or far outside the frame."""
    a = np.linspace(0.0, 2.0 * math.pi, n)
    px = ground_to_pixels(np.column_stack((radius_m * np.cos(a), radius_m * np.sin(a))), intr, pose)
    outside = ((px[:, 0] < -EDGE_MARGIN_PX) | (px[:, 0] > intr.width + EDGE_MARGIN_PX) |
               (px[:, 1] < -EDGE_MARGIN_PX) | (px[:, 1] > intr.height + EDGE_MARGIN_PX))
    px[outside] = np.nan
    return px


def horizon_px(intr, pose, n=200):
    """(n, 2) pixels whose ray is horizontal. In undistorted normalized coordinates
    the ray's down component is R[2]·(x, y, 1), so the horizon is the line where
    that is zero. None if the horizon runs vertically through the image."""
    a, b, c = camera_to_ned(pose)[2]
    if abs(b) < 1e-9:
        return None
    x = np.linspace(-intr.cx / intr.fx, (intr.width - intr.cx) / intr.fx, n)
    return normalized_to_pixels(np.column_stack((x, -(a * x + c) / b)), intr)


def unreliable_pixels(intr, pose, limits):
    """(H, W) bool: pixels whose ray geolocate.py does not trust, by the same rule
    it applies to contour vertices (above the horizon, too shallow, too far)."""
    v, u = np.mgrid[0:intr.height, 0:intr.width] + 0.5
    rays = pixels_to_rays_ned(np.column_stack((u.ravel(), v.ravel())), intr, pose)
    ne = FLAT_TERRAIN.intersect(pose.altitude_agl_m, rays)
    ground_range = np.hypot(ne[:, 0], ne[:, 1])
    bad = np.isnan(ground_range)
    if limits.min_depression_deg is not None:
        bad |= depression_deg(rays) < limits.min_depression_deg
    if limits.max_range_m is not None:
        bad |= ground_range > limits.max_range_m
    return bad.reshape(intr.height, intr.width)


def latlon_to_px(lat0, lon0, latlon, intr, pose):
    north, east = latlon_to_offset(lat0, lon0, latlon[:, 0], latlon[:, 1])
    return ground_to_pixels(np.column_stack((north, east)), intr, pose)


# ==================== Drawing ====================
def label_ring(ax, px, text, color, W, H, at=0.5, ha="left"):
    """Label a ring at its visible point closest to x = at·W."""
    visible = ~np.isnan(px[:, 0]) & (px[:, 0] >= 0) & (px[:, 0] <= W) & (px[:, 1] >= 0) & (px[:, 1] <= H)
    if not visible.any():
        return
    i = np.flatnonzero(visible)[np.argmin(np.abs(px[visible, 0] - at * W))]
    ax.text(px[i, 0] + (4 if ha == "left" else -4), px[i, 1] - 3, text, color=color, fontsize=8,
            ha=ha, path_effects=OUTLINE)


def visualize_frame(result, camera, thermal_dir, rings_m, limits, out_dir):
    regions = [r for r in result.get("fire_regions") or [] if "geolocation_status" in r]
    W, H = regions[0]["original_wh"]["width"], regions[0]["original_wh"]["height"]
    intr = camera.intrinsics_for(W, H)
    drone = result["geolocation"]["drone"]
    pose = CameraPose(drone["altitude_agl_m"], drone["gimbal_yaw_deg"],
                      drone["gimbal_pitch_deg"], drone["gimbal_roll_deg"])
    lat0, lon0 = drone["lat"], drone["lon"]

    fig, (ax, panel) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [W / H * 1.0, 0.75]})
    fig.suptitle(f"{result['frame_id']}  |  altitude {pose.altitude_agl_m:.1f} m ({drone['altitude_reference']})"
                 f"  |  pitch {pose.pitch_deg}°  yaw {pose.yaw_deg}°  |  {camera.name}",
                 fontsize=10, fontweight="bold")
    ax.imshow(Image.open(os.path.join(thermal_dir, result["filename"])).convert("RGB"))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    for r in rings_m:
        px = ground_ring_px(r, intr, pose)
        ax.plot(px[:, 0], px[:, 1], color="white", linewidth=0.8, alpha=0.8)
        label_ring(ax, px, f"{r:g} m", "white", W, H)

    radius = reliable_radius_m(pose.altitude_agl_m, limits)
    if radius is not None:
        px = ground_ring_px(radius, intr, pose)
        ax.plot(px[:, 0], px[:, 1], color="red", linewidth=1.5, linestyle="--")
        parts = [f"{limits.min_depression_deg:g}° below horizon"] if limits.min_depression_deg else []
        parts += [f"{limits.max_range_m:g} m"] if limits.max_range_m is not None else []
        label_ring(ax, px, f"reliable limit ({', '.join(parts)}) = {radius:.0f} m", "red", W, H,
                   at=0.98, ha="right")
    hz = horizon_px(intr, pose)
    if hz is not None:
        ax.plot(hz[:, 0], hz[:, 1], color="cyan", linewidth=1, linestyle=":")

    lines = []
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(regions), 1)))
    hatch = np.zeros((H, W, 4))     # "not geolocated" stripes, only where a region is unreliable
    bad = stripes = None
    for i, (region, color) in enumerate(zip(regions, colors), start=1):
        original = np.array([[p["x"] * W, p["y"] * H] for p in region["normalized"]])
        status = region["geolocation_status"]
        if region["gps_polygon"] is not None:
            kept = latlon_to_px(lat0, lon0, np.array([[v["lat"], v["lon"]] for v in region["gps_polygon"]]),
                                intr, pose)
            ax.add_patch(MplPolygon(kept, closed=True, facecolor=(*color[:3], 0.3),
                                    edgecolor=(*color[:3], 1.0), linewidth=2))
        if status != "ok":    # hatch the part that was not geolocated, outline the original
            if bad is None:
                bad = unreliable_pixels(intr, pose, limits)
                v, u = np.mgrid[0:H, 0:W]
                stripes = ((u + v) // HATCH_PX) % 2 == 0
            inside = np.zeros((H, W), np.uint8)
            cv2.fillPoly(inside, [np.round(original).astype(np.int32)], 1)
            hatch[(inside > 0) & bad & stripes] = (*color[:3], 0.85)
            ax.add_patch(MplPolygon(original, closed=True, fill=False,
                                    edgecolor=(*color[:3], 0.8), linewidth=1, linestyle="--"))

        if region["gps_centroid"] is not None:
            c = region["gps_centroid"]
            cpx = latlon_to_px(lat0, lon0, np.array([[c["lat"], c["lon"]]]), intr, pose)[0]
            north, east = latlon_to_offset(lat0, lon0, c["lat"], c["lon"])
            note = f"  [clipped {region['removed_area_fraction']:.0%}]" if region["clipped"] else ""
            burning = (f"  |  burning {region['fire_area_m2']:,.0f} m²"
                       if region.get("fire_area_m2") is not None else "")
            lines.append((color, f"#{i}  {c['lat']:.6f}, {c['lon']:.6f}  |  {math.hypot(north, east):.0f} m away\n"
                                 f"     zone {region['ground_area_m2']:,.0f} m²{burning}{note}"))
        else:
            cpx = original.mean(axis=0)
            lines.append((color, f"#{i}  {status}: no reliable part"))
        ax.text(cpx[0], cpx[1], f"#{i}", color="white", fontsize=10, fontweight="bold",
                ha="center", va="center", path_effects=OUTLINE)

    if bad is not None:
        ax.imshow(hatch)
    panel.axis("off")
    panel.set_title("Fire regions (GPS centroid, distance from drone, zone and burning area)", fontsize=9)
    y = 0.97
    for color, text in lines:
        panel.text(0.0, y, text, color=color[:3], fontsize=8.5, family="monospace", va="top",
                   transform=panel.transAxes)
        y -= 0.075
    panel.text(0.0, 0.02, f"drone {lat0:.6f}, {lon0:.6f}\n"
                          f"white: ground distance from drone  |  red dashed: reliable limit\n"
                          f"cyan dotted: horizon  |  hatched: not geolocated (unreliable)",
               fontsize=8, va="bottom", transform=panel.transAxes)

    plt.tight_layout()
    path = os.path.join(out_dir, f"{result['frame_id']}_geolocation.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Draw geolocation results on the thermal frames")
    parser.add_argument("--geolocated_json", required=True, help="Output of geolocate.py (json format)")
    parser.add_argument("--camera_config",   required=True, help="The camera config geolocate.py used")
    parser.add_argument("--thermal_dir",     required=True, help="Folder of the thermal images")
    parser.add_argument("--vis_dir",         default="geolocation_vis")
    parser.add_argument("--rings_m",         default="100,200,400",
                        help="Ground-distance rings to draw, in metres (default: 100,200,400)")
    args = parser.parse_args()

    with open(args.geolocated_json) as f:
        data = json.load(f)
    camera = load_camera_config(args.camera_config)
    run = data["summary"].get("geolocation", {})
    limits = ReliabilityLimits(run.get("min_depression_deg"), run.get("max_range_m"))
    rings_m = [float(r) for r in args.rings_m.split(",") if r.strip()]
    os.makedirs(args.vis_dir, exist_ok=True)

    drawn = 0
    for result in data["results"]:
        if (result.get("geolocation") or {}).get("status") != "ok" or not result.get("fire_regions"):
            continue
        path = visualize_frame(result, camera, args.thermal_dir, rings_m, limits, args.vis_dir)
        drawn += 1
        print(f"  [✓] {result['frame_id']:<32} → {path}")

    print(f"\n[✓] Drew {drawn} frame(s) into {args.vis_dir}/")


if __name__ == "__main__":
    main()
