"""
laser_check.py — Geolocation of the boresight vs the M30T laser rangefinder (FLAME 3)
=====================================================================================
For every raw RGB frame whose laser has a reading, projects the camera boresight
(the principal point of the camera config) to the ground with the same loaders
geolocate.py uses, and compares it with the laser's ground point
(drone-dji:LRFTargetLat/Lon).

Run it once per altitude source:
    metadata  take-off-relative altitude, the honest baseline: no rangefinder needed.
    laser     height above the laser ground point. The laser's distance then sets
              the height, so the remaining error mostly tests the angles and DJI's
              own projection, not altitude. It is not an independent accuracy number.

The boresight pixel is independent of the focal length, so this checks altitude and
angles, not fx.

Usage:
    python experiments/flame3/laser_check.py --dataset_root data/flame3 \
        --camera_config camera_configs/dji_m30t_thermal.json --output_csv laser_check.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geoloc.camera import load_camera_config                                   # noqa: E402
from geoloc.metadata import ExifXmpLoader, LaserHeightLoader, MetadataError, read_xmp  # noqa: E402
from geoloc.projection import latlon_to_offset, pixels_to_ground                # noqa: E402

CLASSES = ("Fire", "No Fire")
DEPRESSION_BINS = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 90)]


def frames(root):
    for cls in CLASSES:
        raw_dir = os.path.join(root, cls, "RGB", "Raw")
        for name in sorted(os.listdir(raw_dir)):
            yield cls, name, os.path.join(raw_dir, name)


def main():
    parser = argparse.ArgumentParser(description="Boresight geolocation vs laser rangefinder")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--camera_config", required=True)
    parser.add_argument("--laser_max_terrain_offset_m", type=float, default=50.0)
    parser.add_argument("--output_csv", default=None, help="Per-frame results (optional)")
    args = parser.parse_args()

    camera = load_camera_config(args.camera_config)
    intr = camera.intrinsics_for(camera.calibration_width, camera.calibration_height)
    boresight = np.array([[intr.cx, intr.cy]])
    loaders = {"metadata": ExifXmpLoader(camera),
               "laser": LaserHeightLoader(camera, args.laser_max_terrain_offset_m)}

    rows = []
    for cls, name, path in frames(args.dataset_root):
        xmp = read_xmp(path)
        if float(xmp.get("drone-dji:LRFTargetDistance", 0) or 0) <= 0:
            continue
        laser_lat, laser_lon = float(xmp["drone-dji:LRFTargetLat"]), float(xmp["drone-dji:LRFTargetLon"])
        row = {"class": cls, "file": name}
        for source, loader in loaders.items():
            try:
                meta = loader.load(name, path)
            except MetadataError as e:
                print(f"  [!] {cls}/{name} ({source}): {e}")
                break
            ln, le = latlon_to_offset(meta.latitude_deg, meta.longitude_deg, laser_lat, laser_lon)
            n, e = pixels_to_ground(boresight, intr, meta.pose())[0]
            row.update({"pitch_deg": meta.gimbal_pitch_deg,
                        "laser_range_m": float(np.hypot(ln, le)),
                        f"{source}_altitude_m": meta.altitude_agl_m,
                        f"{source}_reference": meta.altitude_reference,
                        f"{source}_error_m": float(np.hypot(n - ln, e - le))})   # NaN above horizon
        else:
            rows.append(row)

    print(f"Frames with a laser reading: {len(rows)}")
    used = [r for r in rows if r["laser_reference"] == "laser_target"]
    print(f"  laser height used: {len(used)}   rejected (fell back to take-off altitude): "
          f"{len(rows) - len(used)}\n")
    print("Boresight error vs laser ground point, frames where the laser height was accepted")
    print(f"{'depression':>12} {'n':>4} {'range med':>10} | {'take-off alt: med':>18} {'p90':>7} {'% of range':>11}"
          f" | {'laser height: med':>18} {'p90':>6}")
    for lo, hi in DEPRESSION_BINS:
        b = [r for r in used if lo <= -r["pitch_deg"] < hi]
        if not b:
            continue
        rng = np.array([r["laser_range_m"] for r in b])
        em = np.array([r["metadata_error_m"] for r in b])
        el = np.array([r["laser_error_m"] for r in b])
        print(f"{f'{lo}-{hi} deg':>12} {len(b):>4} {np.median(rng):>9.0f}m | {np.nanmedian(em):>17.1f}m"
              f" {np.nanpercentile(em, 90):>6.1f}m {np.nanmedian(em / rng) * 100:>10.1f}%"
              f" | {np.nanmedian(el):>17.1f}m {np.nanpercentile(el, 90):>5.1f}m")

    if args.output_csv:
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[ok] Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
