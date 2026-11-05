"""
run_absolute_threshold.py — Otsu vs absolute-temperature localization on FLAME 3
================================================================================
No classifier: every frame goes to localization. Measures, on all 738 frames:
    No Fire frames   how many produce at least one region (false-alarm frames)
    Fire frames      how many produce no region (misses)
and how much of the truly hot area (>= seed temperature) ends up inside the outlines,
including what the contour size filter throws away. On the Fire test subset, how far the absolute-mode outlines and areas move
from Otsu's, plus the geolocation status counts. Repeated per --low_temp_c value.

Writes summary.json and per_frame.csv. The hottest pixel over the No Fire frames is
the margin to the 80 °C seed. FLAME 3's No Fire frames hold no engines or sun-heated
metal, so this cannot show how the rule behaves on those.

Usage:
    python experiments/flame3/run_absolute_threshold.py --dataset_root data/flame3 \
        --test_subset_dir data/flame3_test/Fire/Thermal \
        --camera_config camera_configs/dji_m30t_thermal.json --output_dir results/run_C_absolute
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import geolocate                                                          # noqa: E402
import localize                                                           # noqa: E402
from detect_by_temperature import load_temperatures                       # noqa: E402
from geoloc.camera import load_camera_config                              # noqa: E402
from geoloc.metadata import ExifXmpLoader                                 # noqa: E402

CLASSES = ("Fire", "No Fire")
# localize.py's defaults
BLUR, SCALE, MERGE, MIN_AREA, MAX_AREA, EPSILON = 5, 0.85, 15, 200, 50000, 0.01


def regions_from_mask(mask, original_size):
    dilated = localize.dilate_mask_for_merging(mask, MERGE)
    contours = localize.extract_contours(dilated, MIN_AREA, MAX_AREA, EPSILON)
    return contours, localize.encode_contours(contours, original_size,
                                              localize.fire_pixel_runs(mask, dilated, contours))


def area_filter_counts(mask):
    """How many merged regions the contour size filter drops: (kept, too small, too big)."""
    found, _ = cv2.findContours(localize.dilate_mask_for_merging(mask, MERGE), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in found]
    small, big = sum(a < MIN_AREA for a in areas), sum(a > MAX_AREA for a in areas)
    return len(areas) - small - big, small, big


def hot_cells(temps, seed_temp_c):
    """Processing-grid cells that are at least half covered by pixels >= seed_temp_c."""
    size = (localize.PROCESS_SIZE, localize.PROCESS_SIZE)
    return cv2.resize((temps >= seed_temp_c).astype(np.float32), size, interpolation=cv2.INTER_AREA) >= 0.5


def filled(contours):
    m = np.zeros((localize.PROCESS_SIZE, localize.PROCESS_SIZE), np.uint8)
    if contours:
        cv2.fillPoly(m, list(contours), 1)
    return m > 0


def iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else float("nan")


def stats(values):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    return {"n": len(v), "median": round(float(np.median(v)), 3), "p10": round(float(np.percentile(v, 10)), 3),
            "p90": round(float(np.percentile(v, 90)), 3)}


def geolocate_frames(frames, camera, raw_dir):
    """frames: {frame_id: encoded regions}. Returns (status counts, {frame_id: result})."""
    results = [{"frame_id": k, "filename": k + ".JPG", "localization_run": True, "fire_regions": v}
               for k, v in frames.items()]
    with contextlib.redirect_stdout(io.StringIO()):
        geolocate.geolocate_results(results, camera, ExifXmpLoader(camera), raw_dir, MERGE)
    statuses = [g["geolocation_status"] for r in results for g in r["fire_regions"]]
    counts = {s: statuses.count(s) for s in ("ok", "clipped_to_reliable", "unreliable", "missing_metadata")}
    return counts, {r["frame_id"]: r for r in results}


def frame_areas(result):
    regions = result["fire_regions"]
    zone = sum(g.get("ground_area_m2") or 0.0 for g in regions)
    fire = sum(g.get("fire_area_m2") or 0.0 for g in regions)
    return zone, fire


def main():
    parser = argparse.ArgumentParser(description="Otsu vs absolute-temperature localization on FLAME 3")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--test_subset_dir", required=True, help="Folder whose images name the Fire test frames")
    parser.add_argument("--camera_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed_temp_c", type=float, default=80.0)
    parser.add_argument("--low_temps_c", default="60,50")
    parser.add_argument("--min_seed_pixels", type=int, default=10)
    args = parser.parse_args()

    lows = [float(x) for x in args.low_temps_c.split(",")]
    camera = load_camera_config(args.camera_config)
    subset = {os.path.splitext(f)[0] for f in os.listdir(args.test_subset_dir)
              if f.upper().endswith(".JPG")}
    os.makedirs(args.output_dir, exist_ok=True)

    per_frame = []
    per_class = {c: {"frames": 0, "otsu_frames_with_regions": 0, "otsu_regions": 0,
                     "absolute": {str(t): {"frames_with_regions": 0, "regions": 0, "frames_without_regions": [],
                                           "dropped_too_small": 0, "dropped_too_big": 0, "hot_cells_inside": 0}
                                  for t in lows},
                     "otsu_too_small": 0, "otsu_too_big": 0, "otsu_frames_only_too_big": 0,
                     "hot_cells": 0, "hot_cells_inside_otsu": 0,
                     "hottest_pixel_c": [], "frames_with_10_px_over_seed": 0} for c in CLASSES}
    subset_otsu, subset_abs = {}, {t: {} for t in lows}
    outline_iou = {t: [] for t in lows}
    region_iou = {t: [] for t in lows}

    for cls in CLASSES:
        thermal_dir = os.path.join(args.dataset_root, cls, "Thermal", "Raw JPG")
        temp_dir = os.path.join(args.dataset_root, cls, "Thermal", "Celsius TIFF")
        c = per_class[cls]
        for name in sorted(os.listdir(thermal_dir)):
            stem = os.path.splitext(name)[0]
            gray, size = localize.load_thermal_gray(os.path.join(thermal_dir, name))
            temps = load_temperatures(os.path.join(temp_dir, stem + ".TIFF"))
            otsu_mask = localize.otsu_fire_mask(gray, BLUR, SCALE)[0]
            otsu_contours, otsu_regions = regions_from_mask(otsu_mask, size)

            c["frames"] += 1
            c["otsu_frames_with_regions"] += bool(otsu_regions)
            c["otsu_regions"] += len(otsu_regions)
            kept, small, big = area_filter_counts(otsu_mask)
            c["otsu_too_small"] += small
            c["otsu_too_big"] += big
            c["otsu_frames_only_too_big"] += kept == 0 and big > 0
            hot = hot_cells(temps, args.seed_temp_c)
            c["hot_cells"] += int(np.count_nonzero(hot))
            c["hot_cells_inside_otsu"] += int(np.count_nonzero(hot & filled(otsu_contours)))
            hottest = float(np.nanmax(temps))
            c["hottest_pixel_c"].append(hottest)
            c["frames_with_10_px_over_seed"] += int(np.count_nonzero(temps >= args.seed_temp_c)) >= args.min_seed_pixels
            row = {"class": cls, "frame": stem, "hottest_c": round(hottest, 1), "otsu_regions": len(otsu_regions)}
            test_frame = cls == "Fire" and stem in subset
            if test_frame:
                subset_otsu[stem] = otsu_regions

            for t in lows:
                mask, seeds = localize.absolute_fire_mask(temps, args.seed_temp_c, t, args.min_seed_pixels)
                contours, regions = regions_from_mask(mask, size)
                a = c["absolute"][str(t)]
                a["frames_with_regions"] += bool(regions)
                a["regions"] += len(regions)
                _, small, big = area_filter_counts(mask)
                a["dropped_too_small"] += small
                a["dropped_too_big"] += big
                a["hot_cells_inside"] += int(np.count_nonzero(hot & filled(contours)))
                if not regions:
                    a["frames_without_regions"].append(stem)
                row[f"absolute_{t:g}_regions"] = len(regions)
                row[f"absolute_{t:g}_seed_pixels"] = seeds
                if test_frame:
                    subset_abs[t][stem] = regions
                    outline_iou[t].append(iou(filled(otsu_contours), filled(contours)))
                    # each Otsu region against the absolute region it overlaps most
                    masks = [filled([k]) for k in contours]
                    for k in otsu_contours:
                        m = filled([k])
                        region_iou[t].append(max((iou(m, other) for other in masks), default=0.0))
            per_frame.append(row)

    summary = {"parameters": {"seed_temp_c": args.seed_temp_c, "low_temps_c": lows,
                              "min_seed_pixels": args.min_seed_pixels, "merge_kernel_size": MERGE,
                              "min_contour_area": MIN_AREA, "max_contour_area": MAX_AREA,
                              "otsu_threshold_scale": SCALE},
               "classes": {}}
    for cls in CLASSES:
        c = per_class[cls]
        hottest = np.array(c.pop("hottest_pixel_c"))
        summary["classes"][cls] = {**c, "hottest_pixel_c": {"min": round(float(hottest.min()), 1),
                                                            "median": round(float(np.median(hottest)), 1),
                                                            "max": round(float(hottest.max()), 1)}}

    raw_dir = os.path.join(args.dataset_root, "Fire", "RGB", "Raw")
    otsu_counts, otsu_geo = geolocate_frames(subset_otsu, camera, raw_dir)
    subset_summary = {"frames": len(subset), "otsu": {"geolocation_status": otsu_counts}, "absolute": {}}
    for t in lows:
        counts, abs_geo = geolocate_frames(subset_abs[t], camera, raw_dir)
        zone_ratio, fire_ratio, zone_abs, fire_abs = [], [], [], []
        for frame in subset:
            oz, of = frame_areas(otsu_geo[frame])
            az, af = frame_areas(abs_geo[frame])
            zone_abs.append(az)
            fire_abs.append(af)
            zone_ratio.append(az / oz if oz else float("nan"))
            fire_ratio.append(af / of if of else float("nan"))
        subset_summary["absolute"][str(t)] = {
            "geolocation_status": counts,
            "frame_outline_iou_vs_otsu": stats(outline_iou[t]),
            "region_iou_vs_otsu": stats(region_iou[t]),
            "frame_zone_area_ratio_vs_otsu": stats(zone_ratio),
            "frame_fire_area_ratio_vs_otsu": stats(fire_ratio),
            "total_zone_area_m2": round(sum(zone_abs)),
            "total_fire_area_m2": round(sum(fire_abs)),
        }
    subset_summary["otsu"]["total_zone_area_m2"] = round(sum(frame_areas(g)[0] for g in otsu_geo.values()))
    subset_summary["otsu"]["total_fire_area_m2"] = round(sum(frame_areas(g)[1] for g in otsu_geo.values()))
    summary["fire_test_subset"] = subset_summary

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, "per_frame.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame[0]))
        writer.writeheader()
        writer.writerows(per_frame)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
