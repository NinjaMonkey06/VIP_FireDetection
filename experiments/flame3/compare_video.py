"""
compare_video.py — Does frame-by-frame video processing give the same, consistent results?
==========================================================================================
Takes process_video.py's output for a FLAME 3 test video (make_video.py) and checks:

1. Video vs stills: the same frames processed from the original still JPGs (same
   telemetry, same settings). Per frame: whether each found regions, outline IoU
   in the 256x256 processing space, and how far the video's geolocated border
   points lie from the stills' border (metres on the ground).
2. Frame-to-frame consistency: consecutive frames see the same fire from slightly
   different positions. Each frame's fire pixels (the localized hot pixels, not
   the merged outline) are projected onto one ground grid. Within the ground both
   frames see, it reports their overlap (IoU) and agreement: the share of one
   frame's fire that lies within 2 / 5 / 10 m of the other frame's fire, averaged
   both ways. Pairs are split by how far the drone moved between them: when it
   hovers, the pose barely changes, so only a moving drone tests the geolocation.
   Fire spread in the 2-4 s between stills and perspective (the same fire looks
   different from 100 m and 140 m away) also lower the scores.

Writes a summary JSON and prints it.

Usage:
    python experiments/flame3/compare_video.py --dataset_root data/flame3 --cls Fire \
        --video_run results/run_D_video/fire/lossy --telemetry_csv results/run_D_video/fire/telemetry.csv \
        --camera_config camera_configs/dji_m30t_thermal.json --output_json results/run_D_video/fire/lossy/comparison.json
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
from geoloc.camera import load_camera_config                              # noqa: E402
from geoloc.metadata import SidecarCsvLoader                              # noqa: E402
from geoloc.projection import (VERTEX_OK, CameraPose, ReliabilityLimits, classify_vertices, depression_deg,  # noqa: E402
                               latlon_to_offset, pixels_to_rays_ned, project_contour, FLAT_TERRAIN)

GRID_M = 0.5          # ground raster cell
MIN_SHARED_M2 = 500   # ground both frames must see before comparing them
MIN_FIRE_M2 = 20      # fire each frame must show inside it
TOLERANCES_M = (2, 5, 10)
HOVER_M = 5           # drone moved less than this between the frames


def stats(values, digits=2):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), digits),
            "p10": round(float(np.percentile(v, 10)), digits), "p90": round(float(np.percentile(v, 90)), digits)}


def filled_outline(result):
    m = np.zeros((localize.PROCESS_SIZE, localize.PROCESS_SIZE), np.uint8)
    for g in result.get("fire_regions") or []:
        pts = np.array([[p["x"], p["y"]] for p in g["pixel_256"]], np.int32)
        cv2.fillPoly(m, [pts], 1)
    return m > 0


def border_points_ne(result):
    """Each geolocated contour's border as (N, 2) north/east metres from the frame's drone."""
    d = result["geolocation"]["drone"]
    out = []
    for g in result.get("fire_regions") or []:
        if g.get("gps_polygon"):
            n, e = latlon_to_offset(d["lat"], d["lon"], [v["lat"] for v in g["gps_polygon"]],
                                    [v["lon"] for v in g["gps_polygon"]])
            out.append(np.column_stack((n, e)))
    return out


def distance_to_borders(points, polygons):
    """For each point, the distance to the nearest edge of any polygon."""
    best = np.full(len(points), np.inf)
    for poly in polygons:
        a, b = poly, np.roll(poly, -1, axis=0)
        ab = b - a
        t = np.clip(((points[:, None, :] - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-12), 0, 1)
        nearest = a + t[..., None] * ab
        best = np.minimum(best, np.hypot(*(points[:, None, :] - nearest).transpose(2, 0, 1)).min(axis=1))
    return best


def ground_quads(result, camera, limits, origin):
    """The frame's fire pixels as ground quadrilaterals (N, 4, 2) in a common north/east
    frame (metres from origin), reliable ones only."""
    d = result["geolocation"]["drone"]
    pose = CameraPose(d["altitude_agl_m"], d["gimbal_yaw_deg"], d["gimbal_pitch_deg"], d["gimbal_roll_deg"])
    quads = []
    for g in result.get("fire_regions") or []:
        W, H = g["original_wh"]["width"], g["original_wh"]["height"]
        intr = camera.intrinsics_for(W, H)
        runs = np.asarray(g.get("fire_pixels_256") or [], float).reshape(-1, 3)
        if not len(runs):
            continue
        r, c0, c1 = runs[:, 0], runs[:, 1], runs[:, 2] + 1
        corners = np.stack([np.column_stack((c0, r)), np.column_stack((c1, r)), np.column_stack((c1, r + 1)),
                            np.column_stack((c0, r + 1))], axis=1) * [W / 256, H / 256]
        rays = pixels_to_rays_ned(corners.reshape(-1, 2), intr, pose)
        ne = FLAT_TERRAIN.intersect(pose.altitude_agl_m, rays)
        ok = np.array([s == VERTEX_OK for s in classify_vertices(np.hypot(*ne.T), depression_deg(rays), limits)])
        ok = ok.reshape(-1, 4).all(axis=1)
        quads.append(ne.reshape(-1, 4, 2)[ok])
    dn, de = latlon_to_offset(origin[0], origin[1], d["lat"], d["lon"])
    return (np.concatenate(quads) if quads else np.empty((0, 4, 2))) + [dn, de]


def footprint(result, camera, limits, origin, size):
    """Ground the frame sees reliably, as a polygon in the common north/east frame."""
    d = result["geolocation"]["drone"]
    pose = CameraPose(d["altitude_agl_m"], d["gimbal_yaw_deg"], d["gimbal_pitch_deg"], d["gimbal_roll_deg"])
    W, H = size
    s = np.linspace(0, 1, 40, endpoint=False)
    border = np.concatenate([np.column_stack((s * W, 0 * s)), np.column_stack((W + 0 * s, s * H)),
                             np.column_stack((W - s * W, H + 0 * s)), np.column_stack((0 * s, H - s * H))])
    pc = project_contour(border, camera.intrinsics_for(W, H), pose, d["lat"], d["lon"], limits=limits)
    if pc.polygon_north_east is None:
        return None
    dn, de = latlon_to_offset(origin[0], origin[1], d["lat"], d["lon"])
    return pc.polygon_north_east + [dn, de]


def raster(polys, n_max, e_min, shape):
    m = np.zeros(shape, np.uint8)
    pts = [np.round(np.column_stack(((p[:, 1] - e_min) / GRID_M, (n_max - p[:, 0]) / GRID_M))).astype(np.int32)
           for p in polys if len(p) >= 3]
    if pts:
        cv2.fillPoly(m, pts, 1)
    return m > 0


def main():
    parser = argparse.ArgumentParser(description="Video vs stills, and frame-to-frame consistency")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--cls", required=True, choices=["Fire", "No Fire"])
    parser.add_argument("--video_run", required=True, help="process_video.py output folder")
    parser.add_argument("--telemetry_csv", required=True)
    parser.add_argument("--camera_config", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.video_run, "geolocated_results.json")) as f:
        video = json.load(f)
    p = video["summary"]["parameters"]
    camera = load_camera_config(args.camera_config)
    loader = SidecarCsvLoader(args.telemetry_csv, p["csv_altitude_reference"])
    limits = ReliabilityLimits(p["min_depression_deg"] if p["min_depression_deg"] >= 0 else None, p["max_range_m"])
    with open(args.telemetry_csv) as f:
        source = {r["frame_id"]: r["source_frame"] for r in csv.DictReader(f)}

    # 1. the same frames from the original stills
    thermal_dir = os.path.join(args.dataset_root, args.cls, "Thermal", "Raw JPG")
    stills = []
    for v in video["results"]:
        gray, size = localize.load_thermal_gray(os.path.join(thermal_dir, source[v["frame_id"]] + ".JPG"))
        mask = localize.otsu_fire_mask(gray, p["otsu_blur_kernel"], p["threshold_scale"])[0]
        _, _, regions = localize.regions_from_mask(mask, size, p["merge_kernel_size"], p["min_contour_area"],
                                                   p["max_contour_area"], p["epsilon_factor"])
        stills.append({"frame_id": v["frame_id"], "localization_run": True, "fire_regions": regions})
    with contextlib.redirect_stdout(io.StringIO()):
        geolocate.geolocate_results(stills, camera, loader, None, p["merge_kernel_size"], limits=limits)

    found = {"both": 0, "video_only": 0, "still_only": 0, "neither": 0}
    ious, border_dist, identical = [], [], 0
    for v, s in zip(video["results"], stills):
        fv, fs = bool(v["fire_regions"]), bool(s["fire_regions"])
        found["both" if fv and fs else "video_only" if fv else "still_only" if fs else "neither"] += 1
        identical += [g["pixel_256"] for g in v["fire_regions"]] == [g["pixel_256"] for g in s["fire_regions"]]
        if fv and fs:
            a, b = filled_outline(v), filled_outline(s)
            ious.append((a & b).sum() / max((a | b).sum(), 1))
            sv, ss = border_points_ne(v), border_points_ne(s)
            if sv and ss:
                border_dist.extend(distance_to_borders(np.concatenate(sv), ss).tolist())
    comparison = {"frames": len(stills), "frames_with_regions": found, "frames_with_identical_contours": identical,
                  "outline_iou_video_vs_still": stats(ious, 3),
                  "video_border_point_distance_to_still_border_m": stats(border_dist, 2),
                  "still_regions": sum(len(s["fire_regions"]) for s in stills),
                  "video_regions": sum(len(v["fire_regions"]) for v in video["results"])}

    # 2. consecutive frames on one ground grid
    frames = [r for r in video["results"] if r.get("geolocation", {}).get("status") == "ok"]
    origin = (frames[0]["geolocation"]["drone"]["lat"], frames[0]["geolocation"]["drone"]["lon"])
    size = (frames[0]["fire_regions"][0]["original_wh"]["width"], frames[0]["fire_regions"][0]["original_wh"]["height"]) \
        if frames[0]["fire_regions"] else (640, 512)
    fire = {r["frame_id"]: ground_quads(r, camera, limits, origin) for r in frames}
    view = {r["frame_id"]: footprint(r, camera, limits, origin, size) for r in frames}
    pairs = []
    for a, b in zip(frames, frames[1:]):
        fa, fb = view[a["frame_id"]], view[b["frame_id"]]
        if fa is None or fb is None:
            continue
        allp = np.concatenate([fa, fb])
        n_max, e_min = allp[:, 0].max(), allp[:, 1].min()
        shape = (int((n_max - allp[:, 0].min()) / GRID_M) + 2, int((allp[:, 1].max() - e_min) / GRID_M) + 2)
        shared = raster([fa], n_max, e_min, shape) & raster([fb], n_max, e_min, shape)
        A = raster(list(fire[a["frame_id"]]), n_max, e_min, shape) & shared
        B = raster(list(fire[b["frame_id"]]), n_max, e_min, shape) & shared
        cell = GRID_M ** 2
        rec = {"from": a["frame_id"], "to": b["frame_id"], "shared_m2": round(shared.sum() * cell),
               "fire_a_m2": round(A.sum() * cell), "fire_b_m2": round(B.sum() * cell)}
        if rec["shared_m2"] >= MIN_SHARED_M2 and min(rec["fire_a_m2"], rec["fire_b_m2"]) >= MIN_FIRE_M2:
            rec["iou"] = round(float((A & B).sum() / (A | B).sum()), 3)
            to_a = cv2.distanceTransform((~A).astype(np.uint8), cv2.DIST_L2, 5) * GRID_M
            to_b = cv2.distanceTransform((~B).astype(np.uint8), cv2.DIST_L2, 5) * GRID_M
            for tol in TOLERANCES_M:
                rec[f"agreement_{tol}m"] = round(float(((to_b[A] <= tol).mean() + (to_a[B] <= tol).mean()) / 2), 3)
            da, db = a["geolocation"]["drone"], b["geolocation"]["drone"]
            rec["drone_moved_m"] = round(float(np.hypot(*latlon_to_offset(da["lat"], da["lon"], db["lat"], db["lon"]))), 1)
        pairs.append(rec)
    compared = [r for r in pairs if "iou" in r]
    consistency = {"consecutive_pairs": len(frames) - 1, "pairs_compared": len(compared),
                   "pairs_skipped_little_shared_view_or_fire": len(pairs) - len(compared)}
    for name, group in (("hovering", [r for r in compared if r["drone_moved_m"] < HOVER_M]),
                        ("moving", [r for r in compared if r["drone_moved_m"] >= HOVER_M])):
        consistency[name] = {"pairs": len(group), "drone_moved_m": stats([r["drone_moved_m"] for r in group], 1),
                             "iou": stats([r["iou"] for r in group], 3),
                             **{f"agreement_{t}m": stats([r[f"agreement_{t}m"] for r in group], 3)
                                for t in TOLERANCES_M}}
    consistency["pairs"] = pairs

    out = {"video_run": args.video_run, "video_vs_stills": comparison, "frame_to_frame": consistency}
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({**out, "frame_to_frame": {k: v for k, v in consistency.items() if k != "pairs"}}, indent=2))


if __name__ == "__main__":
    main()
