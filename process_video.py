"""
process_video.py — Localization + geolocation on a thermal video, frame by frame
================================================================================
Reads a thermal video one frame at a time, the way a live stream arrives, and for
each frame: localizes fire (Otsu, exactly as localize.py) → geolocates the border
with that frame's telemetry row (as geolocate.py) → contour tables.

A video carries no per-frame metadata, so the pose comes from a telemetry CSV with
one row per frame in geolocate.py's sidecar format; frame_id is the frame number,
zero-padded to --frame_id_digits. A plain video has no temperatures either, so
localization uses Otsu, on every frame: there is no detector in front, and a frame
without fire can still produce regions. For the same reason second-level Otsu
(localize.py's default) is off here unless --second_level_otsu: it recovers fire
frames, but also turns more fire-free frames into false alarms.

Writes to --output_dir:
    geolocated_results.json   per-frame results (localize.py + geolocate.py fields,
                              plus frame_index, video_time_s and timing_ms)
    contour_tables/           <frame_id>_contour_<NN>.csv, as geolocate.py --tables_dir
    annotated.mp4             (--annotate) the video with geolocated borders drawn in

Usage:
    python process_video.py --video flight.mp4 --telemetry_csv telemetry.csv \
                            --camera_config camera_configs/<your_camera>.json \
                            --csv_altitude_reference takeoff --output_dir results/my_video --annotate
"""

import argparse
import contextlib
import io
import json
import os
import time

import cv2
import numpy as np
from PIL import Image

import geolocate
import localize
from geoloc.camera import load_camera_config
from geoloc.metadata import SidecarCsvLoader
from geoloc.projection import CameraPose, ReliabilityLimits
from visualize_geolocation import latlon_to_px

COLORS = [(76, 76, 230), (230, 160, 60), (80, 190, 80), (200, 90, 200), (40, 160, 250),
          (220, 220, 60), (170, 120, 60), (230, 130, 200), (160, 160, 160), (90, 200, 220)]   # BGR


def localize_frame(frame_bgr, args):
    """localize.py's Otsu steps on one decoded frame. Returns a localize.py-style result."""
    gray, size = localize.thermal_gray(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    if args.second_level_otsu:
        mask, otsu_thresh, applied_thresh, rounds = localize.second_level_otsu_mask(
            gray, args.otsu_blur_kernel, args.threshold_scale, args.merge_kernel_size, args.max_contour_area)
    else:
        mask, otsu_thresh, applied_thresh = localize.otsu_fire_mask(gray, args.otsu_blur_kernel, args.threshold_scale)
        rounds = 0
    _, contours, regions = localize.regions_from_mask(mask, size, args.merge_kernel_size, args.min_contour_area,
                                                      args.max_contour_area, args.epsilon_factor)
    result = {"localization_run": True, "otsu_threshold": otsu_thresh, "applied_threshold": applied_thresh,
              "threshold_scale": args.threshold_scale, "merge_kernel_size": args.merge_kernel_size,
              "n_raw_contours": len(contours), "fire_regions": regions}
    if rounds:
        result["second_level_otsu_rounds"] = rounds
    return result


def annotate(frame, result, camera):
    """Draw each contour's geolocated border, its numbered points and a header line."""
    out = frame.copy()
    H, W = out.shape[:2]
    geo = result.get("geolocation") or {}
    if geo.get("status") == "ok":
        d = geo["drone"]
        intr = camera.intrinsics_for(W, H)
        pose = CameraPose(d["altitude_agl_m"], d["gimbal_yaw_deg"], d["gimbal_pitch_deg"], d["gimbal_roll_deg"])
    for n, region in enumerate(result["fire_regions"], start=1):
        color = COLORS[(n - 1) % len(COLORS)]
        original = np.array([[p["x"] * W, p["y"] * H] for p in region["normalized"]])
        if region.get("gps_polygon"):
            pts = latlon_to_px(d["lat"], d["lon"], np.array([[v["lat"], v["lon"]] for v in region["gps_polygon"]]),
                               intr, pose)
            cv2.polylines(out, [np.round(pts).astype(np.int32)], True, color, 1, cv2.LINE_AA)
            for k, (u, v) in enumerate(pts, start=1):
                cv2.circle(out, (int(round(u)), int(round(v))), 2, color, -1, cv2.LINE_AA)
                cv2.putText(out, str(k), (int(u) + 3, int(v) - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
            label_at = pts.mean(axis=0)
        else:    # nothing geolocated: grey outline only
            cv2.polylines(out, [np.round(original).astype(np.int32)], True, (128, 128, 128), 1, cv2.LINE_AA)
            label_at = original.mean(axis=0)
        cv2.putText(out, f"#{n}", (int(label_at[0]) - 8, int(label_at[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"#{n}", (int(label_at[0]) - 8, int(label_at[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 1, cv2.LINE_AA)
    statuses = [r.get("geolocation_status", "-") for r in result["fire_regions"]]
    header = f"frame {result['frame_id']}  t={result['video_time_s']:.1f}s  contours={len(statuses)}"
    if geo.get("status") == "ok":
        header += f"  alt {d['altitude_agl_m']:.0f} m  pitch {d['gimbal_pitch_deg']:.1f}  yaw {d['gimbal_yaw_deg']:.1f}"
    else:
        header += "  NO TELEMETRY"
    cv2.rectangle(out, (0, 0), (W, 18), (0, 0, 0), -1)
    cv2.putText(out, header, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Localization + geolocation on a thermal video, frame by frame")
    parser.add_argument("--video",            required=True)
    parser.add_argument("--telemetry_csv",    required=True, help="One row per frame, geolocate.py sidecar format")
    parser.add_argument("--camera_config",    required=True)
    parser.add_argument("--csv_altitude_reference", default="agl", choices=["agl", "takeoff", "msl"])
    parser.add_argument("--frame_id_digits",  type=int, default=5,
                        help="frame_id in the telemetry = frame number zero-padded to this many digits (default: 5)")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--annotate",         action="store_true", help="Also write annotated.mp4")
    # localization (Otsu), localize.py's defaults
    parser.add_argument("--otsu_blur_kernel",  type=int,   default=5)
    parser.add_argument("--second_level_otsu", action="store_true",
                        help="Re-threshold within the bright pixels instead of dropping an oversized region. "
                             "Off by default here: with no detector in front it also adds false alarms")
    parser.add_argument("--threshold_scale",   type=float, default=0.85)
    parser.add_argument("--merge_kernel_size", type=int,   default=15)
    parser.add_argument("--min_contour_area",  type=int,   default=200)
    parser.add_argument("--max_contour_area",  type=int,   default=50000)
    parser.add_argument("--epsilon_factor",    type=float, default=0.01)
    # geolocation reliability, geolocate.py's defaults
    parser.add_argument("--min_depression_deg", type=float, default=10.0)
    parser.add_argument("--max_range_m",        type=float, default=None)
    args = parser.parse_args()

    camera = load_camera_config(args.camera_config)
    loader = SidecarCsvLoader(args.telemetry_csv, args.csv_altitude_reference)
    limits = ReliabilityLimits(args.min_depression_deg if args.min_depression_deg >= 0 else None, args.max_range_m)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    writer = None

    results = []
    counts = {"frames_geolocated": 0, "frames_missing_metadata": 0, "regions_geolocated": 0,
              "regions_clipped_to_reliable": 0, "regions_unreliable": 0}
    print(f"[✓] {args.video}: {fps:g} fps  |  telemetry: {args.telemetry_csv}  |  camera: {camera.name}\n")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        result = {"frame_id": f"{n:0{args.frame_id_digits}d}", "frame_index": n,
                  "video_time_s": round(n / fps, 3) if fps else None, "fire_detected": True,
                  **localize_frame(frame, args)}
        t1 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):     # geolocate_results prints per frame
            c = geolocate.geolocate_results([result], camera, loader, None, args.merge_kernel_size, limits=limits)
        for k in counts:
            counts[k] += c[k]
        t2 = time.perf_counter()
        if args.annotate:
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(args.output_dir, "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps or 2.0, (frame.shape[1], frame.shape[0]))
            writer.write(annotate(frame, result, camera))
        t3 = time.perf_counter()
        result["timing_ms"] = {"localize": round((t1 - t0) * 1000, 1), "geolocate": round((t2 - t1) * 1000, 1),
                               "annotate": round((t3 - t2) * 1000, 1)}
        results.append(result)

        statuses = [r.get("geolocation_status", "-") for r in result["fire_regions"]]
        note = "" if result.get("geolocation", {}).get("status") == "ok" else "  [no telemetry]"
        print(f"  frame {result['frame_id']}  contours={len(statuses)}  ok={statuses.count('ok')}  "
              f"clipped={statuses.count('clipped_to_reliable')}  unreliable={statuses.count('unreliable')}  "
              f"{(t2 - t0) * 1000:5.1f} ms{note}")
        n += 1
    cap.release()
    if writer is not None:
        writer.release()

    timing = {k: round(float(np.median([r["timing_ms"][k] for r in results])), 1)
              for k in ("localize", "geolocate", "annotate")} if results else {}
    summary = {"video": args.video, "telemetry_csv": args.telemetry_csv, "camera": camera.name, "frames": n,
               "fps": fps, "frames_with_regions": sum(bool(r["fire_regions"]) for r in results),
               "regions": sum(len(r["fire_regions"]) for r in results), "geolocation": counts,
               "median_timing_ms": timing,
               "parameters": {k: v for k, v in vars(args).items()
                              if k not in ("video", "telemetry_csv", "camera_config", "output_dir")}}
    with open(os.path.join(args.output_dir, "geolocated_results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, allow_nan=False)
    tables = geolocate.write_contour_tables(results, os.path.join(args.output_dir, "contour_tables"))

    print(f"\n[✓] Frames={n}  with regions={summary['frames_with_regions']}  regions={summary['regions']}  "
          f"ok={counts['regions_geolocated']}  clipped={counts['regions_clipped_to_reliable']}  "
          f"unreliable={counts['regions_unreliable']}  missing telemetry={counts['frames_missing_metadata']}")
    print(f"[✓] Median per frame: localize {timing.get('localize')} ms, geolocate {timing.get('geolocate')} ms"
          + (f", annotate {timing.get('annotate')} ms" if args.annotate else ""))
    print(f"[✓] Saved: {args.output_dir}/geolocated_results.json, {tables} contour tables"
          + (", annotated.mp4" if args.annotate else ""))


if __name__ == "__main__":
    main()
