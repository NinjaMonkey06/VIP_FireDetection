"""
process_video.py — Localization + geolocation on a thermal video, frame by frame
================================================================================
Reads a thermal video one frame at a time, the way a live stream arrives, and for
each frame: localizes fire (Otsu, exactly as localize.py) → geolocates the border
with the pose at that frame's time (as geolocate.py) → writes the results as it goes.

Pose. A video carries no per-frame metadata, so it comes from one of:
    --timed_telemetry_csv   telemetry at its own rate, matched to each frame by its
                            time in the video and interpolated (geoloc.metadata.
                            TimedTelemetryLoader; srt_to_telemetry.py makes this file
                            from a DJI subtitle track). Use this for real footage.
    --telemetry_csv         one row per frame, frame_id = frame number zero-padded to
                            --frame_id_digits (geolocate.py's sidecar format). Only
                            right when no frame was ever dropped, e.g. test videos
                            from experiments/flame3/make_video.py.
Frame times come from the decoder, so dropped frames do not shift the matching. A
video recorded from a live camera (record_camera.py) does not keep real time: the
camera delivers frames unevenly while the file claims a fixed rate. Pass its
frame_times.csv with --frame_times_csv and each frame gets its real capture time. A
frame without usable telemetry is reported as missing_metadata and the run goes on.

Localization. A plain video has no temperatures, so Otsu runs on every frame with
no detector in front: a frame without fire can still produce regions. For the same
reason second-level Otsu (localize.py's default) is off unless --second_level_otsu.

--max_background_gray N is a cheap stand-in for a detector. A thermal camera that
scales its picture to the scene (auto gain) shows the ground dark once something as
hot as fire is in view, and mid-grey when nothing is. A frame whose median grey level
is above N is taken as fire-free and not localized. Off by default: it depends on the
camera's gain behaviour, and 60 was chosen on FLAME 3 (DJI M30T) only. Every frame's
result carries background_gray, so a threshold can be checked on new footage.

Outputs in --output_dir:
    frames.jsonl              one JSON line per processed frame, written as each
                              frame finishes (what a live consumer would read)
    border_points.csv         one row per geolocated border point of every contour
                              of every frame: frame_id, video_time_s, timestamp,
                              contour, point, lat, lon, ground_range_m,
                              depression_deg, point_type
    fire_borders.geojson      every geolocated border as a polygon, for a map
    geolocated_results.json   all frames in one file (localize.py + geolocate.py
                              fields, plus frame_index, video_time_s, timing_ms)
    annotated.mp4             (--annotate) the processed frames with borders drawn in
    <--tables_dir>/           (optional) one CSV per contour, as geolocate.py

Usage:
    python process_video.py --video flight.mp4 --timed_telemetry_csv telemetry.csv \
                            --camera_config camera_configs/<your_camera>.json \
                            --csv_altitude_reference takeoff --output_dir results/my_video --annotate
"""

import argparse
import csv
import json
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

import geolocate
import localize
from geoloc.camera import load_camera_config
from geoloc.metadata import MetadataError, SidecarCsvLoader, TimedTelemetryLoader
from geoloc.projection import CameraPose, ReliabilityLimits, latlon_to_px
from record_camera import load_frame_times

MAX_POINT_LABELS = 40   # a contour with more border points has only every Nth one numbered

COLORS = [(76, 76, 230), (230, 160, 60), (80, 190, 80), (200, 90, 200), (40, 160, 250),
          (220, 220, 60), (170, 120, 60), (230, 130, 200), (160, 160, 160), (90, 200, 220)]   # BGR
POINT_COLUMNS = ("frame_id", "video_time_s", "timestamp", *geolocate.CONTOUR_TABLE_COLUMNS[2:])


class ResolvedMetadata:
    """Hands geolocate_results a pose resolved beforehand (or the reason there is none)."""

    def __init__(self, meta=None, error=None):
        self.meta, self.error = meta, error

    def load(self, frame_id, image_path=None):
        if self.error is not None:
            raise self.error
        return self.meta


def frame_time_s(cap, index, fps):
    """Time of the frame just decoded, from the decoder; frame index / fps if it has none."""
    pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    if pos_ms > 0 or index == 0:
        return pos_ms / 1000.0
    return index / fps if fps else None


def localize_frame(frame_bgr, args):
    """localize.py's Otsu steps on one decoded frame. Returns a localize.py-style result."""
    gray, size = localize.thermal_gray(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
    background_gray = float(np.median(gray))
    if args.max_background_gray is not None and background_gray > args.max_background_gray:
        return {"localization_run": False, "background_gray": background_gray, "fire_regions": [],
                "localization_note": f"Skipped: background too bright for a fire scene (median grey "
                                     f"{background_gray:g} > {args.max_background_gray:g})"}
    if args.second_level_otsu:
        mask, otsu_thresh, applied_thresh, rounds = localize.second_level_otsu_mask(
            gray, args.otsu_blur_kernel, args.threshold_scale, args.merge_kernel_size, args.max_contour_area)
    else:
        mask, otsu_thresh, applied_thresh = localize.otsu_fire_mask(gray, args.otsu_blur_kernel, args.threshold_scale)
        rounds = 0
    _, contours, regions = localize.regions_from_mask(mask, size, args.merge_kernel_size, args.min_contour_area,
                                                      args.max_contour_area, args.epsilon_factor, args.max_epsilon_px)
    result = {"localization_run": True, "background_gray": background_gray,
              "otsu_threshold": otsu_thresh, "applied_threshold": applied_thresh,
              "threshold_scale": args.threshold_scale, "merge_kernel_size": args.merge_kernel_size,
              "n_raw_contours": len(contours), "fire_regions": regions}
    if rounds:
        result["second_level_otsu_rounds"] = rounds
    return result


def point_rows(result):
    """border_points.csv rows for one frame."""
    stamp = (result.get("geolocation") or {}).get("timestamp") or ""
    rows = []
    for n, region in enumerate(result["fire_regions"], start=1):
        if "geolocation_status" not in region:
            continue
        for r in geolocate.contour_table_rows(result["frame_id"], n, region):
            rows.append({"video_time_s": result["video_time_s"], "timestamp": stamp,
                         **{k: r[k] for k in POINT_COLUMNS if k in r}})
    return rows


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
            step = max(1, -(-len(pts) // MAX_POINT_LABELS))
            for k, (u, v) in enumerate(pts, start=1):
                cv2.circle(out, (int(round(u)), int(round(v))), 2, color, -1, cv2.LINE_AA)
                if (k - 1) % step:
                    continue
                cv2.putText(out, str(k), (int(u) + 3, int(v) - 3), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 255, 255), 1)
            label_at = pts.mean(axis=0)
        else:    # nothing geolocated: grey outline only
            cv2.polylines(out, [np.round(original).astype(np.int32)], True, (128, 128, 128), 1, cv2.LINE_AA)
            label_at = original.mean(axis=0)
        for thickness, c in ((2, (255, 255, 255)), (1, color)):
            cv2.putText(out, f"#{n}", (int(label_at[0]) - 8, int(label_at[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        c, thickness, cv2.LINE_AA)
    t = result["video_time_s"]
    header = (f"frame {result['frame_id']}  t={t:.2f}s  " if t is not None else f"frame {result['frame_id']}  ") + \
             f"contours={len(result['fire_regions'])}"
    if geo.get("status") == "ok":
        header += f"  alt {d['altitude_agl_m']:.0f} m  pitch {d['gimbal_pitch_deg']:.1f}  yaw {d['gimbal_yaw_deg']:.1f}"
    elif geo.get("status") == "missing_metadata":
        header += "  NO TELEMETRY"
    elif not result["localization_run"]:
        header += "  skipped: bright background"
    cv2.rectangle(out, (0, 0), (W, 18), (0, 0, 0), -1)
    cv2.putText(out, header, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles default to cp1252, which cannot print ✓
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Localization + geolocation on a thermal video, frame by frame")
    parser.add_argument("--video",            required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--timed_telemetry_csv", help="Telemetry samples with time_s, matched to frames by time")
    source.add_argument("--telemetry_csv",       help="One row per frame (frame_id = frame number), no dropped frames")
    parser.add_argument("--camera_config",    required=True)
    parser.add_argument("--csv_altitude_reference", default="agl", choices=["agl", "takeoff", "msl"])
    parser.add_argument("--telemetry_max_gap_s", type=float, default=1.0,
                        help="Timed telemetry: a frame farther than this from its nearest sample gets no pose")
    parser.add_argument("--frame_times_csv",  default=None,
                        help="record_camera.py's frame_times.csv: each frame's real capture time, used instead "
                             "of the video file's own clock. Give it for anything recorded from a live camera")
    parser.add_argument("--frame_id_digits",  type=int, default=5,
                        help="--telemetry_csv: frame_id = frame number zero-padded to this many digits")
    parser.add_argument("--frame_step",       type=int, default=1,
                        help="Process every Nth frame (default: 1 = all); skipped frames are not decoded")
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--annotate",         action="store_true", help="Also write annotated.mp4")
    parser.add_argument("--tables_dir",       default=None, help="Also write one CSV per contour here")
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
    parser.add_argument("--max_epsilon_px",    type=float, default=2.0)
    parser.add_argument("--max_background_gray", type=float, default=None,
                        help="Do not localize a frame whose median grey level (0-255) is above this: with auto "
                             "gain, a scene without fire has no dark background. Off by default; 60 suited "
                             "FLAME 3 (DJI M30T). Camera-dependent, check background_gray on your own footage")
    # geolocation reliability, geolocate.py's defaults
    parser.add_argument("--min_depression_deg", type=float, default=10.0)
    parser.add_argument("--max_range_m",        type=float, default=None)
    args = parser.parse_args()
    if args.frame_step < 1:
        parser.error("--frame_step must be at least 1")

    camera = load_camera_config(args.camera_config)
    try:
        if args.timed_telemetry_csv:
            timed = TimedTelemetryLoader(args.timed_telemetry_csv, args.csv_altitude_reference, args.telemetry_max_gap_s)
            per_frame = None
        else:
            timed = None
            per_frame = SidecarCsvLoader(args.telemetry_csv, args.csv_altitude_reference)
    except MetadataError as e:
        parser.error(str(e))
    limits = ReliabilityLimits(args.min_depression_deg if args.min_depression_deg >= 0 else None, args.max_range_m)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        parser.error(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_times = None
    if args.frame_times_csv:
        try:
            frame_times = load_frame_times(args.frame_times_csv)
        except (OSError, ValueError) as e:
            parser.error(str(e))
    os.makedirs(args.output_dir, exist_ok=True)
    telemetry = args.timed_telemetry_csv or args.telemetry_csv
    print(f"[✓] {args.video}: {fps:g} fps  |  telemetry: {telemetry} "
          f"({'by time' if timed else 'by frame number'})  |  frame times: "
          f"{args.frame_times_csv or 'from the video file'}  |  camera: {camera.name}\n")

    results = []
    counts = {"frames_geolocated": 0, "frames_missing_metadata": 0, "regions_geolocated": 0,
              "regions_clipped_to_reliable": 0, "regions_unreliable": 0}
    writer = None
    index = processed = 0
    with open(os.path.join(args.output_dir, "frames.jsonl"), "w", encoding="utf-8") as jsonl, \
            open(os.path.join(args.output_dir, "border_points.csv"), "w", newline="", encoding="utf-8") as points:
        points_csv = csv.DictWriter(points, fieldnames=POINT_COLUMNS)
        points_csv.writeheader()
        while True:
            if index % args.frame_step:
                if not cap.grab():
                    break
                index += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            t_video = frame_times.get(index) if frame_times is not None else frame_time_s(cap, index, fps)
            t0 = time.perf_counter()
            result = {"frame_id": f"{index:0{args.frame_id_digits}d}", "frame_index": index,
                      "video_time_s": None if t_video is None else round(t_video, 3),
                      **localize_frame(frame, args)}
            result["fire_detected"] = bool(result["fire_regions"])     # no detector ran, so localization decides
            t1 = time.perf_counter()
            if timed is not None:
                if t_video is None:
                    loader = ResolvedMetadata(error=MetadataError(
                        f"no time for this frame in {args.frame_times_csv}" if frame_times is not None else
                        "the video reports no frame times and no fps"))
                else:
                    try:
                        loader = ResolvedMetadata(meta=timed.at(t_video))
                    except MetadataError as e:
                        loader = ResolvedMetadata(error=e)
            else:
                loader = per_frame
            c = geolocate.geolocate_results([result], camera, loader, None, args.merge_kernel_size, limits=limits)
            for k in counts:
                counts[k] += c[k]
            t2 = time.perf_counter()
            if args.annotate:
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(args.output_dir, "annotated.mp4"),
                                             cv2.VideoWriter_fourcc(*"mp4v"), (fps or 2.0) / args.frame_step,
                                             (frame.shape[1], frame.shape[0]))
                writer.write(annotate(frame, result, camera))
            t3 = time.perf_counter()
            result["timing_ms"] = {"localize": round((t1 - t0) * 1000, 1), "geolocate": round((t2 - t1) * 1000, 1),
                                   "annotate": round((t3 - t2) * 1000, 1)}
            results.append(result)
            jsonl.write(json.dumps(result, allow_nan=False) + "\n")
            jsonl.flush()
            points_csv.writerows(point_rows(result))

            statuses = [r.get("geolocation_status", "-") for r in result["fire_regions"]]
            geo = result.get("geolocation") or {}
            note = "" if geo.get("status") == "ok" else f"  [{geo.get('status', 'no geolocation')}]"
            t_txt = f"t={t_video:7.2f}s" if t_video is not None else "t=      ?"
            print(f"  frame {result['frame_id']}  {t_txt}  contours={len(statuses)}  ok={statuses.count('ok')}  "
                  f"clipped={statuses.count('clipped_to_reliable')}  unreliable={statuses.count('unreliable')}  "
                  f"{(t2 - t0) * 1000:5.1f} ms{note}")
            index += 1
            processed += 1
    cap.release()
    if writer is not None:
        writer.release()

    timing = {k: round(float(np.median([r["timing_ms"][k] for r in results])), 1)
              for k in ("localize", "geolocate", "annotate")} if results else {}
    summary = {"video": args.video, "telemetry": telemetry, "telemetry_matching": "time" if timed else "frame_number",
               "camera": camera.name, "fps": fps,
               "frame_times": "frame_times_csv" if frame_times is not None else "video file", "frames_read": index, "frames_processed": processed,
               "frame_step": args.frame_step,
               "frames_with_regions": sum(bool(r["fire_regions"]) for r in results),
               "frames_skipped_bright_background": sum(not r["localization_run"] for r in results),
               "regions": sum(len(r["fire_regions"]) for r in results), "geolocation": counts,
               "median_timing_ms": timing,
               "parameters": {k: v for k, v in vars(args).items()
                              if k not in ("video", "telemetry_csv", "timed_telemetry_csv", "frame_times_csv", "camera_config",
                                           "output_dir", "tables_dir")}}
    with open(os.path.join(args.output_dir, "geolocated_results.json"), "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, allow_nan=False)
    geojson = geolocate.to_geojson(results)
    times = {r["frame_id"]: r["video_time_s"] for r in results}
    for feature in geojson["features"]:
        feature["properties"]["video_time_s"] = times[feature["properties"]["frame_id"]]
    with open(os.path.join(args.output_dir, "fire_borders.geojson"), "w") as f:
        json.dump(geojson, f, indent=2, allow_nan=False)
    tables = geolocate.write_contour_tables(results, args.tables_dir) if args.tables_dir else 0

    print(f"\n[✓] Frames read={index}  processed={processed}  with regions={summary['frames_with_regions']}  "
          f"regions={summary['regions']}  ok={counts['regions_geolocated']}  "
          f"clipped={counts['regions_clipped_to_reliable']}  unreliable={counts['regions_unreliable']}  "
          f"missing telemetry={counts['frames_missing_metadata']}"
          + (f"  skipped (bright background)={summary['frames_skipped_bright_background']}"
             if args.max_background_gray is not None else ""))
    print(f"[✓] Median per frame: localize {timing.get('localize')} ms, geolocate {timing.get('geolocate')} ms"
          + (f", annotate {timing.get('annotate')} ms" if args.annotate else ""))
    print(f"[✓] Saved in {args.output_dir}/: frames.jsonl, border_points.csv, fire_borders.geojson "
          f"({len(geojson['features'])} borders), geolocated_results.json"
          + (", annotated.mp4" if args.annotate else "") + (f", {tables} contour tables" if args.tables_dir else ""))


if __name__ == "__main__":
    main()
