"""
make_video.py — Build a thermal test video + telemetry from a FLAME 3 capture pass
==================================================================================
FLAME 3 has stills, not video. This takes the pass (frames no more than --max_gap_s
apart) that contains --first_frame, sorts it by capture time and writes:

    thermal_lossy.mp4      one video frame per still, MPEG-4 (mp4v): lossy, like a
                           drone's video stream
    thermal_lossless.avi   the same frames, FFV1 (lossless), to separate the effect
                           of compression from the effect of video processing
    telemetry.csv          one row per video frame: geolocate.py's sidecar format
                           (frame_id = frame number) plus time_s (the frame's time in
                           the video), timestamp and the source still, so it works
                           with both --telemetry_csv and --timed_telemetry_csv of
                           process_video.py. A video carries no per-frame EXIF, so
                           this stands in for the drone's telemetry stream.

The stills are 2-4 s apart (timestamps to the second), so the video is closer to a
time-lapse than to 30 fps footage. Pose comes from the paired raw RGB's XMP, read
with the camera config's metadata fields.

Usage:
    python experiments/flame3/make_video.py --dataset_root data/flame3 --cls Fire --first_frame 00109 \
        --camera_config camera_configs/dji_m30t_thermal.json --output_dir results/run_D_video/fire
"""

import argparse
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from geoloc.camera import load_camera_config                              # noqa: E402
from geoloc.metadata import CSV_COLUMNS, ExifXmpLoader                    # noqa: E402

CODECS = (("thermal_lossy.mp4", "mp4v"), ("thermal_lossless.avi", "FFV1"))


def capture_pass(metas, first_frame, max_gap_s):
    """The frames, in capture order, of the pass that contains first_frame."""
    order = sorted(metas, key=lambda k: (metas[k].timestamp, k))
    passes, current = [], [order[0]]
    for a, b in zip(order, order[1:]):
        if (metas[b].timestamp - metas[a].timestamp).total_seconds() > max_gap_s:
            passes.append(current)
            current = []
        current.append(b)
    passes.append(current)
    return next(p for p in passes if first_frame in p)


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles default to cp1252, which cannot print ✓
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Build a thermal test video + telemetry from a FLAME 3 pass")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--cls", required=True, choices=["Fire", "No Fire"])
    parser.add_argument("--first_frame", required=True, help="Any still of the pass, e.g. 00109")
    parser.add_argument("--camera_config", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fps", type=float, default=2.0, help="Playback rate of the video (default: 2)")
    parser.add_argument("--max_gap_s", type=float, default=6.0)
    args = parser.parse_args()

    loader = ExifXmpLoader(load_camera_config(args.camera_config))
    raw_dir = os.path.join(args.dataset_root, args.cls, "RGB", "Raw")
    thermal_dir = os.path.join(args.dataset_root, args.cls, "Thermal", "Raw JPG")
    metas = {os.path.splitext(f)[0]: loader.load(f, os.path.join(raw_dir, f)) for f in sorted(os.listdir(raw_dir))}
    frames = capture_pass(metas, args.first_frame, args.max_gap_s)
    times = [metas[f].timestamp for f in frames]
    gaps = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    if 0 in gaps:
        print(f"[!] {gaps.count(0)} pair(s) of stills share a timestamp second; their order is by file name")

    os.makedirs(args.output_dir, exist_ok=True)
    first = cv2.imread(os.path.join(thermal_dir, frames[0] + ".JPG"))
    size = (first.shape[1], first.shape[0])
    writers = [cv2.VideoWriter(os.path.join(args.output_dir, name), cv2.VideoWriter_fourcc(*code), args.fps, size)
               for name, code in CODECS]
    if not all(w.isOpened() for w in writers):
        sys.exit("cannot open a video writer (codec missing?)")

    rows = []
    for i, f in enumerate(frames):
        img = cv2.imread(os.path.join(thermal_dir, f + ".JPG"))
        if (img.shape[1], img.shape[0]) != size:
            sys.exit(f"{f}: size {img.shape[1]}x{img.shape[0]} differs from the first frame")
        for w in writers:
            w.write(img)
        m = metas[f]
        rows.append({"frame_id": f"{i:05d}", "time_s": round(i / args.fps, 3),
                     "latitude": m.latitude_deg, "longitude": m.longitude_deg,
                     "altitude_agl_m": m.altitude_agl_m, "gimbal_yaw_deg": m.gimbal_yaw_deg,
                     "gimbal_pitch_deg": m.gimbal_pitch_deg, "gimbal_roll_deg": m.gimbal_roll_deg,
                     "timestamp": m.timestamp.isoformat(), "source_frame": f})
    for w in writers:
        w.release()
    with open(os.path.join(args.output_dir, "telemetry.csv"), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*CSV_COLUMNS, "time_s", "timestamp", "source_frame"])
        writer.writeheader()
        writer.writerows(rows)

    span = (times[-1] - times[0]).total_seconds()
    print(f"[✓] {len(frames)} frames from {args.cls} ({frames[0]} ... {frames[-1]}), {span:.0f} s of flight, "
          f"stills {min(gaps):.0f}-{max(gaps):.0f} s apart, played at {args.fps:g} fps")
    print(f"[✓] Altitude reference in telemetry.csv: '{loader.camera.altitude_reference}' "
          f"(pass --csv_altitude_reference {loader.camera.altitude_reference} to process_video.py)")
    print(f"[✓] Saved: {', '.join(n for n, _ in CODECS)}, telemetry.csv in {args.output_dir}/")


if __name__ == "__main__":
    main()
