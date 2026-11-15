"""
pose_offset_check.py — Is there a constant error in the gimbal angles or the height?
====================================================================================
Consecutive frames of a FLAME 3 test video (make_video.py) see the same fire from
positions 20-48 m apart. A constant error in yaw, pitch or height moves each frame's
fire by a different amount, so the frames disagree more than they should. This takes
process_video.py's output, shifts every frame's pose by a trial offset, projects the
fire pixels again and measures how well consecutive frames agree (compare_video.py's
frame-to-frame comparison, pairs with a moving drone only). The offset where the
frames agree best estimates the error.

It answers "is the yaw off by something like the magnetic declination?" without
ground truth. It cannot give absolute accuracy: an error that moves every frame's
fire the same way stays invisible, pitch and height trade off against each other,
and the fire itself changes in the 2-4 s between frames.

Writes a JSON and prints one table per quantity. The pose in the video run is not
changed.

Usage:
    python experiments/flame3/pose_offset_check.py --video_run results/run_E_refined/fire/lossless \
        --camera_config camera_configs/dji_m30t_thermal.json \
        --output_json results/run_E_refined/fire/pose_offsets.json
"""

import argparse
import copy
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from compare_video import HOVER_M, compare_pair, footprint, ground_quads   # noqa: E402
from geoloc.camera import load_camera_config                               # noqa: E402
from geoloc.projection import ReliabilityLimits                            # noqa: E402

SWEEPS = {"yaw_deg": [-20, -15, -10, -6, -3, 0, 3, 6, 10, 15, 20],
          "pitch_deg": [-3, -2, -1, 0, 1, 2, 3],
          "height_scale": [0.8, 0.9, 1.0, 1.1, 1.2]}


def agreement(frames, camera, limits, yaw_deg=0.0, pitch_deg=0.0, height_scale=1.0):
    """Frame-to-frame agreement over the moving pairs, with every pose shifted."""
    frames = copy.deepcopy(frames)
    for r in frames:
        d = r["geolocation"]["drone"]
        d["gimbal_yaw_deg"] += yaw_deg
        d["gimbal_pitch_deg"] += pitch_deg
        d["altitude_agl_m"] *= height_scale
    origin = (frames[0]["geolocation"]["drone"]["lat"], frames[0]["geolocation"]["drone"]["lon"])
    wh = frames[0]["fire_regions"][0]["original_wh"]
    size = (wh["width"], wh["height"])
    fire = [ground_quads(r, camera, limits, origin) for r in frames]
    view = [footprint(r, camera, limits, origin, size) for r in frames]
    pairs = [compare_pair(frames[i], frames[i + 1], fire[i], fire[i + 1], view[i], view[i + 1])
             for i in range(len(frames) - 1) if view[i] is not None and view[i + 1] is not None]
    moving = [p for p in pairs if "iou" in p and p["drone_moved_m"] >= HOVER_M]
    return {"pairs": len(moving),
            "distance_m_median": round(float(np.median([p["distance_m"] for p in moving])), 2),
            "agreement_2m_median": round(float(np.median([p["agreement_2m"] for p in moving])), 3),
            "agreement_5m_median": round(float(np.median([p["agreement_5m"] for p in moving])), 3)}


def main():
    parser = argparse.ArgumentParser(description="Constant pose error from frame-to-frame agreement")
    parser.add_argument("--video_run", required=True, help="process_video.py output folder")
    parser.add_argument("--camera_config", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    with open(os.path.join(args.video_run, "geolocated_results.json")) as f:
        video = json.load(f)
    p = video["summary"]["parameters"]
    camera = load_camera_config(args.camera_config)
    limits = ReliabilityLimits(p["min_depression_deg"] if p["min_depression_deg"] >= 0 else None, p["max_range_m"])
    frames = [r for r in video["results"]
              if r.get("geolocation", {}).get("status") == "ok" and r["fire_regions"]]

    out = {"video_run": args.video_run, "frames": len(frames), "sweeps": {}}
    for name, offsets in SWEEPS.items():
        rows = [{"offset": o, **agreement(frames, camera, limits, **{name: o})} for o in offsets]
        best = min(rows, key=lambda r: r["distance_m_median"])
        out["sweeps"][name] = {"rows": rows, "best_offset": best["offset"]}
        print(f"\n{name}: fire-to-fire distance between consecutive frames (median over moving pairs)")
        print(f"{'offset':>8} {'pairs':>6} {'distance':>10} {'within 2 m':>11} {'within 5 m':>11}")
        for r in rows:
            mark = "  <- best" if r is best else ""
            print(f"{r['offset']:>8g} {r['pairs']:>6} {r['distance_m_median']:>9.2f}m {r['agreement_2m_median']:>10.1%} "
                  f"{r['agreement_5m_median']:>10.1%}{mark}")
    with open(args.output_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[ok] Saved: {args.output_json}")


if __name__ == "__main__":
    main()
