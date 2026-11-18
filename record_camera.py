"""
record_camera.py — Record video from a camera, with the capture time of every frame
===================================================================================
Opens a camera the way a webcam is opened (OpenCV's VideoCapture) and records until
stopped. The source can be anything OpenCV can open:
    0, 1, ...        a camera attached to this computer (webcam, USB thermal camera,
                     USB capture card on a drone's video output)
    picamera         a Raspberry Pi camera module (the ribbon-cable camera), read
                     through the picamera2 library
    a URL            a network stream, e.g. rtsp://192.168.1.10/stream
    a file path      an existing video (used by the tests)

Why the frame times matter. A camera does not deliver frames at exactly the rate the
video file claims: frames arrive late or get dropped. Geolocation needs each frame's
real time to pick the drone's position at that moment, so every frame's capture time
is written next to the video, and process_video.py reads it with --frame_times_csv
instead of trusting the video file's own clock.

Outputs in --output_dir:
    video.mp4 / video.avi   the recording (--codec)
    frame_times.csv         one row per frame in the video: frame_index, time_s
                            (seconds since the first frame, from a clock that never
                            jumps) and utc_time (this computer's clock, ISO 8601)
    recording.json          source, frame size, nominal and measured frame rate,
                            frame count, start and end time
    telemetry.csv           with --mavlink: the drone's position and attitude from its
                            flight controller, timed on the same clock as the frames
                            (mavlink_telemetry.py)

Stop with Ctrl+C, with q in the preview window (--preview), or after --duration_s.
The files are complete whichever way it stops.

On a Raspberry Pi: a USB camera is a camera number, as on any computer; the Pi's own
camera module needs --source picamera. --codec mjpg is the lightest on its processor.
A Pi has no battery-backed clock, so without a network its UTC times are wrong after
a restart (time_s is not affected). The picamera source has been tested against a
stand-in for the library, not yet on a real Pi.

Usage:
    python record_camera.py --source 0 --output_dir results/flight_01/recording
    python record_camera.py --source picamera --codec mjpg --duration_s 600 --output_dir ...
    python process_video.py --video results/flight_01/recording/video.mp4 \
        --frame_times_csv results/flight_01/recording/frame_times.csv \
        --timed_telemetry_csv telemetry.csv --camera_config camera_configs/<your_camera>.json \
        --output_dir results/flight_01/processed

The UTC times are only as good as this computer's clock: set it from the internet or
GPS before a flight if they will be compared with a telemetry log.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

import cv2

CODECS = {"mp4v": ("video.mp4", "mp4v"),      # small, lossy: like a drone's video stream
          "mjpg": ("video.avi", "MJPG"),      # larger, light on the processor (small onboard computers)
          "ffv1": ("video.avi", "FFV1")}      # lossless, largest
FRAME_TIME_COLUMNS = ("frame_index", "time_s", "utc_time")
MAX_FAILED_READS = 50      # consecutive frames the camera may fail to deliver before recording stops


class PiCamera:
    """A Raspberry Pi camera module through picamera2, read like an OpenCV capture."""

    def __init__(self, width=None, height=None, fps=None):
        try:
            from picamera2 import Picamera2
        except ImportError:
            raise OSError("--source picamera needs the picamera2 library "
                          "(Raspberry Pi OS: sudo apt install python3-picamera2)")
        self.fps = fps
        main = {"format": "RGB888"}      # picamera2's RGB888 is blue, green, red in memory: OpenCV's order
        if width and height:
            main["size"] = (width, height)
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_video_configuration(
            main=main, controls={"FrameRate": fps} if fps else {}))
        self.camera.start()

    def get(self, prop):
        return float(self.fps or 0.0) if prop == cv2.CAP_PROP_FPS else 0.0      # live: no frame count

    def read(self):
        try:
            return True, self.camera.capture_array()
        except Exception:                # a failed capture counts as a missed frame, as with OpenCV
            return False, None

    def release(self):
        self.camera.stop()
        self.camera.close()


def open_source(source, width=None, height=None, fps=None):
    """A capture for a camera index ('0'), a URL, a file path, or 'picamera' (a Raspberry
    Pi camera module). Width, height and fps are requests: the camera may give something else."""
    if str(source).lower() == "picamera":
        return PiCamera(width, height, fps)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    if not cap.isOpened():
        raise OSError(f"cannot open camera or stream {source!r}")
    for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, width), (cv2.CAP_PROP_FRAME_HEIGHT, height),
                        (cv2.CAP_PROP_FPS, fps)):
        if value:
            cap.set(prop, value)
    return cap


def record(cap, output_dir, codec="mp4v", fps=None, duration_s=None, max_frames=None, preview=False,
           clock=time.monotonic, wall_clock=time.time, log=None, telemetry=None):
    """Records frames from an open capture until it ends, duration_s or max_frames is
    reached, or the user stops it. Returns the summary that is also written to
    recording.json. telemetry: a mavlink_telemetry.TelemetryLogger, told when the first
    frame arrives so its times count from that frame, and stopped with the recording."""
    log = log or (lambda text: None)
    os.makedirs(output_dir, exist_ok=True)
    filename, fourcc = CODECS[codec]
    nominal_fps = fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not 1.0 <= nominal_fps <= 240.0:        # cameras that do not report a rate give 0 or nonsense
        nominal_fps = 30.0

    writer = None
    frames = failed = 0
    t0 = start_utc = last_t = None
    stopped_by = "source ended"
    live = cap.get(cv2.CAP_PROP_FRAME_COUNT) <= 0      # a camera or stream has no frame count; a file does
    with open(os.path.join(output_dir, "frame_times.csv"), "w", newline="", encoding="utf-8") as f:
        times = csv.writer(f)
        times.writerow(FRAME_TIME_COLUMNS)
        try:
            while True:
                ok, frame = cap.read()
                now, utc = clock(), wall_clock()        # as close to the capture as we can get
                if not ok:
                    if not live:                        # a video file read to its end
                        break
                    failed += 1                         # a camera can miss a frame and recover
                    if failed >= MAX_FAILED_READS:
                        stopped_by = f"camera stopped delivering frames ({failed} failed reads in a row)"
                        break
                    continue
                failed = 0
                if writer is None:
                    height, width = frame.shape[:2]
                    writer = cv2.VideoWriter(os.path.join(output_dir, filename), cv2.VideoWriter_fourcc(*fourcc),
                                             nominal_fps, (width, height))
                    if not writer.isOpened():
                        raise OSError(f"cannot write {filename} with codec {codec}; try another --codec")
                    t0, start_utc = now, utc
                    if telemetry is not None:
                        telemetry.start(t0)
                    log(f"[✓] Recording {width}x{height} to {os.path.join(output_dir, filename)} "
                        f"(nominal {nominal_fps:g} fps). Stop with Ctrl+C" + (" or q" if preview else ""))
                writer.write(frame)
                last_t = now - t0
                times.writerow([frames, f"{last_t:.4f}",
                                datetime.fromtimestamp(utc, timezone.utc).isoformat(timespec="milliseconds")])
                f.flush()
                frames += 1
                if preview:
                    cv2.imshow("record_camera (q to stop)", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stopped_by = "q pressed"
                        break
                if duration_s is not None and last_t >= duration_s:
                    stopped_by = f"duration {duration_s:g} s reached"
                    break
                if max_frames is not None and frames >= max_frames:
                    stopped_by = f"{max_frames} frames reached"
                    break
        except KeyboardInterrupt:
            stopped_by = "Ctrl+C"
        finally:
            if writer is not None:
                writer.release()
            if preview:
                cv2.destroyAllWindows()
            telemetry_summary = telemetry.stop() if telemetry is not None else None

    if frames == 0:
        raise OSError("the camera opened but delivered no frames")
    summary = {"video": filename, "codec": codec, "width": width, "height": height,
               "frames": frames, "duration_s": round(last_t, 3),
               "nominal_fps": nominal_fps,
               "measured_fps": round((frames - 1) / last_t, 3) if last_t else None,
               "start_utc": datetime.fromtimestamp(start_utc, timezone.utc).isoformat(timespec="milliseconds"),
               "end_utc": datetime.fromtimestamp(start_utc + last_t, timezone.utc).isoformat(timespec="milliseconds"),
               "stopped_by": stopped_by}
    if telemetry_summary is not None:
        summary["telemetry"] = telemetry_summary
    with open(os.path.join(output_dir, "recording.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_frame_times(csv_path):
    """{frame_index: time_s} from a frame_times.csv."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in FRAME_TIME_COLUMNS[:2] if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path}: missing columns {missing}")
        try:
            return {int(row["frame_index"]): float(row["time_s"]) for row in reader}
        except ValueError as e:
            raise ValueError(f"{csv_path}: {e}")


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles default to cp1252, which cannot print ✓
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Record video from a camera, with the capture time of every frame")
    parser.add_argument("--source", default="0",
                        help="Camera number (0 = the first camera), 'picamera' (Raspberry Pi camera module), "
                             "stream URL, or video file (default: 0)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--codec", default="mp4v", choices=sorted(CODECS),
                        help="mp4v: small, lossy (default). mjpg: larger, light on the processor. ffv1: lossless")
    parser.add_argument("--width", type=int, default=None, help="Frame width to ask the camera for")
    parser.add_argument("--height", type=int, default=None, help="Frame height to ask the camera for")
    parser.add_argument("--fps", type=float, default=None, help="Frame rate to ask the camera for")
    parser.add_argument("--duration_s", type=float, default=None, help="Stop after this many seconds")
    parser.add_argument("--preview", action="store_true", help="Show the picture while recording; q stops")
    parser.add_argument("--mavlink", default=None, metavar="CONNECTION",
                        help="Also log the drone's position and attitude from its flight controller into "
                             "telemetry.csv, on the video's clock. E.g. /dev/ttyACM0 (USB), /dev/serial0,57600 "
                             "(serial pins and baud rate), udpin:0.0.0.0:14550 (see mavlink_telemetry.py)")
    parser.add_argument("--telemetry_rate_hz", type=float, default=20.0,
                        help="Attitude and position messages per second to ask the flight controller for (default: 20)")
    args = parser.parse_args()

    telemetry = None
    try:
        if args.mavlink:        # first, so a dead telemetry link is known before the camera starts
            import mavlink_telemetry
            link = mavlink_telemetry.connect(args.mavlink, args.telemetry_rate_hz)
            print(f"[✓] Flight controller found on {args.mavlink}")
            os.makedirs(args.output_dir, exist_ok=True)
            telemetry = mavlink_telemetry.TelemetryLogger(link, os.path.join(args.output_dir, "telemetry.csv"))
        cap = open_source(args.source, args.width, args.height, args.fps)
    except OSError as e:
        if telemetry is not None:
            telemetry.stop()
        parser.error(str(e))
    try:
        summary = record(cap, args.output_dir, args.codec, args.fps, args.duration_s, preview=args.preview, log=print,
                         telemetry=telemetry)
    except OSError as e:
        parser.error(str(e))
    finally:
        cap.release()

    print(f"[✓] Stopped: {summary['stopped_by']}. {summary['frames']} frames in {summary['duration_s']:.1f} s "
          f"({summary['measured_fps']} fps measured, {summary['nominal_fps']:g} nominal)")
    saved = f"{summary['video']}, frame_times.csv, recording.json"
    command = (f"python process_video.py --video {os.path.join(args.output_dir, summary['video'])} "
               f"--frame_times_csv {os.path.join(args.output_dir, 'frame_times.csv')}")
    if telemetry is not None:
        t = summary["telemetry"]
        print(f"[✓] Telemetry: {t['samples']} samples" + (f" at {t['rate_hz']} per second" if t["rate_hz"] else "")
              + (f"; {t['attitude_without_position']} attitude messages had no position yet (no GPS fix?)"
                 if t["attitude_without_position"] else ""))
        if t["samples"] == 0:
            print("[!] No telemetry was logged: the flight controller sent no attitude with a position")
        saved += ", telemetry.csv"
        command += (f" --timed_telemetry_csv {os.path.join(args.output_dir, 'telemetry.csv')} "
                    f"--csv_altitude_reference takeoff --camera_mount_deg 0 -90 0")
    print(f"[✓] Saved in {args.output_dir}/: {saved}")
    print(f"[✓] Process with: {command} ...")


if __name__ == "__main__":
    main()
