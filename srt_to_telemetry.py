"""
srt_to_telemetry.py — Per-frame telemetry CSV from a drone video's subtitle track
=================================================================================
DJI drones write a .SRT file next to each video, with one subtitle entry per frame
that carries the aircraft position and (on enterprise models) the gimbal angles:

    3
    00:00:00,066 --> 00:00:00,099
    <font size="28">SrtCnt : 3, DiffTime : 33ms
    2022-10-27 15:40:11.102
    [latitude: 42.866843] [longitude: -121.134371] [rel_alt: 60.919 abs_alt: 1568.589]
    [gb_yaw: -110.4 gb_pitch: -18.9 gb_roll: 0.0] ...</font>

This turns such a file into the timed telemetry CSV that process_video.py reads with
--timed_telemetry_csv: time_s = the entry's start time in the video, plus the fields
named by --field. The keys differ between drone models and firmware, and not every
model writes the gimbal angles, so on a new file run --show_keys first, then map each
required field to a key (--field) or fix it (--const, e.g. a rigid nadir mount):

    python srt_to_telemetry.py --srt flight.SRT --show_keys
    python srt_to_telemetry.py --srt flight.SRT --output_csv telemetry.csv \
        --field altitude_agl_m=rel_alt --const gimbal_roll_deg=0

Entries without a GPS fix (latitude and longitude both 0) are left out. Written
against DJI's documented SRT layout; not yet checked against a file from our drone.
"""

import argparse
import csv
import re
import sys
from collections import Counter

from geoloc.metadata import TIMED_CSV_COLUMNS

FIELDS = TIMED_CSV_COLUMNS[1:]
DEFAULT_KEYS = {"latitude": "latitude", "longitude": "longitude", "altitude_agl_m": "rel_alt",
                "gimbal_yaw_deg": "gb_yaw", "gimbal_pitch_deg": "gb_pitch", "gimbal_roll_deg": "gb_roll"}
_TIME_RANGE = re.compile(r"(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->")
_BRACKETS = re.compile(r"\[([^\]]*)\]")
_PAIR = re.compile(r"([A-Za-z_][\w.]*)\s*:\s*([-+]?\d+(?:\.\d+)?)")
_DATETIME = re.compile(r"(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2}(?:\.\d+)?)")
_TAGS = re.compile(r"<[^>]+>")


def parse_srt(text):
    """[(index, start_s, {key: value}, timestamp_iso or None), ...], one per subtitle entry."""
    entries = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n").strip()):
        lines = block.strip().split("\n")
        timing = next((i for i, l in enumerate(lines) if _TIME_RANGE.search(l)), None)
        if timing is None:
            continue
        h, m, s, frac = _TIME_RANGE.search(lines[timing]).groups()
        start_s = int(h) * 3600 + int(m) * 60 + int(s) + int(frac) / 10 ** len(frac)
        index = int(lines[0].strip()) if timing > 0 and lines[0].strip().isdigit() else len(entries) + 1
        body = _TAGS.sub(" ", " ".join(lines[timing + 1:]))
        segments = _BRACKETS.findall(body) or [body]
        keys = {k: float(v) for seg in segments for k, v in _PAIR.findall(seg)}
        stamp = _DATETIME.search(body)
        entries.append((index, start_s, keys, f"{stamp.group(1)}T{stamp.group(2)}" if stamp else None))
    return entries


def telemetry_rows(entries, key_for, const):
    """Timed telemetry rows from parsed entries. Raises KeyError naming the first missing
    key and the keys that are present."""
    rows, no_fix = [], 0
    for index, start_s, keys, stamp in entries:
        row = {"time_s": round(start_s, 3)}
        for field in FIELDS:
            if field in const:
                row[field] = const[field]
            elif key_for[field] in keys:
                row[field] = keys[key_for[field]]
            else:
                raise KeyError(f"entry {index} has no key {key_for[field]!r} for {field}; keys present: "
                               f"{sorted(keys)}. Map it with --field {field}=<key> or fix it with --const {field}=<value>")
        if row["latitude"] == 0.0 and row["longitude"] == 0.0:
            no_fix += 1
            continue
        row["timestamp"] = stamp or ""
        row["srt_index"] = index
        rows.append(row)
    return rows, no_fix


def main():
    if hasattr(sys.stdout, "reconfigure"):      # Windows consoles default to cp1252, which cannot print ✓
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Timed telemetry CSV from a drone video's .SRT subtitle track")
    parser.add_argument("--srt", required=True)
    parser.add_argument("--output_csv", help="Where to write the timed telemetry (required unless --show_keys)")
    parser.add_argument("--field", action="append", default=[], metavar="FIELD=KEY",
                        help=f"Which SRT key holds a field; fields: {', '.join(FIELDS)}. "
                             f"Defaults: {' '.join(f'{k}={v}' for k, v in DEFAULT_KEYS.items())}")
    parser.add_argument("--const", action="append", default=[], metavar="FIELD=VALUE",
                        help="Fixed value for a field the SRT does not carry, e.g. gimbal_pitch_deg=-90")
    parser.add_argument("--show_keys", action="store_true", help="List the keys found and how often, then stop")
    args = parser.parse_args()

    with open(args.srt, encoding="utf-8", errors="replace") as f:
        entries = parse_srt(f.read())
    if not entries:
        parser.error(f"{args.srt}: no subtitle entries with a time range found")
    if args.show_keys:
        seen = Counter(k for _, _, keys, _ in entries for k in keys)
        print(f"{len(entries)} entries, {entries[0][1]:.3f}-{entries[-1][1]:.3f} s. Keys (entries carrying them):")
        for k, n in seen.most_common():
            print(f"  {k:24s} {n}")
        sample = entries[len(entries) // 2]
        print(f"\nEntry {sample[0]} at {sample[1]:.3f} s: {sample[2]}")
        return
    if not args.output_csv:
        parser.error("--output_csv is required (or use --show_keys)")

    key_for = dict(DEFAULT_KEYS)
    const = {}
    for spec, target in ((args.field, key_for), (args.const, const)):
        for item in spec:
            field, _, value = item.partition("=")
            if field not in FIELDS or not value:
                parser.error(f"expected FIELD=VALUE with FIELD one of {', '.join(FIELDS)}, got {item!r}")
            target[field] = float(value) if target is const else value
    try:
        rows, no_fix = telemetry_rows(entries, key_for, const)
    except KeyError as e:
        parser.error(str(e).strip("'\""))
    if not rows:
        parser.error("every entry lacks a GPS fix (latitude and longitude 0)")

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[*TIMED_CSV_COLUMNS, "timestamp", "srt_index"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[✓] {len(rows)} telemetry samples, {rows[0]['time_s']:.3f}-{rows[-1]['time_s']:.3f} s"
          + (f", {no_fix} entries without a GPS fix left out" if no_fix else ""))
    print(f"[✓] Saved: {args.output_csv}  (use with process_video.py --timed_telemetry_csv)")


if __name__ == "__main__":
    main()
