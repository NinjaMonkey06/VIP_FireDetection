"""
build_geolocation_viewer.py — embeds a geolocate.py JSON output into
geolocation_viewer.html so it can be opened directly in a browser.

Usage:
    python build_geolocation_viewer.py [input_json] [--output out.html] [--template template.html]

    input_json   Path to a geolocate.py JSON result (default: geolocated_results.json,
                 falling back to sample_geolocated_results.json if that doesn't exist).
    --output     Where to write the viewer (default: geolocation_viewer.html — i.e.
                 update the viewer in place).
    --template   HTML file containing the __GEOLOCATE_DATA_START__/END__ markers to
                 fill in (default: geolocation_viewer.html itself, so re-running
                 against the same file just refreshes its embedded data).
"""

import argparse
import json
import os
import re
import sys

START_MARKER = "/*__GEOLOCATE_DATA_START__*/"
END_MARKER = "/*__GEOLOCATE_DATA_END__*/"


def main():
    parser = argparse.ArgumentParser(description="Embed geolocate.py results into the HTML viewer")
    parser.add_argument("input_json", nargs="?", default=None,
                         help="geolocate.py JSON output (default: geolocated_results.json, "
                              "falling back to sample_geolocated_results.json)")
    parser.add_argument("--output", default="geolocation_viewer.html",
                         help="Viewer HTML file to write (default: geolocation_viewer.html)")
    parser.add_argument("--template", default="geolocation_viewer.html",
                         help="HTML file to read the page template from (default: geolocation_viewer.html)")
    args = parser.parse_args()

    input_json = args.input_json
    if input_json is None:
        input_json = "geolocated_results.json" if os.path.exists("geolocated_results.json") \
            else "sample_geolocated_results.json"
        print(f"[i] No input given, using {input_json}")

    if not os.path.exists(input_json):
        sys.exit(f"[!] Input file not found: {input_json}")
    if not os.path.exists(args.template):
        sys.exit(f"[!] Template file not found: {args.template}")

    with open(input_json, encoding="utf-8") as f:
        data = json.load(f)  # validates it's well-formed JSON before we embed it

    with open(args.template, encoding="utf-8") as f:
        html = f.read()

    pattern = re.compile(re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER), re.DOTALL)
    if not pattern.search(html):
        sys.exit(f"[!] Could not find {START_MARKER} ... {END_MARKER} markers in {args.template}")

    new_block = START_MARKER + json.dumps(data, indent=2) + END_MARKER
    html = pattern.sub(lambda _: new_block, html, count=1)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(html)

    n_frames = len(data.get("results", []))
    n_fire = sum(1 for r in data.get("results", []) if r.get("fire_detected"))
    print(f"[OK] Embedded {input_json} ({n_frames} frames, {n_fire} with fire) into {args.output}")


if __name__ == "__main__":
    main()
