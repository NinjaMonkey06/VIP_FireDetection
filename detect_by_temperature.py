"""
detect_by_temperature.py — Fire / no-fire decision from radiometric temperatures
================================================================================
Decides per frame whether there is fire from the thermal camera's measured
temperatures: fire if at least --min_hot_pixels pixels are hotter than
--fire_temp_c. Writes inference_results.json in the same format as infer.py, so
localize.py runs on it unchanged.

With --classifier_json (infer.py's output), a frame is fire if EITHER the
classifier or the temperatures say fire, so a classifier miss no longer skips
localization. The classifier's prediction and probabilities are kept.

Why: the classifier was trained on FLAME 2 palette images and missed 9 of 34 FLAME 3
fire frames (a camera gain mode it never saw). Temperatures do not depend on the
palette or the gain mode. On all 738 FLAME 3 frames, >= 10 pixels above 80 °C
caught 622/622 fire frames with 0/116 false alarms.

Needs a radiometric camera: one float temperature image in °C per thermal frame,
with the same file stem (thermal 00001.JPG <-> temperatures 00001.TIFF). Sun-heated
rock, metal or engines can reach 60-70 °C, so keep the threshold above that.

Usage:
    python detect_by_temperature.py --thermal_dir custom_test/thermal \
                                    --temperature_dir custom_test/temperature \
                                    [--classifier_json inference_results.json] \
                                    [--fire_temp_c 80] [--min_hot_pixels 10] \
                                    [--output_json inference_results.json]
"""

import argparse
import json
import os

import cv2
import numpy as np

TEMPERATURE_EXTS = (".tif", ".tiff")
THERMAL_EXTS = (".jpg", ".jpeg", ".png")


def index_temperature_files(temperature_dir):
    """{file stem: path} of the temperature images in a folder."""
    return {os.path.splitext(f)[0]: os.path.join(temperature_dir, f)
            for f in sorted(os.listdir(temperature_dir))
            if os.path.splitext(f)[1].lower() in TEMPERATURE_EXTS}


def load_temperatures(path):
    """Per-pixel temperatures in °C. Refuses 8-bit images: those are palette or
    grayscale renderings, not measured temperatures."""
    temps = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if temps is None:
        raise ValueError(f"cannot read temperature image {path}")
    if not np.issubdtype(temps.dtype, np.floating):
        raise ValueError(f"{path} holds {temps.dtype} values, not temperatures in °C "
                         f"(needs a radiometric float image)")
    return temps


def temperature_decision(temps, fire_temp_c, min_hot_pixels):
    """(fire, hot pixel count, hottest pixel °C)."""
    hot = int(np.count_nonzero(temps > fire_temp_c))
    return hot >= min_hot_pixels, hot, float(np.nanmax(temps))


def build_results(filenames, temperature_files, fire_temp_c, min_hot_pixels, classifier=None):
    """One infer.py-style result per thermal filename. classifier: {filename: infer.py
    result}, or None. Raises if a frame has no temperature image: a missing reading is
    never treated as 'no fire'."""
    missing = [f for f in filenames if os.path.splitext(f)[0] not in temperature_files]
    if missing:
        raise ValueError(f"no temperature image for {len(missing)} frame(s), e.g. {missing[:3]}")

    results = []
    for filename in filenames:
        temps = load_temperatures(temperature_files[os.path.splitext(filename)[0]])
        hot_fire, hot_pixels, max_c = temperature_decision(temps, fire_temp_c, min_hot_pixels)
        temperature = {"fire": hot_fire, "hot_pixels": hot_pixels, "max_temp_c": round(max_c, 1)}

        if classifier is not None:
            result = dict(classifier[filename])
            clf_fire = bool(result["fire_detected"])
            detected_by = [name for name, fire in (("classifier", clf_fire), ("temperature", hot_fire)) if fire]
        else:
            result = {"frame_id": os.path.splitext(filename)[0], "filename": filename,
                      "classifier_prediction": "temperature"}
            detected_by = ["temperature"] if hot_fire else []

        fire = bool(detected_by)
        result.update({"fire_detected": fire, "phase2_localization": fire,
                       "temperature": temperature, "detected_by": detected_by})
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Fire / no-fire decision from radiometric temperatures")
    parser.add_argument("--thermal_dir",     required=True,
                        help="Thermal images localize.py will use (frames are taken from here)")
    parser.add_argument("--temperature_dir", required=True,
                        help="Radiometric temperature images (float °C), same file stems")
    parser.add_argument("--classifier_json", default=None,
                        help="infer.py output: fire if the classifier OR the temperatures say fire")
    parser.add_argument("--fire_temp_c",     type=float, default=80.0,
                        help="A pixel hotter than this counts as hot (default: 80)")
    parser.add_argument("--min_hot_pixels",  type=int, default=10,
                        help="Hot pixels needed to call the frame fire (default: 10)")
    parser.add_argument("--output_json",     default="inference_results.json")
    args = parser.parse_args()

    classifier = None
    if args.classifier_json:
        with open(args.classifier_json) as f:
            classifier = {r["filename"]: r for r in json.load(f)["results"]}
        filenames = sorted(classifier)
    else:
        filenames = sorted(f for f in os.listdir(args.thermal_dir)
                           if os.path.splitext(f)[1].lower() in THERMAL_EXTS)
    print(f"[✓] {len(filenames)} frames  |  fire = at least {args.min_hot_pixels} pixels above "
          f"{args.fire_temp_c:g} °C" + ("  OR classifier" if classifier else "") + "\n")

    results = build_results(filenames, index_temperature_files(args.temperature_dir),
                            args.fire_temp_c, args.min_hot_pixels, classifier)
    for r in results:
        t = r["temperature"]
        status = "🔥 FIRE " if r["fire_detected"] else "✓  CLEAR"
        by = "+".join(r["detected_by"]) or "-"
        print(f"  {status}  |  {r['filename']:<30}  |  max {t['max_temp_c']:6.1f} °C  "
              f"hot px {t['hot_pixels']:6d}  |  by: {by}")

    fire_count = sum(r["fire_detected"] for r in results)
    summary = {
        "total_frames": len(results),
        "fire_detected": fire_count,
        "clear_frames": len(results) - fire_count,
        "fire_rate": round(fire_count / len(results), 4) if results else 0.0,
        "fire_rule": {"fire_temp_c": args.fire_temp_c, "min_hot_pixels": args.min_hot_pixels,
                      "combined_with_classifier": classifier is not None},
    }
    if classifier is not None:
        summary["detected_by_temperature_only"] = sum(r["detected_by"] == ["temperature"] for r in results)
    with open(args.output_json, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n[✓] Done. {fire_count}/{len(results)} frames flagged as fire.")
    if classifier is not None:
        print(f"[✓] Caught by temperature only (classifier missed): {summary['detected_by_temperature_only']}")
    print(f"[✓] Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()
