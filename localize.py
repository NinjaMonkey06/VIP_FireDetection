"""
localize.py — Phase 2: Fire Region Localization (v4 — Tuned)
=============================================================
Reads inference_results.json produced by infer.py and runs the classical CV
localization pipeline on all fire-positive frames.

Pipeline per frame:
    Thermal image
        → Otsu thresholding + optional threshold scaling
        → Mask dilation              (merges nearby fire regions)
        → Contour extraction + approxPolyDP simplification
        → Area filtering
        → Polyline arc output        (pixel_256 + normalized + GeoJSON/CSV)

Changes v3 → v4 (based on supervisor feedback):
    1. Region merging via mask dilation:
       Nearby fire regions are merged before contour extraction by dilating
       the binary mask. A dilation kernel of --merge_kernel_size pixels causes
       regions within that distance of each other to merge into a single contour.

    2. Denser contour vertices:
       epsilon_factor default lowered from 0.02 → 0.01, retaining more contour
       detail for better fire boundary coverage.

    3. Threshold scaling for missed fire regions:
       Pure Otsu can set the threshold too high when fire occupies a small
       fraction of the frame. A --threshold_scale factor (default 0.85)
       multiplies the Otsu threshold, expanding the fire mask to capture
       dimmer fire regions. Set to 1.0 for pure Otsu.

Parameter history:
    v1: MSER delta=5, min_area=60,  max_area=4000, iou=0.3  → up to 44 boxes
    v2: MSER delta=3, min_area=150, max_area=6000, iou=0.2  → 2-15 boxes
    v3: contour extraction, epsilon=0.02, pure Otsu, no merging
    v4: dilation merging, threshold_scale=0.85, epsilon=0.01
"""

import os
import json
import csv
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon

PROCESS_SIZE = 256
LABEL_DESCRIPTIONS = {
    "NN": "No fire",
    "YN": "Fire (no heavy smoke)",
    "YY": "Fire + heavy smoke"
}


def load_thermal_gray(path):
    img = Image.open(path).convert("L")
    original_size = img.size
    img_resized = img.resize((PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
    return np.array(img_resized, dtype=np.uint8), original_size


def otsu_fire_mask(gray, blur_kernel, threshold_scale=0.85):
    """
    Otsu threshold + scaling to expand fire mask for dimmer fire regions.
    threshold_scale < 1.0 lowers the threshold, capturing more fire pixels.
    Returns: mask, raw otsu value, applied threshold value.
    """
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray_blurred = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blurred = gray

    otsu_thresh, _ = cv2.threshold(
        gray_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    applied_thresh = max(1, int(otsu_thresh * threshold_scale))
    _, mask = cv2.threshold(gray_blurred, applied_thresh, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, int(otsu_thresh), applied_thresh


def dilate_mask_for_merging(mask, kernel_size):
    """
    Dilates binary mask to merge nearby fire regions into single contours.
    Regions within ~kernel_size/2 pixels of each other are merged.
    Set kernel_size <= 1 to disable.
    """
    if kernel_size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(mask, kernel, iterations=1)


def extract_contours(mask, min_area, max_area, epsilon_factor):
    """
    findContours on binary mask + approxPolyDP simplification.
    epsilon_factor=0.01 (v4 default) produces denser vertices than v3's 0.02,
    improving arc coverage for the fire propagation model.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        perimeter  = cv2.arcLength(cnt, closed=True)
        epsilon    = epsilon_factor * perimeter
        simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(simplified) >= 3:
            filtered.append(simplified)
    filtered.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    return filtered


def encode_contours(contours, original_size):
    orig_w, orig_h = original_size
    encoded = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2)
        encoded.append({
            "pixel_256":      [{"x": int(p[0]), "y": int(p[1])} for p in pts],
            "normalized":     [{"x": round(float(p[0]) / PROCESS_SIZE, 6),
                                "y": round(float(p[1]) / PROCESS_SIZE, 6)} for p in pts],
            "area_px":        round(float(cv2.contourArea(cnt)), 2),
            "n_points":       len(pts),
            "original_wh":    {"width": orig_w, "height": orig_h},
            "gps_polygon":    None,   # filled by geolocate.py
            "fire_front_arc": None    # filled by geolocate.py
        })
    return encoded


def to_geojson(results):
    features = []
    for result in results:
        if not result.get("localization_run"):
            continue
        for i, region in enumerate(result.get("fire_regions", [])):
            ring = [[p["x"], p["y"]] for p in region["normalized"]]
            if ring and ring[0] != ring[-1]:
                ring.append(ring[0])
            features.append({
                "type": "Feature",
                "properties": {
                    "frame_id":              result["frame_id"],
                    "region_index":          i,
                    "area_px":               region["area_px"],
                    "n_points":              region["n_points"],
                    "classifier_prediction": result["classifier_prediction"],
                    "note": "normalized [0,1] placeholders — replace with GPS via geolocate.py"
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]}
            })
    return {"type": "FeatureCollection", "features": features}


def to_csv_rows(results):
    rows = []
    for result in results:
        if not result.get("localization_run"):
            continue
        for ri, region in enumerate(result.get("fire_regions", [])):
            for pi, (np_pt, px_pt) in enumerate(zip(region["normalized"], region["pixel_256"])):
                rows.append({
                    "frame_id": result["frame_id"], "region_index": ri,
                    "point_index": pi, "x_norm": np_pt["x"], "y_norm": np_pt["y"],
                    "x_px": px_pt["x"], "y_px": px_pt["y"], "area_px": region["area_px"]
                })
    return rows


def visualize_localization(thermal_path, rgb_path, gray, mask, dilated_mask,
                           contours, frame_id, pred_label,
                           otsu_thresh, applied_thresh, out_dir):
    """5-panel (4 without RGB): RGB | Thermal | Otsu mask | Dilated mask | Contour overlay."""
    os.makedirs(out_dir, exist_ok=True)
    th_color_rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    has_rgb = rgb_path is not None and os.path.exists(rgb_path)
    ncols   = 5 if has_rgb else 4
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(
        f"{frame_id}  |  {pred_label}  |  Otsu={otsu_thresh} → applied={applied_thresh}"
        f"  |  Regions={len(contours)}",
        fontsize=10, fontweight="bold"
    )

    col = 0
    if has_rgb:
        rgb_img = Image.open(rgb_path).convert("RGB").resize(
            (PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
        axes[col].imshow(rgb_img); axes[col].set_title("RGB"); axes[col].axis("off"); col += 1

    axes[col].imshow(th_color_rgb);     axes[col].set_title("Thermal");              axes[col].axis("off"); col += 1
    axes[col].imshow(mask, cmap="gray"); axes[col].set_title(f"Otsu (t={applied_thresh})"); axes[col].axis("off"); col += 1
    axes[col].imshow(dilated_mask, cmap="gray"); axes[col].set_title("Dilated (merged)"); axes[col].axis("off"); col += 1

    axes[col].imshow(th_color_rgb)
    axes[col].set_title(f"Contours ({len(contours)})")
    axes[col].axis("off")
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(contours), 1)))
    for cnt, color in zip(contours, colors):
        pts = cnt.reshape(-1, 2)
        axes[col].add_patch(MplPolygon(pts, closed=True,
                                       facecolor=(*color[:3], 0.25),
                                       edgecolor=(*color[:3], 1.0), linewidth=2))
        axes[col].plot(pts[:, 0].mean(), pts[:, 1].mean(), 'x',
                       color=color[:3], markersize=8, markeredgewidth=2)
    if not contours:
        axes[col].legend(handles=[mpatches.Patch(color="orange",
                         label="No stable contours found")], loc="lower left", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{frame_id}_localization.png"), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fire Localization v4")

    parser.add_argument("--inference_json",  required=True)
    parser.add_argument("--thermal_dir",     required=True)
    parser.add_argument("--rgb_dir",         default=None)
    parser.add_argument("--output_json",     default="localization_results.json")
    parser.add_argument("--output_format",   default="json", choices=["json", "geojson", "csv"])
    parser.add_argument("--visualize",       action="store_true")
    parser.add_argument("--vis_dir",         default="localization_vis")

    # Thresholding
    parser.add_argument("--otsu_blur_kernel",  type=int,   default=5,
                        help="Gaussian blur kernel before Otsu (default: 5, set 1 to disable)")
    parser.add_argument("--threshold_scale",   type=float, default=0.85,
                        help="Scale applied to Otsu threshold (default: 0.85). "
                             "< 1.0 expands fire mask. Set 1.0 for pure Otsu.")

    # Region merging
    parser.add_argument("--merge_kernel_size", type=int,   default=15,
                        help="Dilation kernel for merging nearby regions (default: 15). "
                             "Set 0 or 1 to disable.")

    # Contour parameters
    parser.add_argument("--min_contour_area",  type=int,   default=200,
                        help="Min contour area px at 256x256 (default: 200)")
    parser.add_argument("--max_contour_area",  type=int,   default=50000,
                        help="Max contour area px at 256x256 (default: 50000)")
    parser.add_argument("--epsilon_factor",    type=float, default=0.01,
                        help="approxPolyDP epsilon as fraction of perimeter (default: 0.01). "
                             "Lower = denser vertices.")

    args = parser.parse_args()

    with open(args.inference_json) as f:
        inference_data = json.load(f)

    results      = inference_data["results"]
    fire_frames  = [r for r in results if r["fire_detected"]]
    clear_frames = [r for r in results if not r["fire_detected"]]

    print(f"[✓] Loaded {len(results)} frames — Fire: {len(fire_frames)}  Clear: {len(clear_frames)}\n")

    localized_count = 0
    no_region_count = 0

    for result in results:
        if not result["fire_detected"]:
            result["localization_run"]  = False
            result["localization_note"] = "Skipped — classifier predicted no fire"
            continue

        filename   = result["filename"]
        frame_id   = result["frame_id"]
        pred_label = result["classifier_prediction"]
        thermal_path = os.path.join(args.thermal_dir, filename)

        if not os.path.exists(thermal_path):
            print(f"  [!] Not found: {filename}")
            result["localization_run"] = False; result["localization_note"] = "File not found"
            continue

        gray, original_size = load_thermal_gray(thermal_path)
        mask, otsu_thresh, applied_thresh = otsu_fire_mask(gray, args.otsu_blur_kernel, args.threshold_scale)
        dilated_mask    = dilate_mask_for_merging(mask, args.merge_kernel_size)
        contours        = extract_contours(dilated_mask, args.min_contour_area, args.max_contour_area, args.epsilon_factor)
        encoded_regions = encode_contours(contours, original_size)

        result.update({
            "localization_run":  True,
            "otsu_threshold":    otsu_thresh,
            "applied_threshold": applied_thresh,
            "threshold_scale":   args.threshold_scale,
            "merge_kernel_size": args.merge_kernel_size,
            "n_raw_contours":    len(contours),
            "fire_regions":      encoded_regions,
        })
        result.pop("bounding_boxes", None)

        if encoded_regions:
            localized_count += 1
            total_pts = sum(r["n_points"] for r in encoded_regions)
            status = f"🔥 {len(encoded_regions)} region(s)  {total_pts} pts"
        else:
            no_region_count += 1
            status = "⚠  no stable contours"
            result["localization_note"] = "Try lowering --min_contour_area or --threshold_scale"

        print(f"  {status:<38}  |  {filename:<28}  |  "
              f"Otsu={otsu_thresh} applied={applied_thresh}  regions={len(contours)}")

        if args.visualize:
            visualize_localization(
                thermal_path, os.path.join(args.rgb_dir, filename) if args.rgb_dir else None,
                gray, mask, dilated_mask, contours, frame_id, pred_label,
                otsu_thresh, applied_thresh, args.vis_dir
            )

    summary = inference_data.get("summary", {})
    summary.update({
        "localization_run_on":      len(fire_frames),
        "frames_with_regions":      localized_count,
        "fire_detected_no_regions": no_region_count,
        "parameters": {
            "otsu_blur_kernel":  args.otsu_blur_kernel,
            "threshold_scale":   args.threshold_scale,
            "merge_kernel_size": args.merge_kernel_size,
            "min_contour_area":  args.min_contour_area,
            "max_contour_area":  args.max_contour_area,
            "epsilon_factor":    args.epsilon_factor,
            "output_format":     args.output_format,
        }
    })

    base = os.path.splitext(args.output_json)[0]
    if args.output_format == "json":
        out_path = base + ".json"
        with open(out_path, "w") as f: json.dump({"summary": summary, "results": results}, f, indent=2)
    elif args.output_format == "geojson":
        out_path = base + ".geojson"
        with open(out_path, "w") as f: json.dump(to_geojson(results), f, indent=2)
    elif args.output_format == "csv":
        out_path = base + ".csv"
        rows = to_csv_rows(results)
        if rows:
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
        else:
            open(out_path, "w").close()

    print(f"\n[✓] Fire processed={len(fire_frames)}  with regions={localized_count}"
          f"  no regions={no_region_count}  clear skipped={len(clear_frames)}")
    print(f"[✓] Saved: {out_path}  ({args.output_format})")
    if args.visualize: print(f"[✓] Vis: {args.vis_dir}/")


if __name__ == "__main__":
    main()
