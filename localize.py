"""
localize.py — Phase 2: Fire Region Localization (v3 — Contour Edition)
=======================================================================
Reads inference_results.json produced by infer.py and runs the classical CV
localization pipeline on all fire-positive frames.

Pipeline per frame:
    Thermal image
        → Otsu thresholding          (binary fire mask)
        → Connected component extraction + approxPolyDP simplification
        → Area filtering             (replaces NMS — contours are spatially distinct)
        → Polyline arc output        (pixel_256 + normalized + optional GPS-ready GeoJSON/CSV)

Change from v2 (bounding boxes) → v3 (contours):
    MSER + NMS + containment suppression have been replaced by contour extraction
    directly from the Otsu mask. This produces geometrically meaningful fire boundary
    polygons rather than axis-aligned rectangles, which is the correct primitive for
    fire wavefront arc representation in the downstream propagation model.

    approxPolyDP epsilon controls the simplification trade-off:
        Low epsilon  → dense point cloud, closely follows fire boundary
        High epsilon → sparse simplified polyline, easier for propagation model to consume
    Default epsilon_factor=0.02 (2% of contour perimeter) is a good starting point.

Output formats (--output_format):
    json     : native pipeline format, matches existing inference_results.json structure
    geojson  : GeoJSON FeatureCollection (pixel normalized coords as lon/lat placeholders
               until geolocation stage fills real GPS coords)
    csv      : flat CSV of all contour points across all frames

Outputs:
    - localization_results.json / .geojson / .csv
    - localization_vis/   (optional 4-panel figures)

Usage:
    python localize.py \\
        --inference_json inference_results.json \\
        --thermal_dir    custom_test/thermal \\
        [--rgb_dir       custom_test/rgb] \\
        [--visualize] \\
        [--output_format json|geojson|csv] \\
        [--otsu_blur_kernel 5] \\
        [--min_contour_area 200] \\
        [--max_contour_area 50000] \\
        [--epsilon_factor 0.02]

Parameter history:
    v1: MSER delta=5, min_area=60,  max_area=4000, iou=0.3  → up to 44 boxes/frame
    v2: MSER delta=3, min_area=150, max_area=6000, iou=0.2,
        containment suppression added                        → 2–15 boxes/frame
    v3: MSER + NMS replaced by contour extraction from Otsu mask
        approxPolyDP simplification for arc representation  → fire boundary polygons

Notes:
    - Frames with fire_detected=false are passed through unchanged (no CV).
    - Contour points stored in pixel_256 and normalized [0,1] coordinates.
    - GeoJSON output uses normalized coords as placeholder lon/lat — the geolocation
      module will overwrite these with real GPS coordinates.
    - Otsu threshold logged per-frame for full reproducibility.
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
from matplotlib.collections import PatchCollection

# ── Constants ────────────────────────────────────────────────────────────────
PROCESS_SIZE = 256
LABEL_DESCRIPTIONS = {
    "NN": "No fire",
    "YN": "Fire (no heavy smoke)",
    "YY": "Fire + heavy smoke"
}


# ── Stage 1: Thermal Loading ──────────────────────────────────────────────────
def load_thermal_gray(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Loads thermal image as uint8 grayscale, resized to PROCESS_SIZE × PROCESS_SIZE.
    Returns grayscale array and original (width, height) for coordinate encoding.
    """
    img = Image.open(path).convert("L")
    original_size = img.size
    img_resized = img.resize((PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
    return np.array(img_resized, dtype=np.uint8), original_size


# ── Stage 2: Otsu Thresholding ────────────────────────────────────────────────
def otsu_fire_mask(gray: np.ndarray, blur_kernel: int) -> tuple[np.ndarray, int]:
    """
    Gaussian blur + Otsu threshold → binary fire mask.
    Morphological open+close removes salt noise and fills small holes.

    Returns:
        mask      : uint8 binary array (255 = fire candidate)
        threshold : computed Otsu value (logged for reproducibility)
    """
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray_blurred = cv2.GaussianBlur(gray, (k, k), 0)
    else:
        gray_blurred = gray

    threshold, mask = cv2.threshold(
        gray_blurred, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, int(threshold)


# ── Stage 3: Contour Extraction + Simplification ─────────────────────────────
def extract_contours(mask: np.ndarray,
                     min_area: int,
                     max_area: int,
                     epsilon_factor: float) -> list[np.ndarray]:
    """
    Extracts fire region contours from the binary mask and simplifies them
    using approxPolyDP (Ramer-Douglas-Peucker algorithm).

    Why contours instead of MSER + bounding boxes:
        The downstream fire propagation model requires the shape of the fire
        boundary (wavefront arc), not a rectangle approximating it. A bounding
        box over a crescent-shaped fire front captures mostly empty space and
        loses all geometric information about the front's curvature.
        Contours extracted directly from the thermal mask preserve this geometry.

    approxPolyDP epsilon:
        epsilon = epsilon_factor × contour_perimeter
        Controls the simplification trade-off:
            Low  (0.01) → dense polyline, closely traces fire boundary
            High (0.05) → sparse polygon, coarser arc representation
        Default 0.02 (2% of perimeter) balances precision with downstream usability.

    Area filters (min_contour_area, max_contour_area) replace the role of
    MSER min/max_area — small regions are likely noise, very large regions
    are likely background sky captured at the mask boundary.

    Returns list of simplified contour arrays, each shape (N, 1, 2) in pixel_256 space.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        epsilon = epsilon_factor * perimeter
        simplified = cv2.approxPolyDP(cnt, epsilon, closed=True)

        # Need at least 3 points to form a meaningful polygon
        if len(simplified) >= 3:
            filtered.append(simplified)

    return filtered


# ── Stage 4: Coordinate Encoding ─────────────────────────────────────────────
def encode_contours(contours: list[np.ndarray],
                    original_size: tuple[int, int]) -> list[dict]:
    """
    Encodes each contour in two coordinate systems:

    pixel_256:   points in 256×256 processing space. For visualization.

    normalized:  points in [0,1] range, resolution-agnostic.
                 The geolocation module will consume these and, given camera
                 intrinsics + drone telemetry, back-project each point to a
                 GPS coordinate. The resulting GPS polygon IS the fire wavefront
                 arc that the propagation model ingests.

    also stores:
        area_px   : contour area in pixel_256 space (proxy for fire region size)
        n_points  : number of polyline points after simplification
        original_wh : actual thermal image dimensions at inference time
    """
    orig_w, orig_h = original_size
    encoded = []

    for cnt in contours:
        pts = cnt.reshape(-1, 2)  # (N, 2) array of [x, y]

        pixel_points = [{"x": int(p[0]), "y": int(p[1])} for p in pts]
        norm_points  = [{"x": round(p[0] / PROCESS_SIZE, 6),
                         "y": round(p[1] / PROCESS_SIZE, 6)} for p in pts]

        encoded.append({
            "pixel_256":   pixel_points,
            "normalized":  norm_points,
            "area_px":     round(cv2.contourArea(cnt), 2),
            "n_points":    len(pts),
            "original_wh": {"width": orig_w, "height": orig_h},
            # Placeholder fields — populated by geolocate.py
            "gps_polygon": None,
            "fire_front_arc": None
        })

    # Sort by area descending — largest fire region first
    encoded.sort(key=lambda r: r["area_px"], reverse=True)
    return encoded


# ── Output Formatters ─────────────────────────────────────────────────────────
def to_geojson(results: list[dict]) -> dict:
    """
    Produces a GeoJSON FeatureCollection.
    Each fire region contour becomes a GeoJSON Polygon feature.
    Normalized [0,1] coords are stored as placeholder lon/lat (x→lon, y→lat).
    The geolocation module overwrites these with real GPS coordinates.

    GeoJSON polygon rings must be closed (first point == last point).
    """
    features = []
    for result in results:
        if not result.get("localization_run"):
            continue
        frame_id = result["frame_id"]
        for i, region in enumerate(result.get("fire_regions", [])):
            pts = region["normalized"]
            # Close the ring
            ring = [[p["x"], p["y"]] for p in pts]
            if ring[0] != ring[-1]:
                ring.append(ring[0])

            features.append({
                "type": "Feature",
                "properties": {
                    "frame_id": frame_id,
                    "region_index": i,
                    "area_px": region["area_px"],
                    "n_points": region["n_points"],
                    "classifier_prediction": result["classifier_prediction"],
                    "note": "coords are normalized [0,1] placeholders — replace with GPS via geolocate.py"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ring]
                }
            })

    return {"type": "FeatureCollection", "features": features}


def to_csv_rows(results: list[dict]) -> list[dict]:
    """
    Flat CSV: one row per contour point.
    Columns: frame_id, region_index, point_index, x_norm, y_norm, x_px, y_px, area_px
    """
    rows = []
    for result in results:
        if not result.get("localization_run"):
            continue
        frame_id = result["frame_id"]
        for ri, region in enumerate(result.get("fire_regions", [])):
            for pi, (np_pt, px_pt) in enumerate(
                    zip(region["normalized"], region["pixel_256"])):
                rows.append({
                    "frame_id":     frame_id,
                    "region_index": ri,
                    "point_index":  pi,
                    "x_norm":       np_pt["x"],
                    "y_norm":       np_pt["y"],
                    "x_px":         px_pt["x"],
                    "y_px":         px_pt["y"],
                    "area_px":      region["area_px"]
                })
    return rows


# ── Visualization ─────────────────────────────────────────────────────────────
def visualize_localization(thermal_path, rgb_path,
                           gray, mask, contours,
                           frame_id, pred_label, otsu_thresh,
                           out_dir):
    """
    4-panel figure:
        1. RGB (if available)
        2. Thermal (inferno colormap)
        3. Otsu binary mask
        4. Thermal with fire contour polygons overlaid
    """
    os.makedirs(out_dir, exist_ok=True)

    th_color     = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    th_color_rgb = cv2.cvtColor(th_color, cv2.COLOR_BGR2RGB)

    has_rgb = rgb_path is not None and os.path.exists(rgb_path)
    ncols   = 4 if has_rgb else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    fig.suptitle(
        f"{frame_id}  |  {pred_label} ({LABEL_DESCRIPTIONS.get(pred_label, '')})  "
        f"|  Otsu={otsu_thresh}  |  Regions={len(contours)}",
        fontsize=11, fontweight="bold"
    )

    col = 0
    if has_rgb:
        rgb_img = Image.open(rgb_path).convert("RGB").resize(
            (PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
        axes[col].imshow(rgb_img)
        axes[col].set_title("RGB")
        axes[col].axis("off")
        col += 1

    axes[col].imshow(th_color_rgb)
    axes[col].set_title("Thermal")
    axes[col].axis("off")
    col += 1

    axes[col].imshow(mask, cmap="gray")
    axes[col].set_title(f"Otsu Mask (t={otsu_thresh})")
    axes[col].axis("off")
    col += 1

    # Contour overlay panel
    axes[col].imshow(th_color_rgb)
    axes[col].set_title(f"Fire Contours ({len(contours)} regions)")
    axes[col].axis("off")

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(contours), 1)))
    for cnt, color in zip(contours, colors):
        pts = cnt.reshape(-1, 2)
        # Draw filled polygon with transparency
        poly = MplPolygon(pts, closed=True,
                          facecolor=(*color[:3], 0.25),
                          edgecolor=(*color[:3], 1.0),
                          linewidth=2)
        axes[col].add_patch(poly)
        # Mark centroid
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        axes[col].plot(cx, cy, 'x', color=color[:3], markersize=8, markeredgewidth=2)

    if not contours:
        patch = mpatches.Patch(color="orange",
                               label="Fire detected but no stable contours found")
        axes[col].legend(handles=[patch], loc="lower left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{frame_id}_localization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Fire Localization Pipeline (contour edition)")

    # I/O
    parser.add_argument("--inference_json", required=True,
                        help="Path to inference_results.json from infer.py")
    parser.add_argument("--thermal_dir", required=True,
                        help="Path to folder containing thermal images")
    parser.add_argument("--rgb_dir", default=None,
                        help="(Optional) RGB folder — used only for visualization")
    parser.add_argument("--output_json", default="localization_results.json",
                        help="Output file path. Extension overridden by --output_format.")
    parser.add_argument("--output_format", default="json",
                        choices=["json", "geojson", "csv"],
                        help="Output format: json (default) | geojson | csv")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Save per-frame 4-panel visualization figures")
    parser.add_argument("--vis_dir", default="localization_vis",
                        help="Output folder for visualizations (default: localization_vis/)")

    # Otsu
    parser.add_argument("--otsu_blur_kernel", type=int, default=5,
                        help="Gaussian blur kernel before Otsu (odd int, default: 5). "
                             "Set to 1 to disable.")

    # Contour parameters
    parser.add_argument("--min_contour_area", type=int, default=200,
                        help="Minimum contour area in pixels at 256×256 (default: 200). "
                             "Filters noise and very small fragments.")
    parser.add_argument("--max_contour_area", type=int, default=50000,
                        help="Maximum contour area (default: 50000). "
                             "Prevents full-frame background regions.")
    parser.add_argument("--epsilon_factor", type=float, default=0.02,
                        help="approxPolyDP epsilon as fraction of contour perimeter (default: 0.02). "
                             "Lower = denser points, higher = coarser arc.")

    args = parser.parse_args()

    # ── Load inference results ────────────────────────────────────────────────
    with open(args.inference_json, "r") as f:
        inference_data = json.load(f)

    results      = inference_data["results"]
    fire_frames  = [r for r in results if r["fire_detected"]]
    clear_frames = [r for r in results if not r["fire_detected"]]

    print(f"[✓] Loaded {len(results)} frames from {args.inference_json}")
    print(f"[✓] Fire-positive: {len(fire_frames)}  |  Clear (skipped): {len(clear_frames)}\n")

    localized_count = 0
    no_region_count = 0

    for result in results:

        # Gate: skip clear frames
        if not result["fire_detected"]:
            result["localization_run"]  = False
            result["localization_note"] = "Skipped — classifier predicted no fire"
            continue

        filename   = result["filename"]
        frame_id   = result["frame_id"]
        pred_label = result["classifier_prediction"]

        thermal_path = os.path.join(args.thermal_dir, filename)
        if not os.path.exists(thermal_path):
            print(f"  [!] Thermal image not found, skipping: {filename}")
            result["localization_run"]  = False
            result["localization_note"] = "Thermal image file not found"
            continue

        # Stage 1: load thermal
        gray, original_size = load_thermal_gray(thermal_path)

        # Stage 2: Otsu mask
        mask, otsu_thresh = otsu_fire_mask(gray, args.otsu_blur_kernel)

        # Stage 3: contour extraction + simplification
        contours = extract_contours(
            mask,
            min_area=args.min_contour_area,
            max_area=args.max_contour_area,
            epsilon_factor=args.epsilon_factor
        )

        # Stage 4: encode
        encoded_regions = encode_contours(contours, original_size)

        # Update result
        result["localization_run"]  = True
        result["otsu_threshold"]    = otsu_thresh
        result["n_raw_contours"]    = len(contours)
        result["fire_regions"]      = encoded_regions

        # Remove legacy bounding_boxes field if present from old infer.py output
        result.pop("bounding_boxes", None)

        if encoded_regions:
            localized_count += 1
            status = f"🔥 {len(encoded_regions)} region(s)"
        else:
            no_region_count += 1
            status = "⚠  fire detected, no stable contours"
            result["localization_note"] = (
                "Classifier predicted fire but no contours passed area filter. "
                "Consider lowering --min_contour_area."
            )

        total_pts = sum(r["n_points"] for r in encoded_regions)
        print(f"  {status:<30}  |  {filename:<30}  |  "
              f"Otsu={otsu_thresh}  contours={len(contours)}  pts={total_pts}")

        if args.visualize:
            rgb_path = (os.path.join(args.rgb_dir, filename)
                        if args.rgb_dir else None)
            visualize_localization(
                thermal_path=thermal_path,
                rgb_path=rgb_path,
                gray=gray,
                mask=mask,
                contours=contours,
                frame_id=frame_id,
                pred_label=pred_label,
                otsu_thresh=otsu_thresh,
                out_dir=args.vis_dir
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = inference_data.get("summary", {})
    summary["localization_run_on"]      = len(fire_frames)
    summary["frames_with_regions"]      = localized_count
    summary["fire_detected_no_regions"] = no_region_count
    summary["parameters"] = {
        "otsu_blur_kernel":  args.otsu_blur_kernel,
        "min_contour_area":  args.min_contour_area,
        "max_contour_area":  args.max_contour_area,
        "epsilon_factor":    args.epsilon_factor,
        "output_format":     args.output_format,
    }

    output = {"summary": summary, "results": results}

    # ── Write output in chosen format ─────────────────────────────────────────
    base = os.path.splitext(args.output_json)[0]

    if args.output_format == "json":
        out_path = base + ".json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

    elif args.output_format == "geojson":
        out_path = base + ".geojson"
        geojson  = to_geojson(results)
        with open(out_path, "w") as f:
            json.dump(geojson, f, indent=2)

    elif args.output_format == "csv":
        out_path = base + ".csv"
        rows     = to_csv_rows(results)
        if rows:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        else:
            open(out_path, "w").close()

    print(f"\n[✓] Done.")
    print(f"    Fire frames processed : {len(fire_frames)}")
    print(f"    Frames with regions   : {localized_count}")
    print(f"    Fire but no regions   : {no_region_count}")
    print(f"    Clear frames skipped  : {len(clear_frames)}")
    print(f"[✓] Results saved to: {out_path}  (format: {args.output_format})")
    if args.visualize:
        print(f"[✓] Visualizations saved to: {args.vis_dir}/")


if __name__ == "__main__":
    main()
