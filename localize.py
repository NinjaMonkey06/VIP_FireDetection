"""
localize.py — Phase 2: Fire Region Localization
================================================
Reads inference_results.json produced by infer.py and runs the classical CV
localization pipeline on all fire-positive frames.

Pipeline per frame:
    Thermal image
        → Otsu thresholding          (binary fire mask)
        → MSER region extraction     (candidate blobs)
        → Mask × MSER fusion         (physics-filtered candidates)
        → NMS + containment pruning  (suppress overlapping AND contained boxes)
        → Bounding boxes (256px + normalized)

Outputs:
    - localization_results.json      (extends inference JSON with boxes)
    - localization_vis/              (optional, visualizations per frame)

Usage:
    python localize.py \
        --inference_json inference_results.json \
        --thermal_dir    custom_test/thermal \
        [--visualize] \
        [--rgb_dir       custom_test/rgb] \
        [--mser_delta 3] [--mser_min_area 150] [--mser_max_area 6000] \
        [--nms_iou_threshold 0.2] [--mask_overlap_ratio 0.35] \
        [--otsu_blur_kernel 5]

Parameter history:
    v1 defaults: delta=5, min_area=60,  max_area=4000, iou=0.3, overlap=0.4
                 → produced too many boxes (up to 44), containment issues,
                   occasional zero-detection on valid fire frames.
    v2 defaults: delta=3, min_area=150, max_area=6000, iou=0.2, overlap=0.35
                 → fewer, larger, cleaner boxes; containment suppression added;
                   delta lowered to recover unstable fire regions.

Notes:
    - Frames with fire_detected=false are passed through unchanged (no CV).
    - Bounding boxes are stored in both 256px space and normalized [0,1]
      coordinates for resolution-agnostic downstream geolocation.
    - The Otsu threshold value is logged per-frame for reproducibility.
"""

import os
import json
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# ── Constants ────────────────────────────────────────────────────────────────
PROCESS_SIZE = 256          # must match classifier input size
LABEL_DESCRIPTIONS = {
    "NN": "No fire",
    "YN": "Fire (no heavy smoke)",
    "YY": "Fire + heavy smoke"
}


# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_thermal_gray(path: str) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Loads thermal image, returns:
        - uint8 grayscale array resized to PROCESS_SIZE x PROCESS_SIZE
        - original (width, height) for scale factor computation
    """
    img = Image.open(path).convert("L")
    original_size = img.size  # (width, height)
    img_resized = img.resize((PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR)
    return np.array(img_resized, dtype=np.uint8), original_size


# ── Stage 1: Otsu Thresholding ───────────────────────────────────────────────
def otsu_fire_mask(gray: np.ndarray, blur_kernel: int) -> tuple[np.ndarray, int]:
    """
    Applies Gaussian blur then Otsu's thresholding to produce a binary fire mask.

    Blur rationale: MSER and thresholding are both sensitive to JPEG compression
    artifacts and thermal sensor noise. A mild blur suppresses these without
    destroying genuine high-temperature region boundaries.

    Returns:
        mask        : binary uint8 array (255 = fire candidate)
        threshold   : the computed Otsu threshold value (logged for reproducibility)
    """
    if blur_kernel > 1:
        k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    threshold, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Morphological cleanup: remove salt noise and fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask, int(threshold)


# ── Stage 2: MSER Region Extraction ─────────────────────────────────────────
def extract_mser_boxes(gray: np.ndarray,
                       delta: int,
                       min_area: int,
                       max_area: int) -> list[tuple[int, int, int, int]]:
    """
    Runs MSER on the grayscale thermal image.

    MSER (Maximally Stable Extremal Regions) detects regions whose pixel
    intensity boundary remains stable across a range of thresholds — ideal
    for fire blobs which have a stable high-intensity core.

    Parameters:
        delta     : MSER stability threshold. Lower = more regions detected,
                    higher = only very stable blobs.
                    Lowered from 5 to 3 to recover frames where fire blobs
                    were too thermally variable to pass the stricter threshold.
        min_area  : minimum region area in pixels (at 256x256).
                    Raised from 60 to 150 to suppress small fragment detections
                    that caused excessive box counts in v1.
        max_area  : maximum region area. Raised from 4000 to 6000 to allow
                    larger contiguous fire regions to be captured as single blobs.

    Returns list of (x, y, w, h) bounding boxes in 256px space.
    """
    mser = cv2.MSER_create(
        delta=delta,
        min_area=min_area,
        max_area=max_area,
    )

    regions, _ = mser.detectRegions(gray)

    boxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region)
        boxes.append((x, y, w, h))

    return boxes


# ── Stage 3: Mask × MSER Fusion ─────────────────────────────────────────────
def filter_boxes_by_mask(boxes: list[tuple[int, int, int, int]],
                         mask: np.ndarray,
                         min_overlap_ratio: float = 0.35) -> list[tuple[int, int, int, int]]:
    """
    Retains only MSER boxes that have sufficient overlap with the Otsu mask.

    Rationale: MSER alone can pick up structural edges (terrain, horizon).
    The thermal mask is physics-grounded — high pixel intensity = high thermal
    emission = genuine fire candidate. Requiring overlap with the mask eliminates
    false MSER regions that straddle cold boundaries.

    min_overlap_ratio lowered from 0.4 to 0.35 to recover valid detections on
    frames where fire regions had softer thermal boundaries (contributing to the
    zero-detection cases observed in v1 testing).
    """
    filtered = []
    for (x, y, w, h) in boxes:
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, PROCESS_SIZE), min(y + h, PROCESS_SIZE)

        if x2 <= x1 or y2 <= y1:
            continue

        roi = mask[y1:y2, x1:x2]
        overlap_ratio = np.count_nonzero(roi) / (roi.size + 1e-6)

        if overlap_ratio >= min_overlap_ratio:
            filtered.append((x, y, w, h))

    return filtered


# ── Stage 4: NMS + Containment Suppression ───────────────────────────────────
def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU between two (x, y, w, h) boxes."""
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = box_a[2] * box_a[3]
    area_b = box_b[2] * box_b[3]
    union_area = area_a + area_b - inter_area + 1e-6

    return inter_area / union_area


def is_contained(inner: tuple, outer: tuple,
                 containment_threshold: float = 0.85) -> bool:
    """
    Returns True if `inner` is substantially contained within `outer`.

    Standard NMS uses IoU, which is low when a small box sits inside a large
    one (small intersection relative to large union). This means contained
    boxes survive NMS despite being redundant. This check catches that case
    explicitly.

    containment_threshold: fraction of `inner`'s area that must lie inside
    `outer` to trigger suppression. Default 0.85 — suppresses boxes that are
    85%+ inside a larger box, while preserving genuinely adjacent small regions.
    """
    ix1, iy1 = inner[0], inner[1]
    ix2, iy2 = ix1 + inner[2], iy1 + inner[3]
    ox1, oy1 = outer[0], outer[1]
    ox2, oy2 = ox1 + outer[2], oy1 + outer[3]

    # Intersection of inner with outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    inner_area = inner[2] * inner[3] + 1e-6
    return (inter_area / inner_area) >= containment_threshold


def nms(boxes: list[tuple[int, int, int, int]],
        iou_threshold: float,
        containment_threshold: float = 0.85) -> list[tuple[int, int, int, int]]:
    """
    Greedy NMS with containment suppression.

    Two-pass suppression:
        Pass 1 — containment: if a smaller box is ≥85% inside a larger box,
                 suppress the smaller one. Handles the nested-box problem that
                 standard IoU-based NMS misses.
        Pass 2 — standard IoU NMS: suppress boxes with high overlap with an
                 already-kept larger box.

    Boxes are ranked by area descending — larger fire regions are preferred
    as anchors, which is physically sensible (larger thermal blob = more
    confident detection).

    iou_threshold lowered from 0.3 to 0.2 to be more aggressive at merging
    boxes that correspond to the same fire region.
    """
    if not boxes:
        return []

    # Sort by area descending
    boxes_sorted = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    # Pass 1: remove boxes substantially contained within a larger box
    after_containment = []
    for i, box in enumerate(boxes_sorted):
        contained = False
        for larger in boxes_sorted[:i]:
            if is_contained(box, larger, containment_threshold):
                contained = True
                break
        if not contained:
            after_containment.append(box)

    # Pass 2: standard greedy IoU NMS
    kept = []
    while after_containment:
        current = after_containment.pop(0)
        kept.append(current)
        after_containment = [
            b for b in after_containment
            if compute_iou(current, b) < iou_threshold
        ]

    return kept


# ── Stage 5: Coordinate Encoding ────────────────────────────────────────────
def encode_boxes(boxes: list[tuple[int, int, int, int]],
                 original_size: tuple[int, int]) -> list[dict]:
    """
    Encodes each bounding box in two coordinate systems:

    pixel_256:    coordinates in the 256x256 processing space.
                  Useful for overlaying on model input images.

    normalized:   [0,1] range, resolution-agnostic.
                  Multiply by any target resolution to map boxes onto it.
                  This is what the geolocation stage will consume — it will
                  receive drone altitude + gimbal angle + GPS and back-project
                  normalized image coordinates to world coordinates.

    original_wh:  stored for reference — the actual thermal image dimensions
                  at inference time (may vary by deployment platform).
    """
    orig_w, orig_h = original_size
    encoded = []

    for (x, y, w, h) in boxes:
        encoded.append({
            "pixel_256": {"x": x, "y": y, "w": w, "h": h},
            "normalized": {
                "x":  round(x / PROCESS_SIZE, 6),
                "y":  round(y / PROCESS_SIZE, 6),
                "w":  round(w / PROCESS_SIZE, 6),
                "h":  round(h / PROCESS_SIZE, 6),
            },
            "original_wh": {"width": orig_w, "height": orig_h}
        })

    return encoded


# ── Visualization ────────────────────────────────────────────────────────────
def visualize_localization(thermal_path, rgb_path,
                           gray, mask, final_boxes,
                           frame_id, pred_label, otsu_thresh,
                           out_dir):
    """
    4-panel figure:
        1. RGB (if available) or blank
        2. Thermal (inferno colormap)
        3. Otsu binary mask
        4. Thermal with final bounding boxes overlaid
    """
    os.makedirs(out_dir, exist_ok=True)

    th_color = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    th_color_rgb = cv2.cvtColor(th_color, cv2.COLOR_BGR2RGB)

    has_rgb = rgb_path is not None and os.path.exists(rgb_path)
    ncols = 4 if has_rgb else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))

    fig.suptitle(
        f"{frame_id}  |  Prediction: {pred_label} ({LABEL_DESCRIPTIONS.get(pred_label, '')})  "
        f"|  Otsu threshold: {otsu_thresh}  |  Detections: {len(final_boxes)}",
        fontsize=11, fontweight="bold"
    )

    col = 0
    if has_rgb:
        rgb_img = Image.open(rgb_path).convert("RGB").resize(
            (PROCESS_SIZE, PROCESS_SIZE), Image.BILINEAR
        )
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

    axes[col].imshow(th_color_rgb)
    axes[col].set_title(f"Detections ({len(final_boxes)} boxes)")
    axes[col].axis("off")
    for (x, y, w, h) in final_boxes:
        rect = Rectangle((x, y), w, h,
                          linewidth=2, edgecolor="#FF4500", facecolor="none")
        axes[col].add_patch(rect)

    if not final_boxes:
        patch = mpatches.Patch(color="orange",
                               label="Fire detected but no stable regions found")
        axes[col].legend(handles=[patch], loc="lower left", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{frame_id}_localization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fire Localization Pipeline")

    # I/O
    parser.add_argument("--inference_json", required=True,
                        help="Path to inference_results.json from infer.py")
    parser.add_argument("--thermal_dir", required=True,
                        help="Path to folder containing thermal images")
    parser.add_argument("--rgb_dir", default=None,
                        help="(Optional) RGB folder — used only for visualization")
    parser.add_argument("--output_json", default="localization_results.json",
                        help="Output JSON path (default: localization_results.json)")

    # Visualization
    parser.add_argument("--visualize", action="store_true",
                        help="Save per-frame visualization panels")
    parser.add_argument("--vis_dir", default="localization_vis",
                        help="Output folder for visualizations (default: localization_vis/)")

    # Otsu parameters
    parser.add_argument("--otsu_blur_kernel", type=int, default=5,
                        help="Gaussian blur kernel size before Otsu (odd int, default: 5). "
                             "Set to 1 to disable blur.")

    # MSER parameters
    parser.add_argument("--mser_delta", type=int, default=3,
                        help="MSER delta: stability threshold (default: 3). "
                             "Lower = more regions detected; higher = only very stable blobs.")
    parser.add_argument("--mser_min_area", type=int, default=150,
                        help="MSER minimum region area in pixels at 256x256 (default: 150). "
                             "Increase to filter out small noise/fragment regions.")
    parser.add_argument("--mser_max_area", type=int, default=6000,
                        help="MSER maximum region area in pixels at 256x256 (default: 6000). "
                             "Decrease if large background regions are being captured.")

    # Fusion parameter
    parser.add_argument("--mask_overlap_ratio", type=float, default=0.35,
                        help="Minimum fraction of MSER box pixels inside Otsu mask (default: 0.35). "
                             "Increase to be stricter about thermal confirmation.")

    # NMS parameters
    parser.add_argument("--nms_iou_threshold", type=float, default=0.2,
                        help="IoU threshold for NMS (default: 0.2). "
                             "Lower = more aggressive suppression of overlapping boxes.")
    parser.add_argument("--containment_threshold", type=float, default=0.85,
                        help="Fraction of a smaller box that must lie inside a larger box "
                             "to be suppressed (default: 0.85). "
                             "Handles nested boxes that IoU-based NMS misses.")

    args = parser.parse_args()

    # ── Load inference results ────────────────────────────────────────────────
    with open(args.inference_json, "r") as f:
        inference_data = json.load(f)

    results = inference_data["results"]
    fire_frames  = [r for r in results if r["fire_detected"]]
    clear_frames = [r for r in results if not r["fire_detected"]]

    print(f"[✓] Loaded {len(results)} frames from {args.inference_json}")
    print(f"[✓] Fire-positive: {len(fire_frames)}  |  Clear (skipped): {len(clear_frames)}\n")

    # ── Process fire-positive frames only ────────────────────────────────────
    localized_count  = 0
    no_region_count  = 0

    for result in results:

        # Gate: skip clear frames — no CV processing
        if not result["fire_detected"]:
            result["localization_run"] = False
            result["localization_note"] = "Skipped — classifier predicted no fire"
            continue

        filename   = result["filename"]
        frame_id   = result["frame_id"]
        pred_label = result["classifier_prediction"]

        thermal_path = os.path.join(args.thermal_dir, filename)

        if not os.path.exists(thermal_path):
            print(f"  [!] Thermal image not found, skipping: {filename}")
            result["localization_run"] = False
            result["localization_note"] = "Thermal image file not found"
            continue

        # Stage 1: load + resize thermal
        gray, original_size = load_thermal_gray(thermal_path)

        # Stage 2: Otsu threshold → binary mask
        mask, otsu_thresh = otsu_fire_mask(gray, args.otsu_blur_kernel)

        # Stage 3: MSER on grayscale thermal
        mser_boxes = extract_mser_boxes(
            gray,
            delta=args.mser_delta,
            min_area=args.mser_min_area,
            max_area=args.mser_max_area
        )

        # Stage 4: filter MSER boxes by mask overlap
        filtered_boxes = filter_boxes_by_mask(
            mser_boxes, mask, args.mask_overlap_ratio
        )

        # Stage 5: NMS + containment suppression
        final_boxes = nms(
            filtered_boxes,
            iou_threshold=args.nms_iou_threshold,
            containment_threshold=args.containment_threshold
        )

        # Stage 6: encode coordinates
        encoded_boxes = encode_boxes(final_boxes, original_size)

        # Update result record
        result["localization_run"]        = True
        result["otsu_threshold"]          = otsu_thresh
        result["mser_candidates"]         = len(mser_boxes)
        result["post_fusion_candidates"]  = len(filtered_boxes)
        result["bounding_boxes"]          = encoded_boxes

        if encoded_boxes:
            localized_count += 1
            status = f"🔥 {len(encoded_boxes)} box(es)"
        else:
            no_region_count += 1
            status = "⚠  fire detected, no stable regions found"
            result["localization_note"] = (
                "Classifier predicted fire but no MSER regions survived "
                "mask fusion. Consider lowering --mask_overlap_ratio or "
                "--mser_min_area."
            )

        print(f"  {status:<30}  |  {filename:<30}  |  "
              f"Otsu={otsu_thresh}  MSER={len(mser_boxes)}→{len(filtered_boxes)}→{len(final_boxes)}")

        if args.visualize:
            rgb_path = (
                os.path.join(args.rgb_dir, filename)
                if args.rgb_dir else None
            )
            visualize_localization(
                thermal_path=thermal_path,
                rgb_path=rgb_path,
                gray=gray,
                mask=mask,
                final_boxes=final_boxes,
                frame_id=frame_id,
                pred_label=pred_label,
                otsu_thresh=otsu_thresh,
                out_dir=args.vis_dir
            )

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = inference_data.get("summary", {})
    summary["localization_run_on"]       = len(fire_frames)
    summary["frames_with_boxes"]         = localized_count
    summary["fire_detected_no_regions"]  = no_region_count
    summary["parameters"] = {
        "otsu_blur_kernel":      args.otsu_blur_kernel,
        "mser_delta":            args.mser_delta,
        "mser_min_area":         args.mser_min_area,
        "mser_max_area":         args.mser_max_area,
        "mask_overlap_ratio":    args.mask_overlap_ratio,
        "nms_iou_threshold":     args.nms_iou_threshold,
        "containment_threshold": args.containment_threshold,
    }

    output = {"summary": summary, "results": results}

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[✓] Done.")
    print(f"    Fire frames processed : {len(fire_frames)}")
    print(f"    Frames with boxes     : {localized_count}")
    print(f"    Fire but no regions   : {no_region_count}")
    print(f"    Clear frames skipped  : {len(clear_frames)}")
    print(f"[✓] Results saved to: {args.output_json}")
    if args.visualize:
        print(f"[✓] Visualizations saved to: {args.vis_dir}/")


if __name__ == "__main__":
    main()
