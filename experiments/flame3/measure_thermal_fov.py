"""
measure_thermal_fov.py — Thermal intrinsics of the DJI M30T from FLAME 3 image pairs
=====================================================================================
The thermal fx is uncertain: 9.1 mm / 12 µm gives ~758 px, DJI's 61° diagonal FOV
gives ~696 px. This measures it from the data, relative to the wide camera, in two
ways. Each fits thermal pixel (u, v) -> raw RGB pixel (a·u + tx, b·v + ty):

  authors_crop  where each "Corrected FOV" RGB (640x512) sits inside its raw RGB
                (4000x3000), by SIFT + RANSAC. This is the dataset authors' manual
                RGB-to-thermal alignment, not a measurement of the thermal camera.
  direct        the Celsius TIFF registered to the raw RGB by maximizing the
                correlation of edge maps (tree lines, terrain edges; fire is clipped
                out so it does not dominate), starting from the authors' crop.
                Frames without structure (smoke, featureless meadow) give a flat
                peak and are filtered out by --min_peak.

With the wide camera's focal length fx_w and principal point (cx_w, cy_w), a thermal
pixel and the raw pixel it maps to see the same direction, so
    fx = fx_w / a,  fy = fx_w / b,  cx = (cx_w - tx) / a,  cy = (cy_w - ty) / b.
The principal point found this way is the pixel along the wide camera's axis, i.e.
along the gimbal angles. Parallax between the two lenses (a few cm) is negligible at
these ranges.

Usage:
    python experiments/flame3/measure_thermal_fov.py --dataset_root data/flame3 \
        --step 5 --workers 6 --output_csv thermal_fov.csv
"""

import argparse
import csv
import os
from multiprocessing import Pool

import cv2
import numpy as np

CLASSES = ("Fire", "No Fire")
THERMAL_W, THERMAL_H = 640, 512
DOWNSCALE = 4.0                  # raw RGB is matched at 1000x750
AUTHORS_CROP_INIT = (4.106, 3.969, 620.0, 454.0)
# Wide camera: DJI's 84° DFOV (2777 px) and the 24 mm equivalent (2773 px) agree;
# 4.4 mm / 1.6 µm gives 2750 px. Not calibrated — the result scales with it.
WIDE_FX = 2775.0
WIDE_CX, WIDE_CY = 2000.0, 1500.0


# ==================== Authors' crop (SIFT) ====================
def authors_crop(raw_gray, corrected_gray):
    sift = cv2.SIFT_create(4000)
    small = cv2.resize(raw_gray, None, fx=1 / DOWNSCALE, fy=1 / DOWNSCALE, interpolation=cv2.INTER_AREA)
    k1, d1 = sift.detectAndCompute(corrected_gray, None)
    k2, d2 = sift.detectAndCompute(small, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None
    good = [m[0] for m in cv2.BFMatcher().knnMatch(d1, d2, k=2)
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < 10:
        return None
    p1 = np.float32([k1[g.queryIdx].pt for g in good])
    p2 = np.float32([k2[g.trainIdx].pt for g in good]) * DOWNSCALE
    A, inliers = cv2.estimateAffine2D(p1, p2, ransacReprojThreshold=8, maxIters=5000, confidence=0.999)
    if A is None:
        return None
    return {"crop_a": A[0, 0], "crop_b": A[1, 1], "crop_tx": A[0, 2], "crop_ty": A[1, 2],
            "crop_inliers": int(inliers.sum())}


# ==================== Direct thermal ↔ RGB registration ====================
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def edge_map(img8, sigma):
    g = cv2.GaussianBlur(img8, (0, 0), 1.0).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.GaussianBlur(m, (0, 0), sigma)


def correlation(params, thermal_edges, raw_edges):
    a, b, tx, ty = params
    M = np.float32([[a / DOWNSCALE, 0, tx / DOWNSCALE], [0, b / DOWNSCALE, ty / DOWNSCALE]])
    warped = cv2.warpAffine(raw_edges, M, (THERMAL_W, THERMAL_H),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=-1)
    inside = warped >= 0
    if inside.mean() < 0.9:
        return -1.0
    x, y = thermal_edges[inside], warped[inside]
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (np.sqrt((x * x).sum() * (y * y).sum()) + 1e-9))


def pattern_search(start, steps, score):
    """Hooke–Jeeves style coordinate search; halves the steps when nothing improves."""
    p, best = np.array(start, float), score(start)
    steps = np.array(steps, float)
    while (steps > np.array([0.002, 0.002, 0.5, 0.5])).any():
        improved = False
        for i in range(len(p)):
            for sign in (1, -1):
                q = p.copy()
                q[i] += sign * steps[i]
                v = score(q)
                if v > best:
                    p, best, improved = q, v, True
        if not improved:
            steps /= 2
    return p, best


def direct_registration(raw_gray, celsius):
    lo, hi = np.percentile(celsius, 1), min(np.percentile(celsius, 99), 60.0)   # clip out fire
    t8 = cv2.normalize(np.clip(celsius, lo, hi), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    small = cv2.resize(raw_gray, None, fx=1 / DOWNSCALE, fy=1 / DOWNSCALE, interpolation=cv2.INTER_AREA)
    thermal_edges = edge_map(_CLAHE.apply(t8), 3.0)
    raw_edges = edge_map(_CLAHE.apply(small), 3.0 * AUTHORS_CROP_INIT[0] / DOWNSCALE)

    def score(p):
        return correlation(p, thermal_edges, raw_edges)

    a0, b0, tx0, ty0 = AUTHORS_CROP_INIT
    best = max((pattern_search((a0 + da, b0 + db, tx0, ty0), (0.05, 0.05, 16, 16), score)
                for da in (-0.15, 0, 0.15) for db in (-0.15, 0, 0.15)), key=lambda r: r[1])
    p, v = best
    # Peak sharpness: correlation lost when shifting 40 raw px (~10 thermal px) off the optimum.
    peak = v - max(score(p + d) for d in ([0, 0, 40, 0], [0, 0, -40, 0], [0, 0, 0, 40], [0, 0, 0, -40]))
    return {"direct_a": p[0], "direct_b": p[1], "direct_tx": p[2], "direct_ty": p[3],
            "direct_corr": v, "direct_peak": peak}


def measure(job):
    root, cls, name, do_direct = job
    raw = cv2.imread(os.path.join(root, cls, "RGB", "Raw", name), cv2.IMREAD_GRAYSCALE)
    corrected = cv2.imread(os.path.join(root, cls, "RGB", "Corrected FOV", name), cv2.IMREAD_GRAYSCALE)
    row = {"class": cls, "file": name}
    row.update(authors_crop(raw, corrected) or {})
    if do_direct:
        celsius = cv2.imread(os.path.join(root, cls, "Thermal", "Celsius TIFF",
                                          os.path.splitext(name)[0] + ".TIFF"), cv2.IMREAD_UNCHANGED)
        row.update(direct_registration(raw, celsius))
    return row


# ==================== Summary ====================
def intrinsics(a, b, tx, ty, wide_fx):
    return wide_fx / a, wide_fx / b, (WIDE_CX - tx) / a, (WIDE_CY - ty) / b


def summarize(rows, prefix, keep, wide_fx):
    sel = [r for r in rows if keep(r)]
    if not sel:
        print(f"{prefix}: no usable frames")
        return
    v = {k: np.array([r[f"{prefix}_{k}"] for r in sel]) for k in ("a", "b", "tx", "ty")}
    fx, fy, cx, cy = intrinsics(v["a"], v["b"], v["tx"], v["ty"], wide_fx)
    print(f"\n{prefix}: {len(sel)} frames")
    for label, x in (("scale a", v["a"]), ("scale b", v["b"]), ("fx", fx), ("fy", fy),
                     ("cx", cx), ("cy", cy), ("fx/fy", fx / fy)):
        q = np.percentile(x, [10, 50, 90])
        print(f"  {label:8s} median {q[1]:9.3f}   10-90%: {q[0]:9.3f} .. {q[2]:9.3f}   "
              f"MAD {np.median(np.abs(x - q[1])):.3f}")
    f_sq = wide_fx / np.median((v["a"] + v["b"]) / 2)
    print(f"  square-pixel fit: fx = fy = {f_sq:.1f} px  (61 deg DFOV: 696.0, 9.1 mm / 12 um: 758.3)")


def main():
    parser = argparse.ArgumentParser(description="Measure M30T thermal intrinsics from FLAME 3 pairs")
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--step", type=int, default=1,
                        help="Direct registration on every Nth frame (~10 s each; default: all)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min_peak", type=float, default=0.02,
                        help="Minimum peak sharpness for a direct registration to count (default: 0.02)")
    parser.add_argument("--wide_fx", type=float, default=WIDE_FX)
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    jobs, i = [], 0
    for cls in CLASSES:
        for name in sorted(os.listdir(os.path.join(args.dataset_root, cls, "RGB", "Raw"))):
            jobs.append((args.dataset_root, cls, name, i % args.step == 0))
            i += 1
    with Pool(args.workers) as pool:
        rows = pool.map(measure, jobs, chunksize=4)

    summarize(rows, "crop", lambda r: r.get("crop_inliers", 0) >= 30, args.wide_fx)
    summarize(rows, "direct", lambda r: r.get("direct_peak", -1) >= args.min_peak, args.wide_fx)

    if args.output_csv:
        fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("class", "file"), k))
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n[ok] Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
