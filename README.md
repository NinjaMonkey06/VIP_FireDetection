# VIP_FireDetection

A hybrid deep learning + classical computer vision pipeline for real-time wildfire detection and localization from UAV-mounted multispectral sensors. The system ingests paired **RGB + Thermal/IR** frames and outputs fire presence decisions, spatial fire boundary contours, and structured coordinate data ready for handoff to a fire propagation model or geolocation module.

> **Design philosophy:** Deep learning answers *"Is there fire?"* — Classical CV answers *"Where is the fire?"*
> This separation prioritizes computational efficiency, explainability, and onboard UAV deployability.

---

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Training](#2-training)
  - [3. Evaluation](#3-evaluation)
  - [4. Inference](#4-inference)
  - [5. Localization](#5-localization)
  - [6. Geolocation](#6-geolocation)
- [Output Format](#output-format)
- [Script Reference](#script-reference)
- [Model Architecture](#model-architecture)
- [Known Limitations](#known-limitations)
- [Acknowledgments](#acknowledgments)

---

## Pipeline Overview

```
RGB + Thermal Frame Pair
        │
        ▼
┌─────────────────────────────────────────────┐
│   Phase 1: MidFusionNet                     │  ← dual-branch ResNet-18 classifier
│   Fire presence decision (NN / YN / YY)     │
└─────────────────────────────────────────────┘
        │
        ├── No fire → skip (zero localization compute)
        │
        ▼
┌─────────────────────────────────────────────┐
│   Phase 2: Localization                     │  ← classical CV pipeline
│   Otsu Thresholding + Threshold Scaling     │
│   Mask Dilation (region merging)            │
│   Contour Extraction (approxPolyDP)         │
│   Area Filtering                            │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│   Phase 3: Geolocation                      │  ← geolocate.py + geoloc/
│   Per-frame metadata (EXIF/XMP or CSV)      │
│   Pinhole ray casting (any gimbal pitch)    │
│   Flat-ground intersection → lat/lon        │
│   GPS polygon + centroid + ground area      │
└─────────────────────────────────────────────┘
        │
        ▼
  Fire propagation model input
```

**Label classes:**
| Label | Meaning |
|-------|---------|
| `NN`  | No fire |
| `YN`  | Fire present, no heavy smoke |
| `YY`  | Fire + heavy smoke (smoke occupies ≥50% of RGB frame) |

---

## Dataset

This pipeline was built and evaluated on the **[FLAME-2 dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)** — a large-scale collection of paired RGB and thermal aerial wildfire frames.

| Property | Value |
|----------|-------|
| Total frame pairs | 53,451 |
| RGB resolution | 3840 × 2160 (4K) |
| Thermal resolution | 640 × 512 |
| Format | RGB (.jpg) + Thermal (.jpg) |

A subsample of **5,000 frame pairs** was used, selected via regular interval sampling across labeled temporal segments to maximize scene diversity and avoid near-duplicate frames. Class balance was preserved across all splits.

---

## Project Structure

```
├── dataset.py          # PyTorch Dataset class for paired RGB+thermal data
├── split_dataset.py    # Stratified train/val/test split with file copying
├── model.py            # MidFusionNet architecture definition
├── train.py            # Training loop
├── evaluate.py         # Test set evaluation with confusion matrix
├── infer.py            # Phase 1: inference on arbitrary image pairs
├── detect_by_temperature.py  # Phase 1 option: fire / no fire from radiometric temperatures
├── localize.py         # Phase 2: fire region localization (contour-based)
├── geolocate.py        # Phase 3: contour pixel → GPS (CLI)
├── visualize_geolocation.py  # draws geolocate.py results on the thermal frames
├── geoloc/             # Camera-agnostic geolocation library (no rendering code)
│   ├── camera.py       #   CameraConfig: intrinsics / FOV type, distortion, crop, metadata fields
│   ├── metadata.py     #   FrameMetadata + loaders (EXIF/XMP, sidecar CSV)
│   └── projection.py   #   Pure math: pixel → ray → ground → lat/lon
├── camera_configs/     # dji_m30t_thermal.json (FLAME 3), dji_mini4pro.json (sample), template.json
├── experiments/flame3/ # thermal FOV measurement + laser-rangefinder check on FLAME 3
├── tests/              # unittest suite (synthetic inputs only)
├── docs/               # geolocation_worked_example.md (one pixel projected by hand),
│                       # FLAME3_GEOLOCATION_HANDOFF.md (real-data test plan and results)
├── results/            # One subfolder per test run (outputs, figures)
│   ├── flame2_sample/        # 16 FLAME 2 frames: inference + localization outputs, panels
│   └── flame2_sample_early/  # earlier 7-frame FLAME 2 localization output
├── data/               # Datasets, git-ignored (download separately)
│   ├── flame3/         #   FLAME 3 CV subset (Sycan Marsh): Fire/, No Fire/
│   └── flame3_test/    #   39-frame test subset (34 Fire, 5 No Fire): <class>/RGB, <class>/Thermal (+ Celsius TIFF/)
├── THIRD_PARTY_LICENSES/
│
├── dataset_processed/  # Output of preprocessing (aligned 256×256 pairs)
│   ├── rgb/
│   ├── thermal/
│   └── labels.csv
│
├── dataset_split/      # Output of split_dataset.py
│   ├── train/  rgb/  thermal/  labels.csv
│   ├── val/    rgb/  thermal/  labels.csv
│   └── test/   rgb/  thermal/  labels.csv
│
├── fire_classifier.pth         # Saved model checkpoint (after training)
├── inference_results.json      # Output of infer.py
└── localization_results.json   # Output of localize.py
```

---

## Installation

```bash
git clone https://github.com/NinjaMonkey06/VIP_FireDetection
cd VIP_FireDetection
pip install torch torchvision opencv-python pillow matplotlib scikit-learn pandas numpy
```

Phase 3 geolocation needs no extra packages (numpy, OpenCV and Pillow only).

> **Note:** CUDA is supported automatically. The scripts fall back to CPU if no GPU is available.

---

## Usage

### 1. Data Preparation

Before training, preprocess your raw FLAME-2 frames (alignment, cropping, resizing) and place them in `dataset_processed/` following the structure above. Then run the stratified split:

```bash
python split_dataset.py
```

This produces `dataset_split/` with `train/`, `val/`, and `test/` subfolders, each containing `rgb/`, `thermal/`, and `labels.csv`.

**Split ratios:** 70% train / 15% val / 15% test (stratified by class).

---

### 2. Training

```bash
python train.py
```

Trains MidFusionNet for 15 epochs using Adam (lr=1e-4, batch=16) with Cross-Entropy loss. Saves the model checkpoint to `fire_classifier.pth` on completion.

> **Important:** No ImageNet normalization is applied. The model trains on raw `[0, 1]` pixel values from `transforms.ToTensor()` only. This must be replicated exactly at inference time.

---

### 3. Evaluation

```bash
python evaluate.py
```

Loads `fire_classifier.pth` and evaluates on the test split. Prints and displays a confusion matrix. Requires `dataset_split/test/` to exist.

---

### 4. Inference

Runs the Phase 1 classifier on arbitrary paired RGB + thermal images. **No labels file required.**

```bash
# Minimal — results saved to inference_results.json
python infer.py \
  --rgb_dir custom_test/rgb \
  --thermal_dir custom_test/thermal \
  --checkpoint fire_classifier.pth

# With per-frame visualization panels
python infer.py \
  --rgb_dir custom_test/rgb \
  --thermal_dir custom_test/thermal \
  --checkpoint fire_classifier.pth \
  --visualize
```

**Input requirements:**
- RGB and thermal images must share **identical filenames** (including extension)
- No specific naming convention required beyond this pairing constraint
- Images do not need to be pre-resized — `infer.py` handles resizing to 256×256 internally

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--rgb_dir` | required | Path to folder of RGB images |
| `--thermal_dir` | required | Path to folder of thermal images |
| `--checkpoint` | required | Path to `.pth` model checkpoint |
| `--output_json` | `inference_results.json` | Path to save structured results |
| `--visualize` | off | Save 3-panel visualization figures (RGB / Thermal / confidence bars) |
| `--vis_dir` | `inference_vis/` | Output folder for visualizations |

---

### 4b. Temperature-based fire detection (radiometric cameras, optional)

`detect_by_temperature.py` decides fire / no fire from the measured temperatures: a frame is fire if at least `--min_hot_pixels` pixels are hotter than `--fire_temp_c`. It writes `inference_results.json` in the same format as `infer.py`, so `localize.py` runs on it unchanged. With `--classifier_json`, a frame is fire if **either** the classifier or the temperatures say so, and the classifier's output is kept.

```bash
python detect_by_temperature.py \
  --thermal_dir custom_test/thermal \
  --temperature_dir custom_test/temperature \
  --classifier_json inference_results.json \
  --output_json inference_results_combined.json
```

- Needs one float temperature image in °C per thermal frame, with the same file stem (`00001.JPG` ↔ `00001.TIFF`). 8-bit images are rejected, and a frame without a temperature image is an error, never "no fire".
- Defaults: 80 °C, 10 pixels. Sun-heated rock, metal or engines can reach 60–70 °C, so keep the threshold above that.
- On FLAME 3 the classifier alone missed 9 of 34 fire test frames (a camera gain mode it never saw). Combined with temperature, all 34 were detected and the 5 No Fire frames stayed clear. Over all 738 FLAME 3 frames, the temperature rule alone caught 622/622 fire frames with 0/116 false alarms.

---

### 5. Localization

Runs the Phase 2 classical CV localization pipeline on all fire-positive frames from `inference_results.json`. **Must be run after `infer.py`.** Clear frames are skipped entirely — zero additional compute incurred.

```bash
# Minimal — JSON output
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal

# With 5-panel visualization (include --rgb_dir for the RGB panel)
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal \
  --rgb_dir custom_test/rgb \
  --visualize

# GeoJSON output for mapping tools
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal \
  --output_format geojson

# Parameter tuning example
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal \
  --threshold_scale 0.8 \
  --merge_kernel_size 20 \
  --epsilon_factor 0.005 \
  --visualize
```

**Tunable parameters:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--otsu_blur_kernel` | `5` | Gaussian blur kernel before Otsu. Set to `1` to disable. |
| `--threshold_scale` | `0.85` | Scales Otsu threshold down to capture dimmer fire regions. Set to `1.0` for pure Otsu. |
| `--merge_kernel_size` | `15` | Dilation kernel size for merging nearby fire regions. Regions within ~kernel/2 px of each other are merged. Set to `0` to disable. |
| `--min_contour_area` | `200` | Minimum contour area in pixels at 256×256. Filters noise fragments. |
| `--max_contour_area` | `50000` | Maximum contour area in pixels at 256×256. Prevents background capture. |
| `--epsilon_factor` | `0.01` | `approxPolyDP` epsilon as fraction of contour perimeter. Lower = denser vertices, better arc coverage. |
| `--output_format` | `json` | Output format: `json` / `geojson` / `csv` |
| `--visualize` | off | Save 5-panel figures per frame |
| `--vis_dir` | `localization_vis/` | Output folder for visualizations |

**Localization evolution — parameter presets:**

| Version | Approach | Key settings | Output |
|---------|----------|-------------|--------|
| v1 | MSER + NMS | `mser_min_area=60`, `iou=0.3` | Up to 44 boxes/frame |
| v2 | MSER + NMS (tuned) | `mser_min_area=150`, `iou=0.2`, containment suppression | 2–15 boxes/frame |
| v3 | Contour extraction | `epsilon=0.02`, pure Otsu, no merging | Fire boundary polygons |
| v4 (current) | Contour + merging | `threshold_scale=0.85`, `merge_kernel=15`, `epsilon=0.01` | Merged polygons, denser arcs |

---

### 6. Geolocation

`geolocate.py` converts every fire contour from `localize.py` into GPS coordinates. **Must be run after `localize.py`.** It reads per-frame drone metadata (GPS position, altitude above ground, absolute gimbal yaw/pitch/roll, timestamp) and a camera config, casts a ray through each contour vertex, intersects it with flat ground, and converts the hit to lat/lon.

```bash
# Metadata from a sidecar CSV (one row per frame_id) — use this until real flight data exists
python geolocate.py \
  --localization_json localization_results.json \
  --camera_config camera_configs/<your_camera>.json \
  --metadata_csv frame_metadata.csv

# Metadata read from the images' own EXIF GPS + XMP fields (e.g. DJI drone-dji:*)
python geolocate.py \
  --localization_json localization_results.json \
  --camera_config camera_configs/dji_mini4pro.json \
  --exif_image_dir custom_test/thermal \
  --output_format geojson
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--localization_json` | required | Output of `localize.py` |
| `--camera_config` | required | Camera JSON — see below |
| `--metadata_csv` / `--exif_image_dir` | one required | Metadata source |
| `--altitude_source` | `metadata` | With `--exif_image_dir`: `metadata` uses the altitude field; `laser` uses drone altitude minus the laser rangefinder's hit point where the frame has a reading, and falls back to the altitude field otherwise. The laser measures the first surface it hits (treetops, smoke), so over vegetation `laser` heights are too low |
| `--laser_max_terrain_offset_m` | `50` | Reject a laser height whose ground point lies farther than this above or below the altitude field's reference (laser hit smoke or a distant ridge) |
| `--csv_altitude_reference` | `agl` | What the CSV altitude column is relative to: `agl` / `takeoff` / `msl` |
| `--min_depression_deg` | `10` | Vertices whose ray is less than this many degrees below the horizon are marked unreliable. Negative = off |
| `--max_range_m` | off | Vertices farther than this from the drone's ground position are marked unreliable |
| `--output_json` | `geolocated_results.json` | Output path (extension follows `--output_format`) |
| `--output_format` | `json` | `json` (full results) or `geojson` (one Polygon per region, `[lon, lat]` order) |

**Sidecar CSV columns:** `frame_id, latitude, longitude, altitude_agl_m, gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg[, timestamp]` — yaw clockwise from north, pitch `-90` = straight down, timestamp ISO 8601. Every column except `timestamp` must be filled; frames with missing metadata are reported as `missing_metadata` and never projected with guessed values.

**Camera configs (`camera_configs/`).** All camera-specific values live here, so swapping cameras is a config change. The final camera is not confirmed yet (a FLIR Hadron is the likely candidate); copy `template.json` once it is.
- Intrinsics: either `fx, fy[, cx, cy]` from a checkerboard calibration (preferred) or `fov_deg` + `fov_type` (`horizontal` / `vertical` / `diagonal`). The FOV type is mandatory because most datasheets quote **diagonal** FOV.
- `dist_coeffs` (OpenCV order), optional `crop`, `altitude_reference`, and a `metadata_fields` mapping for the EXIF/XMP loader (`xmp:<prefix:Name>`, `exif:<TagName>` or `const:<value>`).
- Intrinsics are rescaled to each image's resolution, but only for a plain resize: if the image's aspect ratio differs from the calibrated frame by more than rounding, it is rejected — add a `crop` entry instead.
- `dji_mini4pro.json` is a **sample only**, ported from the reference pipeline: 82.1° **diagonal** FOV (only the diagonal reading is consistent with DJI's 24 mm-equivalent spec).
- `dji_m30t_thermal.json` is the DJI M30T thermal camera used in the FLAME 3 dataset. Its intrinsics were measured by registering thermal frames to the wide camera (`experiments/flame3/measure_thermal_fov.py`): square pixels, fx = 697 px (matches DJI's 61° diagonal FOV), principal point off-centre at (334, 266). Pose comes from the paired raw RGB (`--exif_image_dir <class>/RGB/Raw`). See `docs/FLAME3_GEOLOCATION_HANDOFF.md`.

> **Note on this repo's sample frames:** the thermal images behind `results/flame2_sample/` are 958 × 760, not FLAME-2's native 640 × 512, so they were resized and/or cropped during RGB–thermal alignment. To geolocate them, the camera config must describe that processed frame (its own intrinsics, or the native calibration plus a `crop`).

**Oblique views and the horizon.** A ray depressed θ below the horizon from height h lands h / tan θ away, and an attitude error dθ moves it by h · dθ / sin²θ:

| Depression | Ground range at 100 m AGL | Error per 1° attitude error |
|---|---|---|
| 30° | 173 m | 7 m |
| 10° | 567 m | 58 m |
| 5° | 1.1 km | 230 m |
| 1° | 5.7 km | 5.7 km |

Rays at or above the horizon never reach the ground; rays just below it land kilometres away with errors just as large. A ray is therefore reliable only if it is at least `--min_depression_deg` (default 10°) below the horizon and within `--max_range_m`. On flat ground that reliable area is a disk around the point below the drone (radius h / tan(min depression), or the max range if smaller). A region with unreliable vertices is **clipped to that disk** rather than dropped, and its centroid and area come from the clipped polygon. Only a region with no reliable part is `unreliable`. `docs/geolocation_worked_example.md` walks through one oblique projection by hand.

**Swapping the metadata source.** Every loader implements one method, `load(frame_id, image_path) -> FrameMetadata` (`geoloc/metadata.py`). A telemetry-log loader only has to return the same `FrameMetadata`; `geolocate.py` and the math are unchanged.

**What each region gets:**
- `geolocation_status` — `ok` (every vertex reliable, polygon unchanged), `clipped_to_reliable`, `unreliable` (no reliable part), or `missing_metadata`
- `gps_polygon` — the reliable polygon, projected from `normalized × original_wh` (never `pixel_256`, because the 256 × 256 resize is anisotropic). Each vertex is `{lat, lon, ground_range_m, depression_deg}`; vertices created by clipping carry `"clip_vertex": true` (they lie on the reliability boundary, not on the fire edge). `null` for `unreliable` regions
- `gps_centroid`, `ground_area_m2` — area-weighted centroid and area of that polygon, computed on the ground (for oblique views the image-space centroid is not the ground centroid). The polygon is a fire zone (merged hot spots and the gaps between them)
- `fire_area_m2` — ground area of the hot pixels inside the region (`fire_pixels_256` from `localize.py`), each pixel run projected exactly on flat ground. Only reliable pixels count; `fire_area_excluded_fraction` is the share of hot pixels left out. `null` if `localize.py` did not provide hot pixels
- `clipped`, `removed_area_fraction` — share of the original contour that was cut off, **measured in image pixels** (the removed part's ground area is unbounded or unreliable by definition); `unreliable_vertices` counts the original contour vertices by reason (`above_horizon` / `below_min_depression` / `beyond_max_range`)
- `touches_frame_edge` — the fire continues outside the frame
- GeoJSON output contains `ok` and `clipped_to_reliable` regions, with `clipped` and `removed_area_fraction` in the properties
- Clipping assumes flat ground. With both limits disabled, a region that crosses the horizon cannot be bounded and is `unreliable`
- `fire_front_arc` stays `null` (out of scope)

Each frame also gets a `geolocation` block with the drone pose used, `gsd_center_m`, warnings (oblique view, altitude not AGL), and `dilation_buffer_m` — `localize.py`'s merge dilation pushes every polygon outward by roughly this distance, so GPS outlines are larger than the burning area.

**Seeing the results on the image.** `visualize_geolocation.py` draws a `geolocate.py` JSON result back onto each thermal frame. Each picture shows:
- each region, numbered, with its GPS centroid, ground area and distance from the drone
- ground-distance rings around the drone (`--rings_m`, default 100/200/400 m)
- the reliability limit of the run, and the horizon
- parts clipped as unreliable, hatched

It only draws, using the same camera config and the pose stored in the JSON:
```bash
python visualize_geolocation.py \
  --geolocated_json geolocated_results.json \
  --camera_config camera_configs/<your_camera>.json \
  --thermal_dir custom_test/thermal \
  --vis_dir geolocation_vis
```

**Running the tests** (standard library `unittest`, synthetic inputs only):
```bash
python -m unittest discover -s tests -t .
python -m tests.test_citrus_oracle    # prints the reference-code error table
```

**Attribution.** The geolocation approach is adapted from [citrus-uav-pipeline](https://github.com/afif1230/citrus-uav-pipeline) by Afif El Bsat (MIT License — see `THIRD_PARTY_LICENSES/citrus-uav-pipeline.txt`). Differences from the reference: the image is never rotated (yaw rotates the ray, so focal length does not depend on a rotated canvas size), the FOV type is explicit, any gimbal pitch is supported, and missing metadata is never replaced by defaults. `tests/test_citrus_oracle.py` documents the two scale issues found in the reference (FOV interpreted as horizontal, focal length taken from the rotated canvas) with concrete numbers.

---

## Output Format

### `inference_results.json`

```json
{
  "summary": {
    "total_frames": 60,
    "fire_detected": 40,
    "clear_frames": 20,
    "fire_rate": 0.6667
  },
  "results": [
    {
      "frame_id": "frame_001",
      "filename": "frame_001.jpg",
      "classifier_prediction": "YN",
      "fire_detected": true,
      "phase2_localization": true,
      "probabilities": { "NN": 0.02, "YN": 0.91, "YY": 0.07 }
    }
  ]
}
```

### `localization_results.json`

```json
{
  "summary": {
    "localization_run_on": 40,
    "frames_with_regions": 38,
    "fire_detected_no_regions": 2,
    "parameters": {
      "threshold_scale": 0.85,
      "merge_kernel_size": 15,
      "epsilon_factor": 0.01
    }
  },
  "results": [
    {
      "frame_id": "frame_001",
      "fire_detected": true,
      "localization_run": true,
      "otsu_threshold": 187,
      "applied_threshold": 158,
      "n_raw_contours": 3,
      "fire_regions": [
        {
          "pixel_256": [{"x": 112, "y": 88}, {"x": 134, "y": 91}, "..."],
          "normalized": [{"x": 0.4375, "y": 0.3437}, {"x": 0.5234, "y": 0.3554}, "..."],
          "area_px": 1823.5,
          "n_points": 14,
          "original_wh": {"width": 640, "height": 512},
          "fire_area_px": 412,
          "fire_pixels_256": [[88, 118, 125], [89, 116, 127], "..."],
          "gps_polygon": null,
          "fire_front_arc": null
        }
      ]
    }
  ]
}
```

**Coordinate systems:**
- `pixel_256` — ordered contour points in 256×256 processing space
- `normalized` — same points in [0,1] range, resolution-agnostic. This is what `geolocate.py` consumes to back-project to GPS coordinates.
- `original_wh` — actual thermal image dimensions at inference time, for reference
- `fire_pixels_256`, `fire_area_px` — the hot pixels (undilated mask) inside the region, as row runs `[row, first_col, last_col]` in 256×256 space, and their count. The outline is a merged fire zone; these are the pixels actually above threshold. `geolocate.py` turns them into `fire_area_m2`
- `gps_polygon` — null here; filled by `geolocate.py` (see [Geolocation](#6-geolocation)). `fire_front_arc` stays null

**Output format options (`--output_format`):**
| Format | File | Best for |
|--------|------|----------|
| `json` | `localization_results.json` | Pipeline-internal use, full fidelity |
| `geojson` | `localization_results.geojson` | GIS tools (QGIS, Mapbox, Google Earth) |
| `csv` | `localization_results.csv` | Simple downstream consumption, custom propagation model |

---

## Script Reference

| Script | Role |
|--------|------|
| `dataset.py` | `FireDataset` PyTorch Dataset — loads paired RGB+thermal frames and labels |
| `split_dataset.py` | Stratified train/val/test split (70/15/15), copies files to `dataset_split/` |
| `model.py` | `MidFusionNet` — dual-branch ResNet-18 mid-fusion classifier |
| `train.py` | Training loop — Adam, Cross-Entropy, 15 epochs, saves `.pth` checkpoint |
| `evaluate.py` | Test set evaluation — confusion matrix display |
| `infer.py` | Phase 1 inference on arbitrary image pairs — outputs JSON + optional 3-panel visualizations |
| `detect_by_temperature.py` | Phase 1 option — fire / no fire from radiometric temperatures, alone or OR-combined with `infer.py`'s output; same JSON format |
| `localize.py` | Phase 2 localization — Otsu → dilation → contour extraction → JSON/GeoJSON/CSV + optional 5-panel visualizations |
| `geolocate.py` | Phase 3 — projects fire contours to GPS polygons, centroids and ground areas using per-frame metadata (EXIF/XMP or sidecar CSV) and a camera config |
| `visualize_geolocation.py` | Draws `geolocate.py` results on the thermal frames: regions with GPS centroid / area / distance, ground-distance rings, reliability limit, horizon |
| `geoloc/` | Geolocation library: `camera.py` (config), `metadata.py` (loaders), `projection.py` (pure geometry) |

---

## Model Architecture

**MidFusionNet** — dual-branch mid-fusion network:

```
RGB [B, 3, 256, 256]      Thermal [B, 1, 256, 256]
        │                          │
   ResNet-18 encoder          ResNet-18 encoder
   (ImageNet pretrained)      (ImageNet pretrained,
                               conv1 modified: 3ch→1ch)
        │                          │
   512-d features             512-d features
        └──────────┬───────────────┘
                   │  concat
              1024-d vector
                   │
            Linear(1024→256)
                   │
                 ReLU
                   │
              Dropout(0.3)
                   │
             Linear(256→3)
                   │
              NN / YN / YY
```

**Training results on FLAME-2 test set:**

|  | Predicted NN | Predicted YN | Predicted YY |
|--|---|---|---|
| **Actual NN** | 255 | 0 | 0 |
| **Actual YN** | 0 | 248 | 1 |
| **Actual YY** | 0 | 0 | 246 |

**Overall accuracy: 99.87%**

---

## Known Limitations

- **Thermal palette domain shift** — the model was trained on a single sensor platform. Different thermal cameras apply different color mappings, which can cause misclassification on out-of-distribution hardware. Mitigation: cross-sensor normalization or multi-platform fine-tuning.
- **Fixed processing resolution** — both modalities are resized to 256×256. Small or distant fire regions may lose spatial detail. A multiscale inference approach could improve small-source detection.
- **No temporal modeling** — frames are processed independently. A UAV video stream has strong temporal correlation that is currently unused.
- **Flat-ground geolocation** — rays are intersected with a horizontal plane at the drone's altitude above ground. In hilly terrain the error grows with height error × tan(off-nadir angle) (e.g. 20 m of relief at 30° off-axis ≈ 11.5 m). A DEM can be added behind the `Terrain` interface in `geoloc/projection.py`.
- **Altitude reference** — DJI `RelativeAltitude` is height above the take-off point, not above the ground under the drone; it equals AGL only on terrain level with take-off. Such frames carry a warning. An altitude error δh moves every point by about δh / h of its range. On FLAME 3 (Sycan Marsh) an SRTM elevation model shows the burn area is flat (a few metres of relief), so take-off altitude is a reasonable AGL there. Over hilly ground a DEM behind the `Terrain` interface is the general fix (not built yet).
- **Laser-rangefinder height is unreliable over trees and smoke** — the laser measures the first surface it hits. On FLAME 3 only ~27% of laser readings hit the ground (within ±3 m of SRTM); most hit canopy or smoke 3–60 m higher, which makes `--altitude_source laser` heights too low. The comparison in `experiments/flame3/laser_check.py` therefore checks angle conventions, not ground accuracy.
- **No surveyed validation yet** — the math is unit-tested against hand-computed cases and checked against DJI's own laser projection, which shares the gimbal angles, so absolute accuracy (e.g. a yaw bias) is untested. Surveyed ground control points (FLAME 3 NADIR plots) are the next step. The final camera is not confirmed.
- **Outlines are fire zones** — `localize.py`'s region-merging dilation joins nearby hot spots into one outline, so `ground_area_m2` covers the hot spots, the gaps between them and a ~`dilation_buffer_m` margin. On FLAME 3 the hot pixels are a median ~15% of the outline. `fire_area_m2` is the ground area of the hot pixels themselves.
- **No lever-arm or time-sync correction** — the GPS antenna is assumed to be at the camera, and metadata is assumed to be captured at the same instant as the frame.

---

## Acknowledgments

- Geolocation adapted from [citrus-uav-pipeline](https://github.com/afif1230/citrus-uav-pipeline) (MIT License, Copyright (c) 2026 afif1230):
  A. El Bsat, A. Mohanna, B. Kaddouh, "Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment," *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2026.

---

*Autonomous Firefighting UAVs Project*
