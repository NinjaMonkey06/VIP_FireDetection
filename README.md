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
│   Phase 3: Geolocation                      │  ← geolocate.py (stub)
│   MAVLink v2 telemetry parsing              │
│   Pinhole camera back-projection            │
│   GPS polygon + fire wavefront arc output   │
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
├── localize.py         # Phase 2: fire region localization (contour-based)
├── geolocate.py        # Phase 3: GPS back-projection (fully specified stub)
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

For Phase 3 geolocation (once implementing `geolocate.py`):
```bash
pip install pymavlink pyproj
```

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

`geolocate.py` is a **fully specified implementation stub** for Phase 3. It converts normalized image-space fire contour points from `localize.py` to GPS world coordinates using drone telemetry and camera intrinsics, then outputs GPS fire boundary polygons and simplified wavefront arc polylines for the fire propagation model.

**Hardware specification (confirmed):**
| Component | Specification |
|-----------|--------------|
| Thermal camera | FLIR HADRON 640R+, 13.6mm lens, 60Hz |
| Altitude | Barometric + GPS fusion |
| Gimbal | Real-time angle telemetry (pitch, roll, yaw) |
| Telemetry format | MAVLink v2 over serial |

```bash
# Once implemented:
python geolocate.py \
  --localization_json localization_results.json \
  --telemetry_log flight.tlog \
  --output_format geojson \
  --wind_direction_deg 270
```

**To implement `geolocate.py`:**
1. Install dependencies: `pip install pymavlink pyproj`
2. Obtain a MAVLink v2 `.tlog` or `.bin` telemetry log from a test flight
3. Confirm camera model is FLIR HADRON 640R+ with 13.6mm lens (re-derive intrinsics from spec sheet if different)
4. Confirm whether gimbal angles are world-referenced or body-referenced (affects rotation matrix construction)
5. Translate each pseudocode stub function to Python — all math is fully specified in the file

The stub contains: camera intrinsic derivation, MAVLink message parsing, Euler→rotation matrix, pinhole back-projection math, wind-guided wavefront arc extraction, and all three output formatters. No additional design decisions are required.

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
- `gps_polygon` / `fire_front_arc` — null placeholders populated by `geolocate.py`

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
| `localize.py` | Phase 2 localization — Otsu → dilation → contour extraction → JSON/GeoJSON/CSV + optional 5-panel visualizations |
| `geolocate.py` | Phase 3 GPS back-projection stub — fully specified pseudocode, ready to implement |

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
- **Flat-earth geolocation assumption** — the back-projection in `geolocate.py` assumes flat terrain at the drone's GPS altitude. Mountainous terrain requires DEM (Digital Elevation Model) intersection for accurate GPS projection.
- **Geolocation not yet implemented** — `geolocate.py` is a fully specified stub. Normalized contour coordinates are structured and ready; implementation requires a MAVLink telemetry log and `pymavlink` / `pyproj`.

---

*Autonomous Firefighting UAVs Project*
