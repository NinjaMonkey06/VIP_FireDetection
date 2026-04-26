# VIP_FireDetection

A hybrid deep learning + classical computer vision pipeline for real-time wildfire detection and localization from UAV-mounted multispectral sensors. The system ingests paired **RGB + Thermal/IR** frames and outputs fire presence decisions, spatial bounding boxes over detected fire regions, and structured coordinate data ready for handoff to a fire propagation model or geolocation module.

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
┌─────────────────────────┐
│   Phase 1: MidFusionNet │  ← dual-branch ResNet-18 classifier
│   Fire presence decision│    (NN / YN / YY)
└─────────────────────────┘
        │
        ├── No fire → skip (zero localization compute)
        │
        ▼
┌─────────────────────────┐
│   Phase 2: Localization │  ← classical CV pipeline
│   Otsu Thresholding     │
│   MSER Extraction       │
│   Mask × MSER Fusion    │
│   NMS + Containment     │
└─────────────────────────┘
        │
        ▼
  Bounding Boxes (pixel + normalized)
        │
        ▼
  (Future) Geolocation → GPS World Coordinates
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
├── infer.py            # Phase 1 inference on arbitrary image pairs
├── localize.py         # Phase 2 fire region localization pipeline
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
cd uav-fire-detection
pip install torch torchvision opencv-python pillow matplotlib scikit-learn pandas numpy
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
| `--visualize` | off | Save 3-panel visualization figures |
| `--vis_dir` | `inference_vis/` | Output folder for visualizations |

---

### 5. Localization

Runs the Phase 2 classical CV localization pipeline on all fire-positive frames from `inference_results.json`. **Must be run after `infer.py`.**

```bash
# Minimal
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal

# With 4-panel visualization (include --rgb_dir for the RGB panel)
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal \
  --rgb_dir custom_test/rgb \
  --visualize

# Manual parameter tuning
python localize.py \
  --inference_json inference_results.json \
  --thermal_dir custom_test/thermal \
  --mser_min_area 100 \
  --mask_overlap_ratio 0.3 \
  --nms_iou_threshold 0.15 \
  --visualize
```

**Clear frames are skipped entirely** — localization compute is only incurred on fire-positive frames, directly simulating efficient onboard UAV inference.

**Tunable parameters:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--otsu_blur_kernel` | `5` | Gaussian blur kernel before Otsu thresholding. Set to `1` to disable. |
| `--mser_delta` | `3` | MSER stability threshold. Lower = more regions detected. |
| `--mser_min_area` | `150` | Minimum MSER region area (pixels at 256×256). Increase to filter noise. |
| `--mser_max_area` | `6000` | Maximum MSER region area. Decrease if background regions appear. |
| `--mask_overlap_ratio` | `0.35` | Minimum fraction of MSER box pixels that must overlap the Otsu mask. |
| `--nms_iou_threshold` | `0.2` | IoU threshold for NMS. Lower = more aggressive suppression. |
| `--containment_threshold` | `0.85` | Fraction of a smaller box inside a larger box to trigger suppression. |

**Parameter presets observed during testing:**

| Preset | `mser_min_area` | `nms_iou_threshold` | Containment | Boxes per frame |
|--------|----------------|---------------------|-------------|-----------------|
| v1 (fine-grained) | 60 | 0.3 | None | Up to 44 |
| v2 (default) | 150 | 0.2 | 0.85 | 2 – 15 |

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
      "probabilities": { "NN": 0.02, "YN": 0.91, "YY": 0.07 },
      "bounding_boxes": []
    }
  ]
}
```

### `localization_results.json`

```json
{
  "summary": {
    "localization_run_on": 40,
    "frames_with_boxes": 38,
    "fire_detected_no_regions": 2,
    "parameters": {
      "mser_delta": 3,
      "mser_min_area": 150,
      "nms_iou_threshold": 0.2
    }
  },
  "results": [
    {
      "frame_id": "frame_001",
      "fire_detected": true,
      "localization_run": true,
      "otsu_threshold": 187,
      "mser_candidates": 23,
      "post_fusion_candidates": 6,
      "bounding_boxes": [
        {
          "pixel_256": { "x": 112, "y": 88, "w": 45, "h": 38 },
          "normalized": { "x": 0.4375, "y": 0.3437, "w": 0.1757, "h": 0.1484 },
          "original_wh": { "width": 640, "height": 512 }
        }
      ]
    }
  ]
}
```

**Coordinate systems:**
- `pixel_256` — coordinates in the 256×256 processing space
- `normalized` — resolution-agnostic [0,1] coordinates; multiply by any target resolution to remap. This is what the geolocation stage will consume.
- `original_wh` — actual thermal image dimensions at inference time, for reference

---

## Script Reference

| Script | Role |
|--------|------|
| `dataset.py` | `FireDataset` PyTorch Dataset — loads paired RGB+thermal frames and labels |
| `split_dataset.py` | Stratified train/val/test split (70/15/15), copies files to `dataset_split/` |
| `model.py` | `MidFusionNet` — dual-branch ResNet-18 mid-fusion classifier |
| `train.py` | Training loop — Adam, Cross-Entropy, 15 epochs, saves `.pth` checkpoint |
| `evaluate.py` | Test set evaluation — confusion matrix display |
| `infer.py` | Phase 1 inference on arbitrary image pairs, outputs JSON + optional visualizations |
| `localize.py` | Phase 2 localization — Otsu → MSER → mask fusion → NMS, outputs JSON + optional visualizations |

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
- **Geolocation not yet implemented** — normalized bounding box coordinates are structured and ready for the geolocation module (requires drone altitude, GPS, gimbal angle, and thermal camera intrinsics).

---

*Autonomous Firefighting UAVs Project*
