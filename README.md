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
  - [7. Video (frame by frame)](#7-video-frame-by-frame)
  - [8. Recording from a camera](#8-recording-from-a-camera)
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
├── prepare_dataset.py  # Samples and resizes raw FLAME-2 pairs into dataset_processed/
├── split_dataset.py    # Stratified train/val/test split with file copying
├── model.py            # MidFusionNet architecture definition
├── train.py            # Training loop
├── evaluate.py         # Test set evaluation with confusion matrix
├── infer.py            # Phase 1: inference on arbitrary image pairs
├── detect_by_temperature.py  # Phase 1 option: fire / no fire from radiometric temperatures
├── localize.py         # Phase 2: fire region localization (contour-based)
├── geolocate.py        # Phase 3: contour pixel → GPS (CLI)
├── visualize_geolocation.py  # draws geolocate.py results on the thermal frames
├── process_video.py    # localization + geolocation on a thermal video, frame by frame
├── record_camera.py    # records video from a camera, with the capture time of every frame
├── mavlink_telemetry.py # logs the drone's position and attitude from its flight controller (MAVLink)
├── srt_to_telemetry.py # per-frame telemetry CSV from a drone video's .SRT subtitle track
├── geoloc/             # Camera-agnostic geolocation library (no rendering code)
│   ├── camera.py       #   CameraConfig: intrinsics / FOV type, distortion, crop, metadata fields
│   ├── metadata.py     #   FrameMetadata + loaders (EXIF/XMP, sidecar CSV)
│   └── projection.py   #   Pure math: pixel → ray → ground → lat/lon
├── camera_configs/     # ct_ei5btc_m.json (flight-test camera), dji_m30t_thermal.json (FLAME 3), dji_mini4pro.json (sample), template.json
├── experiments/flame3/ # FLAME 3 checks: thermal FOV, laser rangefinder, Otsu vs absolute, video, pose offsets
├── tests/              # unittest suite (synthetic inputs, plus the saved results/flame2_sample output)
├── docs/               # geolocation_worked_example.md (one pixel projected by hand), progress reports
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
| `--max_epsilon_px` | `2` | Cap on that epsilon, in pixels at 256×256. Without it the outline of a large region strays up to 20 px from its border and cuts through hot pixels; with it, large regions keep more vertices and small ones are unchanged. Set to `0` for no cap. |
| `--output_format` | `json` | Output format: `json` / `geojson` / `csv` |
| `--visualize` | off | Save 5-panel figures per frame |
| `--vis_dir` | `localization_vis/` | Output folder for visualizations |

**Absolute-temperature mode (radiometric cameras).** `--threshold_mode absolute` finds fire from measured temperatures instead of Otsu's per-image brightness threshold, which stays the default. It needs no detector: without `--inference_json` it runs on every image in `--thermal_dir`, and a frame with nothing hot enough gives zero regions.

```bash
python localize.py \
  --threshold_mode absolute \
  --temperature_dir custom_test/temperature \
  --thermal_dir custom_test/thermal
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--threshold_mode` | `otsu` | `otsu` or `absolute` |
| `--temperature_dir` | required for `absolute` | Float temperature images in °C, same file stems as the thermal images. A frame without one is an error, never "no fire" |
| `--seed_temp_c` | `80` | A region needs pixels at least this hot |
| `--low_temp_c` | `60` | A region grows into connected (8-neighbour) pixels at least this hot |
| `--min_seed_pixels` | `10` | Seed pixels a region needs, counted at the camera's resolution |

Dilation, contours, the size filter and `fire_pixels_256` then run exactly as in Otsu mode. The results on all 738 FLAME 3 frames, with no classifier (`experiments/flame3/run_absolute_threshold.py`, `results/run_E_refined/all_frames/`):

| | Otsu, plain | Otsu, second-level (default) | Absolute (80 / 60 °C) |
|---|---|---|---|
| Fire frames with at least one region (of 622) | 517 | 617 | 621 |
| No Fire frames with at least one region (of 116) | 24 | 49 | 0 |
| Area at ≥ 80 °C inside the outlines | 92.8% | 95.4% | 99.4% |

**Second-level Otsu.** Otsu's threshold often lands low enough that the bright part fills most of the frame; the size filter then drops that one contour, and any fire in it (105 Fire and 92 No Fire frames). Second-level Otsu, on by default, rethresholds within the bright pixels instead (`--no_second_level_otsu` turns it off). It changes nothing on frames without that problem, and recovers 100 of the 105 fire frames. On No Fire frames it raises false regions from 24 to 49 (mostly one region covering 70–75% of the frame), but Otsu mode only localizes frames a detector has already called fire. The hottest pixel in any No Fire frame was 47.2 °C, 33 °C below the seed, but those scenes hold no engines or sun-heated metal. The one absolute-mode miss (frame `00531`, hottest 108 °C) has its hot pixels too scattered for any region to reach 10 seed pixels.

**Localization evolution — parameter presets:**

| Version | Approach | Key settings | Output |
|---------|----------|-------------|--------|
| v1 | MSER + NMS | `mser_min_area=60`, `iou=0.3` | Up to 44 boxes/frame |
| v2 | MSER + NMS (tuned) | `mser_min_area=150`, `iou=0.2`, containment suppression | 2–15 boxes/frame |
| v3 | Contour extraction | `epsilon=0.02`, pure Otsu, no merging | Fire boundary polygons |
| v4 (current) | Contour + merging | `threshold_scale=0.85`, `merge_kernel=15`, `epsilon=0.01` capped at 2 px | Merged polygons, denser arcs |

**Outline accuracy.** Before the 2 px cap, 3.2% of the hot pixels on the 34 FLAME 3 test frames lay outside their own outline (22% in the worst contour); with it, 0.7%, all at the image border. Which regions are found does not change. Numbers and pictures: `results/run_E_refined/`.

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
| `--tables_dir` | off | Also write one CSV per contour here: `<frame_id>_contour_<NN>.csv` |
| `--output_format` | `json` | `json` (full results) or `geojson` (one Polygon per region, `[lon, lat]` order) |

**Sidecar CSV columns:** `frame_id, latitude, longitude, altitude_agl_m, gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg[, timestamp]` — yaw clockwise from north, pitch `-90` = straight down, timestamp ISO 8601. Every column except `timestamp` must be filled; frames with missing metadata are reported as `missing_metadata` and never projected with guessed values.

**Camera configs (`camera_configs/`).** All camera-specific values live here, so swapping cameras is a config change. For another camera, copy `template.json`.
- Intrinsics: either `fx, fy[, cx, cy]` from a checkerboard calibration (preferred) or `fov_deg` + `fov_type` (`horizontal` / `vertical` / `diagonal`). The FOV type is mandatory because most datasheets quote **diagonal** FOV.
- `dist_coeffs` (OpenCV order), optional `crop`, `altitude_reference`, and a `metadata_fields` mapping for the EXIF/XMP loader (`xmp:<prefix:Name>`, `exif:<TagName>` or `const:<value>`).
- Intrinsics are rescaled to each image's resolution, but only for a plain resize: if the image's aspect ratio differs from the calibrated frame by more than rounding, it is rejected — add a `crop` entry instead.
- `dji_mini4pro.json` is a **sample only**, ported from the reference pipeline: 82.1° **diagonal** FOV (only the diagonal reading is consistent with DJI's 24 mm-equivalent spec).
- `ct_ei5btc_m.json` is the camera for the campus flight test: a CT-EI5BTC-M thermal module, 640 × 512, read over USB like a webcam (`record_camera.py`). Its intrinsics are the datasheet's (45.7° horizontal FOV, consistent with its 9.1 mm lens and 12 µm pixels: fx = 759 px), not calibrated. It is not radiometric, so localization uses Otsu, and it should be set to the white-hot palette so that grey level rises with temperature.
- `dji_m30t_thermal.json` is the DJI M30T thermal camera used in the FLAME 3 dataset. Its intrinsics were measured by registering thermal frames to the wide camera (`experiments/flame3/measure_thermal_fov.py`): square pixels, fx = 697 px (matches DJI's 61° diagonal FOV), principal point off-centre at (334, 266). Pose comes from the paired raw RGB (`--exif_image_dir <class>/RGB/Raw`). The measurement and check scripts are in `experiments/flame3/`.

> **Note on this repo's sample frames:** the thermal images behind `results/flame2_sample/` are 958 × 760, not FLAME-2's native 640 × 512, so they were resized and/or cropped during RGB–thermal alignment. To geolocate them, the camera config must describe that processed frame (its own intrinsics, or the native calibration plus a `crop`).

**Oblique views and the horizon.** A ray depressed θ below the horizon from height h lands h / tan θ away, and an attitude error dθ moves it by h · dθ / sin²θ:

| Depression | Ground range at 100 m AGL | Error per 1° attitude error |
|---|---|---|
| 30° | 173 m | 7 m |
| 10° | 567 m | 58 m |
| 5° | 1.1 km | 230 m |
| 1° | 5.7 km | 5.7 km |

Rays at or above the horizon never reach the ground; rays just below it land kilometres away with errors just as large. A ray is therefore reliable only if it is at least `--min_depression_deg` (default 10°) below the horizon and within `--max_range_m`. On flat ground that reliable area is a disk around the point below the drone (radius h / tan(min depression), or the max range if smaller). A region with unreliable vertices is **clipped to that disk** rather than dropped, and its centroid and area come from the clipped polygon. Only a region with no reliable part is `unreliable`. `docs/geolocation_worked_example.md` walks through one oblique projection by hand.

**Swapping the metadata source.** Every loader implements one method, `load(frame_id, image_path) -> FrameMetadata` (`geoloc/metadata.py`). A telemetry-log loader only has to return the same `FrameMetadata`; `geolocate.py` and the math are unchanged. `TimedTelemetryLoader` is the loader for video: telemetry samples at their own rate, interpolated to a frame's time (yaw along the shorter arc). It never interpolates across a hole in the log: a frame near one side of the hole gets that side's sample, and a frame deeper in gets no pose.

**What each region gets:**
- `geolocation_status` — `ok` (every vertex reliable, polygon unchanged), `clipped_to_reliable`, `unreliable` (no reliable part), or `missing_metadata`
- `gps_polygon` — the reliable polygon, projected from `normalized × original_wh` (never `pixel_256`, because the 256 × 256 resize is anisotropic). Each vertex is `{lat, lon, ground_range_m, depression_deg}`; vertices created by clipping carry `"clip_vertex": true` (they lie on the reliability boundary, not on the fire edge). `null` for `unreliable` regions
- `gps_centroid`, `ground_area_m2` — area-weighted centroid and area of that polygon, computed on the ground (for oblique views the image-space centroid is not the ground centroid). The polygon is a fire zone (merged hot spots and the gaps between them)
- `fire_area_m2` — ground area of the hot pixels inside the region (`fire_pixels_256` from `localize.py`), each pixel run projected exactly on flat ground. Only reliable pixels count; `fire_area_excluded_fraction` is the share of hot pixels left out. `null` if `localize.py` did not provide hot pixels
- `dilation_buffer_range_m` — `[smallest, largest]` distance on the ground by which the merge dilation pushed this region's outline outside its hot pixels, measured around the region's centroid. The smallest is sideways, the largest away from the drone. `[0, 0]` without merging, `null` for a region with no reliable part
- `clipped`, `removed_area_fraction` — share of the original contour that was cut off, **measured in image pixels** (the removed part's ground area is unbounded or unreliable by definition); `unreliable_vertices` counts the original contour vertices by reason (`above_horizon` / `below_min_depression` / `beyond_max_range`)
- `touches_frame_edge` — the fire continues outside the frame. Each `gps_polygon` vertex on the image border carries `"frame_edge_vertex": true`: the view ends there, not the fire
- GeoJSON output contains `ok` and `clipped_to_reliable` regions, with `clipped` and `removed_area_fraction` in the properties
- Clipping assumes flat ground. With both limits disabled, a region that crosses the horizon cannot be bounded and is `unreliable`
- `fire_front_arc` stays `null` (out of scope)

Each frame also gets a `geolocation` block with the drone pose used, `gsd_center_m`, warnings (oblique view, altitude not AGL), and `dilation_buffer_m` — `localize.py`'s merge dilation pushes every polygon outward, so GPS outlines are larger than the burning area. This one value is that distance sideways at the image centre; in an oblique view the margin is longer away from the drone and grows with distance, so use each region's `dilation_buffer_range_m`.

**Contour tables.** With `--tables_dir`, every contour gets its own CSV of geolocated border points, in the order of the border line. Columns: `frame_id, contour, point, lat, lon, ground_range_m, depression_deg, point_type`. `point_type` is `fire_edge`, `image_border` (the view ends here) or `clipped` (the reliability limit ends here). A contour with no coordinates still gets a table: blank coordinates, with `unreliable` or `missing_metadata` as `point_type`. Contour and point numbers match the pictures below.

**Seeing the results on the image.** `visualize_geolocation.py` draws a `geolocate.py` JSON result back onto each thermal frame. Each picture shows:
- each contour, numbered, as a border line through its numbered points (every point is marked; on a contour with more than 40 only every Nth is numbered), with its distance range from the drone, zone area and burning area
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

**Running the tests** (standard library `unittest`; synthetic inputs, plus one check on the saved `results/flame2_sample` output):
```bash
python -m unittest discover -s tests -t .
python -m tests.test_citrus_oracle    # prints the reference-code error table
```

**Attribution.** The geolocation approach is adapted from [citrus-uav-pipeline](https://github.com/afif1230/citrus-uav-pipeline) by Afif El Bsat (MIT License — see `THIRD_PARTY_LICENSES/citrus-uav-pipeline.txt`). Differences from the reference: the image is never rotated (yaw rotates the ray, so focal length does not depend on a rotated canvas size), the FOV type is explicit, any gimbal pitch is supported, missing metadata is never replaced by defaults, and metres are converted to lat/lon on the WGS84 ellipsoid (the reference's sphere is 0.2% off, 1 m at 500 m from the drone). `tests/test_citrus_oracle.py` documents the two scale issues found in the reference (FOV interpreted as horizontal, focal length taken from the rotated canvas) with concrete numbers.

---

### 7. Video (frame by frame)

`process_video.py` runs localization and geolocation on a thermal video one frame at a time, the way a live stream arrives. Each frame goes through exactly the same steps as a still image, and the results are written as each frame finishes.

```bash
# telemetry from the drone's subtitle track (DJI .SRT), matched to each frame by time
python srt_to_telemetry.py --srt flight.SRT --output_csv telemetry.csv
python process_video.py \
  --video flight.mp4 \
  --timed_telemetry_csv telemetry.csv \
  --camera_config camera_configs/<your_camera>.json \
  --csv_altitude_reference takeoff \
  --output_dir results/my_video --annotate
```

**Pose.** A video carries no per-frame metadata, so it comes from a CSV, in one of two forms:

| Option | CSV | Use for |
|---|---|---|
| `--timed_telemetry_csv` | telemetry at its own rate: `time_s` (seconds from the start of the video), `latitude, longitude, altitude_agl_m, gimbal_yaw_deg, gimbal_pitch_deg, gimbal_roll_deg[, timestamp]`. Each frame gets the pose interpolated to its time; a frame farther than `--telemetry_max_gap_s` (default 1 s) from any sample gets none | real footage |
| `--telemetry_csv` | one row per frame, `frame_id` = frame number zero-padded to `--frame_id_digits` (the sidecar format above) | test videos where no frame was ever dropped |

Frame times come from the decoder, so dropped frames do not shift the matching. For a video recorded from a live camera, pass `--frame_times_csv` (see [Recording from a camera](#8-recording-from-a-camera)). A frame without usable telemetry is reported as `missing_metadata` and the run continues. `srt_to_telemetry.py` makes the timed CSV from a DJI `.SRT` subtitle track (one entry per frame with position and gimbal angles): run it with `--show_keys` on a new file first, since the keys differ between drone models, then map each field with `--field` or fix it with `--const`. It was written against DJI's documented layout, not yet checked against a file from our drone.

**Outputs** in `--output_dir`:

| File | Contents |
|---|---|
| `frames.jsonl` | one JSON line per processed frame, written as each frame finishes: what a live consumer would read |
| `border_points.csv` | one row per geolocated border point of every contour of every frame: `frame_id, video_time_s, timestamp, contour, point, lat, lon, ground_range_m, depression_deg, point_type` |
| `fire_borders.geojson` | every geolocated border as a polygon with its frame and time, for a map |
| `geolocated_results.json` | all frames in one file: `localize.py` + `geolocate.py` fields plus `frame_index`, `video_time_s`, `background_gray`, `timing_ms` |
| `annotated.mp4` | with `--annotate`: the processed frames with borders and numbered points drawn in |
| `<--tables_dir>/` | optional: one CSV per contour, as `geolocate.py --tables_dir` |

- **Localization:** plain video has no temperatures, so this uses Otsu on every frame, with no detector in front. Its parameters and the reliability limits match `localize.py` and `geolocate.py`, except that second-level Otsu is off unless `--second_level_otsu`: without a detector it also turns more fire-free frames into false alarms.
- **Hot-spot mode (optional):** `--threshold_mode hotspot` outlines whatever stands well above the picture's typical level: brighter than the median by `--hotspot_sigmas` (default 6) times the picture's spread, and by at least `--hotspot_min_contrast` (default 25) grey levels. Otsu splits the picture into two groups, so it only isolates fire that fills a sizeable share of the picture; a small hot area in a large scene (an early fire, a heated test target) is too few pixels, and Otsu then cuts the background in two. In simulated scenes a 12 × 9 px target was missed by Otsu every time and found by this mode every time, with no regions in target-free scenes. On FLAME 3 it finds regions in 621 of 622 fire frames and 37 of 116 fire-free frames (plain Otsu: 517 and 24), with 91.7% of the area above 80 °C inside the outlines (plain Otsu: 92.8%).
- **Background gate (optional):** `--max_background_gray N` skips a frame whose median grey level is above N. A camera that scales its picture to the scene shows the ground dark once fire is in view and mid-grey when there is none. It is off by default because it depends on the camera's gain behaviour: 60 suited FLAME 3's DJI M30T, the only camera it was tried on. Every frame's result carries `background_gray`, so the limit can be checked on new footage. A skipped frame has `fire_detected: false` and no regions.
- **Slower hardware:** `--frame_step N` processes every Nth frame; skipped frames are not decoded and keep their times.

**Tested on FLAME 3** (`experiments/flame3/make_video.py`, `compare_video.py`; `results/run_D_video/`): 70 fire stills from one pass and 55 No Fire stills, made into videos at one frame per still (3 s apart).
- **Same as stills:** a lossless video gives identical contours to the stills in all 70 frames. A lossy MP4 (7x smaller) moves border points by at most 0.52 m for 90% of them.
- **Consistent between frames:** while the drone moves 20-48 m between frames, 87% of the fire lands within 5 m of where the next frame puts it (median).
- **Fast:** about 10 ms per frame for localization plus geolocation at 640x512.
- **Otsu's limits carry over:** fire was localized in 61 of 70 fire frames, and 21 of 55 No Fire frames produced regions. With `--second_level_otsu`: 70 of 70 fire frames, but 31 of 55 No Fire frames. Adding `--max_background_gray 60`: still 70 of 70, and 8 of 55 (`results/run_E_refined/`; over all 738 stills, 617 of 622 fire frames and 8 of 116 No Fire frames). That limit was chosen on these frames, so a real detector before Otsu remains the safer answer on video.

---

### 8. Recording from a camera

`record_camera.py` records video from a camera the way a webcam is read (OpenCV's `VideoCapture`), and writes the real capture time of every frame next to it.

```bash
python record_camera.py --source 0 --output_dir results/flight_01/recording --preview
python process_video.py   --video results/flight_01/recording/video.mp4   --frame_times_csv results/flight_01/recording/frame_times.csv   --timed_telemetry_csv telemetry.csv   --camera_config camera_configs/<your_camera>.json   --output_dir results/flight_01/processed --annotate
```

| Argument | Default | Description |
|---|---|---|
| `--source` | `0` | Camera number (`0` = the first camera: a webcam, a USB thermal camera, a USB capture card), `picamera` (a Raspberry Pi camera module), a stream URL (`rtsp://...`), or a video file |
| `--output_dir` | required | Where the recording goes |
| `--codec` | `mp4v` | `mp4v`: small, lossy. `mjpg`: larger, light on the processor. `ffv1`: lossless |
| `--width`, `--height`, `--fps` | camera's own | Requests to the camera; it may give something else, and `recording.json` says what it gave |
| `--duration_s` | off | Stop after this many seconds |
| `--preview` | off | Show the picture while recording; `q` stops |
| `--mavlink` | off | Also log the drone's telemetry from its flight controller into `telemetry.csv`: a connection such as `/dev/ttyACM0` (USB), `/dev/serial0,57600` (serial pins, baud rate) or `udpin:0.0.0.0:14550` |
| `--telemetry_rate_hz` | `20` | Attitude and position messages per second to ask the flight controller for |

Stop with Ctrl+C, `q` in the preview, or `--duration_s`. It writes `video.mp4` (or `.avi`), `frame_times.csv` (`frame_index, time_s, utc_time`) and `recording.json` (frame size, nominal and measured frame rate, start and end time).

**On a Raspberry Pi.** The recorder is meant to run on the computer that flies with the camera.
- A USB camera is a camera number, as on any computer. The Pi's own camera module (ribbon cable) needs `--source picamera` and the `picamera2` library (`sudo apt install python3-picamera2 python3-opencv`).
- Use `--codec mjpg`, the lightest on the Pi's processor, and `--duration_s` or Ctrl+C over SSH instead of `--preview`.
- A Pi has no battery-backed clock: without a network it starts with the wrong date, so `utc_time` is wrong unless the clock is set before the flight. `time_s` is not affected.
- The `picamera` source is tested against a stand-in for the library, not yet on a real Pi.

**Telemetry from the flight controller.** On a drone with a MAVLink flight controller (a Pixhawk running PX4 or ArduPilot, as on the Holybro X500), the recording computer can log the drone's position itself, on the same clock as the frames, so no clocks have to be matched afterwards. It needs `pymavlink` (`pip install pymavlink`).

```bash
python record_camera.py --source 0 --codec mjpg --mavlink /dev/ttyACM0 --output_dir flight_01
python process_video.py --video flight_01/video.avi --frame_times_csv flight_01/frame_times.csv --timed_telemetry_csv flight_01/telemetry.csv --csv_altitude_reference takeoff --camera_mount_deg 0 -90 0 --camera_config camera_configs/ct_ei5btc_m.json --output_dir results/flight_01 --annotate
```

- The recorder waits for the flight controller's heartbeat before it starts, so a dead link is known on the ground. If the link drops later, the video keeps recording.
- `telemetry.csv` has one row per attitude message with the latest position: `time_s` (seconds since the first frame), `latitude`, `longitude`, `altitude_agl_m` (height above the take-off point, hence `--csv_altitude_reference takeoff`), the drone's yaw, pitch and roll, plus `position_age_s`, `gps_fix_type` and `satellites` for checking the data.
- **Fixed camera.** Without a gimbal the camera tilts with the drone, and the telemetry holds the drone's attitude, not the camera's. `process_video.py --camera_mount_deg YAW PITCH ROLL` gives the camera's angles on a level drone, and the two are combined for every frame. `0 -90 0` is straight down with the top of the picture towards the nose. The mount is applied when processing, so it can be corrected after the flight.
- Tested with stand-ins for the flight controller and for `pymavlink`, not yet against the library itself or a real flight controller.

**Why the frame times.** A camera does not deliver frames at exactly the rate the video file claims, so the file's clock drifts from real time: 0.08 s in a 5 s webcam test, which would be about 10 s over a 10-minute flight. With `--frame_times_csv`, `process_video.py` uses each frame's real capture time (`time_s`, seconds since the first frame) to pick the drone's pose. The telemetry's `time_s` must count from the same instant: the recording's first frame, whose UTC time is `start_utc` in `recording.json`. `utc_time` comes from the recording computer's clock, so set that clock before a flight if the telemetry is stamped by GPS time.

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
      "epsilon_factor": 0.01,
      "max_epsilon_px": 2.0
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
          "normalized": [{"x": 0.439453, "y": 0.345703}, {"x": 0.525391, "y": 0.357422}, "..."],
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
- `normalized` — the same points as positions across the image in [0,1], resolution-agnostic: the centre of pixel `i` is at `(i + 0.5) / 256`, so `normalized × original_wh` is that spot in the full-size image. This is what `geolocate.py` consumes to back-project to GPS coordinates.
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
| `visualize_geolocation.py` | Draws `geolocate.py` results on the thermal frames: numbered border points per contour, distance, areas, ground-distance rings, reliability limit, horizon |
| `record_camera.py` | Records video from a camera (number, stream URL or file) with the real capture time of every frame, for `process_video.py --frame_times_csv`; with `--mavlink`, also the drone's telemetry |
| `mavlink_telemetry.py` | Logs position and attitude from a MAVLink flight controller (Pixhawk: PX4 or ArduPilot) into the timed telemetry CSV, on the video's clock |
| `process_video.py` | Localization + geolocation on a thermal video, frame by frame, with pose matched by time (or frame number) from a telemetry CSV; streams `frames.jsonl`, `border_points.csv`, `fire_borders.geojson`, optional annotated video |
| `srt_to_telemetry.py` | Timed telemetry CSV from a drone video's `.SRT` subtitle track (DJI layout, configurable keys) |
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
- **No surveyed validation yet** — the math is unit-tested against hand-computed cases and checked against DJI's own laser projection, which shares the gimbal angles, so absolute accuracy is untested. One check without ground truth: consecutive video frames taken 43 m apart agree best with the yaw as recorded (0° to −3°), which rules out a bias the size of the magnetic declination (`experiments/flame3/pose_offset_check.py`, `results/run_E_refined/`). An error that moves every frame the same way would not show there. Surveyed ground control points (FLAME 3 NADIR plots) are the next step.
- **Outlines are fire zones** — `localize.py`'s region-merging dilation joins nearby hot spots into one outline, so `ground_area_m2` covers the hot spots, the gaps between them and a margin around them. On FLAME 3 the hot pixels are a median ~15% of the outline. `fire_area_m2` is the ground area of the hot pixels themselves. Border coordinates sit outside the hot pixels by that margin: 7 px at 256×256 in the image, but on the ground it depends on the direction and on how far away the region is. On the 34 test frames it is a median 6 m sideways and 16 m away from the drone, over 30 m for one region in ten, and up to 62 m for the farthest. Each region reports its own in `dilation_buffer_range_m`; the frame's single `dilation_buffer_m` (median 5.5 m) is only the sideways value at the image centre. For the same reason, hot spots far from the drone are merged across larger gaps than near ones.
- **No lever-arm or time-sync correction** — the GPS antenna is assumed to be at the camera, and metadata is assumed to be captured at the same instant as the frame.

---

## Acknowledgments

- Geolocation adapted from [citrus-uav-pipeline](https://github.com/afif1230/citrus-uav-pipeline) (MIT License, Copyright (c) 2026 afif1230):
  A. El Bsat, A. Mohanna, B. Kaddouh, "Unified RGB-UAV Pipeline for Citrus Tree Detection, Geotagging, and HLB Health Assessment," *IEEE International Geoscience and Remote Sensing Symposium (IGARSS)*, 2026.

---

*Autonomous Firefighting UAVs Project*
