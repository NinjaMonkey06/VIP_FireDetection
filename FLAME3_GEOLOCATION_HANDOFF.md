# FLAME 3 dataset — geolocation test handoff

**For:** Claude Code (or anyone picking this up)
**Project:** UAV wildfire detection: adding GPS geolocation to the existing pipeline
**Status (2026-10-21):** Plan steps 1–3 are done:
- metadata audit of all 738 frames
- thermal intrinsics measured from the data
- M30T camera config
- laser-height altitude option, with a check against the laser rangefinder

**Next:** ground-truth fire masks from the TIFFs, then the full classify → localize → geolocate run (§7).

---

## 1. Goal

Test the **geolocation stage** of the fire-detection pipeline on real, labeled RGB + thermal drone imagery whose camera and per-image pose are fully known. Geolocation must work from **any viewing angle**, not only straight down (supervisor requirement).

Pipeline in this repo (see `README.md`):
- `infer.py`: MidFusionNet classifier, fire yes/no.
- `localize.py`: classical CV on the thermal frame (Otsu → dilation → `approxPolyDP`). It outputs **contour polygons** in `localization_results.json` as `pixel_256` and `normalized` [0,1]. The older MSER/NMS box version (v1/v2) has been replaced.
- `geolocate.py` + `geoloc/`: projects every contour vertex to lat/lon from `normalized × original_wh`. It reports polygon, centroid, ground area, and a reliability status per region.

---

## 2. Dataset: FLAME 3, Computer-Vision subset (Sycan Marsh)

- **Download:** `CVSubset.zip` (6.23 GB) from IEEE DataPort, https://ieee-dataport.org/open-access/flame-3-radiometric-thermal-uav-imagery-wildfire-management (free IEEE account). Kaggle mirror: https://www.kaggle.com/datasets/brycehopkins/flame-3-computer-vision-subset-sycan-marsh. Local copy: `CVSubset/` (git-ignored).
- **Later, for accuracy scoring only:** `NADIRPlots.zip` (32.59 GB) has surveyed ground control points. It was shot straight down at 122 m AGL with an **Autel EVO II Dual 640T (XT709)**, a different camera.
- **Paper:** Hopkins et al., *FLAME 3 Dataset*, arXiv:2412.02831. Dataset DOI 10.21227/w0mz-aq48. No license is stated, so cite both.
- **Why this one:** it's the only public dataset found with labels + paired RGB/thermal + one known camera + full per-image pose. Rejected alternatives: FLAME 2 (video, no telemetry), FireMan-UAV-RGBT (no telemetry), RGBT-3M (not released), FLAME 1 / UCAR (no metadata).

### Layout (verified)
```
FLAME 3 CV Dataset (Sycan Marsh)/
  Fire/     (622 frames)   No Fire/ (116 frames)
    RGB/Raw/00001.JPG            4000x3000 wide camera, carries the pose (DJI XMP)
    RGB/Corrected FOV/00001.JPG  640x512, cropped to roughly the thermal view
    Thermal/Raw JPG/00001.JPG    640x512 palette image, EXIF GPS only, no pose
    Thermal/Celsius TIFF/00001.TIFF  640x512 float32 °C
```
- A frame's four files share one name. RGB and thermal timestamps agree to within ±1 s in all 738 frames: 655 match exactly, and 83 differ by one second (a capture straddling a second boundary).
- **Fire and No Fire reuse the names `00001`–`00116`.** Run the two classes separately.
- **Labels are image-level only.** Ground-truth fire pixels can be made from the TIFF by temperature threshold (the paper uses below ~80 °C = no fire, above ~200 °C = fire).
- The CV subset has **no `metadata` folder or burn map**.

---

## 3. Metadata audit (all 738 frames)

| Item | Result |
|---|---|
| Raw RGB with DJI XMP pose | 738 / 738 |
| `RelativeAltitude` (above take-off) | 49–124 m, median 64 m (Fire) / 77 m (No Fire) |
| Gimbal pitch | −10° to −20°: 551, −20° to −47°: 130, −10° to 0°: 41, above horizon: 16 |
| Frames with pitch ≤ −30° | 46 |
| Gimbal roll | 0 ± 0.3° |
| `GimbalYawDegree` | **absolute**: covers the full circle, tracks `FlightYawDegree` (median difference 0.5°). Values can exceed ±180° (wrapped), which is harmless. True vs magnetic north: **unverified** (declination ~14.5° E here). |
| Laser rangefinder | 643 frames have a reading (`LRFTargetDistance > 0`). `TooFar` 75, `TooClose` 20 (distance written as 0). |
| Laser plausibility | 24 readings are implausible: smoke hits 3–5 m from the drone, or ground 100–230 m below take-off. They are rejected by the ±50 m terrain-offset guard (§5). |
| Ground at laser point vs take-off | median +10 m (IQR +3 to +18 m) |

Thermal JPG pitfall: EXIF `ExifImageWidth/Height` says 1280×1024. **Ignore it**; the images are 640×512. `geolocate.py` uses `original_wh` from the localization JSON.

---

## 4. Thermal camera model (measured)

The thermal fx was the main open question: 9.1 mm / 12 µm gives 758 px, DJI's 61° DFOV gives 696 px. It was **measured** with `experiments/flame3/measure_thermal_fov.py`:

- **Direct registration:** each Celsius TIFF registered to its raw RGB. The fit maximizes edge-map correlation, with fire clipped out so tree lines and terrain edges drive it. It fits thermal (u, v) → raw (a·u + tx, b·v + ty).
- **Intrinsics:** with the wide camera's fx_w = 2775 px and its centre (2000, 1500), fx = fx_w/a, fy = fx_w/b, cx = (2000 − tx)/a, cy = (1500 − ty)/b.
- **Sample:** every 5th frame (148). Of those, 54 have a sharp correlation peak (peak ≥ 0.02); smoke-filled and featureless frames are dropped.

| | fx | fy | cx | cy |
|---|---|---|---|---|
| **Direct registration, 54 frames (median)** | **695.4** | **698.1** | **334.1** | **266.4** |
| Square-pixel fit (used) | 697.3 | 697.3 | | |
| Frame-to-frame spread (MAD) | 5.7 | 5.2 | 2.0 | 1.2 |
| DJI 61° DFOV | 696.0 | 696.0 | | |
| 9.1 mm / 12 µm | 758.3 | 758.3 | | |
| Authors' "Corrected FOV" crop | 675.8 | 699.2 | 336.1 | 263.5 |

**Findings:**
- **Pixels are square** (fx/fy = 0.997), and **fx agrees with DJI's 61° DFOV**. The 758 px value is ruled out.
- The result is stable across peak thresholds (0.015–0.03), Fire vs No Fire (fx 696.5 vs 695.4), and pitch bins (694–697).
- **The principal point is not the image centre.** (cx, cy) = (334.1, 266.4) is the thermal pixel along the wide camera's axis, which is the gimbal axis. The thermal camera points ~1.1° left and ~0.8° up of the wide camera. Near the horizon this matters: for frame 00001 the image centre lands at 504 m, the boresight at 443 m.
- **The authors' crop is one fixed rectangle**: x = 620, y = 454, 2628×2032, identical in all 662 measurable frames (spread < 0.5 px). Its aspect ratio (1.293) is ~3% wider than the thermal frame's (1.25). The direct registration correlates better than the crop in 54/54 frames. **Don't use the Corrected FOV images for geometry.** They are fine as the RGB input to the classifier.
- **Uncertainty:**
  - ±1% systematic from fx_w. DJI's 84° DFOV gives 2777 px and the 24 mm equivalent gives 2773 px, but 4.4 mm / 1.6 µm gives 2750 px. That is ±7 px on the thermal fx.
  - Statistical uncertainty is ~1 px.
  - No distortion model (no checkerboard).
- **Config:** `camera_configs/dji_m30t_thermal.json`, with fx = fy = 697.3 and (cx, cy) = (334.1, 266.4).
- Pose is read from the paired raw RGB: `--exif_image_dir <class>/RGB/Raw`.

---

## 5. Geolocation: where the recipe lives

The original recipe (box → pixel → ray → gimbal rotation → flat-ground intersection → lat/lon) is implemented in `geoloc/projection.py`. It uses `normalized × original_wh` and gimbal (not flight) angles, applied as R = Rz(yaw)·Ry(pitch)·Rx(roll). Beyond the recipe, it also:
- **Handles the horizon:** rays above the horizon, or less than `--min_depression_deg` (default 10°) below it, are unreliable.
- **Clips instead of dropping:** a region is clipped to its reliable part (a disk of radius h / tan 10° on flat ground) rather than discarded.
- **Computes on the ground:** the centroid and area are computed in ground metres.

**Altitude sources** (`geolocate.py --altitude_source`):
- `metadata` (default): `RelativeAltitude`, height above take-off. This is the **honest baseline**; it needs no rangefinder.
- `laser`: `AbsoluteAltitude − LRFTargetAbsAlt`, the height above the laser's ground point (image centre). Both values come from the same altimeter, so the ellipsoid/geoid datum cancels.
  - It falls back to `RelativeAltitude` when there's no reading, or when the laser ground point lies more than `--laser_max_terrain_offset_m` (default 50 m) from take-off level.
  - The fallback reason is written to each frame's `altitude_note`, and the summary counts frames per `altitude_reference`.
  - It's implemented as `LaserHeightLoader` in `geoloc/metadata.py`.
  - **This is a dataset convenience.** The final drone may have no rangefinder, so elevation from a terrain model (`Terrain` interface; USGS 3DEP) stays on the list as the general fix. A laser height is also only correct near the image centre: the ground is still assumed flat across the frame.

**Sanity values, frame `00001`** (h = 55.18 m, yaw 27.6°, pitch −7.1°):

| | Original handoff (fx 758.3, image centre as boresight) | Measured config (fx 697.3, pp 334.1/266.4) |
|---|---|---|
| Boresight | 443 m at 27.6° | 443 m at 27.6° (unit test in `tests/test_camera.py`) |
| Image centre (320, 256) | 443 m | 504 m at 26.4° |
| Bottom-centre (320, 512) | 114 m | 111 m |
| Horizon row | 162 | 180 |
| Ground error per 1° pitch error at boresight | ~60 m | 63 m |

The default 10° depression limit makes almost all of frame `00001` unreliable (its boresight is only 7.1° below the horizon).

---

## 6. Results so far: boresight vs laser rangefinder

`experiments/flame3/laser_check.py` projects the boresight pixel (cx, cy) with the same loaders `geolocate.py` uses, and compares it with the laser's ground point (`LRFTargetLat/Lon`). It covers the 619 frames whose laser height passed the guard, and both columns use the same frames. The boresight doesn't depend on fx, so this checks **altitude and angles, not the focal length**.

| Depression | Frames | Median range | Take-off altitude: median error | p90 | % of range | Laser height: median error | p90 |
|---|---|---|---|---|---|---|---|
| 0–10° | 17 | 378 m | 97.4 m | 186.4 m | 28.0% | 1.3 m | 2.2 m |
| 10–15° | 311 | 240 m | 39.1 m | 102.0 m | 16.9% | 1.1 m | 1.9 m |
| 15–20° | 189 | 197 m | 35.0 m | 72.9 m | 17.4% | 0.4 m | 1.5 m |
| 20–30° | 56 | 170 m | 24.9 m | 46.6 m | 18.6% | 0.9 m | 1.6 m |
| 30–90° | 46 | 85 m | 3.0 m | 24.0 m | 3.6% | 0.8 m | 1.6 m |

**How to read it:**
- **The take-off-altitude column is the honest baseline.** It's the error a drone without a rangefinder would get on this terrain, and it's dominated by the ground being ~10 m above take-off on median.
- **The laser-height column is not an accuracy number.** DJI computes the laser point from the same gimbal angles and the laser distance, and the height is taken from that same measurement. ~1 m agreement only shows that our angle conventions (yaw sense, pitch sign, rotation order) match DJI's.
- **Absolute pointing is still untested.** A yaw bias or a true-vs-magnetic north error would cancel here. Surveyed GCPs (NADIR plots) or a georeferenced basemap are needed for that.

---

## 7. Test plan

1. ~~**Metadata audit**~~ (done, §3).
2. ~~**Pairing check**~~ (done, §2).
3. ~~**Camera model + altitude source**~~ (done, §4–6).
4. **Ground-truth fire masks** from the Celsius TIFFs (>200 °C fire, <80 °C not fire, in between ignored). These score `localize.py` independently of the classifier.
5. **Classifier on FLAME 3.** Compare `infer.py` with the folder labels. The thermal JPGs use a palette colour map that FLAME-2 training never saw, so first **try feeding normalized TIFF temperatures as grayscale**. Only conclude there's a palette/domain accuracy drop if both inputs fail. RGB input: the Corrected FOV images.
6. **Full run, per class, both altitude sources:** `infer.py` → `localize.py` → `geolocate.py --exif_image_dir <class>/RGB/Raw --camera_config camera_configs/dji_m30t_thermal.json`, once with `--altitude_source metadata` and once with `--altitude_source laser`. Report:
   - classifier accuracy
   - localization vs TIFF masks
   - region status counts (ok / clipped / unreliable)
   - frames per altitude reference
   - centre error vs laser with both sources (§6), labelled as above
   - consistency of fire centroids across consecutive frames
   - a map of fire centroids
7. **Terrain model** (USGS 3DEP or SRTM) behind the `Terrain` interface. This is the general replacement for the laser, and matters at Sycan, where a hillside borders the meadow.
8. **North reference check:** confirm DJI yaw is true north. Use a georeferenced feature (road, burn-unit edge) on a satellite basemap, or the GCPs.
9. **Accuracy scoring:** detect GCP markers in the NADIR plots, geolocate them, and compute error against the surveyed coordinates. It's a different camera (XT709: thermal 640×512, 13 mm, 12 µm), straight down only.
10. **Uncertainty ring:** Monte Carlo over ±pitch, ±yaw, ±fx (±1%), GPS and altitude, drawn on the overlay.

---

## 8. Known pitfalls

- Thermal JPGs and TIFFs have **no pose**. Always read it from the paired raw RGB.
- Ignore thermal `ExifImageWidth/Height` (1280×1024); the images are 640×512.
- `RelativeAltitude` is relative to the **take-off point**. At Sycan the ground under the target is a median ~10 m higher, which puts ranges 17–19% too long at 10–30° depression.
- The laser sometimes reports `Normal` for hits on smoke a few metres away. The ±50 m terrain-offset guard catches these.
- Most frames are **oblique**: 608 of 738 are shallower than −20°, and 16 point above the horizon. Expect heavy clipping at the 10° depression limit.
- The **image centre is not the boresight** for the thermal camera. Use the config's principal point.
- The Corrected FOV RGB crop is ~3% too wide horizontally. Don't use it for geometry.
- The raw RGB has a **DJI text stamp burned into the top-left** (model, time, coordinates). Don't run RGB detection on that corner.
- Labels are image-level. "Fire" images still contain mostly non-fire pixels.
- On Windows, the scripts' `✓` output fails on a cp1252 console. Run with `PYTHONUTF8=1`. This predates FLAME 3 work.

---

## 9. Related context

- Geolocation is adapted from Afif's repo https://github.com/afif1230/citrus-uav-pipeline (permission given, MIT). Its Mini 4 Pro / nadir assumptions have been replaced (see README "Attribution").
- The back-projection math (`geoloc/`) is kept separate from rendering/overlay code.
- The velocity overlay (later) needs per-frame timestamps. `DateTimeOriginal` has 1 s resolution here. Check whether the CV subset frames are sequential enough to be useful for velocity.
