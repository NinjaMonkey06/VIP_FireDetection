# Geolocation worked example: one pixel, by hand

This walks one pixel through `geoloc/projection.py` with real numbers, for an
oblique (tilted) camera. Every value below is checked by
`tests/test_worked_example.py`, so if the code changes, this document's numbers
are re-verified.

## Sign conventions

| Quantity | Zero means | Positive means | Notes |
|---|---|---|---|
| **Yaw** | the top of the image points **north** | rotation **clockwise seen from above** (north → east) | yaw 90° = image top points east; absolute (relative to north), not relative to the drone's nose |
| **Pitch** | optical axis **horizontal** | camera tilted **up** towards the sky | negative = tilted down; −60° = optical axis 60° below the horizon; **−90° = straight down (nadir)** |
| **Roll** | image horizon level | **right side of the image down** | rotation about the camera's forward axis |
| **Pixel (u, v)** | top-left corner | u to the right, v **down** | OpenCV convention |
| **Ground offsets** | the point directly below the drone | north, east | in metres |

At pitch −90° the camera looks straight down and the top of the image points in
the yaw direction, so at yaw 0 a pixel above the image center is north of the drone.

Frames used internally:

- **camera**: x right, y down, z along the optical axis (out of the lens)
- **gimbal**: x forward, y right, z down
- **NED**: x north, y east, z down. A ray reaches the ground only if its down component is positive.

The rotation from gimbal to NED is `R = Rz(yaw) · Ry(pitch) · Rx(roll)`: roll is
applied first, then pitch, then yaw.

## Inputs (made-up camera)

| | Value |
|---|---|
| Image | 640 × 512 px |
| Intrinsics | fx = fy = 640 px, cx = 320, cy = 256, no distortion (horizontal FOV = 2·atan(320/640) = 53.13°) |
| Pixel | (u, v) = (480, 160): 160 px right of and 96 px above the image center |
| Drone | lat 34.0°, lon −118.0°, altitude 100 m above ground |
| Gimbal | yaw 30°, pitch −60°, roll 0° |

## Step 1: pixel → normalized camera ray

```
x = (u − cx) / fx = (480 − 320) / 640 =  0.25
y = (v − cy) / fy = (160 − 256) / 640 = −0.15
ray_camera = (x, y, 1) = (0.25, −0.15, 1)
```

The ray points right (+x) and up (−y) of the optical axis.

## Step 2: camera frame → gimbal frame

The gimbal's (forward, right, down) are the camera's (z, x, y):

```
ray_gimbal = (1, 0.25, −0.15)
```

## Step 3: roll (0°)

`Rx(0)` is the identity, so the ray is unchanged: `(1, 0.25, −0.15)`.

## Step 4: pitch −60°

```
Ry(θ) = [[ cos θ, 0, sin θ],      cos(−60°) =  0.5
         [   0,   1,   0  ],      sin(−60°) = −0.8660254
         [−sin θ, 0, cos θ]]

forward = 0.5·1        + (−0.8660254)·(−0.15) = 0.5 + 0.1299038 = 0.6299038
right   = 0.25
down    = 0.8660254·1  + 0.5·(−0.15)          = 0.8660254 − 0.075 = 0.7910254
```

Check: the optical axis (forward = 1) maps to (0.5, 0, 0.866), which is 60°
below the horizon, as pitch −60° should give.

## Step 5: yaw 30° → NED

```
Rz(ψ) = [[cos ψ, −sin ψ, 0],      cos 30° = 0.8660254
         [sin ψ,  cos ψ, 0],      sin 30° = 0.5
         [  0,      0,   1]]

north = 0.8660254·0.6299038 − 0.5·0.25       = 0.5455127 − 0.125     = 0.4205127
east  = 0.5·0.6299038       + 0.8660254·0.25 = 0.3149519 + 0.2165064 = 0.5314583
down  = 0.7910254
```

## Step 6: intersect flat ground 100 m below

```
t     = altitude / down = 100 / 0.7910254 = 126.4182
north = 0.4205127 · 126.4182 = 53.160 m
east  = 0.5314583 · 126.4182 = 67.186 m
```

Ground range from the point below the drone: √(53.160² + 67.186²) = **85.674 m**.
Slant distance from the camera: t · |ray| = **131.68 m**.

## Step 7: reliability check

Depression below the horizon = atan2(down, √(north² + east²)) =
atan2(0.7910254, 0.6777011) = **49.41°**. That is above the default 10° minimum,
and the range is finite, so the vertex is marked `ok`. A 1° attitude error here
moves the point by about 100 · (π/180) / sin²(49.41°) ≈ 3.0 m.

## Step 8: metres → lat/lon

Spherical Earth, R = 6 378 137 m:

```
Δlat = degrees(north / R)                = degrees(53.160 / 6378137)            = 0.00047755°
Δlon = degrees(east / (R · cos(lat0)))   = degrees(67.186 / (6378137 · 0.829038)) = 0.00072800°

lat = 34.0      + 0.00047755 = 34.00047755
lon = −118.0    + 0.00072800 = −117.99927200
```

## Sanity checks

- **Image center**: the optical axis at pitch −60° lands 100 / tan(60°) = 57.735 m
  away, along the yaw bearing of 30°: north = 57.735·cos 30° = **50.000 m**,
  east = 57.735·sin 30° = **28.868 m**.
- **Round trip**: `ground_to_pixels` on (53.160 N, 67.186 E) returns pixel (480, 160).
- **Nadir special case**: at pitch −90° and roll 0, Steps 2–6 reduce to
  right = x·h and forward = −y·h (forward = the image-top direction), then
  north = forward·cos ψ − right·sin ψ and east = forward·sin ψ + right·cos ψ.
  For this pixel at nadir: right = 25 m, forward = 15 m, so north = 0.490 m and
  east = 29.151 m. At yaw 0 this is the reference citrus-pipeline formula.
