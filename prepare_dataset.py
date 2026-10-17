import os
import shutil
import math
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------

RGB_DIR = "dataset_raw/rgb"
THERMAL_DIR = "dataset_raw/thermal"

OUT_RGB = "dataset_processed/rgb"
OUT_THERMAL = "dataset_processed/thermal"

OUTPUT_SIZE = 256

TOTAL_TARGET = 5000

# Sampling targets per class
TARGET_NN = 1700
TARGET_YN = 1650
TARGET_YY = 1650


# Label intervals
intervals = [
    (1,13700,"NN"),
    (13701,14699,"YY"),
    (14700,15980,"YN"),
    (15981,19802,"YY"),
    (19803,19899,"YN"),
    (19900,27183,"YY"),
    (27184,27514,"YN"),
    (27515,31294,"YY"),
    (31295,31509,"YN"),
    (31510,33597,"YY"),
    (33598,33929,"YN"),
    (33930,36550,"YY"),
    (36551,38030,"YN"),
    (38031,38153,"YY"),
    (38154,41642,"YN"),
    (41642,45279,"YY"),
    (45280,51206,"YN"),
    (51207,52286,"YY"),
    (52287,53451,"YN")
]


# ----------------------------
# CREATE OUTPUT FOLDERS
# ----------------------------

os.makedirs(OUT_RGB, exist_ok=True)
os.makedirs(OUT_THERMAL, exist_ok=True)


# ----------------------------
# BUILD FRAME LIST PER CLASS
# ----------------------------

classes = {
    "NN": [],
    "YN": [],
    "YY": []
}

for start, end, label in intervals:
    for i in range(start, end + 1):
        classes[label].append(i)


# ----------------------------
# REGULAR INTERVAL SAMPLING
# ----------------------------

def sample_regular(frame_list, target):

    step = len(frame_list) / target

    selected = []

    for i in range(target):
        index = int(i * step)
        selected.append(frame_list[index])

    return selected


sampled_frames = []

sampled_frames += [(f,"NN") for f in sample_regular(classes["NN"], TARGET_NN)]
sampled_frames += [(f,"YN") for f in sample_regular(classes["YN"], TARGET_YN)]
sampled_frames += [(f,"YY") for f in sample_regular(classes["YY"], TARGET_YY)]


print("Total sampled:", len(sampled_frames))


# ----------------------------
# PROCESS IMAGES
# ----------------------------

def crop_rgb_center(img):

    w, h = img.size

    target_aspect = 1.25
    new_w = int(h * target_aspect)

    # horizontal center crop
    left = (w - new_w) // 2
    right = left + new_w

    # vertical offset correction
    vertical_offset = 51

    top = vertical_offset
    bottom = h - vertical_offset

    return img.crop((left, top, right, bottom))

counter = 1

labels = []

for frame,label in sampled_frames:

    rgb_name = f"Original Sized RGB Frame ({frame}).jpg"
    th_name = f"Original Sized Thermal Frame ({frame}).jpg"

    rgb_path = os.path.join(RGB_DIR, rgb_name)
    th_path = os.path.join(THERMAL_DIR, th_name)

    if not os.path.exists(rgb_path) or not os.path.exists(th_path):
        continue

    rgb = Image.open(rgb_path)
    thermal = Image.open(th_path)

    # crop RGB to match thermal aspect
    rgb = crop_rgb_center(rgb)

    # resize both
    rgb = rgb.resize((OUTPUT_SIZE, OUTPUT_SIZE))
    thermal = thermal.resize((OUTPUT_SIZE, OUTPUT_SIZE))

    out_name = f"{counter:05d}.png"

    rgb.save(os.path.join(OUT_RGB, out_name))
    thermal.save(os.path.join(OUT_THERMAL, out_name))

    labels.append((out_name,label))

    counter += 1


# ----------------------------
# SAVE LABELS
# ----------------------------

import csv

with open("dataset_processed/labels.csv","w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["image","class"])

    for row in labels:
        writer.writerow(row)


print("Processing complete.")