import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

DATA_DIR = "dataset_processed"
OUT_DIR = "dataset_split"

os.makedirs(OUT_DIR, exist_ok=True)

labels = pd.read_csv(f"{DATA_DIR}/labels.csv")

label_map = {"NN":0, "YN":1, "YY":2}
labels["class"] = labels["class"].map(label_map)

train, temp = train_test_split(
    labels,
    test_size=0.30,
    stratify=labels["class"],
    random_state=42
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["class"],
    random_state=42
)

splits = {"train":train, "val":val, "test":test}

for split_name, df in splits.items():

    rgb_out = f"{OUT_DIR}/{split_name}/rgb"
    th_out = f"{OUT_DIR}/{split_name}/thermal"

    os.makedirs(rgb_out, exist_ok=True)
    os.makedirs(th_out, exist_ok=True)

    for img in df["image"]:

        shutil.copy(
            f"{DATA_DIR}/rgb/{img}",
            f"{rgb_out}/{img}"
        )

        shutil.copy(
            f"{DATA_DIR}/thermal/{img}",
            f"{th_out}/{img}"
        )

    df.to_csv(f"{OUT_DIR}/{split_name}/labels.csv", index=False)

print("Dataset split complete.")