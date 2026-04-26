import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms

class FireDataset(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))

        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.th_dir = os.path.join(root_dir, "thermal")

        # transforms
        self.rgb_transform = transforms.ToTensor()
        self.th_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        img_name = row["image"]
        label = row["class"]

        rgb_path = os.path.join(self.rgb_dir, img_name)
        th_path = os.path.join(self.th_dir, img_name)

        rgb = Image.open(rgb_path).convert("RGB")
        th = Image.open(th_path).convert("L")

        rgb = self.rgb_transform(rgb)
        th = self.th_transform(th)

        return rgb, th, torch.tensor(label)
