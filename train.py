import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import FireDataset
from model import MidFusionNet

device = "cuda" if torch.cuda.is_available() else "cpu"

train_ds = FireDataset("dataset_split/train")
val_ds = FireDataset("dataset_split/val")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)

model = MidFusionNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 15

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for rgb, th, labels in train_loader:

        rgb, th, labels = rgb.to(device), th.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(rgb, th)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.3f}")

torch.save(model.state_dict(), "fire_classifier.pth")