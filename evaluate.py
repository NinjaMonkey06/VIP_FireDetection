import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import FireDataset
from model import MidFusionNet

device = "cuda" if torch.cuda.is_available() else "cpu"

test_ds = FireDataset("dataset_split/test")
loader = DataLoader(test_ds, batch_size=16)

model = MidFusionNet().to(device)
model.load_state_dict(torch.load("fire_classifier.pth"))
model.eval()

y_true = []
y_pred = []

with torch.no_grad():

    for rgb, th, labels in loader:

        rgb, th = rgb.to(device), th.to(device)

        outputs = model(rgb, th)

        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

plt.show()