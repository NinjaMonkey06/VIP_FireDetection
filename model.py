import torch
import torch.nn as nn
import torchvision.models as models


class MidFusionNet(nn.Module):

    def __init__(self, num_classes=3):
        super().__init__()

         # RGB branch
        self.rgb_net = models.resnet18(weights="DEFAULT")
        self.rgb_net.fc = nn.Identity()

        # Thermal branch
        self.th_net = models.resnet18(weights="DEFAULT")

        # modify first conv for 1-channel input
        self.th_net.conv1 = nn.Conv2d(
            1,64,kernel_size=7,stride=2,padding=3,bias=False
        )
        self.th_net.fc = nn.Identity()

        # fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(512*2,256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256,num_classes)
        )

    def forward(self, rgb, thermal):

        rgb_feat = self.rgb_net(rgb)
        th_feat = self.th_net(thermal)

        fused = torch.cat([rgb_feat, th_feat], dim=1)

        out = self.classifier(fused)

        return out
