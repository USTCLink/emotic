import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import vgg19_bn, resnet50


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.body_feature_extraction = vgg19_bn(pretrained=True).features  # 基于ImageNet预训练
        self.fc = nn.Linear(25088, 1000)
        self.image_feature_extraction = resnet50()
        self.image_feature_extraction.fc = nn.Linear(2048, 365)  # 为了适应在预训练模型，作此修改
        self.fusion = nn.Sequential(
            nn.Linear(1365, 1000),
            nn.ReLU(),
            nn.Linear(1000, 26),
            nn.Sigmoid()
        )

    def forward(self, body, image):
        body_feature = self.body_feature_extraction(body)
        body_feature = body_feature.view(body_feature.size(0), -1)
        body_feature = self.fc(body_feature)
        image_feature = self.image_feature_extraction(image)
        feature = torch.cat((body_feature, image_feature), dim=1)
        return self.fusion(feature)
