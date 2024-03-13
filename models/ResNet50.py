import torch.nn as nn
import torchvision.models as models


class ResNet50Custom(nn.Module):
    def __init__(self, config):
        super(ResNet50Custom, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.conv1 = nn.Conv2d(config.num_channels, 64, kernel_size=(config.w, config.h), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        self.layer1 = self.resnet50.layer1
        self.layer2 = self.resnet50.layer2
        self.layer3 = self.resnet50.layer3
        self.layer4 = self.resnet50.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, config.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
