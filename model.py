import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ArtNet(nn.Module):
    def __init__(self, num_classes=10, dropout_prob=0.5):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.dropout = nn.Dropout(dropout_prob)
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(64 * 32 * 32, 256)
        # self.dropout = nn.Dropout(dropout_prob)
        # self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        return x
        # x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # x = x.view(-1, 64 * 32 * 32)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.fc2(x)
        # return x
