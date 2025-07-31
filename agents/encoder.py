import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, resnet18, resnet34


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        downsample=False,
    ):
        super().__init__()
        self.downsample = downsample
        stride1 = 2 if downsample else stride
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride1,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out)
        return out


class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        # ((H - f + 2p) / s )+ 1
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 2, 3),  # 224×224×3->112,112,64
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, 2, 1),  # 112,112,64->56,56,64
            BasicBlock(64, 64),
            BasicBlock(64, 128, downsample=True),
            BasicBlock(128, 128),
            BasicBlock(128, 256, downsample=True),
            BasicBlock(256, 256),
            BasicBlock(256, 512, downsample=True),
            BasicBlock(512, 512),
            nn.AvgPool2d(7),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.layers(x)


class ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet1 = Resnet()
        self.resnet2 = Resnet()
        self.resnet3 = Resnet()

    # x (camera,batch,h,w,channle)
    def forward(self, x):
        # (batch,h,w,channle) -> (batch,channle,h,w)
        x1 = x[0, :, :, :, :3].permute(0, 3, 1, 2)
        x2 = x[1, :, :, :, :3].permute(0, 3, 1, 2)
        x3 = x[2, :, :, :, :3].permute(0, 3, 1, 2)
        x1 = self.resnet1(x1)
        x2 = self.resnet2(x2)
        x3 = self.resnet3(x3)
        return torch.cat([x1, x2, x3], dim=1)


class pre_trained_ResnetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for param in resnet1.parameters():
            param.requires_grad = False

        self.resnet1 = nn.Sequential(*list(resnet1.children())[:-1])

    def forward(self, x):
        x1 = x[0, :, :, :, :3].permute(0, 3, 1, 2)
        x2 = x[1, :, :, :, :3].permute(0, 3, 1, 2)
        x3 = x[2, :, :, :, :3].permute(0, 3, 1, 2)
        x1 = self.resnet1(x1)
        x2 = self.resnet1(x2)
        x3 = self.resnet1(x3)
        return torch.cat([x1, x2, x3], dim=1).squeeze()


"""
dummy = torch.rand(3, 10, 224, 224, 4)
model = pre_trained_ResnetEncoder()
print(model(dummy).shape)
"""
