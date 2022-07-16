import torch
import torch.nn as nn


class BaselineArch(nn.Module):
    def __init__(self, in_channels, channels, stride):
        super().__init__()
        self.out_channels = out_channels = channels
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y += self.shortcut(x)
        y = self.relu(y)
        return y


class BottleneckArch(nn.Module):
    def __init__(self, in_channels, channels, stride):
        super().__init__()
        self.out_channels = out_channels = channels * 4
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, channels, 1, stride=stride, padding=0, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        y = self.conv1(x)
        y = self.norm1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.norm2(y)
        y = self.relu(y)
        y = self.conv3(y)
        y = self.norm3(y)
        y += self.shortcut(x)
        y = self.relu(y)
        return y


class Layer(nn.Module):
    def __init__(self, Block, in_channels, channels, stride, count):
        super().__init__()
        blocks = []

        block = Block(in_channels, channels, stride)
        blocks.append(block)
        in_channels = block.out_channels

        for i in range(1, count):
            block = Block(in_channels, channels, 1)
            blocks.append(block)
            in_channels = block.out_channels

        self.blocks = nn.Sequential(*blocks)
        self.out_channels = in_channels

    def forward(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self, Block, counts, start_channels, classes):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(3, start_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm0 = nn.BatchNorm2d(start_channels)
        self.pool0 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = Layer(Block, start_channels, start_channels, 1, counts[0])
        self.layer2 = Layer(Block, self.layer1.out_channels, start_channels*2, 2, counts[1])
        self.layer3 = Layer(Block, self.layer2.out_channels, start_channels*4, 2, counts[2])
        self.layer4 = Layer(Block, self.layer3.out_channels, start_channels*8, 2, counts[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(self.layer4.out_channels, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu(x)
        x = self.pool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


def resnet18(start_channels, classes):
    return ResNet(BaselineArch, [2, 2, 2, 2], start_channels, classes)


def resnet34(start_channels, classes):
    return ResNet(BaselineArch, [3, 4, 6, 3], start_channels, classes)


def resnet50(start_channels, classes):
    return ResNet(BottleneckArch, [3, 4, 6, 3], start_channels, classes)


def resnet101(start_channels, classes):
    return ResNet(BottleneckArch, [3, 4, 23, 3], start_channels, classes)


def resnet152(start_channels, classes):
    return ResNet(BottleneckArch, [3, 8, 36, 3], start_channels, classes)
