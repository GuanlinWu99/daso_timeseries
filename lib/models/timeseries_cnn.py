import torch.nn as nn
from yacs.config import CfgNode


class ResidualBlock1D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, downsample=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)


class TimeSeries1DCNN(nn.Module):

    def __init__(self, input_features=35, num_classes=7, width=2):
        super().__init__()

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        filters = [64, 64 * width, 128 * width, 256 * width]
        self.layer1 = self._make_layer(filters[0], filters[1], 2)
        self.layer2 = self._make_layer(filters[1], filters[2], 2, stride=2)
        self.layer3 = self._make_layer(filters[2], filters[3], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.out_features = filters[3]
        self.num_classes = num_classes
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

        layers = [ResidualBlock1D(in_ch, out_ch, stride=stride, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        return x.squeeze(-1)


def build_timeseries_cnn(cfg: CfgNode):
    return TimeSeries1DCNN(
        input_features=cfg.MODEL.INPUT_FEATURES,
        num_classes=cfg.MODEL.NUM_CLASSES,
        width=cfg.MODEL.WIDTH
    )
