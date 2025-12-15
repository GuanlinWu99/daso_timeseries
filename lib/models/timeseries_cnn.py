import torch.nn as nn
from yacs.config import CfgNode


class ResidualBlock1D(nn.Module):
    """1D Residual Block for time series data"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class TimeSeries1DCNN(nn.Module):
    """
    1D CNN for time series classification
    Designed for ALFA dataset with shape (batch, seq_len=25, features=35)
    """

    def __init__(self, input_features: int = 35, num_classes: int = 7, width: int = 2):
        super(TimeSeries1DCNN, self).__init__()

        # Input shape: (batch, seq_len=25, features=35)
        # Need to transpose to (batch, features=35, seq_len=25) for Conv1d

        # Initial convolution
        self.init_conv = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Residual blocks with increasing channels
        filters = [64, 64 * width, 128 * width, 256 * width]

        # Block 1
        self.layer1 = self._make_layer(filters[0], filters[1], num_blocks=2)

        # Block 2 with downsampling
        self.layer2 = self._make_layer(filters[1], filters[2], num_blocks=2, stride=2)

        # Block 3 with downsampling
        self.layer3 = self._make_layer(filters[2], filters[3], num_blocks=2, stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # Output features for classifier
        self.out_features = filters[3]
        self.num_classes = num_classes

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(ResidualBlock1D(in_channels, out_channels, stride=stride, downsample=downsample))

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
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
        # Input: (batch, seq_len=25, features=35)
        # Transpose to (batch, features=35, seq_len=25)
        x = x.transpose(1, 2)

        x = self.init_conv(x)      # (batch, 64, seq_len/2)
        x = self.layer1(x)          # (batch, 64*width, seq_len/2)
        x = self.layer2(x)          # (batch, 128*width, seq_len/4)
        x = self.layer3(x)          # (batch, 256*width, seq_len/8)
        x = self.avgpool(x)         # (batch, 256*width, 1)

        return x.squeeze(-1)        # (batch, 256*width)


def build_timeseries_cnn(cfg: CfgNode) -> nn.Module:
    """
    Build 1D CNN for time series data
    """
    # fmt: off
    width = cfg.MODEL.WIDTH
    num_classes = cfg.MODEL.NUM_CLASSES
    input_features = cfg.MODEL.INPUT_FEATURES
    # fmt: on
    return TimeSeries1DCNN(input_features=input_features, num_classes=num_classes, width=width)
