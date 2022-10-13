# -- encoding: utf-8 --

# ===================================

# IT - LongTermBiosignals

# Package: research_journal/october/08_10_22 
# Module: ECGResNet
# Description: 

# Contributors: João Saraiva
# Created: 08/10/2022

# ===================================
from torch import nn


class ECGResNet(nn.Module):

    class PadConv1d(nn.Conv1d):
        """
        Conv1d with input padding
        """
        def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
            super().__init__(in_channels, out_channels, kernel_size, **kwargs)

        def forward(self, x):
            net = x
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape before Pad =', net.shape)
            net = nn.functional.pad(net, ECGResNet._get_padding(net, self.stride[0], self.kernel_size[0]), "constant", 0)  # Compute padding
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape after Pad =', net.shape)
            net = super().forward(net)  # Compute convolution
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape after Conv =', net.shape)
            return net

    class PadMaxPool1d(nn.MaxPool1d):
        """
        MaxPool1d with input padding
        """
        def __init__(self, kernel_size, **kwargs):
            super().__init__(kernel_size, **kwargs)

        def forward(self, x):
            net = x
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape before Pad =', net.shape)
            net = nn.functional.pad(net, ECGResNet._get_padding(net, 1, self.kernel_size), "constant", 0)  # Compute padding
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape after Pad =', net.shape)
            net = super().forward(net)  # Compute max pooling
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shape after MaxPool =', net.shape)
            return net

    class InputBlock(nn.Sequential):
        """
        First 3 layers of the network
        """
        def __init__(self, in_features: int, out_features: int, kernel: int, stride: int, groups: int):
            super().__init__()
            # Layers
            self.append(ECGResNet.PadConv1d(in_features, out_features, kernel, stride=stride, groups=groups))
            self.append(nn.BatchNorm1d(out_features))
            self.append(nn.ReLU())

    class ResidualBlock(nn.Module):
        """
        The unit block of hidden layers of the network
        """
        def __init__(self, in_features: int, out_features: int, kernel: int, stride: int, groups: int, downsample: bool = False, _first: bool = False):
            super().__init__()

            # Save to compute paddings
            self.__downsample = downsample
            self.__kernel, self.__is_first = kernel, _first
            self.__stride = stride if downsample else 1
            self.__out_features, self.__in_features = out_features, in_features

            self.__layers = nn.ModuleList()

            # Part 1
            if not _first: # These first 3 layers only occur from the 2nd block onwards.
                self.__layers.append(nn.BatchNorm1d(in_features)),
                self.__layers.append(nn.ReLU())
                self.__layers.append(nn.Dropout(p=0.5))
            self.__layers.append(ECGResNet.PadConv1d(in_features, out_features, kernel, stride=self.__stride, groups=groups))
            # Part 2
            self.__layers.append(nn.BatchNorm1d(out_features))
            self.__layers.append(nn.ReLU())
            self.__layers.append(nn.Dropout(p=0.5))
            self.__layers.append(ECGResNet.PadConv1d(out_features, out_features, kernel, stride=1, groups=groups))

            # Final Part -- Pooling (or downsamling) for the identity only
            self.__downsample = ECGResNet.PadMaxPool1d(kernel_size=self.__stride) if downsample else None

        def forward(self, x):
            identity = x
            out = x

            # Apply all layers on out
            for layer in self.__layers:
                out = layer(out)

            # Maxpool identity if it is to downsample (Fist conv already downsampled out with its stride)
            if self.__downsample is not None:
                identity = self.__downsample(identity)

            # If increasing features, also pad zeros to identity
            if self.__out_features != self.__in_features:
                identity = identity.transpose(-1, -2)
                ch1 = (self.__out_features - self.__in_features) // 2
                ch2 = self.__out_features - self.__in_features - ch1
                identity = nn.functional.pad(identity, (ch1, ch2), "constant", 0)
                identity = identity.transpose(-1, -2)

            # Compute residue
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Shapes of output + identity =', out.shape, identity.shape)
            out += identity
            return out

    class ClassificationBlock(nn.Module):
        """
        Last 3 layers of the network
        """
        def __init__(self, in_features: int, n_classes: int):
            super().__init__()
            # Layers
            self.bn = nn.BatchNorm1d(in_features)
            self.relu = nn.ReLU(inplace=True)
            self.linear = nn.Linear(in_features, n_classes)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.bn(x)
            x = self.relu(x)
            x = x.mean(-1)
            x = self.linear(x)
            x = self.sigmoid(x)
            return x

    def __init__(self, n_classes: int, n_residual_blocks: int, initial_filters: int, kernel: int, stride: int, groups: int):
        super().__init__()
        ECGResNet.DEBUG = False

        # Assert minima
        assert n_residual_blocks >= 1
        assert n_classes >= 1

        # Input Block
        self.__input_block = ECGResNet.InputBlock(1, initial_filters, kernel, stride=1, groups=1)

        # Constants to increase filters and downsample
        upfilter_gap = 12
        downsample_gap = 2

        # Residual Blocks
        self.__residual_blocks = nn.Sequential()
        for i in range(n_residual_blocks):
            if i == 0:  # First Residual block (no filter increase)
                in_features = initial_filters
                out_features = in_features
                is_first = True
                downsample = False
            else:  # Increase filters according to the gap
                in_features = int(initial_filters * 2 ** ((i - 1) // upfilter_gap))
                out_features = in_features * 2 if (i % upfilter_gap == 0) else in_features
                is_first = False
                downsample = i % downsample_gap == 1
            self.__residual_blocks.append(ECGResNet.ResidualBlock(in_features, out_features, kernel, stride, groups, downsample, is_first))

        # Classification Block
        self.__classification_block = ECGResNet.ClassificationBlock(out_features, n_classes)

    def forward(self, x):
        out = x
        if ECGResNet.DEBUG:
            print('ECGResNet | Network Input Shape =', out.shape)

        out = self.__input_block(out)
        if ECGResNet.DEBUG:
            print('ECGResNet | Output Shape of Input Block =', out.shape)

        for i, block in enumerate(self.__residual_blocks):
            out = block(out)
            if ECGResNet.DEBUG:
                print(f'ECGResNet | Output Shape of Residue {i} =', out.shape)

        out = self.__classification_block(out)
        if ECGResNet.DEBUG:
            print(f'ECGResNet | Network Output Shape =', out.shape)

        return out

    @staticmethod
    def _get_padding(x, stride, kernel_size):
        in_dim = x.shape[-1]
        out_dim = (in_dim + stride - 1) // stride
        p = max(0, (out_dim - 1) * stride + kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        return pad_left, pad_right


    def debug_on(self):
        ECGResNet.DEBUG = True

    def debug_off(self):
        ECGResNet.DEBUG = False

