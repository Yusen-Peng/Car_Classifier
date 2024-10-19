import numpy as np
import torch
import torch.nn as nn

class DepthwiseSeparableConv1D(nn.Module):
    """
    A PyTorch module that implements a depthwise separable 1D convolution.

    Depthwise separable convolution reduces the computational cost and model size
    by splitting the convolution operation into two separate layers:
    1. Depthwise convolution: Applies a single convolutional filter per input channel.
    2. Pointwise convolution: Uses a 1x1 convolution to combine the outputs from the depthwise step.

    Parameters:
        in_channels (int): Number of channels in the input signal.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: False
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(DepthwiseSeparableConv1D, self).__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            groups=in_channels,
            bias=bias
        )
        
        self.pointwise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



class HybirdLayer(nn.Module):
    """
    A custom PyTorch module that applies a series of predefined, non-trainable convolutional filters to the input data.
    The filters are designed based on specific mathematical patterns to extract meaningful features from the input.
    
    Parameters:
        input_channels (int): Number of channels in the input data.
        kernel_sizes (list of int): List of kernel sizes to be used for creating filters.
    """
    def __init__(self, input_channels, kernel_sizes=[2, 4, 8, 16, 32, 64]):
        super(HybirdLayer, self).__init__()

        self.convs = nn.ModuleList()
        self.keep_track = 0

        for kernel_size in kernel_sizes:
            filter_ = np.ones((1, input_channels, kernel_size))
            indices_ = np.arange(kernel_size)
            filter_[0, :, indices_ % 2 == 0] *= -1

            conv = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            conv.weight = nn.Parameter(torch.tensor(filter_, dtype=torch.float32))
            conv.weight.requires_grad = False

            self.convs.append(conv)
            self.keep_track += 1

        for kernel_size in kernel_sizes:
            filter_ = np.ones((1, input_channels, kernel_size))
            indices_ = np.arange(kernel_size)
            filter_[0, :, indices_ % 2 > 0] *= -1

            conv = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            )
            conv.weight = nn.Parameter(torch.tensor(filter_, dtype=torch.float32))
            conv.weight.requires_grad = False

            self.convs.append(conv)
            self.keep_track += 1

        for kernel_size in kernel_sizes[1:]:
            filter_size = kernel_size + kernel_size // 2
            filter_ = np.zeros((1, input_channels, filter_size))

            xmash = np.linspace(0, 1, kernel_size // 4 + 1)[1:]

            filter_left = xmash ** 2
            filter_right = filter_left[::-1]

            filter_[0, :, 0 : kernel_size // 4] = -filter_left.reshape(1, -1)
            filter_[0, :, kernel_size // 4 : kernel_size // 2] = -filter_right.reshape(1, -1)
            filter_[0, :, kernel_size // 2 : 3 * kernel_size // 4] = 2 * filter_left.reshape(1, -1)
            filter_[0, :, 3 * kernel_size // 4 : kernel_size] = 2 * filter_right.reshape(1, -1)
            filter_[0, :, kernel_size : 5 * kernel_size // 4] = -filter_left.reshape(1, -1)
            filter_[0, :, 5 * kernel_size // 4 :] = -filter_right.reshape(1, -1)

            conv = nn.Conv1d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=filter_size,
                padding=filter_size // 2,
                bias=False,
            )
            
            conv.weight = nn.Parameter(torch.tensor(filter_, dtype=torch.float32))
            conv.weight.requires_grad = False

            self.convs.append(conv)
            self.keep_track += 1
        self.n_filters = len(self.convs)
        self.activation = nn.ReLU()

    def forward(self, x):
        conv_outputs = []

        for conv in self.convs:
            out = conv(x)
            conv_outputs.append(out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.activation(x)

        return x

class InceptionModule(nn.Module):
    """
    A custom Inception-style module for 1D convolutions in PyTorch.
    This module allows for a combination of standard convolutional layers, optional hybrid layers, 
    and flexibility in using different kernel sizes to enhance feature extraction.

    Parameters:
        input_channels (int): Number of input channels.
        n_filters (int): Number of filters (output channels) per convolutional layer.
        kernel_size (int): Base size of the kernel for the convolutional layers.
        dilation_rate (int, optional): Dilation rate of the convolutions. Default is 1.
        stride (int, optional): Stride size for the convolutions. Default is 1.
        activation (str, optional): Type of activation function to use. Default is "linear".
        use_hybird_layer (bool, optional): If True, includes a hybrid layer with custom filters. Default is False.
        use_multiplexing (bool, optional): If True, applies multiple convolutions with varying kernel sizes. Default is True.
        use_custom_filters (bool, optional): Indicates whether to use custom filters (currently not used in logic). Default is True.
        kernel_sizes (list of int, optional): List of kernel sizes for the hybrid layer. Required if use_hybird_layer is True.
    """
    def __init__(
        self,
        input_channels,
        n_filters,
        kernel_size,
        dilation_rate=1,
        stride=1,
        activation="linear",
        use_hybird_layer=False,
        use_multiplexing=True,
        use_custom_filters=True,
        kernel_sizes=None
    ):
        super(InceptionModule, self).__init__()

        self.use_hybird_layer = use_hybird_layer
        self.use_multiplexing = use_multiplexing
        self.use_custom_filters = use_custom_filters

        if not use_multiplexing:
            n_convs = 1
            n_filters = n_filters * 3
        else:
            n_convs = 3

        kernel_size_s = [kernel_size // (2 ** i) for i in range(n_convs)]

        self.convs = nn.ModuleList()

        for k_size in kernel_size_s:
            conv = nn.Conv1d(
                in_channels=input_channels,
                out_channels=n_filters,
                kernel_size=k_size,
                stride=stride,
                padding=k_size // 2,
                dilation=dilation_rate,
                bias=False,
            )
            self.convs.append(conv)

        if self.use_hybird_layer:
            self.hybird_layer = HybirdLayer(input_channels=input_channels, kernel_sizes=kernel_sizes)
            hybird_filters_total = self.hybird_layer.n_filters
        else:
            self.hybird_layer = None
            hybird_filters_total = 0

        total_filters = n_filters * len(self.convs) + hybird_filters_total
        self.total_filters = total_filters
    
        self.bn = nn.BatchNorm1d(num_features=total_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        conv_outputs = []
        for conv in self.convs:
            out = conv(x)
            conv_outputs.append(out)

        if self.use_hybird_layer:
            hybird_out = self.hybird_layer(x)
            conv_outputs.append(hybird_out)

        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.activation(x)
        return x

class FCNModule(nn.Module):
    """
    A Fully Convolutional Network (FCN) module for 1D data in PyTorch.
    This module applies a 1D convolution followed by batch normalization and a ReLU activation function,
    commonly used for extracting features from sequential data.

    Parameters:
        input_channels (int): Number of channels in the input data.
        kernel_size (int): Size of the convolutional kernel.
        n_filters (int): Number of filters (output channels) in the convolutional layer.
        dilation_rate (int, optional): Dilation rate of the convolution. Default is 1.
        stride (int, optional): Stride of the convolution. Default is 1.
        activation (str, optional): Activation function to be used (currently fixed to ReLU). Default is "relu".
    """
    def __init__(
        self,
        input_channels,
        kernel_size,
        n_filters,
        dilation_rate=1,
        stride=1,
        activation="relu",
    ):
        super(FCNModule, self).__init__()

        self.dwsc = DepthwiseSeparableConv1D(
            in_channels=input_channels,
            out_channels=n_filters,
            kernel_size=kernel_size,
        )
        self.bn = nn.BatchNorm1d(n_filters)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dwsc(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class LITEModel(nn.Module):
    """
    LITEModel: A lightweight neural network architecture for time-series classification.
    Combines Inception modules, FCN (Fully Convolutional Network) layers, and Depthwise Separable Convolution 
    for efficient feature extraction and classification.

    Parameters:
        length_TS (int): Length of the input time-series.
        n_classes (int): Number of output classes for classification.
        n_filters (int, optional): Number of filters for convolutional layers. Default is 32.
        kernel_size (int, optional): Base size of the kernel for the convolutional layers. Default is 41.
        use_custom_filters (bool, optional): If True, uses custom filters in the InceptionModule's hybrid layers. Default is True.
        use_dilation (bool, optional): If True, applies dilated convolutions in FCN modules. Default is True.
        use_multiplexing (bool, optional): If True, applies multiple convolutions with varying kernel sizes in the InceptionModule. Default is True.
    """
    def __init__(
        self,
        length_TS,
        n_classes,
        n_filters=32,
        kernel_size=41,
        use_custom_filters=True,
        use_dilation=True,
        use_multiplexing=True,
    ):
        super(LITEModel, self).__init__()

        self.length_TS = length_TS
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.use_custom_filters = use_custom_filters
        self.use_dilation = use_dilation
        self.use_multiplexing = use_multiplexing
        self.kernel_size = kernel_size - 1
        self.kernel_sizes = [2, 4, 8, 16, 32, 64]

        input_channels = 1

        self.inception_module = InceptionModule(
            input_channels=input_channels,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            dilation_rate=1,
            use_hybird_layer=self.use_custom_filters,
            use_multiplexing=self.use_multiplexing,
            kernel_sizes=self.kernel_sizes
        )

        self.kernel_size //= 2

        # Total output channels from InceptionModule
        total_filters = self.inception_module.total_filters
        input_channels = total_filters

        self.fcn_modules = nn.ModuleList()
        dilation_rate = 1

        for i in range(2):
            if self.use_dilation:
                dilation_rate = 2 ** (i + 1)

            kernel_size = self.kernel_size // (2 ** i)
            n_filters = self.n_filters

            fcn_module = FCNModule(
                input_channels=input_channels,
                kernel_size=kernel_size,
                n_filters=n_filters,
                dilation_rate=dilation_rate,
            )

            self.fcn_modules.append(fcn_module)
            input_channels = n_filters

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(self.n_filters, self.n_classes)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Adds channel dimension if missing
        elif x.dim() == 4:
            x = x.squeeze(1)  # Removes extra dimension if present
            
        x = self.inception_module(x)

        for fcn_module in self.fcn_modules:
            x = fcn_module(x)

        x = self.gap(x).squeeze(-1)  # (batch_size, n_filters)
        x = self.output_layer(x)
        return x
