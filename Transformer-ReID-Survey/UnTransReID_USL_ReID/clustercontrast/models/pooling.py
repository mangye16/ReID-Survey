# Credit to https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/layers/pooling.py
from abc import ABC

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "GeneralizedMeanPoolingPFpn",
    "GeneralizedMeanPoolingList",
    "GeneralizedMeanPoolingP",
    "AdaptiveAvgMaxPool2d",
    "FastGlobalAvgPool2d",
    "avg_pooling",
    "max_pooling",
]


class GeneralizedMeanPoolingList(nn.Module, ABC):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of
    several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size
                     will be the same as that of the input.
    """

    def __init__(self, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingList, self).__init__()
        self.output_size = output_size
        self.eps = eps

    def forward(self, x_list):
        outs = []
        for x in x_list:
            x = x.clamp(min=self.eps)
            out = torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)
            outs.append(out)
        return torch.stack(outs, -1).mean(-1)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPooling(nn.Module, ABC):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of
    several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size
                     will be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
            1.0 / self.p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPoolingP(GeneralizedMeanPooling, ABC):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class GeneralizedMeanPoolingFpn(nn.Module, ABC):
    r"""Applies a 2D power-average adaptive pooling over an input signal composed of
    several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size
                     will be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingFpn, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x_lists):
        outs = []
        for x in x_lists:
            x = x.clamp(min=self.eps).pow(self.p)
            out = torch.nn.functional.adaptive_avg_pool2d(x, self.output_size).pow(
                1.0 / self.p
            )
            outs.append(out)
        return torch.cat(outs, 1)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + str(self.p)
            + ", "
            + "output_size="
            + str(self.output_size)
            + ")"
        )


class GeneralizedMeanPoolingPFpn(GeneralizedMeanPoolingFpn, ABC):
    """ Same, but norm is trainable
    """

    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingPFpn, self).__init__(norm, output_size, eps)
        self.p = nn.Parameter(torch.ones(1) * norm)


class AdaptiveAvgMaxPool2d(nn.Module, ABC):
    def __init__(self):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.avgpool = FastGlobalAvgPool2d()

    def forward(self, x):
        x_avg = self.avgpool(x, self.output_size)
        x_max = F.adaptive_max_pool2d(x, 1)
        x = x_max + x_avg
        return x


class FastGlobalAvgPool2d(nn.Module, ABC):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        else:
            return (
                x.view(x.size(0), x.size(1), -1)
                .mean(-1)
                .view(x.size(0), x.size(1), 1, 1)
            )


def avg_pooling():
    return nn.AdaptiveAvgPool2d(1)
    # return FastGlobalAvgPool2d()


def max_pooling():
    return nn.AdaptiveMaxPool2d(1)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


__pooling_factory = {
    "avg": avg_pooling,
    "max": max_pooling,
    "gem": GeneralizedMeanPoolingP,
    "gemFpn": GeneralizedMeanPoolingPFpn,
    "gemList": GeneralizedMeanPoolingList,
    "avg+max": AdaptiveAvgMaxPool2d,
}


def pooling_names():
    return sorted(__pooling_factory.keys())


def build_pooling_layer(name):
    """
    Create a pooling layer.
    Parameters
    ----------
    name : str
        The backbone name.
    """
    if name not in __pooling_factory:
        raise KeyError("Unknown pooling layer:", name)
    return __pooling_factory[name]()