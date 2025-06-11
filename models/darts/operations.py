from collections.abc import Callable
import torch
import torch.nn as nn

OPS: dict[str, Callable[[int, int, bool], nn.Module]] = {
    "none":
    lambda C, stride, affine: Zero(stride),
    "avg_pool_3x3":
    lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    "max_pool_3x3":
    lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    "skip_connect":
    lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    "sep_conv_3x3":
    lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    "sep_conv_5x5":
    lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    "sep_conv_7x7":
    lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    "dil_conv_3x3":
    lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    "dil_conv_5x5":
    lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    "conv_7x1_1x7":
    lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine),
    ),  # this one is not used in original darts paper
}


class ReLUConvBN(nn.Module):

  def __init__(self,
               C_in: int,
               C_out: int,
               kernel_size: int,
               stride: int,
               padding: int,
               affine: bool = True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.op(x)


class DilConv(nn.Module):

  def __init__(self,
               C_in: int,
               C_out: int,
               kernel_size: int,
               stride: int,
               padding: int,
               dilation: int,
               affine: bool = True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(
            C_in,
            C_in,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=C_in,
            bias=False,
        ),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.op(x)


class SepConv(nn.Module):

  def __init__(self,
               C_in: int,
               C_out: int,
               kernel_size: int,
               stride: int,
               padding: int,
               affine: bool = True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in,
                  C_in,
                  kernel_size=kernel_size,
                  stride=stride,
                  padding=padding,
                  groups=C_in,
                  bias=False),
        nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_in, affine=affine),
        nn.ReLU(inplace=False),
        nn.Conv2d(C_in,
                  C_in,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=padding,
                  groups=C_in,
                  bias=False),
        nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
        nn.BatchNorm2d(C_out, affine=affine),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x


class Zero(nn.Module):

  def __init__(self, stride: int):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.stride == 1:
      return x.mul(0.0)
    return x[:, :, ::self.stride, ::self.stride].mul(0.0)


class FactorizedReduce(nn.Module):
  """
    This module will half the image resolution using a 1x1 Conv with stride=2. 
    Half of the output channels will be created based all even pixels, 
    and the other half using a second convolution on all odd pixels. 
    """

  def __init__(self, C_in: int, C_out: int, affine: bool = True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.relu(x)
    # x with stride 2 is every other pixel
    # x[:, :, 1:, 1:] are exactly the complementary pixels
    out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
    out = self.bn(out)
    return out
