from __future__ import annotations
import torch
import torch.nn as nn
from torch.autograd import Variable
from pathlib import Path

from models.darts.genotypes import Genotype
from models.darts.operations import *


class Cell(nn.Module):
  """
    This class represents a cell after the search process.
    Every node is connected by two operations.
    Genotype describes the cell.
    """

  def __init__(
      self,
      genotype: Genotype,
      C_prev_prev: int,
      C_prev: int,
      C: int,
      reduction: bool,
      reduction_prev: bool,
      device: torch.device,
  ):
    super(Cell, self).__init__()

    if reduction_prev:
      self.preprocess0: nn.Module = FactorizedReduce(C_prev_prev, C)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

    op_names: tuple[str]
    indices: tuple[int]
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat: list[int] = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self.device = device
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C: int, op_names: tuple[str], indices: tuple[int], concat: list[int],
               reduction: bool):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2 * i]]
      h2 = states[self._indices[2 * i + 1]]
      op1 = self._ops[2 * i]
      op2 = self._ops[2 * i + 1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.0:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob, self.device)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob, self.device)
      s = h1 + h2
      states += [s]
    return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
  """
    The Auxiliary head is a debug measure during training.
    It's a classifier based on the intermediate representation after 2/3-th layer.
    """

  def __init__(self, C, num_classes):
    """assuming input size 8x8"""
    super(AuxiliaryHeadCIFAR, self).__init__()
    self.features = nn.Sequential(
        nn.ReLU(inplace=True),
        nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
        nn.Conv2d(C, 128, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 768, 2, bias=False),
        nn.BatchNorm2d(768),
        nn.ReLU(inplace=True),
    )
    self.classifier = nn.Linear(768, num_classes)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x.view(x.size(0), -1))
    return x


class NetworkCIFAR(nn.Module):
  """
    NetworkCIFAR represents the CIFAR network after search phase.
    It adds input ConvNet, cells including reductions,
    an auxiliary head for debugging and a classifier in the end.
    """

  def __init__(self, C: int, num_classes: int, layers: int, genotype: Genotype, device: torch.device,
               drop_path_prob: float, auxiliary: bool, dropout: float):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers
    self._genotype = genotype
    self._C = C
    self._num_classes = num_classes
    self._steps = len(genotype.normal) // 2
    self._auxilary = auxiliary
    self._dropout_prob = dropout
    self.device = device

    stem_multiplier = 3
    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

    self.drop_path_prob = drop_path_prob

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):

      # reduction cells double the amount of channels
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False

      cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, device)
      reduction_prev = reduction
      self.cells += [cell]

      # the output of the cell is the concatination of the concat nodes
      # therefor the channel output is the length of concat nodes times the number of channels
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
      if i == 2 * layers // 3:
        C_to_auxiliary = C_prev

    if auxiliary:
      self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(dropout)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # first two inputs are output of a small convNet
    logits_aux = None
    s0 = s1 = self.stem(input)

    for i, cell in enumerate(self.cells):
      # output of two last cells is input to next cell
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
      if i == 2 * self._layers // 3:
        if self._auxilary and self.training:
          logits_aux = self.auxiliary_head(s1)

    # classifier
    out = self.global_pooling(s1)
    out = out.view(out.size(0), -1)
    out = self.dropout(out)
    logits = self.classifier(out)
    return logits, logits_aux  # type: ignore

  def save_to_file(self, path: Path):
    """
    Safes model as pickle to path.
    """
    torch.save(self, path)

  @staticmethod
  def load_from_file(path: Path) -> NetworkCIFAR:
    """
    Reads model as pickle from path.
    """
    return torch.load(path, weights_only=False)

  def clone(self) -> NetworkCIFAR:
    model_new = NetworkCIFAR(self._C, self._num_classes, self._layers, self._genotype, self.device,
                             self.drop_path_prob, self._auxilary, self._dropout_prob).to(self.device)
    for x, y in zip(model_new.parameters(), self.parameters()):
      x.data.copy_(y.data)
    return model_new


def drop_path(x: torch.Tensor, drop_prob: float, device: torch.device) -> torch.Tensor:
  """
    This drops drop_prob of the samples in the batch
    """
  if drop_prob > 0.0:
    keep_prob = 1.0 - drop_prob
    mask = Variable(torch.FloatTensor(x.size(0), 1, 1, 1).to(device).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x
