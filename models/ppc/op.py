import torch
import torch.nn as nn

from models.darts.genotypes import PRIMITIVES
from models.darts.operations import OPS, ReLUConvBN


class PartialMixedOp(nn.Module):
  """
  MixedOp with Partial Connection.
  """

  def __init__(self, C: int, stride: int, prob: float, edge_switch: list[bool]):
    super(PartialMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.C_sampled = int(prob * C)
    self._stride = stride
    if self._stride > 1:
      self.pool = nn.MaxPool2d(2, self._stride)
    self.C = C
    for primitive, activated in zip(PRIMITIVES, edge_switch):
      if activated is False:
        continue
      op = OPS[primitive](self.C_sampled, stride, False)
      if "pool" in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(self.C_sampled, affine=False))
      self._ops.append(op)

  def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """

    Args:
        x: edge input
        weights: the alphas of the operations
    """
    if self._stride > 1:  # reduction cell
      rest = self.pool(x)
    else:
      rest = x

    indices = torch.randperm(self.C)
    sampled_indices = indices[:self.C_sampled].sort()[0]
    sampled_x = x[:, sampled_indices, :, :]

    sampled_result = sum(w * op(sampled_x) for w, op in zip(weights, self._ops))
    rest[:, sampled_indices, :, :] = sampled_result

    return rest


class SEBlock(nn.Module):
  """
  The Attention module used in PPC paper.
  """

  def __init__(self, C: int, reduction_ratio: int):
    super(SEBlock, self).__init__()

    reduced_C = max(1, C // reduction_ratio)

    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    self.w1 = nn.Linear(C, reduced_C)
    self.a1 = nn.ReLU()
    self.w2 = nn.Linear(reduced_C, reduced_C)
    self.a2 = nn.ReLU()
    self.w3 = nn.Linear(reduced_C, C)
    self.a3 = nn.Sigmoid()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = x.shape
    attn: torch.Tensor = self.avg_pool(x).view(B, C)
    attn = self.a1(self.w1(attn))
    attn = self.a2(self.w2(attn))
    attn = self.a3(self.w3(attn))
    return x * attn.unsqueeze(-1).unsqueeze(-1)
