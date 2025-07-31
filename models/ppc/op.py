import torch
import torch.nn as nn
import numpy as np

from models.darts.genotypes import PRIMITIVES
from models.darts.operations import OPS, ReLUConvBN, activatition

def channel_shuffle(x, groups):
  batchsize, num_channels, height, width = x.data.size()

  channels_per_group = num_channels // groups
  
  # reshape
  x = x.view(batchsize, groups, 
      channels_per_group, height, width)

  x = torch.transpose(x, 1, 2).contiguous()

  # flatten
  x = x.view(batchsize, -1, height, width)

  return x

def random_shuffle(x):
  batchsize, num_channels, height, width = x.data.size()
  indices = torch.randperm(num_channels)
  x = x[:,indices]
  return x

class PartialMixedOp(nn.Module):

  def __init__(self, C: int, stride: int, prob: float, edge_switch: list[bool], gelu: bool):
    super(PartialMixedOp, self).__init__()
    self._ops = nn.ModuleList()
    self.C_sampled = int(prob * C)
    self.k = int(1/prob)
    self.mp = nn.MaxPool2d(2,2)
    for primitive, activated in zip(PRIMITIVES, edge_switch):
      if activated is False:
        continue
      op = OPS[primitive](self.C_sampled, stride, False, gelu)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(self.C_sampled, affine=False))
      self._ops.append(op)


  def forward(self, x, weights):
    #channel proportion k=4  
    x = random_shuffle(x)
    xtemp = x[ : , :  self.C_sampled, :, :]
    xtemp2 = x[ : ,  self.C_sampled:, :, :]
    temp1 = sum(w * op(xtemp) for w, op in zip(weights, self._ops))
    #reduction cell needs pooling before concat
    if temp1.shape[2] == x.shape[2]:
      ans = torch.cat([temp1,xtemp2],dim=1)
    else:
      ans = torch.cat([temp1,self.mp(xtemp2)], dim=1)
    #ans = channel_shuffle(ans,self.k)
    #ans = torch.cat([ans[ : ,  dim_2//4:, :, :],ans[ : , :  dim_2//4, :, :]],dim=1)
    #except channe shuffle, channel shift also works
    return ans

#class PartialMixedOp(nn.Module):
#  """
#  MixedOp with Partial Connection.
#  """
#
#  def __init__(self, C: int, stride: int, prob: float, edge_switch: list[bool], gelu: bool):
#    super(PartialMixedOp, self).__init__()
#    self._ops = nn.ModuleList()
#    self.C_sampled = int(prob * C)
#    self._stride = stride
#    if self._stride > 1:
#      self.pool = nn.MaxPool2d(2, self._stride)
#    self.C = C
#    for primitive, activated in zip(PRIMITIVES, edge_switch):
#      if activated is False:
#        continue
#      op = OPS[primitive](self.C_sampled, stride, False, gelu)
#      if "pool" in primitive:
#        op = nn.Sequential(op, nn.BatchNorm2d(self.C_sampled, affine=False))
#      self._ops.append(op)
#    self.alphas: None | np.ndarray = None
#
#  def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
#    """
#
#    Args:
#        x: edge input
#        weights: the alphas of the operations
#    """
#    self.alphas = weights.detach().cpu().numpy()
#
#    if self._stride > 1:  # reduction cell
#      rest = self.pool(x)
#    else:
#      rest = x
#
#    # Clone rest to avoid in-place modification of a view
#    rest = rest.clone()
#
#    indices = torch.randperm(self.C)
#    sampled_indices = indices[:self.C_sampled].sort()[0]
#    sampled_x = x[:, sampled_indices, :, :]
#
#    assert weights.shape[0] == len(self._ops)
#    sampled_result = sum(w * op(sampled_x) for w, op in zip(weights, self._ops))
#    rest[:, sampled_indices, :, :] = sampled_result
#
#    return rest


class SEBlock(nn.Module):
  """
  The Attention module used in PPC paper.
  """

  def __init__(self, C: int, reduction_ratio: int, gelu: bool):
    super(SEBlock, self).__init__()

    reduced_C = max(1, C // reduction_ratio)

    self.avg_pool = nn.AdaptiveAvgPool2d(1)

    self.w1 = nn.Linear(C, reduced_C)
    self.a1 = activatition(gelu)
    self.w2 = nn.Linear(reduced_C, reduced_C)
    self.a2 = activatition(gelu)
    self.w3 = nn.Linear(reduced_C, C)
    self.a3 = nn.Sigmoid()
    self.attn_last: None | np.ndarray = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, C, _, _ = x.shape
    attn: torch.Tensor = self.avg_pool(x).view(B, C)
    attn = self.a1(self.w1(attn))
    attn = self.a2(self.w2(attn))
    attn = self.a3(self.w3(attn))
    self.attn_last = attn.detach().cpu().numpy()
    return x * attn.unsqueeze(-1).unsqueeze(-1)
