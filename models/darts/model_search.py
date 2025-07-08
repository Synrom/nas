from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch.autograd import Variable
from pathlib import Path
from torch import jit

from models.darts.operations import *
from models.darts.genotypes import PRIMITIVES
from models.darts.genotypes import Genotype


class MixedOp(nn.Module):
  """
    MixedOp is the weighted sum of possible operations.
    """

  def __init__(self, C: int, stride: int, gelu: bool):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in PRIMITIVES:
      op = OPS[primitive](C, stride, False, gelu)
      if "pool" in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
        Args:
          x: edge input
          weights: the alphas of the operations
        """

    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

  def __init__(self, steps: int, multiplier: int, C_prev_prev: int, C_prev: int, C: int,
               reduction: bool, reduction_prev: bool, gelu: bool):
    """
        Args:
          steps: number of intermediate nodes
          multiplier: k-last nodes that will be part of output
          C_prev_prev: number of channels of first input node
          C_prev_prev: number of channels of second input node
          C: number of channels
          reduction: whether this cell is a reduction cell (stride 2 for first two nodes)
          reduction_prev: whether the second input node is was a reduction cell
        """

    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev:
      self.preprocess0: nn.Module = FactorizedReduce(C_prev_prev, C, affine=False, gelu=gelu)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False, gelu=gelu)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False, gelu=gelu)
    self._steps = steps
    self._multiplier = multiplier

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    for i in range(self._steps):
      for j in range(2 + i):  # iterate over all previous nodes i plus two input nodes

        # if this is a reduction cell, stride=2 for edges from first two nodes
        stride = 2 if reduction and j < 2 else 1

        # for each input node, add an edge
        op = MixedOp(C, stride, gelu=gelu)
        self._ops.append(op)

  def forward(self, s0: torch.Tensor, s1: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
        Args:
          s0: first output of first input node
          s1: first output of second input node
          weights: alphas for the edges (flattened list of all edges and their op-alphas)
        """
    # preprocess two input nodes
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps):

      # sum up output of edges
      s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))

      # self._ops is just list of all edges. That's why you need to keep track of offset like this
      offset += len(states)

      states.append(s)

    # return output of last k-nodes
    return torch.cat(states[-self._multiplier:], dim=1)


class Network(nn.Module):
  """
    The network during search.
    """

  def __init__(
      self,
      C: int,
      num_classes: int,
      layers: int,
      criterion: nn.Module,
      device: torch.device,
      gelu: bool,
      steps: int = 4,
      multiplier: int = 4,
      stem_multiplier: int = 3,
      supports_bf16: bool = False,
  ):
    """
        Args:
          C: number of starting channels
          num_classes: for the classifier in the end
          layers: number of cells
          criterion: training loss
          steps: number of nodes per cell
          multiplier: k-last nodes that are used as output of cell
          stem_multiplier: C*stem_multiplier output channels after preprocessing
        """
    super(Network, self).__init__()
    self._gelu = gelu
    self._C = C
    self._num_classes = num_classes
    self._layers = layers
    self._criterion = criterion
    self._steps = steps
    self._multiplier = multiplier
    self.device = device
    self.supports_bf16 = supports_bf16

    C_curr = stem_multiplier * C
    self.stem = nn.Sequential(nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr))

    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, gelu)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, multiplier * C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

    self._initialize_alphas()

  def new(self) -> Network:
    """
        Inits a new Network with the same structure and alphas as self, but new modules.
        """
    model_new = Network(self._C,
                        self._num_classes,
                        self._layers,
                        self._criterion,
                        self.device,
                        gelu=self._gelu).to(self.device)
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
      x.data.copy_(y.data)
    return model_new

  def clone(self) -> Network:
    """
    Clones this network.
    """
    model = self.new()
    for x, y in zip(model.parameters(), self.parameters()):
      x.data.copy_(y.data)
    return model

  def forward(self, input: torch.Tensor) -> torch.Tensor:
    # get inputs cells from stem module
    s0 = s1 = self.stem(input)

    # do the normal cell thing
    for i, cell in enumerate(self.cells):

      if cell.reduction:
        weights = F.softmax(self.alphas_reduce, dim=-1)
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)

      # input to cell is output from last two cells
      s0, s1 = s1, cell(s0, s1, weights)

    # classifier
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

  def _loss(self, input, target):
    if self.supports_bf16 is True:
      with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
        logits = self(input)
        return self._criterion(logits, target)
    logits = self(input)
    return self._criterion(logits, target)

  def _initialize_alphas(self):
    """
        Initializes alphas of network sampled from Norm[0,1].
        """

    # for each node in cell, count all input edges
    k = sum(1 for i in range(self._steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    # for each edge, get num_ops weights from Norm[0,1]
    self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).to(self.device), requires_grad=True)
    self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).to(self.device), requires_grad=True)
    self._arch_parameters = [
        self.alphas_normal,
        self.alphas_reduce,
    ]

  def arch_parameters(self) -> list[Variable]:
    return self._arch_parameters

  def genotype(self) -> Genotype:
    """
        Create genotype that describes learned network with only the top-2 edges going into a node.
        """

    def _parse(weights: ndarray) -> list[tuple[str, int]]:
      """
            Given weights, select best two edges with one opeartion per edge.
            Return Genotype.normal from that.
            """
      # gene is list of edges
      # each node i has two edges going into it: gene[2*i] and gene[2*i+1]
      gene: list[tuple[str, int]] = []
      n = 2
      start = 0
      for i in range(self._steps):  # go over all nodes
        end = start + n

        # weights of edged going into node i
        W = weights[start:end].copy()  # shape (num_edges, num_operations)

        # for each edge, look at maximal alpha value of any operation
        # in the end return idxs of two edges with highest such value
        edges = sorted(range(i + 2),
                       key=lambda x: -max(W[x][k] for k in range(len(W[x]))
                                          if k != PRIMITIVES.index("none")))[:2]

        # iterate over selected edges
        for j in edges:
          k_best = None

          # iterate over operations on edge j and select highest
          for k in range(len(W[j])):
            if k != PRIMITIVES.index("none"):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k

          # primitives[k_best] gives us the best operation along edge j
          gene.append((PRIMITIVES[k_best], j))  # type: ignore
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    # result of node are the concats of the last k-nodes where k=self._multiplier
    concat = list(range(2 + self._steps - self._multiplier, self._steps + 2))

    genotype = Genotype(normal=gene_normal,
                        normal_concat=concat,
                        reduce=gene_reduce,
                        reduce_concat=concat)
    return genotype

  def save_to_file(self, path: Path):
    """
    Safes model as pickle to path.
    """
    torch.save(self, path)

  @staticmethod
  def load_from_file(path: Path) -> Network:
    """
    Reads model as pickle from path.
    """
    return torch.load(path, weights_only=False)
