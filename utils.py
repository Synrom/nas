import copy
import torch.optim as optim
import torch.nn as nn


def clone_optimizer(optimizer: optim.Optimizer, model: nn.Module) -> optim.Optimizer:
  optimizer_class = type(optimizer)

  # Extract relevant constructor parameters from the original optimizer
  kwargs = {}
  for key, value in optimizer.defaults.items():
    kwargs[key] = value

  # Create a new optimizer with the same parameters for the cloned model
  return optimizer_class(model.parameters(), **kwargs)


def models_eq(m1: nn.Module, m2: nn.Module) -> bool:
  for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
    if p1.data.ne(p2.data).abs().sum() > 0:
      print(f"Parameters {n1} and {n2} differ by {p1.data.ne(p2.data).abs().sum()}")
      return False
    else:
      print(f"Parameters {n1} and {n2} are equal")
  return True


def clone_model(m: nn.Module) -> nn.Module:
  return copy.deepcopy(m)
