from __future__ import annotations
import torch
import numpy as np
import torch.nn as nn
from abc import ABC, abstractmethod
from torch.autograd import Variable
from torch.optim import Optimizer
from typing import Iterable, Callable
from numpy.linalg import eigvals

from config import Config
from models.darts.model_search import Network


def _concat(xs: Iterable[torch.Tensor] | torch.Tensor) -> torch.Tensor:
  """
    Returned concatination of flattened xs.
    """
  return torch.cat([x.view(-1) for x in xs])


HookFn = Callable[[torch.Tensor], None]


class Hook(ABC):

  def __init__(self, architect: Architect):
    self.architect = architect

  @abstractmethod
  def remove(self):
    ...


class AlphaGradHook(Hook):

  def remove(self):
    self.architect.alpha_grad_hook = lambda _: None


class HessianHook(Hook):

  def remove(self):
    self.architect.hessian_hook = lambda _: None


class Architect(object):
  """
    This class handles the training of the network during search stage.
    """

  def __init__(self, model: Network, args: Config, optimizer: Optimizer):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = optimizer
    self.arch_weight_decay = args.arch_weight_decay
    self.hessian_hook: HookFn = lambda _: None
    self.alpha_grad_hook: HookFn = lambda _: None

  def _compute_unrolled_model(self, input: torch.Tensor, target: torch.Tensor, eta: float,
                              network_optimizer: Optimizer):
    """
        Create model after a single gradient step.

        Args:
          input: intput tensors into model
          target: target tensors to calculate loss
          eta: learning rate for the model parameters
          network_optimizer: optimizer of model parameters
        """
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data  # this is the w

    # I think this is network_optimizer specific.
    try:
      moment = _concat(network_optimizer.state[v]["momentum_buffer"]
                       for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)

    # gradient of w + L2 regularization
    dtheta = (_concat(torch.autograd.grad(loss, list(self.model.parameters()))).data +
              self.network_weight_decay * theta)

    # unrolled model is model after single SGD update
    unrolled_model = self._construct_model_from_theta(theta.sub(moment + dtheta, alpha=eta))
    return unrolled_model

  def step(
      self,
      input_train: torch.Tensor,
      target_train: torch.Tensor,
      input_valid: torch.Tensor,
      target_valid: torch.Tensor,
      eta: float,
      network_optimizer: Optimizer,
      unrolled: bool,
  ):
    """
        If unrolled, make gradient step on alpha.
        Otherwise, make gradient step on ws.
        """
    self.optimizer.zero_grad()
    if unrolled:
      self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta,
                                   network_optimizer)
    else:
      # do normal gradient step on validation set
      self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid: torch.Tensor, target_valid: torch.Tensor):
    """
        Compute validation loss and take gradients
        """
    loss = self.model._loss(input_valid, target_valid)
    loss.backward()

  def _backward_step_unrolled(
      self,
      input_train: torch.Tensor,
      target_train: torch.Tensor,
      input_valid: torch.Tensor,
      target_valid: torch.Tensor,
      eta: float,
      network_optimizer: Optimizer,
  ):
    """
        Computes gradients of alphas according to loss in DARTS paper.

        Args:
          input_train: training batch
          target_train: training labels
          input_valid: validation batch
          target_valid: validation labels
          eta: eta from paper to calculate w' = w - eta * dw Ltrain(w, alpha)
          network_optimizer: optimizer of ws
        """

    # compute Lval(w', alpha) where w' = w - eta dw Ltrain(w, alpha)
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)
    unrolled_loss.backward()

    # dalpha is the first term of the final loss dalpha Lval(w', alpha)
    dalpha: list[torch.Tensor] = [v.grad for v in unrolled_model.arch_parameters()]

    # vector is dw' Lval(w', alpha)
    vector: list[torch.Tensor] = [v.grad.data for v in unrolled_model.parameters()]

    # implicit_grads are the gradients of the second term from the paper
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    # compute final loss
    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(ig.data, alpha=eta)

    # copy dalpha to model.arch_parameters
    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    self.alpha_grad_hook(torch.stack(dalpha))

  def _construct_model_from_theta(self, theta: torch.Tensor) -> Network:
    """
        Create new network with theta for model weights.
        """
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset:offset + v_length].view(v.size())
      offset += int(v_length)

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict, assign=False)
    return model_new.to(self.model.device)

  def _hessian_vector_product(self,
                              vector: list[torch.Tensor],
                              input: torch.Tensor,
                              target: torch.Tensor,
                              r: float = 1e-2) -> torch.Tensor:
    """
        Compute second gradient loss term: d^2,alpha,w Ltrain(w, alpha) dw' Lval(w', alpha)

        Args:
          vector: dw' Lval(w', alpha) where w' = w - eta * dw Ltrain(w, alpha)
          input: training batch
          target: training labels
          r: eps to compute w+-
        """
    R = r / _concat(vector).norm()

    # set model parameters to w + r * dw' / R
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(v, alpha=R)

    # calculate Ltrain(w+, alpha)
    loss = self.model._loss(input, target)

    # grads_p is dalpha Ltrain(w+, alpha)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    # set model parameters to w - r * dw' / R
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(v, alpha=2 * R)  # 2 bc we first added R, v

    # calculate Ltrain(w-, alpha)
    loss = self.model._loss(input, target)

    # grap_n is dalpha Ltrain(w-, alpha)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    # set parameter values back to values before
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(v, alpha=R)

    # Finally calculate second gradient loss term: d^2,alpha,w Ltrain(w, alpha) dw' Lval(w', alpha)
    hessian = torch.stack([(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)])
    self.hessian_hook(hessian)
    return hessian

  def add_alpha_hook(self, hook: HookFn):
    self.alpha_grad_hook = hook
    return AlphaGradHook(self)

  def add_hessian_hook(self, hook: HookFn):
    self.hessian_hook = hook
    return HessianHook(self)

  def compute_hessian_eigenvalues(self, input_valid: torch.Tensor, target_valid: torch.Tensor):

    def zero_grads(parameters):
      for p in parameters:
        if p.grad is not None:
          p.grad.detach_()
          p.grad.zero_()

    def gradient(_outputs, _inputs, grad_outputs=None, retain_graph=None, create_graph=False):
      if torch.is_tensor(_inputs):
        _inputs = [_inputs]
      else:
        _inputs = list(_inputs)
      grads = torch.autograd.grad(_outputs,
                                  _inputs,
                                  grad_outputs,
                                  allow_unused=True,
                                  retain_graph=retain_graph,
                                  create_graph=create_graph)
      grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads, _inputs)]
      return torch.cat([x.contiguous().view(-1) for x in grads])

    zero_grads(self.model.parameters())
    zero_grads(self.model.arch_parameters())
    loss = self.model._loss(input_valid, target_valid)
    inputs = self.model.arch_parameters()
    if torch.is_tensor(inputs):
      inputs = [inputs]  # type: ignore
    else:
      inputs = list(inputs)

    n = sum(p.numel() for p in inputs)
    hessian = Variable(torch.zeros(n, n)).type_as(loss)

    ai = 0
    for i, inp in enumerate(inputs):
      [grad] = torch.autograd.grad(loss, inp, create_graph=True, allow_unused=False)
      grad = grad.contiguous().view(-1) + self.arch_weight_decay * inp.view(-1)

      for j in range(inp.numel()):
        if grad[j].requires_grad:
          row = gradient(grad[j], inputs[i:], retain_graph=True)[j:]
        else:
          n = sum(x.numel() for x in inputs[i:]) - j
          row = Variable(torch.zeros(n)).type_as(grad[j])
          #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

        hessian.data[ai, ai:].add_(row.clone().type_as(hessian).data)  # ai's row
        if ai + 1 < n:
          hessian.data[ai + 1:, ai].add_(row.clone().type_as(hessian).data[1:])  # ai's column
        del row
        ai += 1
      del grad
    return eigvals(hessian.cpu().data.numpy())
