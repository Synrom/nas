import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Optimizer
from typing import Iterable

from config import Config
from models.darts.model_search import Network


def _concat(xs: Iterable[torch.Tensor] | torch.Tensor) -> torch.Tensor:
  """
    Returned concatination of flattened xs.
    """
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):
  """
    This class handles the training of the network during search stage.
    """

  def __init__(self, model: Network, args: Config):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(
        self.model.arch_parameters(),
        lr=args.arch_learning_rate,
        betas=(0.5, 0.999),
        weight_decay=args.arch_weight_decay,
    )

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
    model_new.load_state_dict(model_dict)
    return model_new.to(self.model.device)

  def _hessian_vector_product(self,
                              vector: list[torch.Tensor],
                              input: torch.Tensor,
                              target: torch.Tensor,
                              r: float = 1e-2) -> list[torch.Tensor]:
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
    return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
