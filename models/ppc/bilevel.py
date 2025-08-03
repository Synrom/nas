import torch
from models.ppc.model_search import Network
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import math


class InverseSqrtLRScheduler(_LRScheduler):

  def __init__(self, optimizer, base_lr: float, last_epoch=-1):
    self.base_lr = base_lr
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    t = self.last_epoch + 1
    return [self.base_lr / math.sqrt(t) for _ in self.optimizer.param_groups]


class SPNGBiO:

  def __init__(self, model: Network, criterion: nn.Module, gamma: float = 0.1):
    self.model = model
    self.theta_model = model.new()
    self.theta_model.alphas_normal = self.model.alphas_normal
    self.theta_model.alphas_reduce = self.model.alphas_reduce
    self.w_optimizer = torch.optim.SGD(self.model.parameters())
    self.alpha_optimizer = torch.optim.SGD(self.model.arch_parameters())
    self.theta_optimizer = torch.optim.SGD(self.theta_model.parameters(), lr=0.05)
    self.w_scheduler = InverseSqrtLRScheduler(self.w_optimizer, 0.5)
    self.alpha_scheduler = InverseSqrtLRScheduler(self.alpha_optimizer, 0.085)
    self.gamma = gamma
    self.criterion = criterion

  def theta_step(self, input: torch.Tensor, target: torch.Tensor):
    self.theta_optimizer.zero_grad()
    logits = self.theta_model(input)
    loss = self.criterion(logits, target)
    loss.backward(retain_graph=True)
    for thetai, yi in zip(self.theta_model.parameters(), self.model.parameters()):
      assert thetai.grad is not None
      thetai.grad.data += 1 / self.gamma * (thetai - yi)
    self.theta_optimizer.step()
    return loss

  def alpha_step(self, input_search: torch.Tensor, target_search: torch.Tensor,
                 input_train: torch.Tensor, target_train: torch.Tensor, ct: float):
    self.alpha_optimizer.zero_grad()
    flogits = self.model(input_search)
    glogits = self.model(input_train)
    theta_logits = self.theta_model(input_train)
    floss = self.criterion(flogits, target_search)
    gloss = self.criterion(glogits, target_train)
    theta_loss = self.criterion(theta_logits, target_train)
    loss = 1 / ct * floss + gloss - theta_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.arch_parameters(), max_norm=1.0)
    self.alpha_optimizer.step()
    return loss

  def wstep(self, input_search: torch.Tensor, target_search: torch.Tensor, input_train: torch.Tensor,
            target_train: torch.Tensor, ct: float):
    self.w_optimizer.zero_grad()
    flogits = self.model(input_search)
    glogits = self.model(input_train)
    floss = self.criterion(flogits, target_search)
    gloss = self.criterion(glogits, target_train)
    loss = 1 / ct * floss + gloss
    loss.backward()
    for thetai, yi in zip(self.theta_model.parameters(), self.model.parameters()):
      assert yi.grad is not None
      yi.grad.data += 1 / self.gamma * (thetai - yi)
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.w_optimizer.step()
    return loss

  def step(self, t: int, input_search: torch.Tensor, target_search: torch.Tensor,
           input_train: torch.Tensor, target_train: torch.Tensor, input_theta: torch.Tensor,
           target_theta: torch.Tensor, ct: float):
    ct = (t + 1)**2
    theta_loss = self.theta_step(input_theta, target_theta)
    alpha_loss = self.alpha_step(input_search, target_search, input_train, target_train, ct)
    wloss = self.wstep(input_search, target_search, input_train, target_train, ct)
    self.alpha_scheduler.step()
    self.w_scheduler.step()
    return theta_loss, alpha_loss, wloss
