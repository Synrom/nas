from pathlib import Path
import logging
import sys
import copy
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import random

from dataset.cifar import cifar_label2name
from utils import clone_optimizer, clone_model
from models.darts.model_search import Network as SearchNetwork
from models.darts.model import NetworkCIFAR
from models.darts.architect import Architect
from config import SearchConfig
from monitor.live import Live, LiveGrid, nparray
from monitor.plot import Line, Grid, Bar, Image, plot_load_data, Hist, TwoLines, ColorMapOptions, VisAlpha, GenotypeGraph
from models.darts.genotypes import PRIMITIVES, Genotype
from models.darts.architect import Architect, Hook


class Monitor:

  def __init__(
      self,
      model: SearchNetwork | NetworkCIFAR,
      test_dataset: torch.utils.data.Dataset,
      device: torch.device,
      criterion: nn.Module,
      vis_interval: int,
      vis_acts_and_grads: bool,
      num_steps_per_epoch: int,
      epoch: int = 0,
      batch_size: int = 8,
      logdir: str = "log",
      runid: str = "train",
      loglevel: int = logging.DEBUG,
      label2name: dict[int, str] = cifar_label2name,
      primitives: list[str] = PRIMITIVES,
      debug: bool = True,
      architect: Architect | None = None,
  ):
    self.vis_interval = vis_interval
    self.num_steps_per_epoch = num_steps_per_epoch
    self.path = Path(logdir) / runid
    if self.path.exists() is False:
      self.path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(runid)
    logger.setLevel(loglevel)
    if logger.hasHandlers():
      logger.handlers.clear()
    file_handler = logging.FileHandler((self.path / "log").as_posix())
    stdout_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%m/%d %I:%M:%S %p")
    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    self.logger = logger
    self.label2name = label2name
    self.primitives = primitives
    self.logger.info(f" --- Starting new run {runid} --- ")
    self.training_loss = Live(self.path / "training_loss.png",
                              Line(title="Training Loss", ylabel="Loss", grid=True))
    self.smoothed_training_loss = Live(self.path / "smooth_training_loss.png",
                                       Line(title="Smoothed Training Loss", ylabel="Loss", grid=True))
    self.steps = 0
    self.epoch = epoch
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    self.test_batch = next(iter(test_loader))
    self.device = device
    self.criterion = criterion
    self.valid_loss = Live(
        self.path / "validation_loss.png",
        TwoLines(label1="Validation Loss",
                 label2="Training Loss",
                 title="Validation Loss",
                 ylabel="Loss",
                 grid=True))
    self.valid_acc = Live(self.path / "validation_acc.png",
                          Line(title="Validation accuracy", ylabel="Loss", grid=True))
    self.valid_topk_acc = Live(self.path / "validation_topk_acc.png",
                               Line(title="Validation topk accuracy", ylabel="Loss", grid=True))
    self.valid_err_rate = Live(
        self.path / "validation_error_rate.png",
        Line(xlabel="CIFAR-10 Test Error (%)", ylabel="Training Epoch", grid=True))
    last_5 = (self.epoch // 5) * 5
    if isinstance(model, SearchNetwork):
      self.vis_alphas_normal = Live(self.path / f"alphas_normal-{last_5}-{last_5+4}.png",
                                    data_path=self.path / "alphas_normal.png.npy",
                                    plot=VisAlpha(steps=model._steps, primitives=self.primitives))
      self.vis_alphas_reduce = Live(self.path / f"alphas_reduce-{last_5}-{last_5+4}.png",
                                    data_path=self.path / "alphas_reduce.png.npy",
                                    plot=VisAlpha(steps=model._steps, primitives=self.primitives))
      self.vis_alphas_distribution: LiveGrid[Hist] = LiveGrid(self.path / "alphas_distribution.png",
                                                              Grid(Hist(50), rows=0, cols=2))
      self.normal_alphas_log = model.alphas_normal.detach().cpu().numpy()[np.newaxis, :, :]
      self.reduce_alphas_log = model.alphas_reduce.detach().cpu().numpy()[np.newaxis, :, :]
    self.valid_train_ref_idx: int = 0
    self.plot_interval = int(len(test_dataset) / 4)  # type: ignore
    self.debug = debug
    self.forward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.backward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.forward_checks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.backward_checks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.hook_vis: None | LiveGrid = None
    self.test_batch_vis: None | LiveGrid = None
    self.vis_genotypes: LiveGrid[GenotypeGraph] = LiveGrid(
        self.path / "graphs.png", Grid(GenotypeGraph(), cols=2, rows=0, col_size=6, row_size=4))
    self.vis_lrs = Live(self.path / "learning_rates.png", Line("Model Learning Rate", grid=True))
    self.vis_acts_and_grads = vis_acts_and_grads
    self.alpha_hook: Hook | None = None
    self.hessian_hook: Hook | None = None
    self.vis_eigvals = Live(self.path / "eigenvalues.png",
                            Line(title="Hessian Eigenvalues", ylabel="Dominant Eigenvalue"))
    if self.vis_acts_and_grads:
      self.add_hooks(model, architect)

  def commit(self):
    self.training_loss.commit()
    self.smoothed_training_loss.commit()
    self.valid_loss.commit()
    self.valid_acc.commit()
    self.valid_topk_acc.commit()
    self.valid_err_rate.commit()
    if vis := getattr(self, "vis_alphas_normal", None): vis.commit()
    if vis := getattr(self, "vis_alphas_reduce", None): vis.commit()
    if vis := getattr(self, "vis_alphas_distribution", None): vis.commit()
    if self.test_batch_vis: self.test_batch_vis.commit()
    self.vis_genotypes.commit()
    self.vis_lrs.commit()
    self.vis_eigvals.commit()

  def add_hooks(self, model: SearchNetwork | NetworkCIFAR, architect: None | Architect = None):
    if self.hook_vis:
      self.logger.info("Logging activation inputs and gradients")
      self.hook_vis.commit()
    i = 1
    nr_relus = len([m for m in model.modules() if isinstance(m, nn.ReLU)])
    interval = nr_relus // 7 if nr_relus > 7 else nr_relus
    rows = 0

    # check for inf or Nan values
    def forward_check(module, input, output):
      if any(torch.isnan(x).any() or torch.isinf(x).any() for x in input):
        self.logger.error(f"[FORWARD] NaN or Inf detected in input of {module.__class__.__name__}")
      if torch.isnan(output).any() or torch.isinf(output).any():
        self.logger.error(f"[FORWARD] NaN or Inf detected in output of {module.__class__.__name__}")
      if module in self.forward_checks:
        self.forward_checks[module].remove()
        self.forward_checks.pop(module)

    def backward_check(module, grad_input, grad_output):
      if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grad_input if g is not None):
        self.logger.error(
            f"[BACKWARD] NaN or Inf detected in grad_input of {module.__class__.__name__}")
      if any(torch.isnan(g).any() or torch.isinf(g).any() for g in grad_output if g is not None):
        self.logger.error(
            f"[BACKWARD] NaN or Inf detected in grad_output of {module.__class__.__name__}")
      if module in self.backward_checks:
        self.backward_checks[module].remove()
        self.backward_checks.pop(module)

    for module in model.modules():
      if isinstance(module, nn.ReLU):
        if not isinstance(module, torch.nn.Sequential) and not len(list(module.children())) > 0:
          self.forward_checks[module] = module.register_forward_hook(forward_check)
          self.backward_checks[module] = module.register_full_backward_hook(backward_check)
        if i % interval == 0:
          self.forward_hooks[module] = module.register_forward_hook(
              self.plot_inputs(f"{i}th ReLU Layer Inputs", rows))
          self.backward_hooks[module] = module.register_full_backward_hook(
              self.plot_gradients(f"{i}th ReLu Layer Gradients", rows))
          rows += 1
        i += 1
    if architect is not None:
      self.register_alpha_hook(rows, 0, architect)
      self.register_hessian_hook(rows, 1, architect)
    rows += 1
    self.hook_vis = LiveGrid(self.path / f"epoch-{self.epoch-1}-activations-and-gradients.png",
                             Grid(Hist(50), rows, 2))

  def register_alpha_hook(self, row: int, col: int, architect: Architect):

    def alpha_hook(tensor: torch.Tensor):
      if self.hook_vis is None:
        return
      numbers = tensor.detach().cpu().flatten().numpy()
      self.hook_vis.add_idx(
          numbers, row, col,
          r"$\nabla_{\alpha} L_{val}(w', \alpha) - \xi \nabla^2_{\alpha,w} L_{train}(w, \alpha) \nabla_{w'} L_{val}(w', \alpha)$"
      )
      if self.alpha_hook is not None:
        self.alpha_hook.remove()
        self.alpha_hook = None

    self.alpha_hook = architect.add_alpha_hook(alpha_hook)

  def register_hessian_hook(self, row: int, col: int, architect: Architect):

    def hessian_hook(tensor: torch.Tensor):
      if self.hook_vis is None:
        return
      numbers = tensor.detach().cpu().flatten().numpy()
      self.hook_vis.add_idx(
          numbers, row, col,
          r"$\nabla^2_{\alpha,w} L_{train}(w, \alpha) \nabla_{w'} L_{val}(w', \alpha)$")
      if self.hessian_hook is not None:
        self.hessian_hook.remove()
        self.hessian_hook = None

    self.hessian_hook = architect.add_hessian_hook(hessian_hook)

  def plot_inputs(self, title: str, row: int):

    def hook(module, input, output):
      numbers = input[0].detach().cpu().float().flatten().numpy()
      self.hook_vis.add_idx(numbers, row, 0, title)
      if module in self.forward_hooks:
        self.forward_hooks[module].remove()
        self.forward_hooks.pop(module)

    return hook

  def plot_gradients(self, title: str, row: int):

    def hook(module, grad_input, grad_output):
      numbers = grad_input[0].detach().float().cpu().flatten().numpy()
      self.hook_vis.add_idx(numbers, row, 1, title)
      if module in self.backward_hooks:
        self.backward_hooks[module].remove()
        self.backward_hooks.pop(module)

    return hook

  def first_batch(self, imgs: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, loss: float):
    """ Visualize input to the model. """
    self.logger.info(f"First loss is {loss:.2f} (should be around {-np.log(1/10):.2f})")
    num_samples = imgs.shape[0]
    imgs = imgs.permute(0, 2, 3, 1)
    data = X.detach().cpu().numpy()
    plot: LiveGrid[Image] = LiveGrid(
        self.path / "first_batch_model_inputs.png",
        Grid(
            Image(cmap="viridis",
                  aspect="auto",
                  vmin=data.min(),
                  vmax=data.max(),
                  colormap=ColorMapOptions(fraction=0.046, pad=0.04)), num_samples, 4, 2, 3))
    for i in range(num_samples):
      plot.plot.manual[(i, 0)] = plot_load_data(Image(), imgs[i].cpu().numpy())
      plot.plot.titles[(i, 0)] = f"Label: {self.label2name[int(Y[i].item())]}"
      for j in range(1, 4):
        plot.add_idx(data[i, j - 1], i, j)

    plot.commit()

  def visualize_lrs(self, model_lr: float):
    self.logger.info("Visualize LRs ...")
    self.vis_lrs.add(nparray(model_lr))

  def add_training_loss(self, loss: float):
    self.steps += 1
    self.training_loss.add(nparray(loss))

  def end_epoch(self,
                model: SearchNetwork | NetworkCIFAR,
                architect: Architect | None = None,
                visualize: bool = True):
    self.epoch += 1
    self.steps = 0
    if self.vis_acts_and_grads and visualize:
      self.add_hooks(model,
                     architect)  # visualize activations and gradients at beginning of each epoch
    if self.training_loss.data is not None:
      loss = np.array(self.training_loss.data[-self.vis_interval:]).mean()
      self.smoothed_training_loss.add(nparray(loss))

  def eval_test_batch(self, title: str, model: SearchNetwork | NetworkCIFAR):
    self.logger.info("Eval test batch ...")
    imgs, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    out = model(input)
    loss = self.criterion(out, target)
    self.logger.info(f"Loss on test batch is {loss.item():.2f}")
    probs = F.softmax(out, dim=1).cpu().detach().numpy()

    labels = list(self.label2name.values())
    cols = imgs.shape[0]
    if self.test_batch_vis is None:
      imgs = imgs.permute(0, 2, 3, 1)
      plot: Grid[Bar] = Grid(Bar(), rows=1, cols=cols, col_size=3, row_size=2)
      self.test_batch_vis = LiveGrid(self.path / "test_batch_predictions_later.png", grid=plot)
      for i, col in enumerate(range(cols)):
        label = self.label2name[target[i].item()]
        plot.manual[0, col] = plot_load_data(Image(), imgs[col])
        plot.col_plots[col] = Bar(labels=labels,
                                  ylim=(0, 1),
                                  xticks=list(range(len(labels))),
                                  rotation=45,
                                  highlight_label=label)
        plot.titles[0, col] = label

    row = self.test_batch_vis.add_row()
    for col in range(cols):
      self.test_batch_vis.add_idx(probs[col], row, col, title)
    self.logger.debug(f"Append predictions of test batch: {title}")

  def add_validation_loss(self, loss: float, acc: float, topk_acc: float):
    if self.training_loss.data is None:
      return
    train_loss = np.array(self.training_loss.data[self.valid_train_ref_idx:]).mean()
    self.valid_train_ref_idx = len(self.training_loss.data)
    self.valid_loss.add(np.array([loss, train_loss])[:, np.newaxis], axis=1)
    self.logger.info(f"After {self.steps} batches of epoch {self.epoch}:")
    self.logger.info(f"\t- validation loss {loss:.2f}")
    self.valid_acc.add(np.array([acc]))
    self.logger.info(f"\t- validation accuracy {acc:.2f}")
    self.valid_topk_acc.add(np.array([topk_acc]))
    self.logger.info(f"\t- validation topk accuracy {topk_acc:.2f}")

  def add_error_rate(self, error_rate: float):
    self.valid_err_rate.add(nparray(error_rate))

  def visualize_eigenvalues(self, input_valid: torch.Tensor, target_valid: torch.Tensor,
                            architect: Architect):
    self.logger.info("Calculating Hessian Eigenvalues ... This may take a while")
    eigvals = architect.compute_hessian_eigenvalues(input_valid, target_valid)
    self.logger.info("Done calculating the Hessian Eigenvalues")
    dom_eigval = np.max(np.abs(eigvals))
    self.vis_eigvals.add(nparray(dom_eigval))

  def visualize_alphas(self, alpha_normal: np.ndarray, alpha_reduce: np.ndarray):
    self.logger.info("Visualize alphas ...")
    self.vis_alphas_normal.add(alpha_normal[np.newaxis, :, :], axis=0)
    self.vis_alphas_reduce.add(alpha_reduce[np.newaxis, :, :], axis=0)
    row = self.vis_alphas_distribution.add_row()
    self.vis_alphas_distribution.add_idx(alpha_normal.flatten(), row, 0,
                                         f"Normal alphas at epoch {self.epoch}")
    self.vis_alphas_distribution.add_idx(alpha_reduce.flatten(), row, 1,
                                         f"Reduction alphas at epoch {self.epoch}")

  def visualize_genotypes(self, genotype: Genotype):
    self.logger.info("Visualize genotypes ...")
    row = self.vis_genotypes.add_row()
    self.vis_genotypes.add_idx(
        self.vis_genotypes.plot.default.convert_genotype_to_array(genotype.normal), row, 0,
        f"Normal cell at {self.epoch}th epoch")
    self.vis_genotypes.add_idx(
        self.vis_genotypes.plot.default.convert_genotype_to_array(genotype.reduce), row, 1,
        f"Reduction cell at {self.epoch}th epoch")

  def input_dependent_baseline(self, model: nn.Module, criterion: nn.Module):
    _, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    logits = model(input)
    loss_real = criterion(logits, target)

    zeroes = torch.zeros_like(input)
    out_zeroes = model(zeroes)
    loss_zeroes = criterion(out_zeroes, target)

    if loss_zeroes < loss_real:
      self.logger.warning("##################################################################")
      self.logger.warning("#                                                                #")
      self.logger.warning("# It seems that the model learns some input-independent features #")
      self.logger.warning("#                                                                #")
      self.logger.warning("##################################################################")
      self.logger.warning("Loss on test batch:")
      self.logger.warning(f"\t- with real input is {loss_real:.2f}")
      self.logger.warning(f"\t- with only zeroes is {loss_zeroes:.2f}")

  def test_data_sharing_inbetween_batch(self, model: nn.Module, X: torch.Tensor):
    model.eval()
    input = X.clone().detach().requires_grad_(True)
    input.retain_grad()
    out = model(input)
    assert input.shape[0] == out.shape[0]
    i = random.choice(range(input.shape[0]))
    loss = out[i].sum()
    loss.backward()
    # check that only the i-th input has non-zero gradients
    assert input.shape[0] == out.shape[0]
    assert input.grad.shape == input.shape  # type: ignore
    assert input.grad[i].sum() != 0.0  # type: ignore
    for idx in range(input.shape[0]):
      if idx == i:
        continue
      assert input.grad[idx].sum() == 0.0  # type: ignore
    model.zero_grad()

  def overfit_single_batch(
      self,
      model: SearchNetwork,
      input_train: torch.Tensor,
      target_train: torch.Tensor,
      input_search: torch.Tensor,
      target_search: torch.Tensor,
      criterion: nn.Module,
      optimizer: optim.Optimizer,
      alpha_optimizer: optim.Optimizer,
      config: SearchConfig,
      lr: float,
      n: int = 1000,
  ):
    # clone everything
    self.logger.info("Start overfitting single batch")
    model = model.clone()
    criterion = clone_model(criterion)
    optimizer = clone_optimizer(optimizer, model)
    alpha_optimizer = clone_optimizer(alpha_optimizer, model)
    architect = Architect(model, config, alpha_optimizer)

    visualization = Live(self.path / "overfit_batch_loss.png",
                         Line(title="Overfitting Batch Loss", ylabel="Loss", grid=True))

    model.train()
    for _ in range(n):
      architect.step(input_train,
                     target_train,
                     input_search,
                     target_search,
                     lr,
                     optimizer,
                     unrolled=config.unrolled)
      optimizer.zero_grad()
      logits = model(input_train)
      assert logits.shape == torch.Size([config.batch_size, 10])
      loss = criterion(logits, target_train)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
      optimizer.step()
      visualization.add(nparray(loss.item()))
      visualization.commit()

    self.logger.info(f"Overfitted single batch for {n} iterations leading to loss {loss.item():.2f}")

  def count_nr_parameters(self, model: nn.Module):
    nr_params = np.sum(
        np.prod(v.size())
        for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6  # type: ignore
    self.logger.info(f"param size = {nr_params}MB")

  def overfit_single_batch_eval(
      self,
      model: NetworkCIFAR,
      input: torch.Tensor,
      target: torch.Tensor,
      criterion: nn.Module,
      optimizer: optim.Optimizer,
      grad_clip: float,
      n: int = 400,
  ):
    # clone everything
    self.logger.info("Start overfitting single batch")
    model = model.clone()
    criterion = clone_model(criterion)
    optimizer = clone_optimizer(optimizer, model)

    visualization = Live(self.path / "overfit_batch_loss.png",
                         Line(title="Overfitting Batch Loss", ylabel="Loss", grid=True))

    model.train()
    for _ in range(n):
      optimizer.zero_grad()
      logits = model(input)
      loss = criterion(logits, target)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()
      visualization.add(nparray(loss.item()))
      visualization.commit()

    self.logger.info(f"Overfitted single batch for {n} iterations leading to loss {loss.item():.2f}")
