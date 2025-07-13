from pathlib import Path
import logging
import sys
import copy
import numpy as np
import matplotlib
import matplotlib.patches as mpatches

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import random
from torch.utils.hooks import RemovableHandle

from dataset.cifar import cifar10_label2name
from utils import clone_optimizer, clone_model
from models.darts.model_search import Network as DartsSearchNetwork
from models.darts.model import NetworkCIFAR as DartsEvalNetwork
from models.ppc.model_search import Network as PPCSearchNetwork
from models.darts.architect import Architect
from config import DartsSearchConfig, PPCSearchConfig
from monitor.live import Live, LiveGrid, nparray
from monitor.plot import Line, Bar, Image, Hist, TwoLines, ColorMapOptions, VisAlpha, GenotypeGraph, MultiLines
from models.darts.genotypes import PRIMITIVES, Genotype, SHORT_PRIMITIVES
from models.darts.architect import Architect, Hook
from models.ppc.config import StageConfig
from models.ppc.model_search import SEBlock
from dashboard.config import write_dashboard_config, DashboardConfig
from models.ppc.op import PartialMixedOp
from models.ppc.model_search import Cell as PpcCell

SearchNetwork = DartsSearchNetwork | PPCSearchNetwork
EvalNetwork = DartsEvalNetwork
SearchConfig = DartsSearchConfig | PPCSearchConfig


class Monitor:

  def __init__(
      self,
      model: SearchNetwork | EvalNetwork,
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
      label2name: dict[int, str] = cifar10_label2name,
      primitives: list[str] = PRIMITIVES,
      debug: bool = True,
      architect: Architect | None = None,
      stage: int = 0,
  ):
    self.vis_interval = vis_interval
    self.num_steps_per_epoch = num_steps_per_epoch
    self.path = Path(logdir) / runid
    if self.path.exists() is False:
      self.path.mkdir(parents=True, exist_ok=True)
    write_dashboard_config(self.path, DashboardConfig(runid))
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
    self.stage = stage
    self.logger = logger
    self.label2name = label2name
    self.num_labels = len(self.label2name.keys())
    self.primitives = primitives
    self.logger.info(f" --- Starting new run {runid} --- ")
    self.training_loss = Live("Training Loss", self.path,
                              Line(title="Training Loss", ylabel="Loss", grid=True))
    self.smoothed_training_loss = Live("Smooth Training Loss", self.path,
                                       Line(title="Smoothed Training Loss", ylabel="Loss", grid=True))
    self.steps = 0
    self.epoch = epoch
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    self.test_batch = next(iter(test_loader))
    self.device = device
    self.criterion = criterion
    self.valid_loss = Live(
        "Validation Loss", self.path,
        TwoLines(label1="Validation Loss",
                 label2="Training Loss",
                 title="Validation Loss",
                 ylabel="Loss",
                 grid=True))
    self.training_aux_loss = Live(
        "Training Aux Loss", self.path,
        TwoLines(label1="Loss",
                 label2="Aux Loss",
                 title="Training Aux Loss",
                 ylabel="Loss",
                 grid=True))
    self.valid_acc = Live("Validation Accuracy", self.path,
                          Line(title="Validation accuracy", ylabel="Loss", grid=True))
    self.valid_topk_acc = Live("Validation Top-K Accuracy", self.path,
                               Line(title="Validation topk accuracy", ylabel="Loss", grid=True))
    self.valid_err_rate = Live(
        "Validation Error Rate", self.path,
        Line(ylabel=f"CIFAR-{self.num_labels} Test Error (%)",
             xlabel="Training Epoch",
             title="Validation Error Rate",
             grid=True))
    if isinstance(model, SearchNetwork):
      self.vis_alphas_normal = LiveGrid("Alphas Normal", self.path)
      self.vis_alphas_reduce = LiveGrid("Alphas Reduce", self.path)
      self.vis_alphas_distribution: LiveGrid = LiveGrid("Alpha Distributions", self.path)
    self.valid_train_ref_idx: int = len(
        self.training_loss.data) if self.training_loss.data is not None else 0
    self.plot_interval = int(len(test_dataset) / 4)  # type: ignore
    self.debug = debug
    self.forward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.backward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.forward_checks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.backward_checks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.hook_vis_acts = LiveGrid(f"Activations", self.path)
    self.hook_vis_grads = LiveGrid(f"Gradients", self.path)
    self.test_batch_vis: LiveGrid = LiveGrid("Test Batch", self.path)
    self.vis_genotypes: LiveGrid = LiveGrid("Genotypes", self.path)
    self.vis_lrs = Live("Learning Rate", self.path, Line("Learning Rate", grid=True))
    self.vis_acts_and_grads = vis_acts_and_grads
    self.alpha_hook: Hook | None = None
    self.se_vis = LiveGrid("SE Block", self.path)
    self.hessian_hook: Hook | None = None
    self.vis_eigvals = Live("Eigenvalues", self.path,
                            Line(title="Hessian Eigenvalues", ylabel="Dominant Eigenvalue"))
    if self.vis_acts_and_grads:
      self.add_hooks(model, architect)

  def commit(self):
    self.training_loss.commit()
    self.training_aux_loss.commit()
    self.smoothed_training_loss.commit()
    self.valid_loss.commit()
    self.valid_acc.commit()
    self.valid_topk_acc.commit()
    self.valid_err_rate.commit()
    self.vis_lrs.commit()
    self.vis_eigvals.commit()

  def reset_hook(self):
    for hook in self.forward_checks.values():
      hook.remove()
    for hook in self.backward_checks.values():
      hook.remove()
    for hook in self.forward_hooks.values():
      hook.remove()
    for hook in self.backward_hooks.values():
      hook.remove()
    self.forward_checks = {}
    self.backward_checks = {}
    self.forward_hooks = {}
    self.backward_hooks = {}

  def next_stage(self, model: PPCSearchNetwork, stage: StageConfig):
    self.stage += 1
    self.epoch = 0
    self.steps = 0
    self.logger.info(f"Entering Stage {self.stage}")
    self.logger.info(f"- {stage.cells} cells")
    self.logger.info(f"- {stage.operations} operations")
    self.logger.info(f"- {stage.channels} init channels")
    self.logger.info(f"- {stage.channel_sampling_prob} channel sampling probability")
    self.logger.info(f"- {stage.epochs} epochs")
    self.logger.info(f"- {stage.dropout} dropout")
    self.count_nr_parameters(model)
    title = f"Stage {self.stage}"
    self.training_loss.add_marker(title)
    self.valid_loss.add_marker(title)
    self.smoothed_training_loss.add_marker(title)
    self.valid_acc.add_marker(title)
    self.valid_err_rate.add_marker(title)
    self.valid_topk_acc.add_marker(title)
    self.vis_lrs.add_marker(title)
    self.commit()

  def add_hooks(self, model: SearchNetwork | EvalNetwork, architect: None | Architect = None):
    i = 1
    nr_relus = len([m for m in model.modules() if isinstance(m, nn.ReLU) or isinstance(m, nn.GELU)])
    interval = nr_relus // 7 if nr_relus > 7 else 1
    rows = min(7, nr_relus)

    self.hook_vis_acts.next_row(f"Stage {self.stage} Epoch {self.epoch}")
    self.hook_vis_grads.next_row(f"Stage {self.stage} Epoch {self.epoch}")

    # check for inf or Nan values
    def forward_check(module, input, output):
      if any(torch.isnan(x.detach()).any() or torch.isinf(x).any() for x in input):
        self.logger.error(f"[FORWARD] NaN or Inf detected in input of {module.__class__.__name__}")
      if torch.isnan(output.detach().clone()).any() or torch.isinf(output).any():
        self.logger.error(f"[FORWARD] NaN or Inf detected in output of {module.__class__.__name__}")
      if module in self.forward_checks:
        self.forward_checks[module].remove()
        self.forward_checks.pop(module)

    def backward_check(module, grad_input, grad_output):
      if any(
          torch.isnan(g.detach()).any() or torch.isinf(g).any() for g in grad_input if g is not None):
        self.logger.error(
            f"[BACKWARD] NaN or Inf detected in grad_input of {module.__class__.__name__}")
      if any(
          torch.isnan(g.detach()).any() or torch.isinf(g).any() for g in grad_output
          if g is not None):
        self.logger.error(
            f"[BACKWARD] NaN or Inf detected in grad_output of {module.__class__.__name__}")
      if module in self.backward_checks:
        self.backward_checks[module].remove()
        self.backward_checks.pop(module)

    row = 0
    for module in model.modules():
      if isinstance(module, nn.ReLU) or isinstance(module, nn.GELU):
        if not isinstance(module, torch.nn.Sequential) and not len(list(module.children())) > 0:
          self.forward_checks[module] = module.register_forward_hook(forward_check)
          self.backward_checks[module] = module.register_full_backward_hook(backward_check)
        if (i + 1) % interval == 0 and i / interval < rows:
          self.forward_hooks[module] = module.register_forward_hook(
              self.plot_inputs(f"{i}th Activation Layer Inputs"))
          self.backward_hooks[module] = module.register_full_backward_hook(
              self.plot_gradients(f"{i}th Activation Layer Gradients"))
          row += 1
        i += 1
    if architect is not None:
      self.register_alpha_hook(architect)
      self.register_hessian_hook(architect)

  def register_alpha_hook(self, architect: Architect):

    def alpha_hook(tensor: torch.Tensor):
      if self.hook_vis_grads is None:
        return
      numbers = tensor.detach().cpu().flatten().numpy()
      self.hook_vis_grads.add_entry(Hist(50).plot(numbers)[0], "Alpha Gradients")
      if self.alpha_hook is not None:
        self.alpha_hook.remove()
        self.alpha_hook = None

    self.alpha_hook = architect.add_alpha_hook(alpha_hook)

  def register_hessian_hook(self, architect: Architect):

    def hessian_hook(tensor: torch.Tensor):
      if self.hook_vis_grads is None:
        return
      numbers = tensor.detach().cpu().flatten().numpy()
      self.hook_vis_grads.add_entry(Hist(50).plot(numbers)[0], "Alpha Hessian")
      if self.hessian_hook is not None:
        self.hessian_hook.remove()
        self.hessian_hook = None

    self.hessian_hook = architect.add_hessian_hook(hessian_hook)

  def plot_inputs(self, title: str):

    def hook(module, input, output):
      numbers = input[0].detach().cpu().float().flatten().numpy()
      self.hook_vis_acts.add_entry(Hist(50).plot(numbers)[0], title)
      if module in self.forward_hooks:
        self.forward_hooks[module].remove()
        self.forward_hooks.pop(module)

    return hook

  def plot_gradients(self, title: str):

    def hook(module, grad_input, grad_output):
      numbers = grad_input[0].detach().float().cpu().flatten().numpy()
      self.hook_vis_grads.add_entry(Hist(50).plot(numbers)[0], title)
      if module in self.backward_hooks:
        self.backward_hooks[module].remove()
        self.backward_hooks.pop(module)

    return hook

  def first_batch(self, imgs: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, loss: float):
    """ Visualize input to the model. """
    self.logger.info(f"First loss is {loss:.2f} (should be around {-np.log(1/self.num_labels):.2f})")
    num_samples = imgs.shape[0]
    imgs = imgs.permute(0, 2, 3, 1)
    data = X.detach().cpu().numpy()
    vis = LiveGrid("First Batch Inputs", self.path)
    plot = Image(cmap="viridis",
                 aspect="auto",
                 vmin=data.min(),
                 vmax=data.max(),
                 colormap=ColorMapOptions(fraction=0.046, pad=0.04))
    for i in range(num_samples):
      vis.add_entry(Image().plot(imgs[i].cpu().numpy())[0],
                    f"Label: {self.label2name[int(Y[i].item())]}")
      for j in range(1, 4):
        vis.add_entry(plot.plot(data[i, j - 1])[0])
      vis.next_row(None)

  def add_aux_loss(self, raw_loss: float, aux_loss: float):
    self.training_aux_loss.add(np.array([raw_loss, aux_loss])[:, np.newaxis], axis=1)

  def visualize_lrs(self, model_lr: float):
    self.logger.info("Visualize LRs ...")
    self.vis_lrs.add(nparray(model_lr))

  def add_training_loss(self, loss: float):
    self.steps += 1
    self.training_loss.add(nparray(loss))
    if (self.steps + 1) % self.vis_interval == 0:
      loss = np.array(self.training_loss.data[-self.vis_interval:]).mean()
      self.smoothed_training_loss.add(nparray(loss))

  def end_epoch(self,
                model: SearchNetwork | EvalNetwork,
                architect: Architect | None = None,
                visualize: bool = True):
    self.epoch += 1
    self.steps = 0
    if self.vis_acts_and_grads and visualize:
      self.add_hooks(model,
                     architect)  # visualize activations and gradients at beginning of each epoch

  def eval_test_batch(self, title: str, model: SearchNetwork | EvalNetwork):
    self.logger.info("Eval test batch ...")
    imgs, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    if isinstance(model, SearchNetwork):
      out = model(input)
    else:
      out, _ = model(input)
    loss = self.criterion(out, target)
    self.logger.info(f"Loss on test batch is {loss.item():.2f}")
    probs = F.softmax(out, dim=1).cpu().detach().numpy()

    labels = list(self.label2name.values())
    cols = imgs.shape[0]
    if self.epoch == 0 and self.stage == 0:
      imgs = imgs.permute(0, 2, 3, 1)
      for i, col in enumerate(range(cols)):
        label = self.label2name[target[i].item()]
        self.test_batch_vis.add_entry(Image().plot(imgs[col])[0], label)
      self.test_batch_vis.next_row(None)

    self.test_batch_vis.next_row(f"Stage {self.stage} Epoch {self.epoch}")
    for col in range(cols):
      label = self.label2name[target[col].item()]
      if len(labels) > 10:
        topk = np.argsort(probs[col])[-10:]
        labels = [labels[i] for i in topk]
        data = np.array([probs[col][i] for i in topk])
      else:
        data = probs[col]
      plot = Bar(labels=labels,
                 ylim=(0, 1),
                 xticks=list(range(len(labels))),
                 rotation=45,
                 highlight_label=label)
      self.test_batch_vis.add_entry(plot.plot(data)[0], title)
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
    self.logger.info(f"\t- validation accuracy {acc * 100:.2f}%")
    self.valid_topk_acc.add(np.array([topk_acc]))
    self.logger.info(f"\t- validation topk accuracy {topk_acc * 100:.2f}%")
    error_rate = (1 - acc) * 100
    self.logger.info(f"\t- error rate {error_rate:.2f}%")
    self.valid_err_rate.add(nparray(error_rate))

  def visualize_eigenvalues(self, input_valid: torch.Tensor, target_valid: torch.Tensor,
                            architect: Architect):
    self.logger.info("Calculating Hessian Eigenvalues ... This may take a while")
    eigvals = architect.compute_hessian_eigenvalues(input_valid, target_valid)
    self.logger.info("Done calculating the Hessian Eigenvalues")
    dom_eigval = np.max(np.abs(eigvals))
    self.vis_eigvals.add(nparray(dom_eigval))

  def visualize_alphas(self, alpha_normal: np.ndarray, alpha_reduce: np.ndarray,
                       model: PPCSearchNetwork):
    self.logger.info("Visualize alphas ...")
    normal_plot = VisAlpha(steps=model._steps,
                           primitives=self.primitives,
                           switch=model._switch_normal,
                           verbose=False,
                           short_primitives=SHORT_PRIMITIVES)
    reduce_plot = VisAlpha(steps=model._steps,
                           primitives=self.primitives,
                           switch=model._switch_reduce,
                           verbose=False,
                           short_primitives=SHORT_PRIMITIVES)
    offset = 0
    self.vis_alphas_normal.next_row(f"Stage {self.stage} Epoch {self.epoch}")
    self.vis_alphas_reduce.next_row(f"Stage {self.stage} Epoch {self.epoch}")

    for col in range(model._steps):
      fig_normal = normal_plot.draw_graph_subplot(col + 2, alpha_normal[offset:offset + col + 2])
      fig_reduce = reduce_plot.draw_graph_subplot(col + 2, alpha_reduce[offset:offset + col + 2])
      self.vis_alphas_normal.add_entry(fig_normal, f"Node {col+2}")
      self.vis_alphas_reduce.add_entry(fig_reduce, f"Node {col+2}")
      offset += col + 2

    self.vis_alphas_distribution.next_row(f"Stage {self.stage} Epoch {self.epoch}")
    self.vis_alphas_distribution.add_entry(Hist(50).plot(alpha_normal.flatten())[0], "Normal")
    self.vis_alphas_distribution.add_entry(Hist(50).plot(alpha_reduce.flatten())[0], "Reduce")

  def visualize_genotypes(self, genotype: Genotype):
    self.logger.info("Visualize genotypes ...")
    plot = GenotypeGraph()
    normal_genotype = plot.convert_genotype_to_array(genotype.normal)
    reduce_genotype = plot.convert_genotype_to_array(genotype.reduce)

    self.vis_genotypes.next_row(f"Stage {self.stage} Epoch {self.epoch}")
    self.vis_genotypes.add_entry(plot.plot(normal_genotype)[0], "Normal")
    self.vis_genotypes.add_entry(plot.plot(reduce_genotype)[0], "Reduce")

  def input_dependent_baseline(self, model: SearchNetwork | EvalNetwork, criterion: nn.Module):
    _, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    if isinstance(model, SearchNetwork):
      logits = model(input)
    else:
      logits, _ = model(input)
    loss_real = criterion(logits, target)

    zeroes = torch.zeros_like(input)
    if isinstance(model, SearchNetwork):
      out_zeroes = model(zeroes)
    else:
      out_zeroes, _ = model(zeroes)
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

  def test_data_sharing_inbetween_batch(self, model: SearchNetwork | EvalNetwork, X: torch.Tensor):
    model.eval()
    input = X.clone().detach().requires_grad_(True)
    input.retain_grad()
    if isinstance(model, SearchNetwork):
      out = model(input)
    else:
      out, _ = model(input)
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

  def visualize_se_blocks(self, model: PPCSearchNetwork):
    self.logger.info(f"Visualize SE Blocks ...")
    nr_se_blocks = len([m for m in model.modules() if isinstance(m, SEBlock)])
    interval = nr_se_blocks // 7 if nr_se_blocks > 7 else 1
    rows = min(7, nr_se_blocks)
    _, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    indices: list[int] = []
    indice_targets: list[int] = []
    j: int = 0
    while len(indices) < 4 and j < target.shape[0]:
      if target[j].item() not in indice_targets:
        indices.append(j)
        indice_targets.append(target[j].item())
      j += 1
    labels = [self.label2name[idx] for idx in indice_targets]
    tensor_indices = torch.tensor(indices).to(self.device)
    input = input[tensor_indices]
    target = target[tensor_indices]
    colors = plt.cm.tab10.colors[:len(labels)]  # type: ignore
    legend = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
    hooks: dict[nn.Module, RemovableHandle] = {}
    plot = MultiLines(ylabel="Activation",
                      xlabel="Channel Index",
                      grid=True,
                      linewidth=0.5,
                      legend=legend)
    self.se_vis.next_row(f"Stage {self.stage} Epoch {self.epoch}")

    def create_hook(i: int):

      def hook(module: SEBlock, _1, _2):
        if module.attn_last is not None:
          self.se_vis.add_entry(plot.plot(module.attn_last)[0], f"SE_{i}")
        hooks[module].remove()
        hooks.pop(module)

      return hook

    i = 0
    for module in model.modules():
      if isinstance(module, SEBlock):
        if (i + 1) % interval == 0 and i / interval < rows:
          hooks[module] = module.register_forward_hook(create_hook(i))
        i += 1
    model.eval()
    with torch.no_grad():
      model(input)
    for hook in hooks.values():
      hook.remove()

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

    visualization = LiveGrid("Overfit Sinlge Batch", self.path)
    visualization.next_row(f"Stage {self.stage} Epoch {self.epoch}")

    losses: list[float] = []
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
      assert logits.shape == torch.Size([config.batch_size, self.num_labels])
      loss = criterion(logits, target_train)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
      optimizer.step()
      losses.append(loss.item())

    visualization.add_entry(Line(ylabel="Loss", grid=True).plot(np.array(losses))[0])

    self.logger.info(f"Overfitted single batch for {n} iterations leading to loss {loss.item():.2f}")

  def count_nr_parameters(self, model: nn.Module):
    nr_params = np.sum(
        np.prod(v.size())
        for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6  # type: ignore
    self.logger.info(f"param size = {nr_params}MB")

  def overfit_single_batch_eval(
      self,
      model: EvalNetwork,
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

    visualization = Live("Overfit Single Batch", self.path,
                         Line(title="Overfitting Batch Loss", ylabel="Loss", grid=True))

    model.train()
    for _ in range(n):
      optimizer.zero_grad()
      logits, _ = model(input)
      loss = criterion(logits, target)
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
      optimizer.step()
      visualization.add(nparray(loss.item()))
      visualization.commit()

    self.logger.info(f"Overfitted single batch for {n} iterations leading to loss {loss.item():.2f}")
