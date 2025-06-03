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
import torch
import os
import random

from dataset.cifar import cifar_label2name
from utils import clone_optimizer, clone_model


class Monitor:

  def __init__(
      self,
      model: nn.Module,
      test_dataset: torch.utils.data.Dataset,
      device: torch.device,
      criterion: nn.Module,
      batch_size: int = 8,
      logdir: str = "log",
      runid: str = "train",
      loglevel: int = logging.DEBUG,
      label2name: dict[int, str] = cifar_label2name,
      debug: bool = True,
  ):
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
    self.model = model
    self.logger.info(f" --- Starting new run {runid} --- ")
    self.training_loss: list[float] = []
    self.steps = 0
    self.epoch = 0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    self.test_batch = next(iter(test_loader))
    self.device = device
    self.criterion = criterion
    self.test_batch_titles: list[str] = []
    self.test_batch_preds: list[np.ndarray] = []
    self.valid_losses: list[float] = []
    self.valid_accs: list[float] = []
    self.valid_topk_accs: list[float] = []
    self.plot_interval = int(len(test_dataset) / 4)  # type: ignore
    self.debug = debug
    self.forward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.backward_hooks: dict[nn.Module, torch.utils.hooks.RemovableHandle] = {}
    self.hook_fig: None | plt.Figure = None
    self.hook_axes: None | plt.Axes = None
    self.add_hooks()

  def add_hooks(self):
    if self.hook_fig:
      self.logger.info("Logging activation inputs and gradients")
      self.hook_fig.tight_layout()
      self.savefig(self.hook_fig, self.path / f"epoch-{self.epoch}-activations-and-gradients.png")
      plt.close(self.hook_fig)
    i = 1
    for module in self.model.modules():
      if isinstance(module, nn.ReLU):
        self.forward_hooks[module] = module.register_forward_hook(
            self.plot_inputs(f"{i}th ReLU Layer Inputs", i - 1))
        self.backward_hooks[module] = module.register_full_backward_hook(
            self.plot_gradients(f"{i}th ReLu Layer Gradients", i - 1))
        i += 1
    self.hook_fig, self.hook_axes = plt.subplots(nrows=i - 1, ncols=2)

  def plot_inputs(self, title: str, row: int):

    def hook(module, input, output):
      path = title.replace(" ", "-") + f"-epoch-{self.epoch}.png"
      path = self.path / path
      numbers = input[0].detach().cpu().flatten().numpy()
      self.hook_axes[row, 0].hist(numbers, bins=50)
      self.hook_axes[row, 0].set_title(title)
      if module in self.forward_hooks:
        self.forward_hooks[module].remove()
        self.forward_hooks.pop(module)

    return hook

  def plot_gradients(self, title: str, row: int):

    def hook(module, grad_input, grad_output):
      path = title.replace(" ", "-") + f"-epoch-{self.epoch}.png"
      path = self.path / path
      numbers = grad_input[0].detach().cpu().flatten().numpy()
      self.hook_axes[row, 1].hist(numbers, bins=50)
      self.hook_axes[row, 1].set_title(title)
      if module in self.backward_hooks:
        self.backward_hooks[module].remove()
        self.backward_hooks.pop(module)

    return hook

  def savefig(self, fig: plt.Figure, path: Path):
    copypath = path.with_name("panding.png")
    fig.savefig(copypath, dpi=600)
    os.rename(copypath, path)

  def first_batch(self, imgs: torch.Tensor, X: torch.Tensor, Y: torch.Tensor, loss: float):
    self.logger.info(f"First loss is {loss:.2f} (should be around {-np.log(1/10):.2f})")

    # visualize input to the model
    num_samples = imgs.shape[0]
    fig, axs = plt.subplots(num_samples, 4, figsize=(12, 2 * num_samples))
    imgs = imgs.permute(0, 2, 3, 1)

    vmin = X.min()
    vmax = X.max()

    for i in range(num_samples):
      # Plot the original image
      axs[i, 0].imshow(imgs[i])
      axs[i, 0].set_title(f"Label: {self.label2name[int(Y[i].item())]}")
      axs[i, 0].axis("off")

      # Plot each channel as a heatmap
      for c in range(3):
        channel_data = X[i][c].cpu().numpy()
        im = axs[i, c + 1].imshow(channel_data, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        axs[i, c + 1].axis("off")
        fig.colorbar(im, ax=axs[i, c + 1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    self.savefig(fig, self.path / "first_batch_model_inputs.png")
    plt.close(fig)

  def add_training_loss(self, loss: float):
    self.steps += 1
    if self.debug is True:
      self.training_loss.append(loss)
    if self.steps % 200 == 0:
      if self.debug is False:
        self.training_loss.append(loss)
      fig = plt.figure()
      ax = fig.add_subplot(1, 1, 1)
      ax.plot(self.training_loss)
      ax.set_title("Training Loss")
      ax.set_ylabel("Loss")
      ax.grid(True)
      self.savefig(fig, self.path / "training_loss.png")
      plt.close(fig)
      self.logger.info(f"After {self.steps} batches of epoch {self.epoch}: loss {loss:.2f}")

      if (length := len(self.training_loss)) > 200:
        # smooth training loss plot
        losses = np.array(self.training_loss)
        losses = losses[:length - (length % 200)]
        losses = losses.reshape(-1, 200).mean(axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(losses)
        ax.set_title("Smoothed Training Loss")
        ax.set_ylabel("Loss")
        ax.grid(True)
        self.savefig(fig, self.path / "smooth_training_loss.png")
        plt.close(fig)

  def end_epoch(self):
    self.epoch += 1
    self.steps = 0
    self.add_hooks()  # visualize activations and gradients at beginning of each epoch

  def eval_test_batch(self, model: nn.Module, title: str):
    imgs, input, target = self.test_batch
    input, target = input.to(self.device), target.to(self.device)
    out = model(input)
    loss = self.criterion(out, target)
    self.logger.info(f"Loss on test batch is {loss.item():.2f}")
    probs = out.cpu().detach().exp().numpy()
    self.test_batch_preds.append(probs)
    self.test_batch_titles.append(title)

    # visualize predictions
    imgs = imgs.permute(0, 2, 3, 1)
    cols = imgs.shape[0]
    rows = len(self.test_batch_preds) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 2))

    for col in range(cols):
      ax = axes[0, col]
      ax.imshow(imgs[col])
      ax.axis("off")

    labels = list(self.label2name.values())

    for i in range(1, rows):
      batchprobs = self.test_batch_preds[i - 1]
      for c in range(cols):
        ax = axes[i, c]
        probs = batchprobs[c]
        ax.bar(range(10), probs, tick_label=labels)
        ax.set_ylim(0, 1)
        ax.set_title(self.test_batch_titles[i - 1], fontsize=10)
        ax.set_xticks(range(10))
        for tick in ax.get_xticklabels():
          tick.set_rotation(45)
          tick.set_ha("right")

    fig.tight_layout()
    self.savefig(fig, self.path / "test_batch_predictions.png")
    plt.close(fig)
    self.logger.debug(f"Append predictions of test batch: {title}")

  def add_validation_loss(self, loss: float, acc: float, topk_acc: float):
    self.valid_accs.append(acc)
    self.valid_losses.append(loss)
    self.valid_topk_accs.append(topk_acc)

    # loss
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(self.valid_losses)
    ax.set_title("Validation Loss")
    ax.set_ylabel("Loss")
    ax.grid(True)
    self.savefig(fig, self.path / "validation_loss.png")
    plt.close(fig)
    self.logger.info(f"After {self.steps} batches of epoch {self.epoch}:")
    self.logger.info(f"\t- validation loss {loss:.2f}")

    # accuracy
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(self.valid_accs)
    ax.set_title("Validation accuracy")
    ax.set_ylabel("Loss")
    ax.grid(True)
    self.savefig(fig, self.path / "validation_acc.png")
    plt.close(fig)
    self.logger.info(f"\t- validation accuracy {acc:.2f}")

    # topk accuracy
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(self.valid_topk_accs)
    ax.set_title("Validation topk accuracy")
    ax.set_ylabel("Loss")
    ax.grid(True)
    self.savefig(fig, self.path / "validation_topk_acc.png")
    plt.close(fig)
    self.logger.info(f"\t- validation topk accuracy {topk_acc:.2f}")

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

  def overfit_single_batch(  # TODO: this should be done with architect
      self,
      model: nn.Module,
      X: torch.Tensor,
      Y: torch.Tensor,
      criterion: nn.Module,
      optimizer: optim.Optimizer,
      n: int = 1000,
  ):
    # clone everything
    self.logger.info("Start overfitting single batch")
    model = clone_model(model)
    criterion = clone_model(criterion)
    optimizer = clone_optimizer(optimizer, model)
    losses = []

    model.train()
    for _ in range(n):
      out = model(X)
      loss = criterion(out, Y)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      losses.append(loss.item())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(losses)
    ax.set_title("Overfitting Batch Loss")
    ax.set_ylabel("Loss")
    ax.grid(True)
    self.savefig(fig, self.path / "overfit_batch_loss.png")
    plt.close(fig)
    self.logger.info(f"Overfitted single batch for {n} iterations leading to loss {losses[-1]:.2f}")
