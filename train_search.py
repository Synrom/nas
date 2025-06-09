import torch
import torch.nn.functional as F
import random
import argparse
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path

from dataset.transform import Cutout
from dataset.cifar import cifar10_means, cifar10_stds
from dataset.wrapper import cifar10
from monitor.monitor import Monitor
from utils import clone_model, models_eq
from config import Config
from models.darts.model_search import Network
from models.darts.architect import Architect


def add_neglatible_bool_to_parser(parser: argparse.ArgumentParser, cmd: str, name: str):
  parser.add_argument(cmd, action="store_false", dest=name)
  parser.set_defaults(**{name: True})


def parse_args() -> Config:
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument("--batch_size", type=int, default=64, help="batch size")
  parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
  parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
  parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
  parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
  parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
  parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
  parser.add_argument("--layers", type=int, default=8, help="total number of layers")
  parser.add_argument("--model_path", type=str, default="saved_models", help="path to save the model")
  parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
  parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
  parser.add_argument("--drop_path_prob", type=float, default=0.3, help="drop path probability")
  parser.add_argument("--runid", type=str, default="train", help="Run ID of this training run")
  parser.add_argument("--logdir", type=str, default="log", help="Directory to write logs to")
  parser.add_argument("--seed", type=int, default=2, help="random seed")
  parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
  parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
  parser.add_argument("--data_num_workers", type=int, default=2, help="num workers of data loaders")
  parser.add_argument("--checkpoint", type=str, help="path to checkpoint pickle-file")
  parser.add_argument("--unrolled",
                      action="store_true",
                      default=False,
                      help="use one-step unrolled validation loss")
  parser.add_argument("--arch_learning_rate",
                      type=float,
                      default=3e-4,
                      help="learning rate for arch encoding")
  parser.add_argument("--arch_weight_decay",
                      type=float,
                      default=1e-3,
                      help="weight decay for arch encoding")
  add_neglatible_bool_to_parser(parser, "--no-vis-acts-and-grads", "vis_activations_and_gradients")
  add_neglatible_bool_to_parser(parser, "--no-vis-fist-batch-inputs", "vis_first_batch_inputs")
  add_neglatible_bool_to_parser(parser, "--no-test-data-sharing", "test_data_sharing_inbetween_batch")
  add_neglatible_bool_to_parser(parser, "--no-overfit-single-batch", "overfit_single_batch")
  add_neglatible_bool_to_parser(parser, "--no-input-dependent-baseline", "input_dependent_baseline")
  add_neglatible_bool_to_parser(parser, "--no-eval-test-batch", "eval_test_batch")
  add_neglatible_bool_to_parser(parser, "--no-live-validate", "live_validate")
  add_neglatible_bool_to_parser(parser, "--debug", "debug")
  add_neglatible_bool_to_parser(parser, "--no-save-checkpoint", "save_checkpoint")
  add_neglatible_bool_to_parser(parser, "--no-vis-alphas", "vis_alphas")
  add_neglatible_bool_to_parser(parser, "--no-vis-genotypes", "vis_genotypes")
  add_neglatible_bool_to_parser(parser, "--no-vis-lrs", "vis_lrs")
  add_neglatible_bool_to_parser(parser, "--no-vis-eigenvalues", "vis_eigenvalues")
  parser.add_argument("--vis_interval",
                      type=int,
                      default=200,
                      help="Interval to visualize training loss")
  args = parser.parse_args()
  return Config(**vars(args))


config = parse_args()
config.save = "search-{}-{}".format(config.save, time.strftime("%Y%m%d-%H%M%S"))

if torch.backends.mps.is_available():
  device = torch.device("mps")
else:
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# set random seed
torch.manual_seed(config.seed)
torch.cuda.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
np.random.seed(config.seed)
random.seed(config.seed)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=cifar10_means, std=cifar10_stds),
])
if config.cutout:
  train_transform.transforms.append(Cutout(config.cutout_length))

eval_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=cifar10_means, std=cifar10_stds)])

train_dataset = cifar10(train=True, transform=train_transform, download=True)
test_dataset = cifar10(train=False, transform=eval_transform, download=True)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(config.train_portion * num_train))

train_queue = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=SubsetRandomSampler(indices[:split]),
    pin_memory=False,
    num_workers=config.data_num_workers,
)

valid_queue = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    sampler=SubsetRandomSampler(indices[split:num_train]),
    pin_memory=False,
    num_workers=config.data_num_workers,
)

criterion = nn.CrossEntropyLoss()
if config.checkpoint is not None:
  model = Network.load_from_file(Path(config.checkpoint))
else:
  model = Network(config.init_channels, 10, config.layers, criterion, device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(),
                            config.learning_rate,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
architect = Architect(model, config)
monitor = Monitor(model,
                  architect=architect,
                  test_dataset=test_dataset,
                  device=device,
                  criterion=criterion,
                  debug=config.debug,
                  runid=config.runid,
                  logdir=config.logdir,
                  vis_interval=config.vis_interval,
                  vis_acts_and_grads=config.vis_activations_and_gradients,
                  num_steps_per_epoch=len(train_queue))

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       config.epochs,
                                                       eta_min=config.learning_rate_min)


def validate_model(model: Network, criterion: nn.Module, monitor: Monitor, valid_queue: DataLoader,
                   config: Config) -> tuple[float, float, float]:
  """
  Run model on valid_queue and report results to monitor.
  """
  model.eval()
  losses, accs, topk_accs = [], [], []
  with torch.no_grad():
    for idx, batch in enumerate(valid_queue):
      imgs, input, target = batch
      input, target = input.to(model.device), target.to(model.device)
      logits = model(input)

      # loss
      loss = criterion(logits, target)
      assert logits.shape == torch.Size([config.batch_size, 10])
      losses.append(loss.item())

      # accuracy
      preds = logits.argmax(dim=1)
      assert preds.shape == target.shape
      acc = torch.sum(preds == target) / target.shape[0]
      accs.append(acc.item())

      # topk-accuracy
      _, topk_preds = logits.topk(k=5, dim=1)
      assert topk_preds.shape == torch.Size([config.batch_size, 5])
      topk_acc = 0.0
      for i, p in enumerate(topk_preds):
        topk_acc += float(p in target[i])
      topk_acc /= target.shape[0]
      topk_accs.append(topk_acc)
  mean_loss = np.array(losses).mean()
  mean_acc = np.array(accs).mean()
  mean_topk_acc = np.array(topk_accs).mean()
  monitor.add_validation_loss(mean_loss, mean_acc, mean_topk_acc)
  return mean_loss, mean_acc, mean_topk_acc


def train(model: Network, criterion: nn.Module, monitor: Monitor, architect: Architect, lr: float,
          epoch: int, optimizer: Optimizer):
  for idx, (imgs, input_train, target_train) in enumerate(train_queue):
    model.train()
    assert target_train.shape == torch.Size([config.batch_size])

    input_train, target_train = input_train.to(model.device), target_train.to(model.device)

    imgs_search, input_search, target_search = next(iter(valid_queue))
    input_search, target_search = input_search.to(model.device), target_search.to(model.device)

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

    # do some sanity tests for very first batch
    if epoch == 0 and idx == 0:
      if config.vis_first_batch_inputs is True:
        monitor.first_batch(imgs, input_train, target_train, loss.item())
      if config.test_data_sharing_inbetween_batch is True:
        monitor.test_data_sharing_inbetween_batch(model, input_train)

      # overfit single batch. Save model and criterion to make sure they dont change
      if config.overfit_single_batch is True:
        bf_model, bf_criterion = model.clone(), clone_model(criterion)
        monitor.overfit_single_batch(model, input_train, target_train, input_search, target_search,
                                     criterion, optimizer, config, lr, 100)

        # model and critertion should not have changed
        assert models_eq(bf_model, model) == True
        assert models_eq(bf_criterion, criterion) == True
        del bf_model
        del bf_criterion

    if idx == 0:  # at beginning of each epoch
      if config.vis_eigenvalues:
        monitor.visualize_eigenvalues(input_search, target_search)

    if idx % config.vis_interval == 0:
      if config.vis_lrs:
        monitor.visualize_lrs(lr)

    monitor.add_training_loss(loss.item())


for epoch in range(config.epochs):
  lr = scheduler.get_last_lr()[0]

  # train single batch
  train(model, criterion, monitor, architect, lr, epoch, optimizer)

  # visualize everything
  monitor.end_epoch()
  model.eval()
  if config.input_dependent_baseline is True:
    monitor.input_dependent_baseline(model, criterion)
  if config.eval_test_batch is True:
    monitor.eval_test_batch(f"After {epoch} epochs")
  if config.live_validate is True:
    validate_model(model, criterion, monitor, valid_queue, config)
  if config.vis_alphas is True:
    monitor.visualize_alphas(
        F.softmax(model.alphas_normal, dim=-1).detach().cpu().numpy(),
        F.softmax(model.alphas_reduce, dim=-1).detach().cpu().numpy())
  if config.vis_genotypes is True:
    monitor.visualize_genotypes(model.genotype())

  # save model
  torch.save(model.state_dict(), f"{config.logdir}/{config.runid}/checkpoint-{epoch}-epochs.pt")

  scheduler.step()
