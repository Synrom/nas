import torch
import random
import argparse
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

from dataset.transform import Cutout
from dataset.cifar import cifar10_means, cifar10_stds
from dataset.wrapper import cifar10
from monitor import Monitor
from utils import clone_model, models_eq
from config import Config
from models.darts.model_search import Network
from models.darts.architect import Architect


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
model = Network(config.init_channels, 10, config.layers, criterion, device)
model.to(device)
optimizer = torch.optim.SGD(model.parameters(),
                            config.learning_rate,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
monitor = Monitor(
    model,
    test_dataset=test_dataset,
    device=device,
    criterion=criterion,
    debug=True,
    runid=config.runid,
    logdir=config.logdir,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       config.epochs,
                                                       eta_min=config.learning_rate_min)

architect = Architect(model, config)


def validate_model(model: nn.Module, criterion: nn.Module, monitor: Monitor):
  model.eval()
  losses, accs, topk_accs = [], [], []
  with torch.no_grad():
    for idx, batch in enumerate(valid_queue):
      imgs, input, target = batch
      input, target = input.to(device), target.to(device)
      logits = model(input)

      # loss
      loss = criterion(logits, target)
      assert logits.shape == torch.Size([config.batch_size, 10])
      losses.append(loss)

      # accuracy
      preds = logits.argmax(dim=1)
      assert preds.shape == target.shape
      acc = torch.sum(preds == target) / target.shape[0]
      accs.append(acc)

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
          epoch: int):
  for idx, (imgs, input_train, target_train) in enumerate(train_queue):
    model.train()
    assert target_train.shape == torch.Size([config.batch_size])

    input_train, target_train = input_train.to(device), target_train.to(device)

    imgs_search, input_search, target_search = next(iter(valid_queue))
    input_search, target_search = input_search.to(device), target_search.to(device)

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
      monitor.first_batch(imgs, input_train, target_train, loss.item())
      monitor.test_data_sharing_inbetween_batch(model, input_train)
      bf_model, bf_criterion = model.new(), clone_model(criterion)
      monitor.overfit_single_batch(model, input_train, target_train, criterion, optimizer, 100)
      assert models_eq(bf_model, model) == False
      assert models_eq(bf_criterion, criterion) == False
      del bf_model
      del bf_criterion

    monitor.add_training_loss(loss.item())
    break  # for debugging


for epoch in range(config.epochs):
  lr = scheduler.get_last_lr()[0]

  # train single batch
  train(model, criterion, monitor, architect, lr, epoch)

  # visualize everything
  monitor.end_epoch()
  model.eval()
  monitor.input_dependent_baseline(model, criterion)
  monitor.eval_test_batch(model, f"After {epoch} epochs")
  validate_model(model, criterion, monitor)

  # save model
  torch.save(model.state_dict(), f"{config.logdir}/{config.runid}/checkpoint-{epoch}-epochs.pt")

  scheduler.step()
