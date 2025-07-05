import json
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import argparse
import torch.nn as nn
from torchvision import transforms
from dataset.transform import Cutout
import torch.backends.cudnn as cudnn
from dataset.cifar import cifar10_means, cifar10_stds
from dataset.wrapper import cifar10
from torch.utils.data import DataLoader

from models.darts.genotypes import load_genotype
from models.darts.model import NetworkCIFAR
from config import EvalConfig, add_neglatible_bool_to_parser, PastTrainRun
from monitor.monitor import Monitor
from utils import clone_model, models_eq


def parse_args() -> EvalConfig:
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--batch_size', type=int, default=96, help='batch size')  # 128 for PPC
  parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
  parser.add_argument("--runid", type=str, default="train", help="Run ID of this training run")
  parser.add_argument('--layers', type=int, default=20, help='total number of layers')
  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
  parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
  parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
  parser.add_argument('--seed', type=int, default=0, help='random seed')
  parser.add_argument('--genotype', type=str, help='Path to genotype')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
  parser.add_argument("--data_num_workers", type=int, default=2, help="num workers of data loaders")
  add_neglatible_bool_to_parser(parser, "--no-vis-acts-and-grads", "vis_activations_and_gradients")
  parser.add_argument("--debug", action="store_true", dest="debug")
  parser.set_defaults(debug=False)
  add_neglatible_bool_to_parser(parser, "--no-vis-fist-batch-inputs", "vis_first_batch_inputs")
  add_neglatible_bool_to_parser(parser, "--no-test-data-sharing", "test_data_sharing_inbetween_batch")
  add_neglatible_bool_to_parser(parser, "--no-overfit-single-batch", "overfit_single_batch")
  add_neglatible_bool_to_parser(parser, "--no-input-dependent-baseline", "input_dependent_baseline")
  add_neglatible_bool_to_parser(parser, "--no-eval-test-batch", "eval_test_batch")
  add_neglatible_bool_to_parser(parser, "--no-count-params", "count_params")
  parser.add_argument("--vis_interval",
                      type=int,
                      default=200,
                      help="Interval to visualize training loss")
  add_neglatible_bool_to_parser(parser, "--no-live-validate", "live_validate")
  add_neglatible_bool_to_parser(parser, "--no-vis-lrs", "vis_lrs")
  parser.add_argument("--past_train", type=str, default=None, help="Optional path to previous train.")

  args = parser.parse_args()
  return EvalConfig(**vars(args))


CIFAR_CLASSES = 10


def train(train_queue: DataLoader, model: NetworkCIFAR, criterion: nn.Module,
          optimizer: torch.optim.Optimizer, monitor: Monitor, epoch: int):
  model.train()
  model.drop_path_prob = config.drop_path_prob * epoch / config.epochs

  for step, (imgs, input, target) in enumerate(train_queue):
    input, target = input.to(model.device), target.to(model.device)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if config.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += config.auxiliary_weight * loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    if epoch == 0 and step == 0:
      if config.vis_first_batch_inputs is True:
        monitor.first_batch(imgs, input, target, loss.item())
      if config.test_data_sharing_inbetween_batch is True:
        monitor.test_data_sharing_inbetween_batch(model, input)
      if config.overfit_single_batch is True:
        bf_model, bf_criterion = model.clone(), clone_model(criterion)
        monitor.overfit_single_batch_eval(model, input, target, criterion, optimizer,
                                          config.grad_clip, 300)
        # model and critertion should not have changed
        assert models_eq(bf_model, model) == True
        assert models_eq(bf_criterion, criterion) == True
        del bf_model
        del bf_criterion
      model.train()

    if step % config.vis_interval == 0:
      monitor.logger.info(f"After {step} steps of {epoch} epoch: {loss.item()}")

    monitor.add_training_loss(loss.item())


def infer(valid_queue: DataLoader, model: NetworkCIFAR, criterion: nn.Module):
  model.eval()
  losses = []
  monitor.logger.info("Validating model ...")
  tp = torch.tensor(0).to(model.device)
  topk_tp = torch.tensor(0).to(model.device)
  nr_samples = 0
  model.drop_path_prob = 0.0

  with torch.no_grad():
    for step, (img, input, target) in enumerate(valid_queue):
      input, target = input.to(model.device), target.to(model.device)

      # loss
      logits, _ = model(input)
      loss = criterion(logits, target)
      losses.append(loss.item())

      # accuracy
      preds = logits.argmax(dim=1)
      assert preds.shape == target.shape
      tp += torch.sum(preds == target)

      # topk-accuracy
      _, topk_preds = logits.topk(k=5, dim=1)
      for i, p in enumerate(topk_preds):
        topk_tp += p in target[i]

      nr_samples += input.shape[0]
  mean_loss = np.array(losses).mean()
  acc = tp / nr_samples
  topk_acc = topk_tp / nr_samples
  monitor.add_validation_loss(mean_loss, acc.item(), topk_acc.item())
  return mean_loss, acc.item(), topk_acc.item()


if __name__ == "__main__":
  start_time = time.time()

  config = parse_args()

  print(f"Torch is available: {torch.cuda.is_available()}")
  if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_str = "mps"
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"Device is {device}")

  cudnn.benchmark = True
  cudnn.enabled = True

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

  train_queue = DataLoader(train_dataset,
                           batch_size=config.batch_size,
                           shuffle=True,
                           pin_memory=False,
                           num_workers=config.data_num_workers)

  valid_queue = DataLoader(test_dataset,
                           batch_size=config.batch_size,
                           shuffle=False,
                           pin_memory=False,
                           num_workers=config.data_num_workers)

  genotype = load_genotype(Path(config.genotype))

  if config.past_train is not None:
    with open(config.past_train) as fstream:
      print("Loading checkpoint ...")
      past = PastTrainRun(**json.load(fstream))
      start_epoch = past.epoch + 1
      print(f"Beginning with epoch {start_epoch}")
      model = NetworkCIFAR.load_from_file(Path(past.checkpoint))
      optimizer = torch.optim.SGD(model.parameters(),
                                  config.learning_rate,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)
      scheduler_states = torch.load(past.scheduler_checkpoint, weights_only=True)
      optimizer.load_state_dict(scheduler_states["optimizer_state"])
      scheduler.load_state_dict(scheduler_states["scheduler_state"])
  else:
    start_epoch = 0
    model = NetworkCIFAR(config.init_channels, CIFAR_CLASSES, config.layers, genotype, device,
                         config.drop_path_prob, config.auxiliary)
    optimizer = torch.optim.SGD(model.parameters(),
                                config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.epochs)

  model.to(device)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.to(device)

  monitor = Monitor(model,
                    test_dataset=test_dataset,
                    device=device,
                    epoch=start_epoch,
                    criterion=criterion,
                    debug=config.debug,
                    runid=config.runid,
                    logdir=config.logdir,
                    vis_interval=config.vis_interval,
                    vis_acts_and_grads=config.vis_activations_and_gradients,
                    num_steps_per_epoch=len(train_queue))

  if config.count_params:
    monitor.count_nr_parameters(model)

  for epoch in range(start_epoch, config.epochs):
    scheduler.step()

    train(train_queue, model, criterion, optimizer, monitor, epoch)

    visualize = (epoch % config.vis_interval) == 0

    model.eval()
    if config.live_validate is True:
      infer(valid_queue, model, criterion)
    if config.eval_test_batch is True and visualize is True:
      monitor.eval_test_batch(f"After {epoch} epochs", model)
    if config.input_dependent_baseline is True:
      monitor.input_dependent_baseline(model, criterion)
    if config.vis_lrs:
      monitor.visualize_lrs(scheduler.get_lr()[0])
    if visualize:
      monitor.smoothed_training_loss.add_marker(f"Epoch {epoch}")
      monitor.training_loss.add_marker(f"Epoch {epoch}")

    stop = False
    current_time = time.time()
    if current_time - start_time >= 60 * 45:  # stop after 45 mins
      monitor.logger.info("Restart after half an hour")
      stop = True

    if stop or visualize or epoch == config.epochs - 1:
      model_checkpoint_path = f"{config.logdir}/{config.runid}/checkpoint-{epoch}-epochs.pkl"
      model.save_to_file(Path(model_checkpoint_path))
      optimizer_checkpoint_path = f"{config.logdir}/{config.runid}/optimizer.pkl"
      train_checkpoint_path = f"{config.logdir}/{config.runid}/last_run.json"
      monitor.logger.info(f"Save training to {train_checkpoint_path}")
      osd = optimizer.state_dict()
      ssd = scheduler.state_dict()
      checkpoint = {"optimizer_state": osd, "scheduler_state": ssd}
      torch.save(checkpoint, optimizer_checkpoint_path)
      train_checkpoint = PastTrainRun(epoch=epoch,
                                      checkpoint=model_checkpoint_path,
                                      scheduler_checkpoint=optimizer_checkpoint_path)
      print(f"Saving {asdict(train_checkpoint)} to {train_checkpoint_path}")
      with open(train_checkpoint_path, "w") as fstream:
        json.dump(asdict(train_checkpoint), fstream)
      monitor.end_epoch(model, visualize=visualize)
      monitor.commit()
    else:
      monitor.end_epoch(model, visualize=visualize)

    if stop:
      break
