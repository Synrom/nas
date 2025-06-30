import json
import time
import torch
import torch.nn.functional as F
import argparse
import time
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.optim import Optimizer
from pathlib import Path
from dataclasses import asdict
from dataset.transform import Cutout
from dataset.cifar import cifar10_means, cifar10_stds
from dataset.wrapper import cifar10
from monitor.monitor import Monitor
from utils import clone_model, models_eq
from config import PPCSearchConfig, PPCPastTrainRun, add_neglatible_bool_to_parser
from models.ppc.model_search import Network
from models.ppc.config import read_stage_config
from models.ppc.switch import init_switch
from models.darts.architect import Architect
from models.darts.genotypes import save_genotype, PRIMITIVES


def parse_args() -> PPCSearchConfig:
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument("--batch_size", type=int, default=64, help="batch size")
  parser.add_argument("--learning_rate", type=float, default=0.025, help="init learning rate")
  parser.add_argument("--learning_rate_min", type=float, default=0.001, help="min learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
  parser.add_argument("--weight_decay", type=float, default=3e-4, help="weight decay")
  parser.add_argument("--report_freq", type=float, default=50, help="report frequency")
  parser.add_argument("--epochs", type=int, default=50, help="num of training epochs")
  parser.add_argument("--steps", type=int, default=4, help="Number of nodes per cell")
  parser.add_argument("--init_channels", type=int, default=16, help="num of init channels")
  parser.add_argument("--layers", type=int, default=8, help="total number of layers")
  parser.add_argument('--stages', type=str, help='Path to stage configuration')
  parser.add_argument("--cutout", action="store_true", default=False, help="use cutout")
  parser.add_argument("--cutout_length", type=int, default=16, help="cutout length")
  parser.add_argument("--runid", type=str, default="train", help="Run ID of this training run")
  parser.add_argument("--logdir", type=str, default="log", help="Directory to write logs to")
  parser.add_argument("--seed", type=int, default=2, help="random seed")
  parser.add_argument("--grad_clip", type=float, default=5, help="gradient clipping")
  parser.add_argument("--train_portion", type=float, default=0.5, help="portion of training data")
  parser.add_argument("--data_num_workers", type=int, default=2, help="num workers of data loaders")
  parser.add_argument("--past_train", type=str, default=None, help="Optional path to previous train.")
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
  add_neglatible_bool_to_parser(parser, "--np-vis-se-block", "vis_se_block")
  parser.add_argument("--debug", action="store_true", dest="debug")
  parser.set_defaults(debug=False)
  add_neglatible_bool_to_parser(parser, "--no-vis-alphas", "vis_alphas")
  add_neglatible_bool_to_parser(parser, "--no-vis-genotypes", "vis_genotypes")
  add_neglatible_bool_to_parser(parser, "--no-vis-lrs", "vis_lrs")
  add_neglatible_bool_to_parser(parser, "--no-vis-eigenvalues", "vis_eigenvalues")
  parser.add_argument("--vis_interval",
                      type=int,
                      default=200,
                      help="Interval to visualize training loss")
  args = parser.parse_args()
  return PPCSearchConfig(**vars(args))


def validate_model(model: Network, criterion: nn.Module, monitor: Monitor,
                   valid_queue: DataLoader) -> tuple[float, float, float]:
  """
  Run model on valid_queue and report results to monitor.
  """
  model.eval()
  losses, accs, topk_accs = [], [], []
  monitor.logger.info("Validating model ...")
  with torch.no_grad():
    for idx, batch in enumerate(valid_queue):
      imgs, input, target = batch
      input, target = input.to(model.device), target.to(model.device)
      logits = model(input)

      # loss
      loss = criterion(logits, target)
      losses.append(loss.item())

      # accuracy
      preds = logits.argmax(dim=1)
      assert preds.shape == target.shape
      acc = torch.sum(preds == target) / target.shape[0]
      accs.append(acc.item())

      # topk-accuracy
      _, topk_preds = logits.topk(k=5, dim=1)
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


def train(model: Network,
          criterion: nn.Module,
          monitor: Monitor,
          architect: Architect,
          lr: float,
          epoch: int,
          optimizer: Optimizer,
          stage: int,
          train_alphas: bool = True):
  monitor.logger.info(f"Train {epoch} epoch")
  for idx, (imgs, input_train, target_train) in enumerate(train_queue):
    model.train()

    input_train, target_train = input_train.to(model.device), target_train.to(model.device)

    imgs_search, input_search, target_search = next(iter(valid_queue))
    input_search, target_search = input_search.to(model.device), target_search.to(model.device)

    if train_alphas is True:
      architect.step(input_train,
                     target_train,
                     input_search,
                     target_search,
                     lr,
                     optimizer,
                     unrolled=config.unrolled)

    optimizer.zero_grad()

    loss = model._loss(input_train, target_train)
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
                                     criterion, optimizer, alpha_optimizer, config, lr, 100)

        # model and critertion should not have changed
        assert models_eq(bf_model, model) == True
        assert models_eq(bf_criterion, criterion) == True
        del bf_model
        del bf_criterion

    if idx % config.vis_interval == 0:
      monitor.logger.info(f"After {idx} steps of {epoch} epoch {stage} stage: {loss.item()}")

    if idx == 0:  # at beginning of each epoch
      if config.vis_eigenvalues:
        monitor.visualize_eigenvalues(input_search, target_search, architect)

    monitor.add_training_loss(loss.item())


if __name__ == '__main__':

  start_time = time.time()

  config = parse_args()

  supports_bf16 = False
  print(f"Torch is available: {torch.cuda.is_available()}")
  if torch.backends.mps.is_available():
    device = torch.device("mps")
    device_str = "mps"
  else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
      supports_bf16 = torch.cuda.is_bf16_supported()
      if supports_bf16:
        print("Using bfloat16")
  print(f"Device is {device}")

  torch.backends.cudnn.benchmark = True
  torch.backends.cudnn.enabled = True

  # set random seed
  #torch.manual_seed(config.seed)
  #torch.cuda.manual_seed(config.seed)
  #torch.cuda.manual_seed_all(config.seed)
  #np.random.seed(config.seed)
  #random.seed(config.seed)

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

  stages = read_stage_config(Path(config.stages))
  criterion = nn.CrossEntropyLoss()
  reduction_ratio = 2

  if config.past_train is not None:
    with open(config.past_train) as fstream:
      print("Loading checkpoint ...")
      past = PPCPastTrainRun(**json.load(fstream))
      start_epoch = past.epoch + 1
      start_stage = past.stage
      stage = stages[start_stage]
      print(f"Beginning with epoch {start_epoch}")
      model = Network.load_from_file(Path(past.checkpoint))
      optimizer = torch.optim.SGD(model.parameters(),
                                  config.learning_rate,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             config.epochs,
                                                             eta_min=config.learning_rate_min)
      alpha_optimizer = torch.optim.Adam(model.arch_parameters(),
                                         lr=config.arch_learning_rate,
                                         betas=(0.5, 0.999),
                                         weight_decay=config.arch_weight_decay)
      scheduler_states = torch.load(past.scheduler_checkpoint, weights_only=True)
      optimizer.load_state_dict(scheduler_states["optimizer_state"])
      scheduler.load_state_dict(scheduler_states["scheduler_state"])
      alpha_optimizer.load_state_dict(scheduler_states["alpha_optimizer_state"])
  else:
    start_epoch = 0
    start_stage = 0
    stage = stages[start_stage]
    switch_normal = init_switch(config.steps, len(PRIMITIVES), True)
    switch_reduce = init_switch(config.steps, len(PRIMITIVES), True)
    model = Network(C=stage.channels,
                    num_classes=10,
                    layers=stage.cells,
                    criterion=criterion,
                    device=device,
                    switch_normal=switch_normal,
                    switch_reduce=switch_reduce,
                    steps=config.steps,
                    reduction_ratio=reduction_ratio,
                    num_ops=stage.operations,
                    channel_sampling_prob=stage.channel_sampling_prob,
                    dropout_rate=stage.dropout,
                    multiplier=config.steps,
                    stem_multiplier=4)
    optimizer = torch.optim.SGD(model.parameters(),
                                config.learning_rate,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           config.epochs,
                                                           eta_min=config.learning_rate_min)
    alpha_optimizer = torch.optim.Adam(model.arch_parameters(),
                                       lr=config.arch_learning_rate,
                                       betas=(0.5, 0.999),
                                       weight_decay=config.arch_weight_decay)

  print(f"Batch size is {config.batch_size} and we'll need {len(train_queue)} steps")

  model.to(device)
  architect = Architect(model, config, alpha_optimizer)
  monitor = Monitor(model,
                    architect=architect,
                    test_dataset=test_dataset,
                    device=device,
                    epoch=start_epoch,
                    criterion=criterion,
                    debug=config.debug,
                    runid=config.runid,
                    logdir=config.logdir,
                    vis_interval=config.vis_interval,
                    vis_acts_and_grads=config.vis_activations_and_gradients,
                    num_steps_per_epoch=len(train_queue),
                    stage=start_stage)
  monitor.count_nr_parameters(model)

  for stage_idx in range(start_stage, len(stages)):
    stage = stages[stage_idx]
    for epoch in range(start_epoch, stage.epochs + 10):
      print(f"Epoch is {epoch}")
      lr = scheduler.get_lr()[0]

      # train single batch
      train(model,
            criterion,
            monitor,
            architect,
            lr,
            epoch,
            optimizer,
            stage_idx,
            train_alphas=epoch >= 10)

      # visualize everything
      visualize = epoch % config.vis_interval == 0
      model.eval()
      if config.input_dependent_baseline is True:
        monitor.input_dependent_baseline(model, criterion)
      if config.eval_test_batch is True:
        monitor.eval_test_batch(f"At stage {stage_idx} and epoch {epoch}", model)
      if config.live_validate is True:
        validate_model(model, criterion, monitor, valid_queue)
      if config.vis_alphas is True:
        monitor.visualize_alphas(
            F.softmax(model.alphas_normal, dim=1).detach().cpu().numpy(),
            F.softmax(model.alphas_reduce, dim=1).detach().cpu().numpy(), model)
      if config.vis_genotypes is True:
        monitor.visualize_genotypes(model.genotype())
      if config.vis_lrs:
        monitor.visualize_lrs(lr)
      if visualize and config.vis_se_block:
        monitor.visualize_se_blocks(model)

      scheduler.step()

      current_time = time.time()
      stop = False
      if current_time - start_time >= 60 * 45:  # stop after 45 mins
        monitor.logger.info("Restart after half an hour")
        stop = True

      # save model
      if stop or epoch % config.vis_interval == 0 or epoch == stage.epochs + 9:
        model_checkpoint_path = f"{config.logdir}/{config.runid}/checkpoint-{stage_idx}-{epoch}-epochs.pkl"
        model.save_to_file(Path(model_checkpoint_path))
        optimizer_checkpoint_path = f"{config.logdir}/{config.runid}/optimizer.pkl"
        train_checkpoint_path = f"{config.logdir}/{config.runid}/last_run.json"
        monitor.logger.info(f"Save training to {train_checkpoint_path}")
        osd = optimizer.state_dict()
        ssd = scheduler.state_dict()
        asd = alpha_optimizer.state_dict()
        checkpoint = {"optimizer_state": osd, "scheduler_state": ssd, "alpha_optimizer_state": asd}
        torch.save(checkpoint, optimizer_checkpoint_path)
        train_checkpoint = PPCPastTrainRun(epoch=epoch,
                                           checkpoint=model_checkpoint_path,
                                           scheduler_checkpoint=optimizer_checkpoint_path,
                                           stage=stage_idx)
        monitor.logger.info(f"Saving {asdict(train_checkpoint)} to {train_checkpoint_path}")
        with open(train_checkpoint_path, "w") as fstream:
          json.dump(asdict(train_checkpoint), fstream)

      monitor.end_epoch(model, architect, visualize)

      # save genotype
      if epoch == config.epochs - 1:
        genotype_path = monitor.path / f"{config.runid}_genotype.json"
        save_genotype(genotype_path, model.genotype())
        monitor.logger.info(f"Saved resulting genotype to {genotype_path.as_posix()}.")

      if visualize or stop:
        monitor.commit()

      if stop:
        exit()

    if stage_idx + 1 < len(stages):
      stage = stages[stage_idx + 1]
      model = model.transfer_to_stage(stage, stage_idx == len(stages) - 1)
      optimizer = torch.optim.SGD(model.parameters(),
                                  config.learning_rate,
                                  momentum=config.momentum,
                                  weight_decay=config.weight_decay)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             config.epochs,
                                                             eta_min=config.learning_rate_min)
      alpha_optimizer = torch.optim.Adam(model.arch_parameters(),
                                         lr=config.arch_learning_rate,
                                         betas=(0.5, 0.999),
                                         weight_decay=config.arch_weight_decay)
      monitor.reset_hook(model, architect)
      monitor.next_stage(model, stage)
    else:
      monitor.commit()
    start_epoch = 0
