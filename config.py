from dataclasses import dataclass
import argparse


@dataclass
class PastTrainRun:
  epoch: int
  checkpoint: str
  scheduler_checkpoint: str


@dataclass
class PPCPastTrainRun:
  epoch: int
  checkpoint: str
  scheduler_checkpoint: str
  stage: int


@dataclass
class DartsSearchConfig:
  momentum: float = 0.9
  weight_decay: float = 3e-4
  arch_learning_rate: float = 3e-4
  arch_weight_decay: float = 1e-3
  unrolled: bool = False
  batch_size: int = 64
  learning_rate: float = 0.025
  learning_rate_min: float = 0.001
  report_freq: int = 50
  epochs: int = 50
  init_channels: int = 16
  layers: int = 8
  model_path: str = "saved_models"
  cutout: bool = False
  cutout_length: int = 16
  drop_path_prob: float = 0.3
  save: str = "EXP"
  seed: int = 2
  grad_clip: float = 5
  train_portion: float = 0.5
  runid: str = "train"
  logdir: str = "log"
  data_num_workers: int = 2
  vis_activations_and_gradients: bool = True
  debug: bool = False
  vis_first_batch_inputs: bool = True
  test_data_sharing_inbetween_batch: bool = True
  overfit_single_batch: bool = True
  input_dependent_baseline: bool = True
  eval_test_batch: bool = True
  vis_interval: int = 200
  live_validate: bool = True
  vis_alphas: bool = True
  vis_genotypes: bool = True
  vis_lrs: bool = True
  vis_eigenvalues: bool = True
  past_train: str | None = None
  gelu: bool = False


@dataclass
class EvalConfig:
  genotype: str
  momentum: float = 0.9
  weight_decay: float = 3e-4
  batch_size: int = 64
  learning_rate: float = 0.025
  epochs: int = 50
  init_channels: int = 16
  layers: int = 8
  cutout: bool = False
  cutout_length: int = 16
  drop_path_prob: float = 0.3
  seed: int = 0
  grad_clip: float = 5
  runid: str = "eval"
  logdir: str = "log"
  data_num_workers: int = 2
  vis_activations_and_gradients: bool = True
  debug: bool = False
  vis_first_batch_inputs: bool = True
  test_data_sharing_inbetween_batch: bool = True
  overfit_single_batch: bool = True
  input_dependent_baseline: bool = True
  eval_test_batch: bool = True
  vis_interval: int = 200
  live_validate: bool = True
  vis_lrs: bool = True
  past_train: str | None = None
  count_params: bool = True
  auxiliary: bool = False
  auxiliary_weight: float = 0.4
  dropout: float = 0.5
  time_hours: int = 24
  gelu: bool = False
  cifar100: bool = False
  cut_aux_loss: int = 600


def add_neglatible_bool_to_parser(parser: argparse.ArgumentParser, cmd: str, name: str):
  parser.add_argument(cmd, action="store_false", dest=name)
  parser.set_defaults(**{name: True})


@dataclass
class PPCSearchConfig:
  stages: str
  momentum: float = 0.9
  weight_decay: float = 3e-4
  arch_learning_rate: float = 3e-4
  arch_weight_decay: float = 1e-3
  unrolled: bool = False
  batch_size: int = 128
  learning_rate: float = 0.025
  learning_rate_min: float = 0.001
  report_freq: int = 50
  epochs: int = 50
  init_channels: int = 16
  layers: int = 8
  cutout: bool = False
  cutout_length: int = 16
  seed: int = 2
  grad_clip: float = 5
  train_portion: float = 0.5
  runid: str = "train"
  logdir: str = "log"
  data_num_workers: int = 2
  vis_activations_and_gradients: bool = True
  debug: bool = False
  vis_first_batch_inputs: bool = True
  test_data_sharing_inbetween_batch: bool = True
  overfit_single_batch: bool = True
  input_dependent_baseline: bool = True
  eval_test_batch: bool = True
  vis_interval: int = 200
  vis_epoch_interval: int = 200
  live_validate: bool = True
  vis_alphas: bool = True
  vis_genotypes: bool = True
  vis_lrs: bool = True
  vis_eigenvalues: bool = True
  past_train: str | None = None
  steps: int = 4
  vis_se_block: bool = True
  fair: bool = False
  aux_loss_weight: float = 10.0
  gelu: bool = False
  time_hours: int = 24