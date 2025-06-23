from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class StageConfig:
  cells: int
  channels: int
  operations: int
  epochs: int
  dropout: float
  channel_sampling_prob: float


def read_stage_config(path: Path) -> list[StageConfig]:
  with open(path.as_posix(), "r") as fstream:
    raw = json.load(fstream)
  return [StageConfig(**item) for item in raw]
