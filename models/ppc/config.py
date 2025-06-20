from dataclasses import dataclass


@dataclass
class StageConfig:
  cells: int
  channels: int
  operations: int
  epochs: int
  dropout: float
