from dataclasses import dataclass, asdict
from pathlib import Path
import json
import os
from dacite import from_dict


@dataclass
class SingleVisualizationConfig:
  title: str | None
  path: str


@dataclass
class Row:
  title: str | None
  visualizations: list[SingleVisualizationConfig]


@dataclass
class GridConfig:
  title: str
  rows: list[Row]


@dataclass
class DashboardConfig:
  title: str


def read_dashboard_config(directory: Path) -> DashboardConfig:
  path = directory / "dashboard.json"
  return DashboardConfig(**json.loads(path.read_text()))


def write_dashboard_config(directory: Path, config: DashboardConfig):
  path = directory / "dashboard.json"
  path.write_text(json.dumps(asdict(config)))


def write_single_visualization(path: Path, config: SingleVisualizationConfig):
  path.write_text(json.dumps(asdict(config)))


def write_grid_visualization(path: Path, config: GridConfig):
  path.write_text(json.dumps(asdict(config)))


def read_grid_visualization(path: Path) -> GridConfig:
  return from_dict(data_class=GridConfig, data=json.loads(path.read_text()))


def single_visualizations(directory: Path) -> list[SingleVisualizationConfig]:
  path = directory / "single"
  vis: list[SingleVisualizationConfig] = []
  for filename in os.listdir(path.as_posix()):
    filepath = path / filename
    vis.append(from_dict(data_class=SingleVisualizationConfig, data=json.loads(filepath.read_text())))
  return vis


def grid_visualizations(directory: Path) -> list[GridConfig]:
  path = directory / "grid"
  vis: list[GridConfig] = []
  for filename in os.listdir(path.as_posix()):
    filepath = path / filename
    vis.append(from_dict(data_class=GridConfig, data=json.loads(filepath.read_text())))
  return vis
