import os
import json
import pickle
from typing import TypeVar, Generic
from pathlib import Path
import matplotlib
from matplotlib.patches import Patch

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np

from monitor.plot import Plot, Grid
from dashboard.config import SingleVisualizationConfig, GridConfig, write_single_visualization, write_grid_visualization, read_grid_visualization, Row

T = TypeVar('T', bound=Plot)


class Live(Generic[T]):
  """
  Parent class to update plots live during training.
  """

  def __init__(self,
               title: str,
               directory: Path,
               plot: T,
               init_data: np.ndarray | None = None,
               data_path: Path | None = None):
    self.data_path = data_path if data_path is not None else directory / f"{title}.npy"
    self.marker_path = directory / f"{title}_markers.json"
    self.path = directory / f"{title}.png"
    self.config_path = directory / "single" / f"{title}.json"
    self.config_path.parent.mkdir(parents=True, exist_ok=True)
    self.title = title
    if self.marker_path.exists():
      with open(self.marker_path.as_posix(), "r") as fstream:
        self.markers: list[tuple[int, str]] = json.load(fstream)
    else:
      self.markers = []
    if self.data_path.exists() and init_data is None:
      self.data = np.load(self.data_path.as_posix())
    else:
      self.data = init_data
    self.plot = plot

  def add_marker(self, title: str):
    idx = self.data.shape[-1] - 1
    self.markers.append((idx, title))

  def savefig(self, fig: plt.Figure):
    """
    Saves figure to self.path
    """
    copypath = self.path.with_name("panding.png")
    fig.savefig(copypath, dpi=600, bbox_inches='tight')
    os.rename(copypath, self.path)
    write_single_visualization(self.config_path,
                               SingleVisualizationConfig(self.title, self.path.as_posix()))

  def commit(self):
    """
    Create and save plot with current data.
    """
    if self.data is None:
      return
    fig = plt.figure()
    fig, axes = self.plot.plot(self.data, fig)
    for step, title in self.markers:
      axes.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
      ylim = axes.get_ylim()
      axes.text(step, ylim[1], title, rotation=90, va='top', ha='right', fontsize=8)
    self.savefig(fig)
    plt.close(fig)
    np.save(self.data_path, self.data)
    with open(self.marker_path.as_posix(), "w") as fstream:
      json.dump(self.markers, fstream)

  def add(self, item: np.ndarray, axis: int = 0):
    if self.data is None:
      self.data = item
      return
    self.data = np.concatenate((self.data, item), axis=axis)


T2 = TypeVar('T2', bound='Plot')


class LiveGrid:
  """
  Live for Grids
  """

  def __init__(self, title: str, directory: Path):
    self.config = GridConfig(title, [Row(None, [])])
    self.title = title
    self.directory = directory
    self.config_path = directory / "grid" / f"{title}.json"
    self.config_path.parent.mkdir(parents=True, exist_ok=True)
    if self.config_path.exists():
      self.config = read_grid_visualization(self.config_path)
      self.next_row(None)

  def add_entry(self, fig: Figure, title: str | None = None):
    path = self.directory / f"{self.title}_{title}.png"
    counter = 0
    while path.exists():
      path = self.directory / f"{self.title}_{title}-{counter}.png"
      counter += 1
    copypath = path.with_name("panding.png")
    fig.savefig(copypath, dpi=600, bbox_inches='tight')
    os.rename(copypath, path)
    self.config.rows[-1].visualizations.append(SingleVisualizationConfig(title, path.as_posix()))
    write_grid_visualization(self.config_path, self.config)
    plt.close(fig)

  def next_row(self, title: str | None):
    self.config.rows.append(Row(title, []))


def nparray(v: float) -> np.ndarray:
  return np.array([v])
