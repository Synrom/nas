import os
from typing import TypeVar, Generic
from pathlib import Path
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from monitor.plot import Plot, Grid

T = TypeVar('T', bound=Plot)


class Live(Generic[T]):
  """
  Parent class to update plots live during training.
  """

  def __init__(self, path: Path, plot: T, init_data: np.ndarray | None = None):
    self.data_path = path.with_name(f"{path.name}.data")
    if self.data_path.exists() and init_data is None:
      self.data = np.load(self.data_path.as_posix())
    else:
      self.data = init_data
    self.path = path
    self.plot = plot

  def savefig(self, fig: plt.Figure):
    """
    Saves figure to self.path
    """
    copypath = self.path.with_name("panding.png")
    fig.savefig(copypath, dpi=600, bbox_inches='tight')
    os.rename(copypath, self.path)

  def commit(self):
    """
    Create and save plot with current data.
    """
    fig = plt.figure()
    fig = self.plot.plot(self.data, fig)
    self.savefig(fig)
    plt.close(fig)
    np.save(self.data_path, self.data)

  def add(self, item: np.ndarray, axis: int = 0):
    if self.data is None:
      self.data = item
      return
    self.data = np.concatenate((self.data, item), axis=axis)


class LiveGrid(Live):
  """
  Live for Grids
  """

  def __init__(self, path: Path, grid: Grid):
    self.path = path
    self.plot = grid
    self.rows = self.plot.rows
    self.cols = self.plot.cols
    self.grid_data: list[list[np.ndarray | None]] = [[None for c in range(self.cols)]
                                                     for r in range(self.rows)]

  def add_idx(self, item: np.ndarray, row: int, col: int, title: str | None = None):
    self.grid_data[row][col] = item
    if title is not None:
      self.plot.titles[(row, col)] = title

  def add_row(self) -> int:
    self.plot.rows += 1
    self.rows += 1
    self.grid_data = self.grid_data + [[None for c in range(self.cols)]]
    return self.plot.rows - 1

  def commit(self):
    fig, axes = plt.subplots(self.rows,
                             self.cols,
                             squeeze=False,
                             figsize=(self.plot.col_size * self.cols, self.plot.row_size * self.rows))
    for row in range(self.rows):
      for col in range(self.cols):
        if (row, col) in self.plot.manual:
          self.plot.manual[(row, col)](fig, axes[row, col])
        elif self.grid_data[row][col] is not None:
          self.plot.default.plot(self.grid_data[row][col], fig, axes[row, col])
        if (row, col) in self.plot.titles:
          axes[row, col].set_title(self.plot.titles[(row, col)])
    fig.tight_layout()
    self.savefig(fig)
    plt.close(fig)


def nparray(v: float) -> np.ndarray:
  return np.array([v])
