"""
See plot as function that visualizes numpy arrays.
"""

from typing import Callable, Literal, Protocol, TypeVar, Generic
from abc import ABC, abstractmethod
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from dataclasses import dataclass
from graphviz import Digraph
from PIL import Image as PILImage
import io


class Plot(ABC):

  @abstractmethod
  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    """
    Visualize data.
    """

  def fig_and_axes(self, fig: Figure | None = None, axes: Axes | None = None) -> tuple[Figure, Axes]:
    if fig is None:
      fig = plt.figure()
    if axes is None:
      axes = fig.add_subplot(1, 1, 1)
    return fig, axes


class TwoLines(Plot):

  def __init__(self,
               label1: str | None = None,
               label2: str | None = None,
               title: str | None = None,
               grid: bool = False,
               ylabel: str | None = None,
               num_steps_per_epoch: int | None = None):
    self.label1 = label1
    self.label2 = label2
    self.title = title
    self.grid = grid
    self.ylabel = ylabel
    self.num_steps_per_epoch = num_steps_per_epoch

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    fig, axes = self.fig_and_axes(fig, axes)
    assert len(data.shape) > 1
    assert data.shape[0] == 2
    if self.label1:
      axes.plot(np.arange(len(data[0])), data[0], label=self.label1)
    else:
      axes.plot(data[0])
    if self.label2:
      axes.plot(np.arange(len(data[1])), data[1], label=self.label2)
    else:
      axes.plot(data[1])
    if self.title:
      axes.set_title(self.title)
    if self.ylabel:
      axes.set_ylabel(self.ylabel)
    axes.grid(self.grid)
    if self.num_steps_per_epoch is not None:
      for epoch in range(1, data.shape[1] // self.num_steps_per_epoch):
        step = epoch * self.num_steps_per_epoch
        plt.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
        plt.text(step, plt.ylim()[1], f'Epoch {epoch}', rotation=90, va='top', ha='right', fontsize=8)
    if self.label1 or self.label2:
      fig.legend()
    return fig


class Line(Plot):

  def __init__(self, title: str | None = None, ylabel: str | None = None, grid: bool = False):
    self.title = title
    self.ylabel = ylabel
    self.grid = grid

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    fig, axes = self.fig_and_axes(fig, axes)
    axes.plot(np.arange(len(data)), data)
    if self.title:
      axes.set_title(self.title)
    if self.ylabel:
      axes.set_ylabel(self.ylabel)
    axes.grid(self.grid)
    return fig


class LoadedPlotClass(Protocol):

  def __call__(self, fig: Figure | None, ax: Axes | None) -> Figure:
    ...


LoadedPlot = Callable[[Figure | None, Axes | None], Figure] | LoadedPlotClass


def plot_load_data(plot: Plot, data: np.ndarray) -> LoadedPlot:

  def visualize(fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    return plot.plot(data, fig, axes)

  return visualize


@dataclass
class ColorMapOptions:
  fraction: float
  pad: float


class Image(Plot):

  def __init__(self,
               cmap: str | None = None,
               aspect: Literal['equal', 'auto'] | float | None = None,
               vmin: float | None = None,
               vmax: float | None = None,
               colormap: ColorMapOptions | None = None):
    self.cmap = cmap
    self.aspect = aspect
    self.vmin = vmin
    self.vmax = vmax
    self.colormap = colormap

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    fig, axes = self.fig_and_axes(fig, axes)
    img = axes.imshow(data, cmap=self.cmap, aspect=self.aspect, vmin=self.vmin, vmax=self.vmax)
    axes.axis("off")
    if self.colormap is not None:
      fig.colorbar(img, ax=axes, fraction=self.colormap.fraction, pad=self.colormap.fraction)
    return fig


class Bar(Plot):

  def __init__(self,
               labels: list[str] | None = None,
               ylim: tuple[int, int] | None = None,
               title: str | None = None,
               xticks: list[int] | None = None,
               rotation: float | None = None):
    self.labels = labels
    self.ylim = ylim
    self.title = title
    self.xticks = xticks
    self.rotation = rotation

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    fig, axes = self.fig_and_axes(fig, axes)
    if self.labels is not None:
      assert len(data) == len(self.labels)
      axes.bar(range(len(data)), data, tick_label=self.labels)
    else:
      axes.bar(range(len(data)), data)
    if self.ylim:
      axes.set_ylim(self.ylim[0], self.ylim[1])
    if self.title:
      axes.set_title(self.title)
    if self.xticks:
      axes.set_xticks(self.xticks)
    if self.rotation:
      for tick in axes.get_xticklabels():
        tick.set_rotation(self.rotation)
        tick.set_horizontalalignment("right")
    return fig


T = TypeVar('T', bound=Plot)


class Grid(Generic[T]):

  def __init__(self,
               default: Plot,
               rows: int = 0,
               cols: int = 0,
               row_size: int = 2,
               col_size: int = 3):
    self.default = default
    self.rows = rows
    self.cols = cols
    self.row_size = row_size
    self.col_size = col_size
    # save manually set grid cells
    self.manual: dict[tuple[int, int], LoadedPlot] = {}
    self.titles: dict[tuple[int, int], str] = {}

  def plot(self, data: np.ndarray, fig: Figure | None = None, ax: Axes | None = None) -> Figure:
    assert len(data.shape) > 2
    assert data.shape[0] >= self.rows
    assert data.shape[1] >= self.cols
    if fig is None:
      fig = plt.figure(figsize=(self.cols * self.col_size, self.rows * self.row_size))
    axes = fig.subplots(self.rows, self.cols)
    for row in range(self.rows):
      for col in range(self.cols):
        if (row, col) in self.manual:
          fig = self.manual[(row, col)](fig, axes[row, col])
        else:
          fig = self.default.plot(data[row, col], fig, axes[row, col])
        if (row, col) in self.titles:
          axes[row, col].set_title(self.titles[(row, col)])
    fig.tight_layout()
    return fig


class Hist(Plot):

  def __init__(self, bins: int):
    self.bins = bins

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    fig, axes = self.fig_and_axes(fig, axes)
    axes.hist(data, bins=50)
    return fig


class VisAlpha(Plot):

  def __init__(self, steps: int, primitives: list[str]):
    self.steps = steps
    self.primitives = primitives
    self.k = sum(1 for i in range(steps) for n in range(2 + i))
    self.num_ops = len(self.primitives)
    self.colors = plt.cm.tab10.colors[:self.num_ops]  # type: ignore

  def draw_graph_subplot(self, ax: Axes, step: int, weights: np.ndarray):
    G = nx.DiGraph()
    input_nodes = ["Input 0", "Input 1"]
    input_nodes += [f"Node {j}" for j in range(2, step)]
    output_node = f"Node {step}"

    # Position input nodes evenly spaced horizontally, output node below center
    positions: dict[str, tuple[int, int]] = {}
    spacing = 100  # horizontal spacing between input nodes
    for idx, node in enumerate(input_nodes):
      positions[node] = (idx * spacing, 1)
    positions[output_node] = ((step - 2) * spacing // 2, 0)

    # Add nodes and edges
    for node in input_nodes:
      G.add_node(node)
      G.add_edge(node, output_node, weight=weights[input_nodes.index(node)])
    G.add_node(output_node)

    # Draw graph
    nx.draw(G,
            ax=ax,
            pos=positions,
            with_labels=True,
            node_size=2500,
            node_color='lightblue',
            arrows=True)
    bar_plot_width, bar_plot_height = 1.5, 1

    # Annotate edges with bars and weight values
    for (u, v, d) in G.edges(data=True):
      weight = d["weight"]
      x = (positions[u][0] + positions[v][0]) / 2
      y = (positions[u][1] + positions[v][1]) / 2
      inset = inset_axes(ax,
                         width=bar_plot_width,
                         height=bar_plot_height,
                         loc='center',
                         bbox_to_anchor=(x, y),
                         bbox_transform=ax.transData,
                         borderpad=0)

      # Draw the bar plot in the inset
      inset.bar(range(len(weight)), weight, color=self.colors)
      inset.set_xticks([])
      inset.set_yticks([])
      inset.set_ylim(0, 1)
      inset.grid(True, axis='y', linestyle='--', alpha=0.5)

    ax.set_title(f"Node {step}", fontsize=20)
    ax.axis("off")

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    # Create one subplot per step
    fig, ax = plt.subplots(data.shape[0],
                           self.steps,
                           squeeze=False,
                           figsize=(25 * self.steps, 7 * data.shape[0]),
                           gridspec_kw={'hspace': 0.5})

    for row in range(data.shape[0]):
      offset = 0
      for col in range(self.steps):
        self.draw_graph_subplot(ax[row, col],
                                step=col + 2,
                                weights=data[row][offset:offset + col + 2])
        offset += col + 2

    for i in range(data.shape[0]):
      y_pos = 0.92 - i * 0.48  # Adjust spacing as needed based on figure size
      fig.text(0.5, y_pos, f"{i}th Epoch", va='center', ha='left', fontsize=25, fontweight='bold')

    legend_labels = self.primitives
    handles = [
        mpatches.Patch(color=self.colors[i], label=legend_labels[i]) for i in range(self.num_ops)
    ]
    fig.legend(handles=handles, loc='upper center', ncol=self.num_ops, bbox_to_anchor=(0.5, 1.05))

    return fig


class GenotypeGraph(Plot):

  def __init__(self):
    self.dtype = np.dtype([('name', 'U10'), ('count', 'i4')])

  def convert_array_to_genotype(self, data: np.ndarray) -> list[tuple[str, int]]:
    assert data.dtype == self.dtype
    return [(str(i["name"]), int(i["count"])) for i in data]

  def convert_genotype_to_array(self, genotype: list[tuple[str, int]]) -> np.ndarray:
    return np.array(genotype, dtype=self.dtype)

  def graph(self, genotype: list[tuple[str, int]]) -> Digraph:
    g = Digraph(format='pdf',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled',
                               shape='rect',
                               align='center',
                               fontsize='20',
                               height='0.5',
                               width='0.5',
                               penwidth='2',
                               fontname="times"),
                engine='dot')
    g.body.extend(['rankdir=LR'])

    g.node("c_{k-2}", fillcolor='darkseagreen2')
    g.node("c_{k-1}", fillcolor='darkseagreen2')
    assert len(genotype) % 2 == 0
    steps = len(genotype) // 2

    for i in range(steps):
      g.node(str(i), fillcolor='lightblue')

    for i in range(steps):
      for k in [2 * i, 2 * i + 1]:
        op, j = genotype[k]
        if j == 0:
          u = "c_{k-2}"
        elif j == 1:
          u = "c_{k-1}"
        else:
          u = str(j - 2)
        v = str(i)
        g.edge(u, v, label=op, fillcolor="gray")

    g.node("c_{k}", fillcolor='palegoldenrod')
    for i in range(steps):
      g.edge(str(i), "c_{k}", fillcolor="gray")
    return g

  def plot(self, data: np.ndarray, fig: Figure | None = None, axes: Axes | None = None) -> Figure:
    if fig is None or axes is None:
      fig, axes = plt.subplots(1, 2, figsize=(6, 4))
    genotype = self.convert_array_to_genotype(data)
    g = self.graph(genotype)
    img_bytes = g.pipe(format='png')
    image = PILImage.open(io.BytesIO(img_bytes))
    axes.imshow(image)  # type: ignore
    axes.axis('off')  # type: ignore
    return fig
