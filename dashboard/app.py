import streamlit as st
import sys
from pathlib import Path
from typing import Literal

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))
from PIL import Image

from dashboard.config import read_dashboard_config, single_visualizations, grid_visualizations, GridConfig

# Get argument (e.g., model name)
if len(sys.argv) > 1:
  directory = Path(sys.argv[1])
  st.session_state["directory"] = directory
else:
  st.error("You need to specify a directory!")
  exit()

config = read_dashboard_config(directory)
single_vis = single_visualizations(directory)
grid_vis = grid_visualizations(directory)

st.title(config.title)


def header(centered: bool = False):
  if centered is False:
    cols = st.columns(len(grid_vis) + 1)
    with cols[0]:
      if st.button("Home", key="nav_home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    for vis, col in zip(grid_vis, cols[1:]):
      with col:
        if st.button(vis.title, key=vis.title, use_container_width=True):
          st.session_state.page = "grid"
          st.session_state.grid = vis
          st.rerun()
  else:
    all_buttons: list[Literal["Home"] | GridConfig] = ["Home"] + grid_vis  # type: ignore
    for i in range(0, len(all_buttons) + 1, 4):
      buttons = all_buttons[i:i + 4]
      cols = st.columns(len(buttons))
      for button, col in zip(buttons, cols):
        with col:
          if button == "Home":
            if st.button("Home", key="nav_home", use_container_width=True):
              st.session_state.page = "home"
              st.rerun()
          else:
            if st.button(button.title, key=button.title, use_container_width=True):
              st.session_state.page = "grid"
              st.session_state.grid = button
              st.rerun()


def show_home():
  st.set_page_config(layout="wide")
  header(centered=False)
  for i in range(0, len(single_vis), 2):
    visualizations = single_vis[i:i + 2]
    cols = st.columns(len(visualizations))
    for vis, col in zip(visualizations, cols):
      with col:
        #st.subheader(vis.title)
        img = Image.open(vis.path)
        st.image(img, use_container_width=False)


def show_grid() -> None:
  st.set_page_config(layout="wide")
  vis: GridConfig = st.session_state.grid
  st.title(vis.title)
  header()
  for row in vis.rows:
    if len(row.visualizations) == 0:
      continue
    if row.title is not None:
      st.write(f"## {row.title}")
    cols = st.columns(len(row.visualizations))
    for col, entry in zip(cols, row.visualizations):
      with col:
        if entry.title is not None:
          st.subheader(entry.title)
        img = Image.open(entry.path)
        st.image(img, use_container_width=True)


if "page" not in st.session_state:
  st.session_state.page = "home"
if st.session_state.page == "home":
  show_home()
elif st.session_state.page == "grid":
  show_grid()
