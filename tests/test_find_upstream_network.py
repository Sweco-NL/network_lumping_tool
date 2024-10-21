# %% start
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import os
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
import random
from dotenv import dotenv_values

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from src.network_lumping.graph_utils.create_graph import create_graph_from_edges
from src.network_lumping.graph_utils.network_functions import find_nodes_edges_for_direction

logging.basicConfig(level=logging.INFO)

# Load data
logging.info("load hydro-objects")
config = dotenv_values("..\\.env")
base_dir = Path(config["DATA_DIR"])

hydro_objects_gpkg = Path(base_dir, "1_data_bewerkt.gpkg")
hydro_objects = gpd.read_file(hydro_objects_gpkg, layer="hydroobjecten").to_crs(28992)

logging.info("hydro-objects loaded")

# Check and connect endpoints
nodes, edges, graph = create_graph_from_edges(hydro_objects)
positions = {n: [n[0], n[1]] for n in list(graph.nodes)}

# no nodes for which to find upstream/downstream nodes and edges
no_nodes = 5
direction = "upstream" # upstream/downstream

for run in range(3):
    # select random nodes
    nodes_selection = np.array(random.sample(range(0, len(nodes)), no_nodes))
    nodes_selection_colors = plt.get_cmap("hsv", no_nodes+1)

    nodes, edges = find_nodes_edges_for_direction(
        nodes=nodes, 
        edges=edges, 
        node_ids=nodes_selection, 
        border_node_ids=nodes_selection,
        direction=direction
    )

    f, ax = plt.subplots(1, 1, figsize=(12, 8))
    hydro_objects.plot(ax=ax, color='lightblue', linewidth=1.0, zorder=0)
    # nx.draw(graph, positions, ax=ax, node_size=8)

    # for x, y, label in zip(nodes.geometry.x, nodes.geometry.y, nodes.nodeID):
    #     ax.annotate(label, xy=(x, y), zorder=1000)

    for i, node_selection in enumerate(nodes_selection):
        c = nodes_selection_colors(i)
        nodes.iloc[[node_selection]].plot(ax=ax, marker="o", markersize=40, linewidth=1, color=c, zorder=1000)
        if len(edges[edges[f"{direction}_node_{node_selection}"]]) > 0:
            edges[edges[f"{direction}_node_{node_selection}"]].plot(ax=ax, color=c, linewidth=2.0, zorder=1000)
    ax.axis("equal")
    plt.tight_layout()
    ax.axis("off")

plt.show()

