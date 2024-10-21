import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
import os
import sys
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from dotenv import dotenv_values

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from src.network_lumping.preprocessing.general import remove_z_dims, connect_endpoints_by_buffer
from src.network_lumping.graph_utils.create_graph import (
    generate_nodes_from_edges,
    create_graph_based_on_nodes_edges,
    create_graph_from_edges,
)

logging.basicConfig(level=logging.INFO)

logging.info("load hydro-objects")
config = dotenv_values("..\\.env")
base_dir = Path(config["DATA_DIR"])

hydro_objects_gpkg = Path(base_dir, "0_basisdata.gpkg")
hydro_objects = gpd.read_file(hydro_objects_gpkg, layer="hydroobjecten").to_crs(28992)

logging.info("hydro-objects loaded")

hydro_objects.rename(columns={'CODE':'code'}, inplace=True)

# LineString Z to 2D
hydro_objects = remove_z_dims(hydro_objects)

# Check and connect endpoints
hydro_objects = connect_endpoints_by_buffer(hydro_objects)

nodes, edges, graph = create_graph_from_edges(hydro_objects)
positions = {n: [n[0], n[1]] for n in list(graph.nodes)}

f, ax = plt.subplots(1, 1, figsize=(10, 6))
hydro_objects.plot(ax=ax, color='lightblue', zorder=0)
hydro_objects[hydro_objects.preprocessing_split=="split"].plot(ax=ax, color='red', zorder=1)
nx.draw(graph, positions, ax=ax, node_size=8)
ax.axis("equal")
plt.tight_layout()
plt.show()

export_gpkg = Path(str(hydro_objects_gpkg).replace("0_basisdata", "1_data_bewerkt"))
hydro_objects.to_file(export_gpkg, layer="hydroobjecten")

logging.info(hydro_objects)

