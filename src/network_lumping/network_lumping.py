from pathlib import Path

import geopandas as gpd
import networkx as nx
import momepy
from pydantic import BaseModel, ConfigDict

from .graph_utils.create_graph import (
    generate_nodes_from_edges,
    create_graph_based_on_nodes_edges,
    create_graph_from_edges,
)
from .preprocessing.general import remove_z_dims


class NetworkLumping(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    name: str = ""
    areas: gpd.GeoDataFrame = None
    edges: gpd.GeoDataFrame = None
    nodes: gpd.GeoDataFrame = None
    G: nx.DiGraph = None

    def read_basis_data_from_gpkg(
        self,
        basis_gpkg: Path,
        edges_layer: str = "network_edges",
        edges_id_column: str = "CODE",
        areas_layer: str = "areas",
        areas_id_column: str = "CODE",
    ):
        if basis_gpkg.suffix != ".gpkg":
            raise ValueError(f"Provide path to gpkg-file, not {basis_gpkg}")
        if not basis_gpkg.exists():
            raise ValueError(f"Provided path {basis_gpkg} not existing")
        edges = gpd.read_file(basis_gpkg, layer=edges_layer)
        edges = remove_z_dims(edges)
        edges["edgeID"] = edges[edges_id_column]
        self.nodes, self.edges, self.G = create_graph_from_edges(edges)
        areas = gpd.read_file(basis_gpkg, layer=areas_layer)
        areas["areaID"] = areas[areas_id_column]
        self.areas = areas
