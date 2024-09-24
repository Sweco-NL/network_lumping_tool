from pydantic import BaseModel, ConfigDict
import geopandas as gpd
from pathlib import Path


class NetworkLumping(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    
    name: str = ""
    network_edges: gpd.GeoDataFrame = None
    areas: gpd.GeoDataFrame = None

    def read_basis_data_from_gpkg(
        self, 
        basis_gpkg: Path, 
        network_edges_layer: str = "network_edges", 
        areas_layer: str = "areas"
    ):
        self.network_edges = gpd.read_file(basis_gpkg, layer=network_edges_layer)
        self.areas = gpd.read_file(basis_gpkg, layer=areas_layer)

    # def preprocess_basis_data(self):
    #     self.network_edges = 