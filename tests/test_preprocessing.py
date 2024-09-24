import os
import sys
import pandas as pd
import geopandas as gpd
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from src.network_lumping import NetworkLumping
from src.network_lumping.preprocessing.preprocessing import preprocess_hydamo_hydroobjects

basis_gpkg = "p:\\5325\\51024343_AaEnMaas_Afwateringseenheden_Lumpen\\300 Werkdocumenten\\3_analyse\\0_Basisdata.gpkg"

# hydro_objects = gpd.read_file(basis_gpkg, layer="Hydro_objecten")
# afwaterende_eenheden = gpd.read_file(basis_gpkg, layer="Afwateringseenheden")

# network = NetworkLumping(
#     name='Aa en Maas',
#     network_edges=hydro_objects,
#     areas=afwaterende_eenheden
# )

network = NetworkLumping()
network.read_basis_data_from_gpkg(
    basis_gpkg=Path(basis_gpkg),
    network_edges_layer="Hydro_objecten",
    areas_layer="Afwateringseenheden"
)

# network.preprocess_basis_data()

print(network)
