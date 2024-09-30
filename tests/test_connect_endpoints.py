import geopandas as gpd
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import snap, split
import itertools
import datetime
import os
import numpy as np
import time
import logging
import warnings
import sys
from pathlib import Path
import fiona

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

import src.network_lumping.preprocessing.general 

basis_gpkg = 'p:\\5325\\51024343_AaEnMaas_Afwateringseenheden_Lumpen\\300 Werkdocumenten\\3_analyse\\aa_en_maas\\0_basisdata.gpkg'
print(basis_gpkg)
hydro_objects = gpd.read_file(basis_gpkg, layer="hydroobjecten").to_crs(28992)
hydro_objects.rename(columns={'CODE':'code'}, inplace=True)

# Function to convert LineString Z to 2D
def convert_line_string_z_to_2d(geometry):
    if geometry.has_z:  # Check if geometry has Z coordinates
        return LineString(list(geometry.coords))  # Convert to 2D LineString
    return geometry

# Apply the function only to valid geometries
hydro_objects['geometry'] = hydro_objects['geometry'].apply(
    lambda geom: convert_line_string_z_to_2d(geom) if isinstance(geom, LineString) else geom
)

# Assuming `lines_gdf` is your original GeoDataFrame containing LineString Z geometries
problematic_lines = []

for index, row in hydro_objects.iterrows():
    try:
        geom = row['geometry']
        # Attempt to access coordinates to check for Z
        if geom.has_z:  # Check if the geometry has Z coordinates
            raise ValueError(f"Line has Z coordinates: {geom}")
        
        # Your processing logic here (if applicable)

    except Exception as e:
        # Collect problematic line info
        problematic_lines.append({
            'code': row['code'],  # Replace 'id' with your actual ID column name
            'geometry': geom,
            'error': str(e)
        })

# Create a GeoDataFrame from problematic lines
problematic_gdf = gpd.GeoDataFrame(problematic_lines)

# Set the geometry column
problematic_gdf = gpd.GeoDataFrame(problematic_gdf, geometry='geometry')

print(problematic_gdf)

# Now you can safely call your connect_endpoints_by_buffer function
result = src.network_lumping.preprocessing.general.connect_endpoints_by_buffer(problematic_gdf)
