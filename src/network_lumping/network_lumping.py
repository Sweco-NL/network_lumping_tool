from pathlib import Path
import logging

import pandas as pd
import geopandas as gpd
import networkx as nx
from pydantic import BaseModel, ConfigDict
import folium
import matplotlib
import matplotlib.pyplot as plt
import webbrowser
import numpy as np
import random
import os

from .graph_utils.create_graph import create_graph_from_edges
from .graph_utils.network_functions import find_nodes_edges_for_direction
from .utils.general_functions import (
    remove_holes_from_polygons,
    define_list_upstream_downstream_edges_ids,
)


class NetworkLumping(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path = None
    name: str = None

    read_results: bool = False
    write_results: bool = False

    direction: str = "upstream"

    hydroobjecten: gpd.GeoDataFrame = None
    buitenwateren: gpd.GeoDataFrame = None
    overige_watergangen: gpd.GeoDataFrame = None
    afwateringseenheden: gpd.GeoDataFrame = None

    afwateringseenheden0: gpd.GeoDataFrame = None
    afwateringseenheden1: gpd.GeoDataFrame = None

    uitstroom_punten: gpd.GeoDataFrame = None
    uitstroom_edges: gpd.GeoDataFrame = None
    uitstroom_nodes: gpd.GeoDataFrame = None
    uitstroom_areas_0: gpd.GeoDataFrame = None
    uitstroom_areas_1: gpd.GeoDataFrame = None
    uitstroom_areas_2: gpd.GeoDataFrame = None
    uitstroom_splits_0: gpd.GeoDataFrame = None
    uitstroom_splits_1: gpd.GeoDataFrame = None

    edges: gpd.GeoDataFrame = None
    nodes: gpd.GeoDataFrame = None
    network_positions: dict = None
    graph: nx.DiGraph = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.path is not None:
            self.check_case_path_directory(path=self.path)
            self.read_data_from_case()

    def check_case_path_directory(self, path: Path):
        """Checks if case directory exists and if required directory structure exists

        Parameters
        ----------
        path : Path
            path to case directory. name of directory is used as case name.
            self.path and self.name are set

        Raises ValueErrors in case directory and 0_basisdata directory not exist
        """
        if not path.exists() and path.is_dir():
            raise ValueError(
                f"provided path [{path}] does not exist or is not a directory"
            )
        self.path = path
        self.name = self.path.name
        logging.info(f' ### Case "{self.name.capitalize()}" ###')

        # check if directories 0_basisdata and 1_tussenresultaat exist
        if not Path(self.path, "0_basisdata").exists():
            raise ValueError(f"provided path [{path}] exists but without a 0_basisdata")
        for folder in ["1_tussenresultaat", "2_resultaat"]:
            if not Path(self.path, folder).exists():
                Path(self.path, folder).mkdir(parents=True, exist_ok=True)

    def read_data_from_case(self, path: Path = None, read_results: bool = None):
        """Read data from case: including basis data and intermediate results

        Parameters
        ----------
        path : Path, optional
            Path to the case directory including directories 0_basisdata and
            1_tussenresultaat. Directory name is used as name for the case,
            by default None
        read_results : bool, optional
            if True, it reads already all resulst from, by default None
        """
        if path is not None and path.exists():
            self.check_case_path_directory(path=path)
        logging.info(f"   x read basisdata")
        basisdata_gpkgs = [
            Path(self.path, "0_basisdata", f + ".gpkg")
            for f in [
                "hydroobjecten",
                "overige_watergangen",
                "krw_lijn",
                "krw_vlak",
                "krw_deelgebieden",
                "uitstroom_punten",
                "afwateringseenheden",
            ]
        ]
        if isinstance(read_results, bool):
            self.read_results = read_results
        baseresults_gpkgs = (
            [
                Path(self.path, "1_tussenresultaat", f + ".gpkg")
                for f in [
                    "uitstroom_edges",
                    "uitstroom_nodes",
                    "uitstroom_areas_0",
                    "uitstroom_areas_1",
                    "uitstroom_areas_2",
                ]
            ]
            if self.read_results
            else []
        )
        for list_gpkgs in [basisdata_gpkgs, baseresults_gpkgs]:
            for x in list_gpkgs:
                if x.is_file():
                    if hasattr(self, x.stem):
                        logging.debug(f"    - get dataset {x.stem}")
                        setattr(self, x.stem, gpd.read_file(x, layer=x.stem))

    def create_graph_from_network(
        self, water_lines=["buitenwateren", "hydroobjecten", "overige_watergangen"]
    ):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        water_lines : list, optional
            _description_, by default ["buitenwateren", "hydroobjecten", "overige_watergangen"]
        """
        logging.info("  x create network graph")
        self.uitstroom_edges = None
        for water_line in water_lines:
            gdf_water_line = getattr(self, water_line)
            if self.uitstroom_edges is None:
                self.uitstroom_edges = gdf_water_line
            else:
                self.uitstroom_edges = pd.concat([self.uitstroom_edges, gdf_water_line])
        self.nodes, self.edges, self.graph = create_graph_from_edges(
            self.uitstroom_edges
        )
        self.network_positions = {n: [n[0], n[1]] for n in list(self.graph.nodes)}

    def find_upstream_downstream_nodes_edges(
        self, direction: str = "upstream", no_uitstroom_punten: int = None
    ):
        """_summary_

        _extended_summary_

        Parameters
        ----------
        direction : str, optional
            _description_, by default "upstream"
        """
        if direction not in ["upstream", "downstream"]:
            raise ValueError(f" x direction needs to be 'upstream' or 'downstream'")
        self.direction = direction
        logging.info(
            f"  x find {direction} nodes and edges for {len(self.uitstroom_punten)} outflow locations"
        )

        if no_uitstroom_punten is not None:
            self.uitstroom_punten = self.nodes.sample(n=no_uitstroom_punten)

        self.uitstroom_punten["representatieve_node"] = (
            self.uitstroom_punten.geometry.apply(
                lambda x: self.nodes.geometry.distance(x).idxmin()
            )
        )

        self.uitstroom_nodes, self.uitstroom_edges = find_nodes_edges_for_direction(
            nodes=self.nodes,
            edges=self.edges,
            node_ids=self.uitstroom_punten["representatieve_node"].to_numpy(),
            border_node_ids=self.uitstroom_punten["representatieve_node"].to_numpy(),
            direction=direction,
        )

    def detect_confrontation_points_and_create_decision_table(self):
        ## TEST DETECT  ##
        uitstroom_nodes = self.uitstroom_punten.representatieve_node.values
        direction = self.direction

        uitstroom_edges = None

        for node in self.uitstroom_nodes.nodeID.values:
            upstream_edges = self.uitstroom_edges[
                (self.uitstroom_edges.node_start == node)
            ].copy()

            uitstroom_punten_columns = [
                f"{direction}_node_{n}" for n in uitstroom_nodes
            ]

            # drop if all columns are False
            upstream_edges[uitstroom_punten_columns] = (
                upstream_edges[uitstroom_punten_columns].replace(False, np.nan).copy()
            )
            upstream_edges = upstream_edges.dropna(
                subset=uitstroom_punten_columns, how="all"
            )

            # checked if all columns are equal
            edges = upstream_edges.drop_duplicates(subset=uitstroom_punten_columns)
            if len(edges) > 1:
                if uitstroom_edges is None:
                    uitstroom_edges = edges.copy()
                else:
                    uitstroom_edges = pd.concat([uitstroom_edges, edges])

        uitstroom_edges[uitstroom_punten_columns] = (
            uitstroom_edges[uitstroom_punten_columns].replace(np.nan, False).copy()
        )

        self.uitstroom_splits_0 = define_list_upstream_downstream_edges_ids(
            uitstroom_edges.node_start.unique(),
            self.uitstroom_nodes,
            self.uitstroom_edges,
        )
        self.uitstroom_splits_0.to_file(
            Path(self.path, "1_tussenresultaat", "uitstroom_splits_0.gpkg")
        )
        return self.uitstroom_splits_0

    def read_direction_splits(self):
        uitstroom_splits_1_path = Path(
            self.path, "1_tussenresultaat", "uitstroom_splits_1.gpkg"
        )
        if os.path.exists(uitstroom_splits_1_path):
            self.uitstroom_splits_1 = gpd.read_file(
                Path(self.path, "1_tussenresultaat", "uitstroom_splits_1.gpkg")
            )

            return self.uitstroom_splits_1
        else:
            print("The file uistroom_splits_1 does not exist in the specified folder.")

            return None

    def select_direction_splits(self, fillna_with_random=False):
        """_summary_: This function can be used to define the random direction at split points. 
        
        Uitstroom_splits_0 is the GeoDataFrame with the detected points, 
        without decisions. These decisions can be made manually in GIS, 
        by adding the correct hydroobject in the column {search_direction}_edges, 
        and saving the file as uitstroom_splits_1. 
        
        This function fill the entire column when there is no uitstroom_splits_1
        GeoDataFrame. When it is present, the function only fills empty instances in the column.
        """
        search_direction = (
            "upstream" if self.direction == "downstream" else "downstream"
        )
        # Now check if self.uitstroom_splits_1 exists
        if hasattr(self, "uitstroom_splits_1"):
            # Fill f"{search_direction}_edge" in self.uitstroom_splits_1 where it is empty
            self.uitstroom_splits_2 = self.uitstroom_splits_1.copy()
            self.uitstroom_splits_2[f"{search_direction}_edge"] = (
                self.uitstroom_splits_2.apply(
                    lambda x: random.choice(x[f"{search_direction}_edges"])
                    if pd.isnull(x[f"{search_direction}_edge"])
                    else x[f"{search_direction}_edge"],
                    axis=1,
                )
            )
        else:
            self.uitstroom_splits_2 = self.uitstroom_splits_0.copy()
            self.uitstroom_splits_2[f"{search_direction}_edge"] = (
                self.uitstroom_splits_2.apply(
                    lambda x: random.choice(x[f"{search_direction}_edges"])
                    if pd.isnull(x[f"{search_direction}_edge"])
                    else x[f"{search_direction}_edge"],
                    axis=1,
                )
            )

    def assign_drainage_units_to_outflow_points_based_on_id(self):
        self.afwateringseenheden["gridcode"] = (
            self.afwateringseenheden["gridcode"].round(0).astype("Int64").astype(str)
        )
        self.uitstroom_edges["code"] = self.uitstroom_edges["code"].astype(str)

        upstream_columns = [
            f"{self.direction}_node_{node}"
            for node in self.uitstroom_punten["representatieve_node"].tolist()
        ]
        self.afwateringseenheden0 = self.afwateringseenheden.merge(
            self.uitstroom_edges[
                ["code"] + [f"{column}" for column in upstream_columns]
            ],
            how="left",
            left_on="gridcode",
            right_on="code",
        )

        self.afwateringseenheden0 = self.afwateringseenheden0.drop(columns=["code"])

        self.afwateringseenheden0[upstream_columns] = self.afwateringseenheden0[
            upstream_columns
        ].fillna(False)

    def assign_drainage_units_to_outflow_points_based_on_length_hydroobject(self):
        self.afwateringseenheden["unique_id"] = self.afwateringseenheden.index
        self.afwateringseenheden["savedgeom"] = self.afwateringseenheden.geometry

        joined = gpd.sjoin(
            self.uitstroom_edges,
            self.afwateringseenheden,
            how="inner",
            predicate="intersects",
        )

        joined["intersection_length"] = joined.apply(
            lambda row: row.geometry.intersection(row.savedgeom).length, axis=1
        )

        merged = self.afwateringseenheden.merge(
            joined[["unique_id", "code", "intersection_length"]],
            on="unique_id",
            how="inner",
        )

        max_intersections = merged.groupby("unique_id")["intersection_length"].idxmax()
        # Select the rows from the merged GeoDataFrame that correspond to those indices
        result = merged.loc[max_intersections]
        # Optionally reset the index if needed
        result.reset_index(drop=True, inplace=True)
        result = result.rename(columns={"code": "code_hydroobject"})
        result = result.drop(columns=["savedgeom"])

        upstream_columns = [
            f"{self.direction}_node_{node}"
            for node in self.uitstroom_punten["representatieve_node"].tolist()
        ]
        self.afwateringseenheden1 = result.merge(
            self.uitstroom_edges[
                ["code"] + [f"{column}" for column in upstream_columns]
            ],
            how="left",
            left_on="code_hydroobject",
            right_on="code",
        )
        self.afwateringseenheden1 = self.afwateringseenheden1.drop(columns=["code"])

        self.afwateringseenheden1[upstream_columns] = self.afwateringseenheden1[
            upstream_columns
        ].fillna(False)

    def dissolve_assigned_drainage_units(self):
        self.uitstroom_areas_0 = gpd.GeoDataFrame()
        for node in self.uitstroom_punten["representatieve_node"].tolist():
            filtered_areas = self.afwateringseenheden1[
                self.afwateringseenheden1[f"{self.direction}_node_{node}"] == True
            ]
            # Step 2: Dissolve the filtered geometries
            dissolved_areas = filtered_areas[["Oppervlakt", "geometry"]].dissolve(
                aggfunc="sum"
            )
            # Optionally, you can reset the index if needed
            dissolved_areas = dissolved_areas.reset_index()

            self.uitstroom_areas_0 = gpd.GeoDataFrame(
                pd.concat([self.uitstroom_areas_0, dissolved_areas])
            )

        # buffered = self.uitstroom_areas_0.geometry.buffer(-1)  # Use a small value

        # # Create a new GeoDataFrame with the buffered geometries
        # buffered_gdf = gpd.GeoDataFrame(geometry=buffered)

        # # Explode the geometries to separate polygons
        # exploded = buffered_gdf.explode(index_parts=False)

        # # Assuming your CRS is in meters; if not, convert your CRS accordingly
        # exploded = exploded[exploded.geometry.area >= 100]

        # # Reset the index, if desired
        # exploded.reset_index(drop=True, inplace=True)
        self.uitstroom_areas_0 = remove_holes_from_polygons(
            self.uitstroom_areas_0, min_area=50
        )
        self.uitstroom_areas_0 = self.uitstroom_areas_0.geometry.buffer(0.1)

    def export_results_all(
        self,
        html_file_name: str = None,
        width_edges: float = 10.0,
        opacity_edges: float = 0.5,
    ):
        """Export results to geopackages and folium html"""
        self.export_results_to_gpkg()
        self.export_results_to_html_file(
            html_file_name=html_file_name,
            width_edges=width_edges,
            opacity_edges=opacity_edges,
        )

    def export_results_to_gpkg(self):
        """Export results to geopackages in folder 1_tussenresultaat"""
        results_dir = Path(self.path, "1_tussenresultaat")
        logging.info(f"  x export results")
        for layer in [
            "uitstroom_punten",
            "uitstroom_edges",
            "uitstroom_nodes",
            "uitstroom_areas_0",
            "uitstroom_areas_1",
            "uitstroom_areas_2",
            "afwateringseenheden0",
            "afwateringseenheden1",
        ]:
            result = getattr(self, layer)
            if result is not None:
                logging.debug(f"   - {layer}")
                result.to_file(Path(results_dir, f"{layer}.gpkg"))

    def export_results_to_html_file(
        self,
        html_file_name: str = None,
        width_edges: float = 10.0,
        opacity_edges: float = 0.5,
    ):
        """Export results to folium html file

        Parameters
        ----------
        html_file_name : str, optional
            filename folium html, by default None
        width_edges : float, optional
            width (meters) of edges in folium html, by default 10.0
        opacity_edges : float, optional
            opacity of edges in folium html, by default 0.5
        """
        nodes_selection = self.uitstroom_punten.representatieve_node.to_numpy()
        no_nodes = len(self.uitstroom_punten)
        nodes_colors = plt.get_cmap("hsv", no_nodes + 1)
        nodes_4326 = self.uitstroom_nodes.to_crs(4326)

        m = folium.Map(
            location=[nodes_4326.geometry.y.mean(), nodes_4326.geometry.x.mean()],
            tiles=None,
            zoom_start=12,
        )

        fg = folium.FeatureGroup(name=f"Watergangen", control=True).add_to(m)

        folium.GeoJson(
            self.uitstroom_edges.buffer(width_edges / 2.0),
            color="grey",
            weight=5,
            z_index=0,
            opacity=0.25,
        ).add_to(fg)

        folium.GeoJson(
            self.uitstroom_nodes,
            marker=folium.Circle(
                radius=width_edges,
                fill_color="darkgrey",
                fill_opacity=0.5,
                color="darkgrey",
                weight=1,
                z_index=1,
            ),
        ).add_to(fg)

        if self.uitstroom_splits_0 is not None:
            folium.GeoJson(
                self.uitstroom_splits_0,
                marker=folium.Circle(
                    radius=width_edges * 3.0,
                    fill_color="black",
                    fill_opacity=1,
                    color="black",
                    weight=1,
                    z_index=1,
                ),
                name="Splitsingen",
                show=False,
            ).add_to(m)

        folium.GeoJson(
            self.afwateringseenheden0,
            fill_opacity=0.0,
            color="grey",
            weight=0.5,
            z_index=10,
            name="Afwateringseenheden",
        ).add_to(m)

        for i, node_selection in enumerate(nodes_selection):
            c = matplotlib.colors.rgb2hex(nodes_colors(i))
            fg = folium.FeatureGroup(
                name=f"Uitstroompunt node {node_selection}", control=True, show=False
            ).add_to(m)

            sel_uitstroom_edges = self.uitstroom_edges[
                self.uitstroom_edges[f"{self.direction}_node_{node_selection}"]
            ].copy()

            # Assign sel_drainage_units to the class instance
            sel_drainage_units = self.afwateringseenheden0[
                self.afwateringseenheden0[f"{self.direction}_node_{node_selection}"]
            ].copy()

            sel_uitstroom_edges.geometry = sel_uitstroom_edges.buffer(width_edges / 2.0)
            if len(sel_uitstroom_edges) > 0:
                folium.GeoJson(
                    sel_uitstroom_edges,
                    color=c,
                    weight=5,
                    z_index=2,
                    opacity=opacity_edges,
                    fill_opacity=opacity_edges,
                ).add_to(fg)

            folium.GeoJson(
                sel_drainage_units,
                color=c,
                weight=1,
                z_index=2,
                opacity=opacity_edges * 0.5,
                fill_opacity=opacity_edges * 0.5,
            ).add_to(fg)

            folium.GeoJson(
                self.uitstroom_nodes.iloc[[node_selection]],
                marker=folium.Circle(
                    radius=width_edges * 5.0,
                    fill_color=c,
                    fill_opacity=1.0,
                    color=c,
                    opacity=1.0,
                    weight=4,
                    z_index=0,
                ),
            ).add_to(fg)

        folium.TileLayer("openstreetmap", name="Open Street Map", show=False).add_to(m)
        folium.TileLayer("cartodbpositron", name="Light Background", show=True).add_to(
            m
        )

        folium.LayerControl(collapsed=False).add_to(m)

        if html_file_name is None:
            html_file_name = self.name
        m.save(Path(self.path, f"{html_file_name}.html"))
        webbrowser.open(Path(self.path, f"{html_file_name}.html"))
