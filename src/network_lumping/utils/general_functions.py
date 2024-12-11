from shapely.geometry import MultiPolygon, Polygon
import geopandas as gpd
from typing import TypeVar


def _remove_holes(geom, min_area):
    def p(p: Polygon, min_area) -> Polygon:
        holes = [i for i in p.interiors if not Polygon(i).area > min_area]
        return Polygon(shell=p.exterior, holes=holes)

    def mp(mp: MultiPolygon, min_area) -> MultiPolygon:
        return MultiPolygon([p(i, min_area) for i in mp.geoms])

    if isinstance(geom, Polygon):
        return p(geom, min_area)
    elif isinstance(geom, MultiPolygon):
        return mp(geom, min_area)
    else:
        return geom


_Geom = TypeVar("_Geom", Polygon, MultiPolygon, gpd.GeoSeries, gpd.GeoDataFrame)


def remove_holes_from_polygons(geom: _Geom, min_area: float) -> _Geom:
    """Remove all holes from a geometry that satisfy the filter function."""
    if isinstance(geom, gpd.GeoSeries):
        return geom.apply(_remove_holes, min_area=min_area)
    elif isinstance(geom, gpd.GeoDataFrame):
        geom = geom.copy()
        geom["geometry"] = remove_holes_from_polygons(
            geom["geometry"], min_area=min_area
        )
        return geom
    return _remove_holes(geom, min_area=min_area)


def remove_holes_from_basin_areas(basin_areas: gpd.GeoDataFrame, min_area: float):
    print(f" - remove holes within basin areas with less than {min_area/10000.0:.2f}ha")
    return remove_holes_from_polygons(geom=basin_areas, min_area=min_area)


def calculate_angle(line, direction):
    if direction == "downstream":
        # Get the first segment for downstream
        coords = list(line.coords)
        p1, p2 = coords[0], coords[1]  # First segment
    elif direction == "upstream":
        # Get the last segment for upstream
        coords = list(line.coords)
        p1, p2 = coords[-2], coords[-1]  # Last segment
    else:
        raise ValueError("Direction must be 'upstream' or 'downstream'")

    # Calculate the angle relative to the north (0 degrees)
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])  # Angle in radians
    angle_degrees = np.degrees(angle)
    angle_degrees = angle_degrees % 360  # Normalize to 0-360 degrees

    return angle_degrees


def calculate_angles_of_edges_at_nodes(
    nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
):
    edges["downstream_angle"] = edges.apply(
        lambda x: calculate_angle(x["geometry"], "downstream").round(2), axis=1
    )
    edges["upstream_angle"] = edges.apply(
        lambda x: calculate_angle(x["geometry"], "upstream").round(2), axis=1
    )

    nodes["upstream_angles"] = ""
    nodes["downstream_angles"] = ""

    def calculate_angles_of_edges_at_node(node, edges):
        for direction, opp_direction in zip(
            ["upstream", "downstream"], ["downstream", "upstream"]
        ):
            dir_edges = node[f"{direction}_edges"].split(",")

            node[f"{direction}_angles"] = ", ".join(
                [
                    str(
                        edges.loc[
                            edges["code"] == str(edge_id), f"{opp_direction}_angle"
                        ].values[0]
                    )
                    for edge_id in dir_edges
                    if edge_id != ""
                ]
            )

        return node

    nodes = nodes.apply(
        lambda node: calculate_angles_of_edges_at_node(node, edges), axis=1
    )
    return nodes, edges


def angle_difference(angle1, angle2):
    diff = abs(angle1 % 360 - angle2 % 360)
    if diff > 180:
        diff = 360 - diff
    return diff


def find_edge_smallest_angle_difference(reference_angle, angles, edge_codes):
    if reference_angle is None:
        return [None for a in angles], None
    reference_angle = float(reference_angle)
    angle_differences = np.array(
        [angle_difference(angle, reference_angle) for angle in angles]
    )
    min_index = np.argmin(angle_differences)

    return angle_differences, edge_codes[min_index]


def define_list_upstream_downstream_edges_ids(
    node_ids, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
):
    nodes_sel = nodes[nodes.nodeID.isin(node_ids)].copy()
    for direction, node in zip(["upstream", "downstream"], ["node_end", "node_start"]):
        nodes_sel[f"{direction}_edges"] = nodes_sel.apply(
            lambda x: ",".join(list(edges[edges[node] == x.nodeID].code.values)), axis=1
        )
        nodes_sel[f"no_{direction}_edges"] = nodes_sel.apply(
            lambda x: len(x[f"{direction}_edges"].split(","))
            if x[f"{direction}_edges"]
            else 0,
            axis=1,
        )
    nodes_sel = nodes_sel.reset_index(drop=True)
    return nodes_sel
