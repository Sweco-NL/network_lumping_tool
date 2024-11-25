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


def define_list_upstream_downstream_edges_ids(
    node_ids, nodes: gpd.GeoDataFrame, edges: gpd.GeoDataFrame
):
    nodes_sel = nodes[nodes.nodeID.isin(node_ids)].copy()
    for direction, node in zip(["upstream", "downstream"], ["node_end", "node_start"]):
        nodes_sel[f"{direction}_edges"] = nodes_sel.apply(
            lambda x: ",".join(list(edges[edges[node] == x.nodeID].code.astype(str).values)), axis=1
        )
        nodes_sel[f"no_{direction}_edges"] = nodes_sel.apply(
            lambda x: len(x[f"{direction}_edges"].split(",")), axis=1
        )
    return nodes_sel.reset_index(drop=True)
