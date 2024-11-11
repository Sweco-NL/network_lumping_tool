from .network_lumping import NetworkLumping
from pathlib import Path


def run_network_lumping_to_generate_basins(
    path: Path,
    direction: str = "upstream",
    no_uitstroom_punten: int = None,
    write_results: bool = False,
    html_file_name: str = None,
    width_edges: float = 10.0,
    opacity_edges: float = 0.5,
):
    network = NetworkLumping()

    network.read_data_from_case(path=path)

    network.create_graph_from_network()

    network.find_upstream_downstream_nodes_edges(
        direction=direction,
        no_uitstroom_punten=no_uitstroom_punten,
    )

    network.assign_drainage_units_to_outflow_points_based_on_id()

    network.assign_drainage_units_to_outflow_points_based_on_length_hydroobject()

    network.dissolve_assigned_drainage_units()

    if write_results:
        network.export_results_to_gpkg()

        network.export_results_to_html_file(
            html_file_name=html_file_name,
            width_edges=width_edges,
            opacity_edges=opacity_edges,
        )

    return network
