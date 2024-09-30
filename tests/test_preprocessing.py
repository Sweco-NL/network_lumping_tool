from pathlib import Path

import matplotlib.pyplot as plt
import momepy
import networkx as nx

from src.network_lumping import NetworkLumping

basis_gpkg = "p:\\5325\\51024343_AaEnMaas_Afwateringseenheden_Lumpen\\300 Werkdocumenten\\3_analyse\\test\\0_basisdata.gpkg"

n = NetworkLumping(name="Aa en Maas")
n.read_basis_data_from_gpkg(
    basis_gpkg=Path(basis_gpkg),
    edges_layer="hydroobjecten",
    edges_id_column="CODE",
    areas_layer="afwateringseenheden",
    areas_id_column="Id",
)

# network.preprocess_basis_data()
positions = {n: [n[0], n[1]] for n in list(n.G.nodes)}

f, ax = plt.subplots(1, 1, figsize=(10, 6))
n.edges.plot(ax=ax, color='lightblue', zorder=0)
nx.draw(n.G, positions, ax=ax, node_size=8)
ax.axis("equal")
plt.tight_layout()
plt.show()

print(momepy.nx_to_gdf(n.G))
