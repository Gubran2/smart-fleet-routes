# prepare_graph.py
import os, osmnx as ox
from shapely.geometry import box

# نفس الإحداثيات اللي بتستعملها في التطبيق
WEST, SOUTH, EAST, NORTH = 36.71519, 34.00945, 36.74922, 34.03680

os.makedirs("graphs", exist_ok=True)
poly = box(WEST, SOUTH, EAST, NORTH)

# حمّل شبكة القيادة مرة واحدة واحفظها
G = ox.graph_from_polygon(poly, network_type="drive", retain_all=False,
                          simplify=True, truncate_by_edge=True)
G = ox.routing.add_edge_speeds(G)  # يحسب speed_kph لكل حافة
ox.save_graphml(G, "graphs/drive_36.71519_34.00945_36.74922_34.03680.graphml")

print("drive_36.71519_34.00945_36.74922_34.03680.graphml")
