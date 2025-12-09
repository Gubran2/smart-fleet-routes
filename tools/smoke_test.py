"""Smoke test for the VRP pipeline (headless).
This duplicates a small subset of the logic in app.py to benchmark loading, sampling,
clustering, cropping and routing. It purposely avoids Streamlit and only depends on
osmnx, networkx, numpy, sklearn (optional), shapely.

Run:
    python tools\smoke_test.py
"""
import time
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import box, LineString, MultiLineString

# Parameters (mirror defaults)
WEST, SOUTH, EAST, NORTH = 36.71519, 34.00945, 36.74922, 34.03680
N_POINTS = 300
N_VEHICLES = 4
MIN_SPACING_M = 80
BUFFER_M = 800
SEED = 42
VERTEX_SKIP = 6

# Helpers (small subset)

def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def load_graph_once(west, south, east, north, city_factor=1.0):
    poly = box(west, south, east, north)
    t0 = time.perf_counter()
    G = ox.graph_from_polygon(poly, network_type="drive", retain_all=False,
                              simplify=True, truncate_by_edge=True)
    t1 = time.perf_counter()
    # Project and add speeds/travel times
    G = ox.project_graph(G, to_crs="EPSG:4326")
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    for _, _, _, d in G.edges(keys=True, data=True):
        if "travel_time" in d:
            d["travel_time"] = float(d["travel_time"]) * float(city_factor)
    G_cc = ox.truncate.largest_component(G, strongly=False)
    t2 = time.perf_counter()
    # collect edges
    edges, cum, tot = [], [], 0.0
    for u, v, k, d in G_cc.edges(keys=True, data=True):
        Lm = float(d.get("length", 0.0))
        if Lm <= 0: continue
        geom = d.get("geometry")
        if geom is None:
            geom = LineString([(G_cc.nodes[u]["x"], G_cc.nodes[u]["y"]),
                               (G_cc.nodes[v]["x"], G_cc.nodes[v]["y"])])
        edges.append((u, v, k, geom, Lm))
        tot += Lm; cum.append(tot)
    t3 = time.perf_counter()
    return G_cc, edges, cum, tot, (t1-t0, t2-t1, t3-t2)


def sample_point_on_edge(edges, cum, tot):
    r = random.random()*tot
    lo, hi = 0, len(cum)-1
    while lo < hi:
        m = (lo+hi)//2
        if cum[m] < r: lo = m+1
        else: hi = m
    u,v,k,geom,Lm = edges[lo]
    t = random.random()
    p = geom.interpolate(t*geom.length)
    return (p.y, p.x)


def generate_points_on_roads(edges, cum, tot, n_points, min_spacing_m, seed=42):
    random.seed(int(seed)); np.random.seed(int(seed))
    pts = []
    tries, max_tries = 0, max(8000, n_points * 20)
    earth = 111_320.0
    while len(pts) < n_points and tries < max_tries:
        lat, lon = sample_point_on_edge(edges, cum, tot)
        if not pts:
            ok = True
        else:
            # simple linear check (approx)
            ok = all(haversine_m(lat, lon, la, lo) >= min_spacing_m for la,lo in pts)
        if ok:
            pts.append((lat, lon))
        tries += 1
    pts = np.array(pts, dtype=float)
    nodes = ox.distance.nearest_nodes(G_cc, X=pts[:,1], Y=pts[:,0])
    nodes = [n for n in dict.fromkeys(nodes) if n in G_cc.nodes]
    return pts.tolist(), nodes


def cluster_buckets(G_cc, depot_node, targets, V, seed=42):
    coords = np.array([[G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]] for n in targets], dtype=float)
    lat0 = float(coords[:, 0].mean())
    scale = math.cos(math.radians(lat0))
    X = np.column_stack((coords[:, 0], coords[:, 1] * scale))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=V, n_init=10, random_state=int(seed))
    _ = km.fit(X)
    labels = km.labels_
    buckets = [[] for _ in range(V)]
    for n, lab in zip(targets, labels):
        buckets[int(lab)].append(n)
    return buckets

# Small crop helper with cache
_SUB_CACHE = {}

def crop_subgraph_for_cluster_local(G_cc, cluster_nodes, depot_node, buffer_m=BUFFER_M):
    lat_list = [G_cc.nodes[n]["y"] for n in cluster_nodes + [depot_node]]
    lon_list = [G_cc.nodes[n]["x"] for n in cluster_nodes + [depot_node]]
    lat_min, lat_max = min(lat_list), max(lat_list)
    lon_min, lon_max = min(lon_list), max(lon_list)
    lat_buf = buffer_m / 111_320.0
    lon_buf = buffer_m / (111_320.0 * max(0.1, math.cos(math.radians((lat_min + lat_max)/2))))
    north = lat_max + lat_buf; south = lat_min - lat_buf; east = lon_max + lon_buf; west = lon_min - lon_buf
    poly = box(west, south, east, north)
    key = (tuple(sorted(cluster_nodes)), int(depot_node), int(buffer_m))
    if key in _SUB_CACHE: return _SUB_CACHE[key]
    G_sub = ox.truncate.truncate_graph_polygon(G_cc, poly, truncate_by_edge=True)
    try:
        G_sub = ox.truncate.largest_component(G_sub, strongly=False)
    except Exception:
        pass
    _SUB_CACHE[key] = G_sub
    return G_sub


def make_routing_funcs_astar_local(G_cc, Gud, fallback_mps):
    def heuristic(u, v):
        uy, ux = G_cc.nodes[u]["y"], G_cc.nodes[u]["x"]
        vy, vx = G_cc.nodes[v]["y"], G_cc.nodes[v]["x"]
        return haversine_m(uy, ux, vy, vx) / fallback_mps
    def tt(u,v):
        if u==v: return 0.0
        try: return nx.astar_path_length(G_cc, u, v, heuristic=heuristic, weight="travel_time")
        except Exception:
            try:
                L = nx.astar_path_length(Gud, u, v, heuristic=lambda a,b: 0.0, weight="length")
                return L / fallback_mps
            except Exception:
                return float('inf')
    def rn(u,v):
        if u==v: return [u]
        try: return nx.astar_path(G_cc, u, v, heuristic=heuristic, weight="travel_time")
        except Exception:
            try: return nx.astar_path(Gud, u, v, heuristic=lambda a,b:0.0, weight="length")
            except Exception: return []
    return tt, rn

if __name__ == '__main__':
    print('Starting smoke test...')
    t0 = time.perf_counter()
    G_cc, edges, cum, tot, timings = load_graph_once(WEST, SOUTH, EAST, NORTH)
    print(f'Graph load phases (s): {timings}')
    # estimate fallback speed
    speeds_kph = [d.get('speed_kph') for *_, d in G_cc.edges(data=True) if d.get('speed_kph') is not None]
    fallback_mps = (np.median(speeds_kph) if speeds_kph else 30.0) * 1000/3600.0
    t1 = time.perf_counter()
    pts, bin_nodes = generate_points_on_roads(edges, cum, tot, N_POINTS, MIN_SPACING_M, seed=SEED)
    print(f'Sampled points: {len(pts)}, unique nodes: {len(bin_nodes)}, sampling time: {time.perf_counter()-t1:.2f}s')
    t2 = time.perf_counter()
    depot_node = ox.distance.nearest_nodes(G_cc, float(np.mean([G_cc.nodes[n]['x'] for n in G_cc.nodes()])),
                                           float(np.mean([G_cc.nodes[n]['y'] for n in G_cc.nodes()])))
    buckets = cluster_buckets(G_cc, depot_node, bin_nodes, N_VEHICLES, seed=SEED)
    print('Buckets sizes:', [len(b) for b in buckets], 'clustering time:', time.perf_counter()-t2)
    # route clusters in parallel to measure
    def process(i, grp):
        G_sub = crop_subgraph_for_cluster_local(G_cc, grp, depot_node)
        Gud_sub = G_sub.to_undirected(as_view=True)
        tt, rn = make_routing_funcs_astar_local(G_sub, Gud_sub, fallback_mps)
        # map nodes
        mapped = []
        for orig in [depot_node] + grp:
            if orig in G_sub.nodes:
                mapped.append(orig)
            else:
                mapped.append(ox.distance.nearest_nodes(G_sub, G_cc.nodes[orig]['x'], G_cc.nodes[orig]['y']))
        seq = [mapped[0]]
        seq += mapped[1:]
        seq.append(mapped[0])
        # compute simple route lengths
        total_t = 0.0
        for a,b in zip(seq[:-1], seq[1:]):
            try:
                total_t += tt(a,b)
            except Exception:
                total_t += 0.0
        return {'vehicle': i, 'stops': max(0, len(grp)), 'travel_min': total_t/60.0}

    t3 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=min(6, N_VEHICLES)) as ex:
        futures = [ex.submit(process, i+1, grp) for i, grp in enumerate(buckets)]
        for f in as_completed(futures):
            results.append(f.result())
    t4 = time.perf_counter()
    print('Routing results:', results)
    print(f'End-to-end time: {t4 - t0:.2f}s (cropping+routing parallel time {t4-t3:.2f}s)')
