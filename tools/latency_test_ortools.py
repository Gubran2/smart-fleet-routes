"""Latency test using OR-Tools TSP per cluster.
Builds travel-time matrix per cluster using single-source Dijkstra and solves TSP with OR-Tools
with a controllable time limit. Prints timings for matrix build and solver.
"""
import time, math, random
import numpy as np
import networkx as nx
import osmnx as ox
from shapely.geometry import box, LineString, MultiLineString
from concurrent.futures import ThreadPoolExecutor, as_completed

# OR-Tools
try:
    from ortools.constraint_solver import pywrapcp
    from ortools.constraint_solver import routing_enums_pb2
    HAVE_ORTOOLS = True
except Exception:
    HAVE_ORTOOLS = False

# Parameters
WEST, SOUTH, EAST, NORTH = 36.71519, 34.00945, 36.74922, 34.03680
N_POINTS = 1000
N_VEHICLES = 2
MIN_SPACING_M = 40
BUFFER_M = 800
SEED = 42
MAX_TWO_OPT = 100
USE_TWO_OPT = True
VERTEX_SKIP = 6

# helpers
def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# build graph
print('Loading graph...')
poly = box(WEST, SOUTH, EAST, NORTH)
G = ox.graph_from_polygon(poly, network_type='drive', retain_all=False, simplify=True, truncate_by_edge=True)
G = ox.project_graph(G, to_crs='EPSG:4326')
G = ox.routing.add_edge_speeds(G)
G = ox.routing.add_edge_travel_times(G)
for _,_,_,d in G.edges(keys=True, data=True):
    if 'travel_time' in d:
        d['travel_time'] = float(d['travel_time'])
G_cc = ox.truncate.largest_component(G, strongly=False)

# collect edges for sampling
edges, cum, tot = [], [], 0.0
for u,v,k,d in G_cc.edges(keys=True, data=True):
    Lm = float(d.get('length', 0.0))
    if Lm <= 0: continue
    geom = d.get('geometry')
    if geom is None:
        geom = LineString([(G_cc.nodes[u]['x'], G_cc.nodes[u]['y']), (G_cc.nodes[v]['x'], G_cc.nodes[v]['y'])])
    edges.append((u,v,k,geom,Lm))
    tot += Lm; cum.append(tot)

# sample points
random.seed(int(SEED)); np.random.seed(int(SEED))
pts = []
tries = 0
max_tries = max(20000, N_POINTS * 40)
while len(pts) < N_POINTS and tries < max_tries:
    r = random.random()*tot
    lo, hi = 0, len(cum)-1
    while lo < hi:
        m = (lo+hi)//2
        if cum[m] < r: lo = m+1
        else: hi = m
    u,v,k,geom,Lm = edges[lo]
    t = random.random(); p = geom.interpolate(t*geom.length)
    pts.append((p.y, p.x))
    tries += 1
pts = np.array(pts)
nodes = ox.distance.nearest_nodes(G_cc, X=pts[:,1], Y=pts[:,0])
nodes = [n for n in dict.fromkeys(nodes) if n in G_cc.nodes]
print(f'Sampled {len(pts)} pts -> {len(nodes)} unique nodes')

# depot
depot = ox.distance.nearest_nodes(G_cc, float(np.mean([G_cc.nodes[n]['x'] for n in G_cc.nodes()])),
                                   float(np.mean([G_cc.nodes[n]['y'] for n in G_cc.nodes()])))

# clustering (simple KMeans on lat/lon)
coords = np.array([[G_cc.nodes[n]['y'], G_cc.nodes[n]['x']] for n in nodes], dtype=float)
from sklearn.cluster import KMeans
km = KMeans(n_clusters=N_VEHICLES, n_init=10, random_state=int(SEED)).fit(coords)
labels = km.labels_
buckets = [[] for _ in range(N_VEHICLES)]
for n,lab in zip(nodes, labels):
    buckets[int(lab)].append(n)
print('Buckets sizes:', [len(b) for b in buckets])

# function to build travel-time matrix via Dijkstra single-source runs

def build_tt_matrix(Gsub, idx_nodes, weight='travel_time'):
    # idx_nodes: list of node ids (depot + stops)
    N = len(idx_nodes)
    node_to_idx = {n:i for i,n in enumerate(idx_nodes)}
    M = np.full((N,N), np.inf, dtype=float)
    for i,src in enumerate(idx_nodes):
        # single-source shortest path lengths
        dist = nx.single_source_dijkstra_path_length(Gsub, src, weight=weight)
        for j, tgt in enumerate(idx_nodes):
            if tgt in dist:
                M[i,j] = dist[tgt]
    # replace inf with large number
    inf_mask = ~np.isfinite(M)
    if inf_mask.any():
        big = np.nanmax(M[np.isfinite(M)]) * 10 if np.any(np.isfinite(M)) else 1e9
        M[inf_mask] = big
    return M

# OR-Tools solver wrapper for TSP (1 vehicle, start=0)

def solve_tsp_ortools(time_matrix, time_limit_s=30):
    if not HAVE_ORTOOLS:
        raise RuntimeError('ortools not installed')
    N = time_matrix.shape[0]
    # integer cost matrix (ms -> int) to satisfy OR-Tools integer costs
    # Convert seconds to integer milliseconds
    int_mat = (time_matrix * 1000).astype(int).tolist()
    manager = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(manager)
    def time_callback(from_index, to_index):
        return int_mat[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    # search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(int(time_limit_s))
    search_parameters.log_search = False
    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        idx = routing.Start(0)
        order = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            order.append(node)
            idx = solution.Value(routing.NextVar(idx))
        order.append(manager.IndexToNode(idx))
        return order
    return None

# Run OR-Tools per cluster and measure
results = []
for i, grp in enumerate(buckets, 1):
    idx_nodes = [depot] + grp
    print(f'Building TT matrix for vehicle {i} ({len(idx_nodes)} nodes)...')
    t0 = time.perf_counter()
    Gsub = crop = ox.truncate.truncate_graph_polygon(G_cc, box(*(
        min(G_cc.nodes[n]['x'] for n in idx_nodes)-0.01,
        min(G_cc.nodes[n]['y'] for n in idx_nodes)-0.01,
        max(G_cc.nodes[n]['x'] for n in idx_nodes)+0.01,
        max(G_cc.nodes[n]['y'] for n in idx_nodes)+0.01)), truncate_by_edge=True)
    try:
        Gsub = ox.truncate.largest_component(Gsub, strongly=False)
    except Exception:
        pass
    tt_mat = build_tt_matrix(Gsub, idx_nodes, weight='travel_time')
    t1 = time.perf_counter()
    print(f'TT matrix built in {t1-t0:.2f}s; solving TSP (time limit 30s)...')
    t2 = time.perf_counter()
    try:
        order = solve_tsp_ortools(tt_mat, time_limit_s=30)
        t3 = time.perf_counter()
        print(f'Solver finished in {t3-t2:.2f}s; order length: {len(order) if order else None}')
        results.append({'vehicle': i, 'tt_build_s': t1-t0, 'solve_s': t3-t2, 'order_len': len(order) if order else None})
    except Exception as e:
        t3 = time.perf_counter()
        print('Solver error:', e)
        results.append({'vehicle': i, 'tt_build_s': t1-t0, 'solve_s': None, 'order_len': None})

print('All done. Results:')
print(results)
