"""Latency test that mirrors the app pipeline (includes adaptive two_opt and path cache).
Run as: python tools\latency_test_appflow.py
"""
import time, math, random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import box, LineString, MultiLineString

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

# Helpers
def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

# two_opt (same behavior as app)
def two_opt(tt_fn, start, stops, iters=100, time_budget_s=None):
    if len(stops) < 3: return stops
    import time
    best = stops[:]
    def cost(path): return sum(tt_fn(a, b) for a,b in zip([start]+path, path+[start]))
    best_cost = cost(best)
    start_t = time.perf_counter()
    max_iters = int(iters)
    i_iter = 0
    while i_iter < max_iters:
        if time_budget_s is not None and (time.perf_counter() - start_t) >= float(time_budget_s):
            break
        improved = False
        n = len(best)
        for i in range(n-1):
            for j in range(i+2, n):
                if time_budget_s is not None and (time.perf_counter() - start_t) >= float(time_budget_s):
                    break
                new = best[:i] + best[i:j][::-1] + best[j:]
                c = cost(new)
                if c < best_cost - 1e-6:
                    best, best_cost = new, c
                    improved = True
                    break
            if improved: break
        if not improved:
            break
        i_iter += 1
    return best

# routing funcs

def make_routing_funcs_astar(G_cc, Gud, fallback_mps):
    def heuristic(u, v):
        uy, ux = G_cc.nodes[u]['y'], G_cc.nodes[u]['x']
        vy, vx = G_cc.nodes[v]['y'], G_cc.nodes[v]['x']
        return haversine_m(uy, ux, vy, vx) / fallback_mps
    def tt(u,v):
        if u==v: return 0.0
        try: return nx.astar_path_length(G_cc, u, v, heuristic=heuristic, weight='travel_time')
        except Exception:
            try:
                L = nx.astar_path_length(Gud, u, v, heuristic=lambda a,b: 0.0, weight='length')
                return L / fallback_mps
            except Exception:
                return float('inf')
    def rn(u,v):
        if u==v: return [u]
        try: return nx.astar_path(G_cc, u, v, heuristic=heuristic, weight='travel_time')
        except Exception:
            try: return nx.astar_path(Gud, u, v, heuristic=lambda a,b:0.0, weight='length')
            except Exception: return []
    return tt, rn

# build poly (lightweight)
def _edge_data_minlen(G, u, v):
    data = G.get_edge_data(u, v, default=None)
    if not data: return None
    k, d = min(data.items(), key=lambda kv: kv[1].get('length', float('inf')))
    return k, d

def _add_edge_segment(G, a, b, fallback_mps, VERTEX_SKIP=6):
    res = _edge_data_minlen(G, a, b)
    reverse = False
    if res is None:
        res = _edge_data_minlen(G, b, a)
        reverse = True if res is not None else False
    if res is not None:
        k, d = res
        y1, x1 = G.nodes[a]['y'], G.nodes[a]['x']
        y2, x2 = G.nodes[b]['y'], G.nodes[b]['x']
        geom = d.get('geometry')
        if geom is None:
            coords = [(x1, y1), (x2, y2)]
        else:
            if isinstance(geom, LineString):
                coords = list(geom.coords)
            elif isinstance(geom, MultiLineString):
                coords = []
                for ls in geom.geoms:
                    coords += list(ls.coords)
            else:
                coords = [(x1, y1), (x2, y2)]
        if reverse:
            coords = list(reversed(coords))
        if VERTEX_SKIP > 1 and len(coords) > (VERTEX_SKIP + 1):
            coords = coords[::VERTEX_SKIP] + [coords[-1]]
        length_m = float(d.get('length', 0.0))
        if length_m <= 0:
            length_m = 0.0
            for (xA,yA),(xB,yB) in zip(coords[:-1], coords[1:]):
                length_m += haversine_m(yA, xA, yB, xB)
        travel_s = float(d.get('travel_time', 0.0)) if d.get('travel_time') is not None else (length_m / (10.0))
        coords_ll = [(lat, lon) for (lon, lat) in coords]
        return coords_ll, length_m, travel_s
    y1, x1 = G.nodes[a]['y'], G.nodes[a]['x']
    y2, x2 = G.nodes[b]['y'], G.nodes[b]['x']
    length_m = haversine_m(y1, x1, y2, x2)
    travel_s = length_m / 10.0
    coords_ll = [(y1, x1), (y2, x2)]
    return coords_ll, length_m, travel_s

# process cluster

def process_cluster_task(G_cc, grp_nodes, depot_node, service_time_sec, fallback_mps,
                         use_two_opt, max_two_opt, buffer_m=BUFFER_M):
    G_sub = crop_subgraph_for_cluster(G_cc, grp_nodes, depot_node, buffer_m)
    Gud_sub = G_sub.to_undirected(as_view=True)
    tt_fn, rn_fn = make_routing_funcs_astar(G_sub, Gud_sub, fallback_mps)
    mapped_nodes = []
    sub_to_orig = {}
    for orig in [depot_node] + grp_nodes:
        if orig in G_sub.nodes:
            mapped = orig
        else:
            y, x = G_cc.nodes[orig]['y'], G_cc.nodes[orig]['x']
            mapped = ox.distance.nearest_nodes(G_sub, x, y)
        mapped_nodes.append(mapped)
        sub_to_orig[mapped] = orig
    start = mapped_nodes[0]
    grp_mapped = mapped_nodes[1:]
    seq = [start] + grp_mapped + [start]
    stops = seq[1:-1]
    if use_two_opt and len(stops) > 2 and max_two_opt > 0:
        if len(stops) > 200:
            time_budget = max(5.0, min(30.0, float(len(stops)) * 0.02))
            iter_budget = min(int(max_two_opt), 200)
        else:
            time_budget = None
            iter_budget = int(max_two_opt)
        seq = [start] + two_opt(tt_fn, start, stops, iters=iter_budget, time_budget_s=time_budget) + [start]
    # build polylines with caching
    path_cache = {}
    travel_sec = 0.0
    dist_m = 0.0
    for a,b in zip(seq[:-1], seq[1:]):
        key = (int(a), int(b))
        if key in path_cache:
            nodes_path = path_cache[key]
        else:
            nodes_path = rn_fn(a,b)
            path_cache[key] = nodes_path
        if len(nodes_path) >= 2:
            _, Lm, Ts = _add_edge_segment(G_sub, nodes_path[0], nodes_path[-1], fallback_mps, VERTEX_SKIP)
            dist_m += Lm
            travel_sec += Ts
    unique_stops = list(dict.fromkeys(seq[1:-1]))
    service_sec = sum(float(service_time_sec.get(sub_to_orig.get(n, n), 0.0)) for n in unique_stops)
    return {'stops': len(unique_stops), 'travel_min': travel_sec/60.0, 'dist_km': dist_m/1000.0}

# crop function similar to app

def crop_subgraph_for_cluster(G_cc, cluster_nodes, depot_node, buffer_m=BUFFER_M):
    lat_list = [G_cc.nodes[n]['y'] for n in cluster_nodes + [depot_node]]
    lon_list = [G_cc.nodes[n]['x'] for n in cluster_nodes + [depot_node]]
    lat_min, lat_max = min(lat_list), max(lat_list)
    lon_min, lon_max = min(lon_list), max(lon_list)
    lat_buf = buffer_m / 111320.0
    lon_buf = buffer_m / (111320.0 * max(0.1, math.cos(math.radians((lat_min + lat_max)/2))))
    north = lat_max + lat_buf; south = lat_min - lat_buf; east = lon_max + lon_buf; west = lon_min - lon_buf
    poly = box(west, south, east, north)
    G_sub = ox.truncate.truncate_graph_polygon(G_cc, poly, truncate_by_edge=True)
    try:
        G_sub = ox.truncate.largest_component(G_sub, strongly=False)
    except Exception:
        pass
    return G_sub

# main
if __name__ == '__main__':
    print('Starting latency appflow test...')
    t0 = time.perf_counter()
    poly = box(WEST, SOUTH, EAST, NORTH)
    G = ox.graph_from_polygon(poly, network_type='drive', retain_all=False, simplify=True, truncate_by_edge=True)
    G = ox.project_graph(G, to_crs='EPSG:4326')
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    for _,_,_,d in G.edges(keys=True, data=True):
        if 'travel_time' in d:
            d['travel_time'] = float(d['travel_time'])
    G_cc = ox.truncate.largest_component(G, strongly=False)
    # collect edges
    edges, cum, tot = [], [], 0.0
    for u,v,k,d in G_cc.edges(keys=True, data=True):
        Lm = float(d.get('length', 0.0))
        if Lm <= 0: continue
        geom = d.get('geometry')
        if geom is None:
            geom = LineString([(G_cc.nodes[u]['x'], G_cc.nodes[u]['y']), (G_cc.nodes[v]['x'], G_cc.nodes[v]['y'])])
        edges.append((u,v,k,geom,Lm))
        tot += Lm; cum.append(tot)
    t1 = time.perf_counter()
    # sampling
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
    t2 = time.perf_counter()
    speeds_kph = [d.get('speed_kph') for *_, d in G_cc.edges(data=True) if d.get('speed_kph') is not None]
    fallback_mps = (np.median(speeds_kph) if speeds_kph else 30.0) * 1000/3600.0
    depot_node = ox.distance.nearest_nodes(G_cc, float(np.mean([G_cc.nodes[n]['x'] for n in G_cc.nodes()])),
                                           float(np.mean([G_cc.nodes[n]['y'] for n in G_cc.nodes()])))
    buckets = []
    # simple kmeans via sklearn
    coords = np.array([[G_cc.nodes[n]['y'], G_cc.nodes[n]['x']] for n in nodes], dtype=float)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=N_VEHICLES, n_init=10, random_state=int(SEED)).fit(coords)
    labels = km.labels_
    for i in range(N_VEHICLES):
        buckets.append([n for n,l in zip(nodes, labels) if l==i])
    t3 = time.perf_counter()
    service_time_sec = {n: float(random.uniform(60,120)) for n in nodes}

    # sequential processing
    tseq0 = time.perf_counter()
    seq_results = []
    for i, grp in enumerate(buckets, 1):
        r = process_cluster_task(G_cc, grp, depot_node, service_time_sec, fallback_mps, USE_TWO_OPT, MAX_TWO_OPT, BUFFER_M)
        seq_results.append(r)
    tseq1 = time.perf_counter()

    # parallel processing
    tpar0 = time.perf_counter()
    par_results = []
    with ThreadPoolExecutor(max_workers=min(4, N_VEHICLES)) as ex:
        futures = [ex.submit(process_cluster_task, G_cc, grp, depot_node, service_time_sec, fallback_mps, USE_TWO_OPT, MAX_TWO_OPT, BUFFER_M) for grp in buckets]
        for f in as_completed(futures):
            par_results.append(f.result())
    tpar1 = time.perf_counter()

    print('Timings:')
    print('graph build+collect: {:.2f}s'.format(t1-t0))
    print('sampling: {:.2f}s'.format(t2-t1))
    print('clustering: {:.2f}s'.format(t3-t2))
    print('sequential clusters processing: {:.2f}s'.format(tseq1-tseq0))
    print('parallel clusters processing: {:.2f}s'.format(tpar1-tpar0))
    print('total: {:.2f}s'.format(time.perf_counter()-t0))
    print('seq results sample:', seq_results[:3])
    print('par results sample:', par_results[:3])
