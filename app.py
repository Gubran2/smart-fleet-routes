# -*- coding: utf-8 -*-
# Streamlit VRP Sandbox (OSMnx v2 + Folium + KMeans) — SIMPLE SYNC VERSION
# - Synkron körning: resultat visas direkt vid första klick
# - Startkarta visas direkt när appen öppnas

import math, random, functools, io, os
from typing import List
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, MultiLineString, box

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import AntPath, TimestampedGeoJson
import matplotlib.pyplot as plt

# Optional libs
try:
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ---------- OSMnx settings & caching ----------
ox.settings.use_cache = True
ox.settings.cache_folder = "osmnx_cache"
ox.settings.log_console = False

GRAPH_DIR = "graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

# ---------- UI ----------
st.set_page_config(page_title="VRP Sandbox (FAST)", layout="wide")
st.caption("VRP Sandbox — SIMPLE SYNC VERSION")
st.title("🚚 VRP Sandbox — OSMnx v2 + Streamlit (K-means, SPEED)")

PALETTE = [
    "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6",
    "#06b6d4", "#ec4899", "#84cc16", "#a855f7", "#f97316", "#22c55e"
]

# Decimate vertices from each edge geometry: keep every Nth coordinate
VERTEX_SKIP = 6
# Subgraph crop buffer (meters around cluster bbox + depot)
CROP_BUFFER_M = 800

# Performance / behavior toggles
CLUSTER_KEEP_LARGEST = True   # after cropping, keep only the largest connected component
# Simple in-memory cache for cropped subgraphs to avoid repeated OSMnx truncation
_SUBGRAPH_CACHE = {}

with st.sidebar:
    st.header("Parameters")
    st.subheader("Area (BBox)")
    WEST  = st.number_input("West (lon)",  value=36.71519, format="%.6f")
    SOUTH = st.number_input("South (lat)", value=34.00945, format="%.6f")
    EAST  = st.number_input("East (lon)",  value=36.74922, format="%.6f")
    NORTH = st.number_input("North (lat)", value=34.03680, format="%.6f")

    st.subheader("Problem Size")
    n_points   = st.slider("Number of points",   50, 1000, 300, step=10)
    n_vehicles = st.slider("Number of vehicles", 1, 12,    5)

    st.subheader("Travel & Service")
    city_factor   = st.slider("City time factor (signals/turns)", 1.00, 1.50, 1.30, step=0.05)
    min_spacing_m = st.slider("Min spacing between generated points (m)", 20, 200, 80, step=5)
    svc_min       = st.slider("Service time min (s)", 10, 240, 60, step=5)
    svc_max       = st.slider("Service time max (s)", 15, 300,120, step=5)

    st.subheader("Animation")
    animate_ant      = st.checkbox("Animate routes (AntPath)", value=False)
    animate_timeline = st.checkbox("Playback timeline (road-following)", value=False)
    timeline_step_s  = st.slider("Timeline step (seconds between points)", 0.5, 5.0, 1.5, 0.5)

    st.subheader("Layout & Extras")
    map_height = st.slider("Map height (px)", 480, 900, 620, step=10)
    use_two_opt = st.checkbox("Use 2-opt improvement", value=True)
    max_two_opt = st.slider("2-opt iterations (max)", 0, 400, 100, step=10)
    seed        = st.number_input("Random seed", value=42, step=1)

    run_btn   = st.button("▶️ Run optimization",  type="primary")
    reset_btn = st.button("♻️ Reset result", help="Rensa resultat från minnet")

if "vrp_result" not in st.session_state:
    st.session_state.vrp_result = None
if "veh_sel" not in st.session_state:
    st.session_state.veh_sel = []

# 🔄 Förbättrad reset – rensa även widget-state
if reset_btn:
    for key in ("vrp_result", "veh_sel", "veh_multiselect"):
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ---------- Utils ----------
def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

def _graph_key(w, s, e, n):
    return f"{w:.5f}_{s:.5f}_{e:.5f}_{n:.5f}"

@st.cache_resource(show_spinner=False)
def load_graph(west, south, east, north, city_factor):
    key = _graph_key(west, south, east, north)
    gpath = os.path.join(GRAPH_DIR, f"drive_{key}.graphml")

    if os.path.exists(gpath):
        G = ox.load_graphml(gpath)
    else:
        poly = box(west, south, east, north)
        G = ox.graph_from_polygon(poly, network_type="drive", retain_all=False,
                                  simplify=True, truncate_by_edge=True)
        ox.save_graphml(G, gpath)

    # Project, travel times, apply city factor
    G = ox.project_graph(G, to_crs="EPSG:4326")
    G = ox.routing.add_edge_speeds(G)              # ensure speed_kph exists
    G = ox.routing.add_edge_travel_times(G)
    for _, _, _, d in G.edges(keys=True, data=True):
        if "travel_time" in d:
            d["travel_time"] = float(d["travel_time"]) * float(city_factor)

    G_cc = ox.truncate.largest_component(G, strongly=False)
    Gud = G_cc.to_undirected(as_view=True)

    speeds_kph = [d.get("speed_kph") for *_, d in G_cc.edges(data=True) if d.get("speed_kph") is not None]
    fallback_mps = (np.median(speeds_kph) if speeds_kph else 30.0) * 1000/3600.0

    # Pre-collect edges list for sampling points
    edges, cum, tot = [], [], 0.0
    for u, v, k, d in G_cc.edges(keys=True, data=True):
        Lm = float(d.get("length", 0.0))
        if Lm <= 0:
            continue
        geom = d.get("geometry")
        if geom is None:
            geom = LineString([(G_cc.nodes[u]["x"], G_cc.nodes[u]["y"]),
                               (G_cc.nodes[v]["x"], G_cc.nodes[v]["y"])])
        edges.append((u, v, k, geom, Lm))
        tot += Lm; cum.append(tot)

    return G_cc, Gud, float(fallback_mps), edges, cum, float(tot)

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

def generate_points_on_roads(G_cc, edges, cum, tot, n_points, min_spacing_m, seed=42):
    random.seed(int(seed)); np.random.seed(int(seed))
    pts = []
    tries, max_tries = 0, max(8000, n_points * 20)

    if HAVE_SCIPY:
        kdt = None
        earth = 111_320.0
        while len(pts) < n_points and tries < max_tries:
            lat, lon = sample_point_on_edge(edges, cum, tot)
            if not pts:
                ok = True
            else:
                if kdt is None:
                    kdt = cKDTree(np.array(pts, dtype=float))
                ddeg, _ = kdt.query([lat, lon], k=1)
                ok = (ddeg * earth) >= float(min_spacing_m)
            if ok:
                pts.append((lat, lon))
                kdt = cKDTree(np.array(pts, dtype=float))
            tries += 1
    else:
        while len(pts) < n_points and tries < max_tries:
            lat, lon = sample_point_on_edge(edges, cum, tot)
            if all(haversine_m(lat, lon, la,lo) >= min_spacing_m for la,lo in pts):
                pts.append((lat, lon))
            tries += 1

    if not pts:
        return [], []
    pts = np.array(pts, dtype=float)
    nodes = ox.distance.nearest_nodes(G_cc, X=pts[:,1], Y=pts[:,0])
    nodes = [n for n in dict.fromkeys(nodes) if n in G_cc.nodes]
    return pts.tolist(), nodes

def pick_depot(G_cc, Gud, bin_nodes):
    xs = np.array([G_cc.nodes[n]["x"] for n in G_cc.nodes()])
    ys = np.array([G_cc.nodes[n]["y"] for n in G_cc.nodes()])
    cx, cy = float(xs.mean()), float(ys.mean())
    center_node = ox.distance.nearest_nodes(G_cc, cx, cy)
    close = nx.closeness_centrality(Gud, distance="length")
    topk = [n for n,_ in sorted(close.items(), key=lambda kv: kv[1], reverse=True)[:8]]
    rnd  = random.sample(bin_nodes, k=min(12, len(bin_nodes)))
    cand = list(dict.fromkeys([center_node] + topk + rnd))

    @functools.lru_cache(maxsize=200_000)
    def dir_time(u, v):
        try: return nx.shortest_path_length(G_cc, u, v, weight="travel_time")
        except nx.NetworkXNoPath: return float("inf")

    def score(src):
        cnt, tsum = 0, 0.0
        for n in bin_nodes:
            tt = dir_time(src, n)
            if math.isfinite(tt): cnt += 1; tsum += tt
        return cnt, tsum

    scored = [(n,)+score(n) for n in cand]
    best = sorted(scored, key=lambda t: (-t[1], t[2]))[0]
    return best[0], scored

# ---------- A* routing (fast) ----------
def make_routing_funcs_astar(G_cc, Gud, fallback_mps):
    def heuristic(u, v):
        uy, ux = G_cc.nodes[u]["y"], G_cc.nodes[u]["x"]
        vy, vx = G_cc.nodes[v]["y"], G_cc.nodes[v]["x"]
        # optimistic time = straight-line meters / fallback speed
        return haversine_m(uy, ux, vy, vx) / fallback_mps

    @functools.lru_cache(maxsize=200_000)
    def tt(u, v):
        if u == v: return 0.0
        try:
            return nx.astar_path_length(G_cc, u, v, heuristic=heuristic, weight="travel_time")
        except Exception:
            try:
                L = nx.astar_path_length(Gud, u, v, heuristic=lambda a,b: 0.0, weight="length")
                return L / fallback_mps
            except Exception:
                return float("inf")

    @functools.lru_cache(maxsize=100_000)
    def rn(u, v):
        if u == v: return [u]
        try:
            return nx.astar_path(G_cc, u, v, heuristic=heuristic, weight="travel_time")
        except Exception:
            try:
                return nx.astar_path(Gud, u, v, heuristic=lambda a,b: 0.0, weight="length")
            except Exception:
                return []

    return tt, rn

def route_nn_fast(G_cc, start, targets):
    if not targets: return [start, start]
    pts = {n: (G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]) for n in [start] + list(targets)}
    unv = set(targets)
    seq, cur = [start], start
    while unv:
        cy, cx = pts[cur]
        nxt = min(unv, key=lambda n: (pts[n][0]-cy)**2 + (pts[n][1]-cx)**2)
        seq.append(nxt); unv.remove(nxt); cur = nxt
    seq.append(start)
    return seq

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

# ---------- KMeans balanced clustering ----------
def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)

def cluster_buckets(G_cc, depot_node, targets: List[int], V: int, seed: int = 42):
    if len(targets) == 0 or V <= 0:
        return [[] for _ in range(max(1, V))]

    coords = np.array([[G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]] for n in targets], dtype=float)
    lat0 = float(coords[:, 0].mean())
    scale = math.cos(math.radians(lat0))
    X = np.column_stack((coords[:, 0], coords[:, 1] * scale))

    N = len(targets); V = min(V, N)
    base = N // V; rem  = N % V
    caps = np.array([base + (i < rem) for i in range(V)], dtype=int)

    if HAVE_SK:
        km = KMeans(n_clusters=V, n_init=10, random_state=int(seed))
        _ = km.fit(X)
        C = km.cluster_centers_
    else:
        rng = np.random.default_rng(int(seed))
        C = X[rng.choice(N, size=V, replace=False)]

    D = _pairwise_sq_dists(X, C)
    order = np.argsort(D.min(axis=1))
    assigned = -np.ones(N, dtype=int)
    cap_left = caps.copy()
    for idx in order:
        for k in np.argsort(D[idx]):
            if cap_left[k] > 0:
                assigned[idx] = k; cap_left[k] -= 1; break
    for idx in np.where(assigned == -1)[0]:
        k = int(np.argmax(cap_left)); assigned[idx] = k; cap_left[k] -= 1

    buckets = [[] for _ in range(V)]
    for i_pt, k in enumerate(assigned):
        buckets[int(k)].append(targets[i_pt])
    return buckets

# ---------- Edge/polyline helpers (with decimation) ----------
def _edge_data_minlen(G, u, v):
    data = G.get_edge_data(u, v, default=None)
    if not data: return None
    k, d = min(data.items(), key=lambda kv: kv[1].get("length", float("inf")))
    return k, d

def _add_edge_segment(G, a, b, fallback_mps):
    res = _edge_data_minlen(G, a, b)
    reverse = False
    if res is None:
        res = _edge_data_minlen(G, b, a)
        reverse = True if res is not None else False
    if res is not None:
        k, d = res
        y1, x1 = G.nodes[a]["y"], G.nodes[a]["x"]
        y2, x2 = G.nodes[b]["y"], G.nodes[b]["x"]
        geom = d.get("geometry")
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
        # Decimate vertices to speed up rendering
        if VERTEX_SKIP > 1 and len(coords) > (VERTEX_SKIP + 1):
            coords = coords[::VERTEX_SKIP] + [coords[-1]]
        length_m = float(d.get("length", 0.0))
        if length_m <= 0:
            length_m = 0.0
            for (xA,yA),(xB,yB) in zip(coords[:-1], coords[1:]):
                length_m += haversine_m(yA, xA, yB, xB)
        travel_s = float(d.get("travel_time", 0.0)) if d.get("travel_time") is not None else (length_m / fallback_mps)
        coords_ll = [(lat, lon) for (lon, lat) in coords]
        return coords_ll, length_m, travel_s
    # straight fallback
    y1, x1 = G.nodes[a]["y"], G.nodes[a]["x"]
    y2, x2 = G.nodes[b]["y"], G.nodes[b]["x"]
    length_m = haversine_m(y1, x1, y2, x2)
    travel_s = length_m / fallback_mps
    coords_ll = [(y1, x1), (y2, x2)]
    return coords_ll, length_m, travel_s

def build_poly_and_stats_for_path(G, nodes_path, fallback_mps):
    poly, dist_m, time_s = [], 0.0, 0.0
    first = True
    for a, b in zip(nodes_path[:-1], nodes_path[1:]):
        coords_ll, Lm, Ts = _add_edge_segment(G, a, b, fallback_mps)
        if not first and coords_ll:
            coords_ll = coords_ll[1:]
        first = False
        poly += coords_ll
        dist_m += Lm
        time_s += Ts
    return poly, dist_m, time_s

def build_timeline_coords_times(nodes_path, G, base_t, fallback_mps):
    coords_xy, times = [], []
    t = base_t
    first_edge = True
    for a, b in zip(nodes_path[:-1], nodes_path[1:]):
        coords_ll, Lm, Ts = _add_edge_segment(G, a, b, fallback_mps)
        coords = [(lon, lat) for (lat, lon) in coords_ll]
        if not coords:
            continue
        if not first_edge and len(coords) >= 1:
            coords = coords[1:]
        first_edge = False
        nseg = max(1, len(coords) - 1)
        for idx, (x, y) in enumerate(coords):
            frac = idx / nseg
            coords_xy.append([x, y])
            times.append((t + timedelta(seconds=Ts * frac)).strftime("%Y-%m-%d %H:%M:%S"))
        t += timedelta(seconds=Ts)
    return coords_xy, times

def process_cluster_task(G_cc, grp_nodes, depot_node, service_time_sec, fallback_mps,
                         use_two_opt, max_two_opt, buffer_m=CROP_BUFFER_M):
    """Top-level worker to process a cluster."""
    G_sub = crop_subgraph_for_cluster(G_cc, grp_nodes, depot_node, buffer_m=buffer_m)
    Gud_sub = G_sub.to_undirected(as_view=True)
    tt_fn, rn_fn = make_routing_funcs_astar(G_sub, Gud_sub, fallback_mps)

    mapped_nodes = []
    sub_to_orig = {}
    for orig in [depot_node] + grp_nodes:
        if orig in G_sub.nodes:
            mapped = orig
        else:
            y, x = G_cc.nodes[orig]["y"], G_cc.nodes[orig]["x"]
            mapped = ox.distance.nearest_nodes(G_sub, x, y)
        mapped_nodes.append(mapped)
        sub_to_orig[mapped] = orig
    start = mapped_nodes[0]
    grp_mapped = mapped_nodes[1:]

    seq = route_nn_fast(G_sub, start, grp_mapped)
    stops = seq[1:-1]
    if use_two_opt and len(stops) > 2 and max_two_opt > 0:
        if len(stops) > 200:
            time_budget = max(5.0, min(30.0, float(len(stops)) * 0.02))
            iter_budget = min(int(max_two_opt), 200)
        else:
            time_budget = None
            iter_budget = int(max_two_opt)
        stops = two_opt(tt_fn, start, stops, iters=iter_budget, time_budget_s=time_budget)
    seq = [start] + stops + [start]

    travel_sec, dist_m, order = 0.0, 0.0, 0
    poly = []
    path_cache = {}
    def rn_cached(a, b):
        key = (int(a), int(b))
        if key in path_cache:
            return path_cache[key]
        p = rn_fn(a, b)
        path_cache[key] = p
        return p

    for a, b in zip(seq[:-1], seq[1:]):
        nodes_path = rn_cached(a, b)
        if len(nodes_path) >= 2:
            seg_poly, seg_L, seg_T = build_poly_and_stats_for_path(G_sub, nodes_path, fallback_mps)
            poly += seg_poly
            dist_m += seg_L
            travel_sec += seg_T

    unique_stops = list(dict.fromkeys(seq[1:-1]))
    service_sec = 0.0
    rows = []
    for n in unique_stops:
        orig = sub_to_orig.get(n, None)
        if orig is not None:
            service_sec += float(service_time_sec.get(orig, 0.0))

    report = {
        "vehicle": -1,
        "stops": len(unique_stops),
        "travel_min": travel_sec/60.0,
        "service_min": service_sec/60.0,
        "total_min": (travel_sec+service_sec)/60.0,
        "dist_km": dist_m/1000.0
    }

    for n in unique_stops:
        order += 1
        y = G_sub.nodes[n]["y"] if n in G_sub.nodes else G_cc.nodes[n]["y"]
        x = G_sub.nodes[n]["x"] if n in G_sub.nodes else G_cc.nodes[n]["x"]
        rows.append({
            "vehicle": -1, "order": order, "node_id": int(n),
            "lat": y, "lon": x,
            "service_s": float(service_time_sec.get(sub_to_orig.get(n, n), 0.0)),
            "cluster": -1,
            "orig_node_id": int(sub_to_orig.get(n, n)),
        })

    return {"report": report, "rows": rows, "poly": poly, "route": seq}

# ---------- Cluster subgraph crop (v2-safe) ----------
def crop_subgraph_for_cluster(G_cc, cluster_nodes: List[int], depot_node: int, buffer_m=CROP_BUFFER_M):
    lat_list = [G_cc.nodes[n]["y"] for n in cluster_nodes + [depot_node]]
    lon_list = [G_cc.nodes[n]["x"] for n in cluster_nodes + [depot_node]]
    lat_min, lat_max = min(lat_list), max(lat_list)
    lon_min, lon_max = min(lon_list), max(lon_list)

    lat_buf = buffer_m / 111_320.0
    lon_buf = buffer_m / (111_320.0 * max(0.1, math.cos(math.radians((lat_min + lat_max)/2))))

    north = lat_max + lat_buf
    south = lat_min - lat_buf
    east  = lon_max + lon_buf
    west  = lon_min - lon_buf

    poly = box(west, south, east, north)

    key = (tuple(sorted(cluster_nodes)), int(depot_node), int(buffer_m),
           round(west, 6), round(south, 6), round(east, 6), round(north, 6))
    if key in _SUBGRAPH_CACHE:
        return _SUBGRAPH_CACHE[key]

    G_sub = ox.truncate.truncate_graph_polygon(G_cc, poly, truncate_by_edge=True)

    if CLUSTER_KEEP_LARGEST:
        try:
            G_sub = ox.truncate.largest_component(G_sub, strongly=False)
        except Exception:
            pass

    _SUBGRAPH_CACHE[key] = G_sub
    return G_sub

# ---------- Run (simple synchronous) ----------
if run_btn:
    with st.spinner("Building network and optimizing routes..."):
        G_cc, Gud, fallback_mps, edges, cum, tot = load_graph(WEST, SOUTH, EAST, NORTH, city_factor)
        pts, bin_nodes = generate_points_on_roads(G_cc, edges, cum, tot, n_points, min_spacing_m, seed)

        if not bin_nodes:
            st.session_state.vrp_result = None
        else:
            depot_node, diag = pick_depot(G_cc, Gud, bin_nodes)

            buckets = cluster_buckets(G_cc, depot_node, bin_nodes, n_vehicles, seed=seed)

            random.seed(int(seed)); np.random.seed(int(seed))
            service_time_sec = {n: float(random.uniform(svc_min, svc_max)) for n in bin_nodes}

            cl_rows = []
            for k, grp in enumerate(buckets):
                for n in grp:
                    cl_rows.append({
                        "node_id": n,
                        "cluster": k,
                        "lat": G_cc.nodes[n]["y"],
                        "lon": G_cc.nodes[n]["x"],
                    })
            cluster_df = pd.DataFrame(cl_rows)
            if not cluster_df.empty:
                centroids_df = (
                    cluster_df.groupby("cluster")[["lat", "lon"]]
                    .mean()
                    .reset_index()
                )
            else:
                centroids_df = pd.DataFrame()

            reports, all_rows, veh_polys, vehicle_routes = [], [], [], []

            # Du kan köra helt sekventiellt (ingen ThreadPool), men vi kan behålla enkel parallelism:
            max_workers = min(8, max(1, n_vehicles))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_cluster_task,
                        G_cc, grp, depot_node,
                        service_time_sec, fallback_mps,
                        use_two_opt, max_two_opt, CROP_BUFFER_M
                    ): i
                    for i, grp in enumerate(buckets, 1)
                }

                results_by_vehicle = {}
                for fut in futures:
                    i = futures[fut]
                    try:
                        resd = fut.result()
                    except Exception:
                        resd = {
                            "report": {
                                "vehicle": i, "stops": 0,
                                "travel_min": 0, "service_min": 0,
                                "total_min": 0, "dist_km": 0
                            },
                            "rows": [],
                            "poly": [],
                            "route": [depot_node, depot_node],
                        }
                    results_by_vehicle[i] = resd

            ordered = [results_by_vehicle.get(i) for i in range(1, len(buckets)+1)]
            for i, resd in enumerate(ordered, start=1):
                if resd is None:
                    reports.append({
                        "vehicle": i, "stops": 0,
                        "travel_min": 0, "service_min": 0,
                        "total_min": 0, "dist_km": 0
                    })
                    veh_polys.append([])
                    vehicle_routes.append([depot_node, depot_node])
                else:
                    rep = resd.get("report", {})
                    rep["vehicle"] = i
                    reports.append(rep)

                    rows = resd.get("rows", [])
                    for r in rows:
                        r["vehicle"] = i
                        r["cluster"] = i - 1
                        r["orig_node_id"] = int(r.get("orig_node_id", r.get("node_id", 0)))
                    all_rows.extend(rows)

                    veh_polys.append(resd.get("poly", []))
                    vehicle_routes.append(resd.get("route", [depot_node, depot_node]))

            timeline_features = []
            if animate_timeline:
                base_t = datetime(2025, 1, 1, 8, 0, 0)
                for i, seq in enumerate(vehicle_routes, 1):
                    grp_nodes = buckets[i-1]
                    G_sub = crop_subgraph_for_cluster(G_cc, grp_nodes, depot_node, buffer_m=CROP_BUFFER_M)
                    Gud_sub = G_sub.to_undirected(as_view=True)
                    _, rn_fn = make_routing_funcs_astar(G_sub, Gud_sub, fallback_mps)

                    all_xy, all_times = [], []
                    cur_t = base_t
                    first_pair = True
                    for a, b in zip(seq[:-1], seq[1:]):
                        aa = a if a in G_sub.nodes else ox.distance.nearest_nodes(
                            G_sub, G_cc.nodes[a]["x"], G_cc.nodes[a]["y"]
                        )
                        bb = b if b in G_sub.nodes else ox.distance.nearest_nodes(
                            G_sub, G_cc.nodes[b]["x"], G_cc.nodes[b]["y"]
                        )
                        nodes_path = rn_fn(aa, bb)
                        if len(nodes_path) < 2:
                            continue
                        xy, tlist = build_timeline_coords_times(nodes_path, G_sub, cur_t, fallback_mps)
                        if not first_pair and len(xy) >= 1:
                            xy = xy[1:]; tlist = tlist[1:]
                        first_pair = False
                        all_xy += xy; all_times += tlist
                        if tlist:
                            last_t = datetime.strptime(tlist[-1], "%Y-%m-%d %H:%M:%S")
                            cur_t = last_t

                    if len(all_xy) >= 2:
                        timeline_features.append({
                            "type": "Feature",
                            "geometry": {"type": "LineString", "coordinates": all_xy},
                            "properties": {
                                "times": all_times,
                                "style": {
                                    "color": PALETTE[(i-1) % len(PALETTE)],
                                    "weight": 5
                                }
                            }
                        })

            depot_lat, depot_lon = G_cc.nodes[depot_node]["y"], G_cc.nodes[depot_node]["x"]
            st.session_state.vrp_result = {
                "depot": (depot_lat, depot_lon),
                "bin_nodes": [(G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]) for n in bin_nodes],
                "reports_df": pd.DataFrame(reports).round(2),
                "stops_df": pd.DataFrame(all_rows),
                "veh_polys": veh_polys,
                "cluster_df": cluster_df,
                "centroids_df": centroids_df,
                "vehicle_routes": vehicle_routes,
                "timeline_features": timeline_features,
            }

# ---------- Render ----------
res = st.session_state.vrp_result

if res is None:
    # Visa en enkel karta direkt när appen öppnas
    st.subheader("Interactive Map", divider="gray")
    center_lat = (NORTH + SOUTH) / 2.0
    center_lon = (EAST + WEST) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles="OpenStreetMap")
    folium.Rectangle(
        bounds=[[SOUTH, WEST], [NORTH, EAST]],
        color="#2563eb",
        fill=False,
        weight=2,
    ).add_to(m)
    st_folium(m, height=int(map_height), width=None)
    st.info("Välj inställningar i sidofältet och klicka **Run optimization** för att generera rutter.")
else:
    st.subheader("Interactive Map", divider="gray")

    # 🚚 Säker hantering av multiselect-state
    veh_labels = [f"Vehicle {i+1}" for i in range(len(res["veh_polys"]))] or []

    # Rensa gamla värden som inte längre finns i veh_labels
    if "veh_multiselect" in st.session_state:
        st.session_state.veh_multiselect = [
            v for v in st.session_state.veh_multiselect
            if v in veh_labels
        ]

    # Bestäm default-val
    current = st.session_state.get("veh_multiselect", None)
    if current:
        default_sel = [v for v in current if v in veh_labels]
    else:
        default_sel = veh_labels

    veh_sel = st.multiselect(
        "Show vehicles",
        options=veh_labels,
        default=default_sel,
        key="veh_multiselect",
    )
    st.session_state.veh_sel = veh_sel

    show_bins = st.checkbox("Show bin points (cluster colors)", value=False, key="bins_chk")

    depot_lat, depot_lon = res["depot"]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker(
        [depot_lat, depot_lon],
        radius=7, color="green", fill=True, fill_opacity=1, tooltip="Depot"
    ).add_to(m)

    if show_bins:
        bins_fg = folium.FeatureGroup(name=f"Bins ({len(res['bin_nodes'])})", show=True)
        if "cluster_df" in res and not res["cluster_df"].empty:
            for row in res["cluster_df"].itertuples():
                color = PALETTE[int(row.cluster) % len(PALETTE)]
                folium.CircleMarker(
                    [row.lat, row.lon], radius=2,
                    color=color, fill=True, fill_opacity=0.9
                ).add_to(bins_fg)
            if "centroids_df" in res and not res["centroids_df"].empty:
                for row in res["centroids_df"].itertuples():
                    folium.CircleMarker(
                        [row.lat, row.lon], radius=6, color="#111",
                        fill=True, fill_opacity=0.6,
                        tooltip=f"Cluster {int(row.cluster)} centroid"
                    ).add_to(bins_fg)
        else:
            for (y, x) in res["bin_nodes"]:
                folium.CircleMarker(
                    [y, x], radius=2,
                    color="#ff3b30", fill=True, fill_opacity=0.9
                ).add_to(bins_fg)
        bins_fg.add_to(m)

    for i, poly in enumerate(res["veh_polys"], 1):
        label = f"Vehicle {i}"
        if label not in veh_sel:
            continue
        grp = folium.FeatureGroup(name=label, show=True)
        if len(poly) >= 2:
            if animate_ant:
                AntPath(
                    locations=poly, weight=4, opacity=0.9,
                    color=PALETTE[(i-1) % len(PALETTE)],
                    dash_array=[10, 20], delay=1000
                ).add_to(grp)
            else:
                folium.PolyLine(
                    poly, weight=4, opacity=0.9,
                    color=PALETTE[(i-1) % len(PALETTE)],
                    tooltip=label
                ).add_to(grp)
        grp.add_to(m)

    if animate_timeline and res.get("timeline_features"):
        fc = {"type": "FeatureCollection", "features": res["timeline_features"]}
        TimestampedGeoJson(
            data=fc,
            transition_time=int(timeline_step_s * 1000),
            loop=True, auto_play=True, add_last_point=True,
            period="PT1S", speed_slider=True, loop_button=True
        ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    all_coords = (
        [res["depot"]] +
        [p for poly in res["veh_polys"] for p in poly] +
        (res["bin_nodes"] if show_bins else [])
    )
    if all_coords:
        m.fit_bounds([
            [min(y for y, _ in all_coords), min(x for _, x in all_coords)],
            [max(y for y, _ in all_coords), max(x for _, x in all_coords)],
        ])

    st_folium(m, height=int(map_height), width=None)

    # Cluster plot
    st.subheader("Cluster plot", divider="gray")
    if "cluster_df" in res and not res["cluster_df"].empty:
        with st.expander("Visa & spara PNG", expanded=True):
            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=150)
            for k, grp in res["cluster_df"].groupby("cluster"):
                ax.scatter(
                    grp["lon"].values, grp["lat"].values,
                    s=12, alpha=0.9, label=f"Cluster {int(k)}"
                )
            dep_lat, dep_lon = res["depot"]
            ax.scatter(
                [dep_lon], [dep_lat],
                marker="*", s=180, edgecolors="k",
                linewidths=0.8, label="Depot"
            )
            if "centroids_df" in res and not res["centroids_df"].empty:
                ax.scatter(
                    res["centroids_df"]["lon"].values,
                    res["centroids_df"]["lat"].values,
                    marker="x", s=60, linewidths=1.5, label="Centroids"
                )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Clusters (colored) with depot / centroids")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "⬇️ Download cluster plot (PNG)",
                data=buf.getvalue(),
                file_name="clusters.png",
                mime="image/png"
            )
            plt.close(fig)

    # KPIs
    st.subheader("KPIs & Details", divider="gray")
    df = res["reports_df"].copy()
    if not df.empty:
        df["travel_h"] = (df["travel_min"] / 60).round(2)
        df["service_h"] = (df["service_min"] / 60).round(2)
        df["total_h"]   = (df["total_min"]  / 60).round(2)
        total_time_h = float(df["total_h"].sum())
        longest_h    = float(df["total_h"].max())
    else:
        total_time_h, longest_h = 0.0, 0.0
    coverage = f"{res['stops_df']['node_id'].nunique()}/{len(res['bin_nodes'])}"

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    c1.metric("Coverage", coverage)
    c2.metric("Fleet total (h)", f"{total_time_h:.2f}")
    c3.metric("Longest vehicle (h)", f"{longest_h:.2f}")
    c4.metric("Vehicles", f"{len(df)}")

    st.markdown("**Per-vehicle hours**")
    st.dataframe(
        df[["vehicle", "stops", "travel_h", "service_h", "total_h", "dist_km"]]
        .rename(columns={"vehicle":"veh"}),
        height=260,
        use_container_width=True
    )

    with st.expander("Stops (sample) & Download", expanded=False):
        st.dataframe(res["stops_df"].head(300), height=240, use_container_width=True)
        st.download_button(
            "⬇️ Download routes CSV",
            data=res["stops_df"].to_csv(index=False),
            file_name="routes.csv",
            mime="text/csv"
        )
