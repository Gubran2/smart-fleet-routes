# -*- coding: utf-8 -*-
# Streamlit VRP Sandbox (OSMnx v2 + Folium + KMeans clustering)
# - Big map on top (adjustable), compact info below
# - Per-vehicle hours + fleet totals
# - Route animation: AntPath and TimestampedGeoJson timeline
# - Cluster coloring on map + downloadable PNG cluster plot

import math, random, functools, io
from typing import List, Tuple, Dict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import networkx as nx
import osmnx as ox
from shapely.geometry import box, LineString

import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import AntPath, TimestampedGeoJson  # animation plugins
import matplotlib.pyplot as plt

# Optional: scikit-learn for KMeans (recommended)
try:
    from sklearn.cluster import KMeans
    HAVE_SK = True
except Exception:
    HAVE_SK = False

# ---------- Page & style ----------
st.set_page_config(page_title="VRP Sandbox (OSMnx v2)", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 0.5rem; padding-bottom: 0.25rem;}
h2, h3, h4 { margin-top: 0.6rem; margin-bottom: 0.4rem; }
.small-info div[data-testid="stMarkdownContainer"] { font-size: 0.95rem; }
.small-info .stDataFrame, .small-info table { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.title("🚚 VRP Sandbox — OSMnx v2 + Streamlit (K-means)")
st.caption("Klustrar punkter per bil (K-means), ruttar varje kluster (NN + 2-opt), och visar KPI:er.")

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Parameters")

    st.subheader("Area (BBox)")
    WEST  = st.number_input("West (lon)",  value=36.71519, format="%.6f")
    SOUTH = st.number_input("South (lat)", value=34.00945, format="%.6f")
    EAST  = st.number_input("East (lon)",  value=36.74922, format="%.6f")
    NORTH = st.number_input("North (lat)", value=34.03680, format="%.6f")

    st.subheader("Problem Size")
    n_points   = st.slider("Number of points",   50, 1000, 300, step=10)
    n_vehicles = st.slider("Number of vehicles", 1, 12,    2)

    st.subheader("Travel & Service")
    city_factor   = st.slider("City time factor (signals/turns)", 1.00, 1.50, 1.30, step=0.05)
    min_spacing_m = st.slider("Min spacing between generated points (m)", 20, 200, 80, step=5)
    svc_min       = st.slider("Service time min (s)", 10, 240, 60, step=5)
    svc_max       = st.slider("Service time max (s)", 15, 300,120, step=5)

    st.subheader("Animation")
    animate_ant      = st.checkbox("Animate routes (AntPath)", value=False,
                                   help="Animerad streckad linje ovanpå rutten.")
    animate_timeline = st.checkbox("Playback timeline (TimestampedGeoJson)", value=False,
                                   help="Tidslinje som spelar upp varje bils rutt.")
    timeline_step_s  = st.slider("Timeline step (seconds between points)", 0.5, 5.0, 1.0, 0.5)
    downsample_n     = st.slider("Downsample every Nth vertex", 1, 20, 6)

    st.subheader("Layout")
    map_height = st.slider("Map height (px)", 480, 900, 620, step=10)

    st.subheader("Extras")
    use_two_opt = st.checkbox("Use 2-opt improvement", value=True)
    max_two_opt = st.slider("2-opt iterations", 0, 500, 150, step=10)
    seed        = st.number_input("Random seed", value=42, step=1)

    run_btn   = st.button("▶️ Run optimization",  type="primary")
    reset_btn = st.button("♻️ Reset result", help="Rensa resultat från minnet")

# ---------- Persistent state ----------
if "vrp_result" not in st.session_state:
    st.session_state.vrp_result = None
if "veh_sel" not in st.session_state:
    st.session_state.veh_sel = []

if reset_btn:
    st.session_state.vrp_result = None
    st.session_state.veh_sel = []
    st.rerun()

# ---------- Custom speeds ----------
CUSTOM_SPEEDS = {
    "motorway":80, "motorway_link":60,
    "trunk":70, "trunk_link":60,
    "primary":60, "primary_link":50,
    "secondary":50, "secondary_link":45,
    "tertiary":40, "tertiary_link":35,
    "residential":30, "living_street":20,
    "service":20, "unclassified":35
}

@st.cache_resource(show_spinner=False)
def load_graph(west, south, east, north, city_factor):
    poly = box(west, south, east, north)
    G = ox.graph_from_polygon(poly, network_type="drive", retain_all=True,
                              simplify=True, truncate_by_edge=True)
    G = ox.routing.add_edge_speeds(G, hwy_speeds=CUSTOM_SPEEDS)
    G = ox.routing.add_edge_travel_times(G)
    for _, _, _, d in G.edges(keys=True, data=True):
        if "travel_time" in d:
            d["travel_time"] = float(d["travel_time"]) * float(city_factor)
    G = ox.project_graph(G, to_crs="EPSG:4326")
    G_cc = ox.truncate.largest_component(G, strongly=False)
    Gud = G_cc.to_undirected(as_view=True)

    speeds_kph = [d.get("speed_kph") for *_, d in G_cc.edges(data=True) if d.get("speed_kph") is not None]
    fallback_mps = (np.median(speeds_kph) if speeds_kph else 30.0) * 1000/3600.0

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
    return G_cc, Gud, float(fallback_mps), edges, cum, float(tot)

def haversine_m(lat1, lon1, lat2, lon2):
    R=6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))

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

def generate_points_on_roads(G_cc, edges, cum, tot, n_points, min_gap_m, seed=42):
    random.seed(int(seed)); np.random.seed(int(seed))
    pts, tries = [], 0
    max_tries = max(10000, n_points*30)
    while len(pts) < n_points and tries < max_tries:
        lat, lon = sample_point_on_edge(edges, cum, tot)
        if all(haversine_m(lat, lon, la, lo) >= min_gap_m for la,lo in pts):
            pts.append((lat, lon))
        tries += 1
    nodes = [ox.distance.nearest_nodes(G_cc, lo, la) for (la,lo) in pts]
    nodes = list(dict.fromkeys([n for n in nodes if n in G_cc.nodes]))
    return pts, nodes

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

def make_routing_funcs(G_cc, Gud, fallback_mps):
    @functools.lru_cache(maxsize=200_000)
    def tt(u, v):
        if u == v: return 0.0
        try: return nx.shortest_path_length(G_cc, u, v, weight="travel_time")
        except nx.NetworkXNoPath:
            try:
                L = nx.shortest_path_length(Gud, u, v, weight="length")
                return L / fallback_mps
            except Exception:
                return float("inf")

    @functools.lru_cache(maxsize=100_000)
    def rn(u, v):
        if u == v: return [u]
        try: return nx.shortest_path(G_cc, u, v, weight="travel_time")
        except nx.NetworkXNoPath:
            try: return nx.shortest_path(Gud, u, v, weight="length")
            except Exception: return []

    return tt, rn

def route_nn(tt_fn, start, targets):
    unvisited = set(targets)
    seq, cur = [start], start
    while unvisited:
        nxt = min(unvisited, key=lambda n: tt_fn(cur, n))
        if math.isinf(tt_fn(cur, nxt)):
            unvisited.remove(nxt); continue
        seq.append(nxt); cur = nxt; unvisited.remove(nxt)
    seq.append(start)
    return seq

def two_opt(tt_fn, start, stops, iters=150):
    if len(stops) < 3: return stops
    best = stops[:]
    def cost(path): return sum(tt_fn(a, b) for a,b in zip([start]+path, path+[start]))
    best_cost = cost(best)
    for _ in range(iters):
        improved = False
        for i in range(len(best)-1):
            for j in range(i+2, len(best)):
                new = best[:i] + best[i:j][::-1] + best[j:]
                c = cost(new)
                if c < best_cost - 1e-6:
                    best, best_cost = new, c; improved = True; break
            if improved: break
        if not improved: break
    return best

# ---------- KMeans-based balanced clustering ----------
def _pairwise_sq_dists(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    # X: (N,2), C: (K,2) -> D: (N,K)
    return ((X[:, None, :] - C[None, :, :])**2).sum(axis=2)

def cluster_buckets(G_cc, depot_node, targets: List[int], V: int, seed: int = 42):
    """
    Dela 'targets' i V geografiska kluster (ett per bil) med K-means.
    Med balanserad tilldelning så att klusterstorlekar ~lika.
    Returnerar: List[List[node_id]] längd V.
    """
    if len(targets) == 0 or V <= 0:
        return [[] for _ in range(max(1, V))]

    coords = np.array([[G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]] for n in targets], dtype=float)
    lat0 = float(coords[:, 0].mean())
    scale = math.cos(math.radians(lat0))  # meter-liknande skalning för lon
    X = np.column_stack((coords[:, 0], coords[:, 1] * scale))

    N = len(targets)
    V = min(V, N)  # aldrig fler kluster än punkter

    base = N // V
    rem = N % V
    caps = np.array([base + (i < rem) for i in range(V)], dtype=int)

    # KMeans för centrumpunkter
    if HAVE_SK:
        km = KMeans(n_clusters=V, n_init=10, random_state=int(seed))
        _ = km.fit(X)
        C = km.cluster_centers_
    else:
        rng = np.random.default_rng(int(seed))
        inds = rng.choice(N, size=V, replace=False)
        C = X[inds]

    # Balanserad närmsta-centroid-tilldelning
    D = _pairwise_sq_dists(X, C)               # (N,V)
    order = np.argsort(D.min(axis=1))          # börja med tydligast nära
    assigned = -np.ones(N, dtype=int)
    cap_left = caps.copy()

    for idx in order:
        prefs = np.argsort(D[idx])             # närmast → längst bort
        for k in prefs:
            if cap_left[k] > 0:
                assigned[idx] = k
                cap_left[k] -= 1
                break

    # Ev. overflow fallback
    for idx in np.where(assigned == -1)[0]:
        k = int(np.argmax(cap_left))
        assigned[idx] = k
        cap_left[k] -= 1

    # Bygg buckets i original node-id
    buckets = [[] for _ in range(V)]
    for i_pt, k in enumerate(assigned):
        buckets[int(k)].append(targets[i_pt])

    return buckets

# ------------- Compute & store in session_state -------------
if run_btn:
    with st.spinner("Building network and optimizing routes..."):
        G_cc, Gud, fallback_mps, edges, cum, tot = load_graph(WEST, SOUTH, EAST, NORTH, city_factor)
        pts, bin_nodes = generate_points_on_roads(G_cc, edges, cum, tot, n_points, min_spacing_m, seed)
        if not bin_nodes:
            st.session_state.vrp_result = None
        else:
            depot_node, diag = pick_depot(G_cc, Gud, bin_nodes)
            depot_lat, depot_lon = G_cc.nodes[depot_node]["y"], G_cc.nodes[depot_node]["x"]
            tt_fn, rn_fn = make_routing_funcs(G_cc, Gud, fallback_mps)

            # === KMEANS CLUSTERING ===
            buckets = cluster_buckets(G_cc, depot_node, bin_nodes, n_vehicles, seed=seed)

            # Service times
            random.seed(int(seed)); np.random.seed(int(seed))
            service_time_sec = {n: float(random.uniform(svc_min, svc_max)) for n in bin_nodes}

            # Build cluster mapping for plotting
            cluster_map = {}
            cl_rows = []
            for k, grp in enumerate(buckets):
                for n in grp:
                    cluster_map[n] = k
                    cl_rows.append({
                        "node_id": n,
                        "cluster": k,
                        "lat": G_cc.nodes[n]["y"],
                        "lon": G_cc.nodes[n]["x"],
                    })
            cluster_df = pd.DataFrame(cl_rows)
            centroids_df = cluster_df.groupby("cluster")[["lat", "lon"]].mean().reset_index() if not cluster_df.empty else pd.DataFrame()

            # === Routing per cluster ===
            vehicle_routes = []
            for g in buckets:
                seq = route_nn(tt_fn, depot_node, g)
                stops = seq[1:-1]
                if use_two_opt and len(stops) > 2:
                    stops = two_opt(tt_fn, depot_node, stops, iters=int(max_two_opt))
                vehicle_routes.append([depot_node] + stops + [depot_node])

            # Prepare polylines + stats
            reports, all_rows, veh_polys = [], [], []
            for i, seq in enumerate(vehicle_routes, 1):
                travel_sec, dist_m, order = 0.0, 0.0, 0
                poly = []
                for a,b in zip(seq[:-1], seq[1:]):
                    nodes = rn_fn(a, b)
                    if len(nodes) >= 2:
                        ys = [G_cc.nodes[z]["y"] for z in nodes]
                        xs = [G_cc.nodes[z]["x"] for z in nodes]
                        poly += list(zip(ys, xs))
                        for u,v in zip(nodes[:-1], nodes[1:]):
                            try:
                                k,d = min(G_cc[u][v].items(), key=lambda kv: kv[1].get("length", float("inf")))
                                dist_m    += float(d.get("length", 0.0))
                                travel_sec += float(d.get("travel_time", 0.0))
                            except Exception:
                                pass
                unique_stops = list(dict.fromkeys(seq[1:-1]))
                service_sec = sum(service_time_sec[n] for n in unique_stops)
                reports.append({
                    "vehicle": i,
                    "stops": len(unique_stops),
                    "travel_min": travel_sec/60.0,
                    "service_min": service_sec/60.0,
                    "total_min": (travel_sec+service_sec)/60.0,
                    "dist_km": dist_m/1000.0
                })
                for n in unique_stops:
                    order += 1
                    all_rows.append({
                        "vehicle": i,
                        "order": order,
                        "node_id": n,
                        "lat": G_cc.nodes[n]["y"],
                        "lon": G_cc.nodes[n]["x"],
                        "service_s": service_time_sec[n],
                        "cluster": int(cluster_map.get(n, -1)),
                    })
                veh_polys.append(poly)

            st.session_state.vrp_result = {
                "depot": (depot_lat, depot_lon),
                "bin_nodes": [(G_cc.nodes[n]["y"], G_cc.nodes[n]["x"]) for n in bin_nodes],
                "reports_df": pd.DataFrame(reports).round(2),
                "stops_df": pd.DataFrame(all_rows),
                "veh_polys": veh_polys,
                "diag": diag,
                "cluster_df": cluster_df,
                "centroids_df": centroids_df,
            }
    st.rerun()

# ------------- Render -------------
res = st.session_state.vrp_result

if res is None:
    st.info("Välj inställningar i sidofältet och klicka **Run optimization**.")
    st.caption("Session State används för att behålla karta och resultat efter rerun.")
else:
    st.subheader("Interactive Map", divider="gray")

    # Vehicle toggle + bins
    veh_labels = [f"Vehicle {i+1}" for i in range(len(res["veh_polys"]))] or []
    default_sel = st.session_state.veh_sel or veh_labels
    veh_sel = st.multiselect("Show vehicles", veh_labels, default=default_sel, key="veh_multiselect")
    st.session_state.veh_sel = veh_sel
    show_bins = st.checkbox("Show bin points (cluster colors)", value=False, key="bins_chk")

    depot_lat, depot_lon = res["depot"]
    m = folium.Map(location=[depot_lat, depot_lon], zoom_start=14, tiles="OpenStreetMap")
    folium.CircleMarker([depot_lat, depot_lon], radius=7, color="green", fill=True,
                        fill_opacity=1, tooltip="Depot").add_to(m)

    # Colors
    palette = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6",
               "#06b6d4", "#ec4899", "#84cc16", "#a855f7", "#f97316", "#22c55e"]

    # (Optional) cluster-colored points
    if show_bins:
        bins_fg = folium.FeatureGroup(name=f"Bins ({len(res['bin_nodes'])})", show=True)
        if "cluster_df" in res and isinstance(res["cluster_df"], pd.DataFrame) and not res["cluster_df"].empty:
            for row in res["cluster_df"].itertuples():
                color = palette[int(row.cluster) % len(palette)]
                folium.CircleMarker(
                    [row.lat, row.lon], radius=2, color=color, fill=True, fill_opacity=0.9
                ).add_to(bins_fg)
            # (Optional) centroids
            if "centroids_df" in res and isinstance(res["centroids_df"], pd.DataFrame) and not res["centroids_df"].empty:
                for row in res["centroids_df"].itertuples():
                    folium.CircleMarker(
                        [row.lat, row.lon], radius=6, color="#111111",
                        fill=True, fill_opacity=0.6, tooltip=f"Cluster {int(row.cluster)} centroid"
                    ).add_to(bins_fg)
        else:
            # fallback: red points
            for (y, x) in res["bin_nodes"]:
                folium.CircleMarker([y, x], radius=2, color="#ff3b30",
                                    fill=True, fill_opacity=0.9).add_to(bins_fg)
        bins_fg.add_to(m)

    # Route polylines (+ optional AntPath)
    for i, poly in enumerate(res["veh_polys"], 1):
        label = f"Vehicle {i}"
        if label not in veh_sel:
            continue
        grp = folium.FeatureGroup(name=label, show=True)
        if len(poly) >= 2:
            if animate_ant:
                AntPath(
                    locations=poly,
                    weight=4, opacity=0.9, color=palette[(i-1) % len(palette)],
                    dash_array=[10, 20], delay=1000
                ).add_to(grp)
            else:
                folium.PolyLine(poly, weight=4, opacity=0.9,
                                color=palette[(i-1) % len(palette)],
                                tooltip=label).add_to(grp)
        grp.add_to(m)

    # Timeline playback (TimestampedGeoJson)
    if animate_timeline:
        features = []
        base_t = datetime(2025, 1, 1, 8, 0, 0)  # referenstid
        for i, poly in enumerate(res["veh_polys"], 1):
            label = f"Vehicle {i}"
            if label not in veh_sel or len(poly) < 2:
                continue
            coords = poly[::max(1, int(downsample_n))]
            if len(coords) < 2:
                continue
            times = [base_t + timedelta(seconds=k * timeline_step_s) for k in range(len(coords))]
            times_iso = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": [[lon, lat] for (lat, lon) in coords]},
                "properties": {"times": times_iso, "style": {"color": palette[(i-1) % len(palette)], "weight": 5}}
            })
        if features:
            fc = {"type": "FeatureCollection", "features": features}
            TimestampedGeoJson(
                data=fc,
                transition_time=int(timeline_step_s*1000),
                loop=True, auto_play=True, add_last_point=True,
                period="PT1S", speed_slider=True, loop_button=True
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Fit bounds
    all_coords = [res["depot"]] + [p for poly in res["veh_polys"] for p in poly] + (res["bin_nodes"] if show_bins else [])
    if all_coords:
        m.fit_bounds([[min(y for y,_ in all_coords), min(x for _,x in all_coords)],
                      [max(y for y,_ in all_coords), max(x for _,x in all_coords)]])

    # Map render
    st_folium(m, height=int(map_height), width=None)

    # === Cluster plot (Matplotlib) with download ===
    st.subheader("Cluster plot", divider="gray")
    if "cluster_df" in res and isinstance(res["cluster_df"], pd.DataFrame) and not res["cluster_df"].empty:
        with st.expander("Visa & spara PNG", expanded=True):
            fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=150)
            for k, grp in res["cluster_df"].groupby("cluster"):
                ax.scatter(grp["lon"].values, grp["lat"].values,
                           s=12, alpha=0.9, label=f"Cluster {int(k)}")
            dep_lat, dep_lon = res["depot"]
            ax.scatter([dep_lon], [dep_lat], marker="*", s=180, edgecolors="k",
                       linewidths=0.8, label="Depot")

            if "centroids_df" in res and isinstance(res["centroids_df"], pd.DataFrame) and not res["centroids_df"].empty:
                ax.scatter(res["centroids_df"]["lon"].values,
                           res["centroids_df"]["lat"].values,
                           marker="x", s=60, linewidths=1.5, label="Centroids")

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
                mime="image/png",
            )
            plt.close(fig)
    else:
        st.info("Ingen klusterdata att plotta ännu. Kör optimering först.")

    # === KPIs & Details (med timmar per bil) ===
    st.subheader("KPIs & Details", divider="gray")
    with st.container():
        st.markdown('<div class="small-info">', unsafe_allow_html=True)

        df = res["reports_df"].copy()
        df["travel_h"] = (df["travel_min"] / 60).round(2)
        df["service_h"] = (df["service_min"] / 60).round(2)
        df["total_h"]  = (df["total_min"]  / 60).round(2)

        total_time_h = float(df["total_h"].sum()) if len(df) else 0.0
        longest_h    = float(df["total_h"].max()) if len(df) else 0.0
        coverage     = f"{res['stops_df']['node_id'].nunique()}/{len(res['bin_nodes'])}"

        c1, c2, c3, c4 = st.columns([1,1,1,1])
        c1.metric("Coverage", coverage)
        c2.metric("Fleet total (h)", f"{total_time_h:.2f}")
        c3.metric("Longest vehicle (h)", f"{longest_h:.2f}")
        c4.metric("Vehicles", f"{len(df)}")

        st.markdown("**Per-vehicle hours**")
        st.dataframe(
            df[["vehicle", "stops", "travel_h", "service_h", "total_h", "dist_km"]].rename(columns={
                "vehicle":"veh", "dist_km":"dist_km"
            }),
            height=260, width='stretch'
        )

        with st.expander("Stops (sample) & Download", expanded=False):
            st.dataframe(res["stops_df"].head(300), height=240, width='stretch')
            st.download_button(
                "⬇️ Download routes CSV",
                data=res["stops_df"].to_csv(index=False),
                file_name="routes.csv",
                mime="text/csv"
            )

        st.markdown('</div>', unsafe_allow_html=True)
