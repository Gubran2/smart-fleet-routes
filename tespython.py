import folium
center = (36.732205, 34.023125)
m = folium.Map(location=center, zoom_start=14, tiles="CartoDB positron")
map_var = m.get_name()

script = f"""
<script>
(function(){{
  var map = {map_var};
  var pts = [], routeLayer=null, markers=[];

  function clearAll(){{
    if(routeLayer) {{ map.removeLayer(routeLayer); routeLayer=null; }}
    markers.forEach(m=>map.removeLayer(m)); markers=[]; pts=[];
  }}

  async function osrmNearest(lat, lon){{
    const r = await fetch(`https://router.project-osrm.org/nearest/v1/driving/${{lon}},${{lat}}?number=1`);
    const j = await r.json();
    const loc = j.waypoints[0].location; // [lon,lat] snapped
    return [loc[1], loc[0]];
  }}

  async function osrmRoute(A, B){{
    const url = `https://router.project-osrm.org/route/v1/driving/${{A[1]}},${{A[0]}};${{B[1]}},${{B[0]}}?overview=full&geometries=geojson`;
    const r = await fetch(url); const j = await r.json();
    const route = j.routes[0];
    const coords = route.geometry.coordinates.map(c=>[c[1], c[0]]);
    return {{coords, minutes: route.duration/60, km: route.distance/1000}};
  }}

  map.on('click', async function(e){{
    if(pts.length===2) clearAll();
    const raw = [e.latlng.lat, e.latlng.lng];

    // 1) snap-to-road
    let snapped;
    try {{ snapped = await osrmNearest(raw[0], raw[1]); }}
    catch(err) {{ alert('nearest failed'); return; }}

    pts.push(snapped);
    const mk = L.marker(snapped).addTo(map);
    mk.bindTooltip(pts.length===1?'A (snapped)':'B (snapped)').openTooltip();
    markers.push(mk);

    if(pts.length===2){{
      try {{
        const r = await osrmRoute(pts[0], pts[1]);
        routeLayer = L.polyline(r.coords, {{weight:5, opacity:0.95}}).addTo(map);
        routeLayer.bindTooltip(`~ ${'{'}(r.minutes).toFixed(1){'}'} min, ${'{'}(r.km).toFixed(2){'}'} km`).openTooltip();
        map.fitBounds(routeLayer.getBounds().pad(0.2));
      }} catch(err) {{
        alert('routing failed');
      }}
    }}
  }});
}})();
</script>
"""
from folium import Element
m.get_root().html.add_child(Element(script))
m.save("pick_and_route_snapped.html")
print("Open pick_and_route_snapped.html")
