// app.js v4 (India edition: ₹, Hyderabad, 99acres-style requirement filters)
const API = "http://127.0.0.1:8000";             // your FastAPI address
const COLORS = ["#2aa3d8", "#8aa04a", "#f2b61f", "#ec6fa8", "#c2573c"]; // cheap -> expensive
const FEATS = [  // key, label, unit, decimals
  ["area_sqft", "Built-up area", " sq ft", 0], ["distance_to_hub_km", "Distance to HITEC City", " km", 1],
  ["bhk", "Bedrooms (BHK)", "", 1], ["age_years", "Property age", " yrs", 0], ["amenity_score", "Amenities (of 5)", "", 1]];
const GROUPS = {   // filter chips, [value, label]
  ptype: ["Apartment", "Independent House", "Villa", "Builder Floor"].map(v => [v, v]),
  bhk: [["1", "1 BHK"], ["2", "2 BHK"], ["3", "3 BHK"], ["4", "4 BHK"], ["5", "4+ BHK"]],
  poss: ["Ready to Move", "In 1 Year", "In 2 Years", "In 3 Years", "After 3 Years"].map(v => [v, v]),
  amen: [["parking", "Parking"], ["lift", "Lift"], ["gated_community", "Gated Community"], ["swimming_pool", "Swimming Pool"], ["gas_pipeline", "Gas Pipeline"]],
};
const $ = id => document.getElementById(id);
const fmtn = (v, d = 0) => v.toLocaleString("en-IN", { maximumFractionDigits: d });
const inr = n => n >= 1e7 ? "₹" + (n / 1e7).toFixed(2).replace(/\.?0+$/, "") + " Cr"
  : n >= 1e5 ? "₹" + (n / 1e5).toFixed(1).replace(/\.0$/, "") + " L" : "₹" + fmtn(Math.round(n));
const colorFor = p => COLORS[Math.min(4, breaks.filter(b => p > b).length)];
let breaks = [], bbox = null, pin = null, active = null;

// ---- map ----
L.GeometryUtil.readableArea = a => (a / 1e6).toFixed(2) + " km²"; // leaflet.draw bug workaround
const map = L.map("map", { zoomControl: false }).setView([17.4435, 78.4], 11);
L.control.zoom({ position: "bottomright" }).addTo(map);
const streets = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap contributors", maxZoom: 19 });
const esri = "Tiles © Esri";
const satellite = L.layerGroup([
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", { attribution: esri, maxZoom: 19 }),
  L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}", { maxZoom: 19 })]);
const esriStreets = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}", { attribution: esri, maxZoom: 19 }).addTo(map);
L.control.layers({ "Streets": esriStreets, "Streets (OSM)": streets, "Satellite": satellite }, null, { position: "topright" }).addTo(map);
const drawn = L.featureGroup().addTo(map), base = L.layerGroup().addTo(map), hits = L.layerGroup().addTo(map);
const shape = { color: "#1b2350", weight: 3, dashArray: "7 6", fillColor: "#ec6fa8", fillOpacity: 0.2 };
window.addEventListener("splash-done", () => map.invalidateSize());

function status(ok, msg) { $("dot").className = "w-2 h-2 rounded-full " + (ok ? "bg-olive" : "bg-rose"); $("statusText").textContent = msg; }

async function init() {
  try {
    const d = await (await fetch(API + "/map-data")).json();
    const sorted = d.points.map(p => p.price).sort((a, b) => a - b);
    breaks = [1, 2, 3, 4].map(i => sorted[Math.floor(sorted.length * i / 5)]);
    d.points.forEach(p => L.circleMarker([p.lat, p.lon], { radius: 5, weight: 1, color: "#fff", fillColor: colorFor(p.price), fillOpacity: 0.8 }).addTo(base));
    map.fitBounds(d.bounds);
    $("legend").innerHTML = `<p class="lbl mb-2">Listed price</p><div class="flex gap-1">${COLORS.map(c => `<i style="background:${c}" class="w-8 h-2.5 rounded-sm"></i>`).join("")}</div>
      <div class="flex justify-between mt-1.5 text-mute"><span>${inr(sorted[0])}</span><span>${inr(sorted.at(-1))}</span></div>`;
    status(true, `${fmtn(d.points.length)} sample listings · Hyderabad`);
  } catch { status(false, "Backend not reachable"); }
}

// ---- drawing ----
function setBBox(b) {
  bbox = { min_lat: b.getSouth(), max_lat: b.getNorth(), min_lon: b.getWest(), max_lon: b.getEast() };
  $("bN").textContent = bbox.max_lat.toFixed(4); $("bS").textContent = bbox.min_lat.toFixed(4);
  $("bE").textContent = bbox.max_lon.toFixed(4); $("bW").textContent = bbox.min_lon.toFixed(4);
  $("searchBtn").disabled = false;
}
const pinIcon = () => L.divIcon({ className: "", html: '<div class="pin"></div>', iconSize: [20, 20] });
function placePin(ll) {
  drawn.clearLayers(); pin = ll; $("radius").disabled = false;
  L.marker(ll, { icon: pinIcon() }).addTo(drawn);
  setBBox(L.circle(ll, { ...shape, radius: $("radius").value * 1000 }).addTo(drawn).getBounds());
}
const handlers = {
  polygon: () => new L.Draw.Polygon(map, { shapeOptions: shape, allowIntersection: false }),
  rectangle: () => new L.Draw.Rectangle(map, { shapeOptions: shape }),
  marker: () => new L.Draw.Marker(map, { icon: pinIcon() }),
};
const clearTools = () => document.querySelectorAll("[data-tool]").forEach(x => x.classList.remove("on"));
document.querySelectorAll("[data-tool]").forEach(b => b.onclick = () => {
  if (active) active.disable(); clearTools();
  b.classList.add("on"); active = handlers[b.dataset.tool](); active.enable();
});
map.on(L.Draw.Event.CREATED, e => {
  clearTools();
  if (e.layerType === "marker") return placePin(e.layer.getLatLng());
  pin = null; $("radius").disabled = true; drawn.clearLayers(); drawn.addLayer(e.layer); setBBox(e.layer.getBounds());
});
$("radius").oninput = e => { $("rOut").textContent = (+e.target.value).toFixed(1) + " km"; if (pin) placePin(pin); };
$("clearBtn").onclick = () => {
  if (active) active.disable(); clearTools(); drawn.clearLayers(); hits.clearLayers(); pin = bbox = null;
  $("searchBtn").disabled = true; $("radius").disabled = true;
  ["bN", "bS", "bE", "bW"].forEach(i => $(i).textContent = "–");
  $("results").innerHTML = ""; $("empty").style.display = "";
};

// ---- requirement filters + budget ----
Object.entries(GROUPS).forEach(([g, items]) => {
  $("g-" + g).innerHTML = items.map(([v, l]) => `<button type="button" class="chip" data-v="${v}" aria-pressed="false">${l}</button>`).join("");
  $("g-" + g).onclick = e => { const b = e.target.closest("[data-v]"); if (b) b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") !== "true"); };
});
const sel = g => [...$("g-" + g).querySelectorAll("[aria-pressed=true]")].map(b => b.dataset.v);
$("resetFilters").onclick = () => {
  document.querySelectorAll("[data-v]").forEach(b => b.setAttribute("aria-pressed", "false"));
  $("minArea").value = $("maxArea").value = "";
};
const showWords = () => $("budgetWords").textContent = +$("budget").value ? "= " + inr(+$("budget").value) : "";
$("budget").oninput = showWords; showWords();
document.querySelectorAll("[data-b]").forEach(b => b.onclick = () => { $("budget").value = b.dataset.b; showWords(); });

// ---- search ----
$("searchBtn").onclick = async e => {
  const btn = e.target; btn.disabled = true; btn.textContent = "Searching…";
  try {
    const r = await fetch(API + "/region-query", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...bbox, model_type: document.querySelector("input[name=model]:checked").value, target_budget: +$("budget").value || null,
        property_types: sel("ptype"), bhk: sel("bhk").map(Number), possession: sel("poss"), amenities: sel("amen"),
        min_area: +$("minArea").value || null, max_area: +$("maxArea").value || null }) });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    render(await r.json());
  } catch (err) { alert("Search failed: " + err.message); }
  btn.disabled = false; btn.textContent = "Search this region";
};

const tile = (l, v, c) => `<div class="rounded-xl border border-line p-4"><p class="lbl flex items-center gap-2"><i class="w-2 h-2 rounded-full" style="background:${c}"></i>${l}</p><p class="font-display text-2xl font-semibold mt-1.5">${v}</p></div>`;

function render(r) {
  $("empty").style.display = "none"; hits.clearLayers();
  if (!r.count) { $("results").innerHTML = '<p class="text-sm text-mute">No homes match this area and these requirements. Try a larger area or fewer filters.</p>'; return; }
  const budget = r.reverse?.target_budget;
  r.points.forEach(p => L.circleMarker([p.lat, p.lon], { radius: 8, weight: budget && p.price <= budget ? 3 : 1.5, color: "#1b2350", fillColor: colorFor(p.price), fillOpacity: 0.95 })
    .bindPopup(`<b>${inr(p.price)}</b><br>${p.bhk} BHK ${p.type} · ${fmtn(p.area)} sq ft<br>${p.locality} · ${p.possession}<br><span style="color:#6b7090">Model estimate ${inr(p.pred)}</span>`).addTo(hits));
  const s = r.stats;
  let h = `<div class="grid grid-cols-2 gap-3">${tile("Homes found", r.count, "#2aa3d8")}${tile("Avg. listed", inr(s.avg_price), "#f2b61f")}
    ${tile("Model estimate", inr(s.avg_predicted), "#ec6fa8")}${tile("Price range", inr(s.min_price) + " – " + inr(s.max_price), "#6b6a2c")}
    ${tile("Avg. ₹ / sq ft", "₹" + fmtn(s.avg_ppsf), "#8aa04a")}${tile("Avg. area", fmtn(s.avg_area) + " sq ft", "#c2573c")}</div>`;
  if (r.reverse) {
    const v = r.reverse, pct = Math.round(v.affordable_count / r.count * 100), off = Math.abs(v.achieved_price - v.target_budget) > v.target_budget * 0.01;
    h += `<div class="mt-6"><div class="flex justify-between text-sm"><span class="font-semibold">Within ${inr(v.target_budget)}</span><span class="text-mute"><b class="text-ink">${v.affordable_count}</b> of ${r.count} homes</span></div>
      <div class="h-2 rounded-full bg-line mt-2"><div class="h-2 rounded-full bg-sky" style="width:${pct}%"></div></div></div>
      <p class="lbl mt-7">Typical home vs. needed to reach budget</p>
      <table class="t mt-2"><tr><th>Feature</th><th>Typical</th><th>Needed</th><th>Change</th></tr>
      ${FEATS.map(([key, l, u, d]) => { const dl = v.config[key] - r.avg_features[key];
        return `<tr><td class="font-medium">${l}</td><td>${fmtn(r.avg_features[key], d)}</td><td>${fmtn(v.config[key], d)}${u}</td><td class="font-semibold ${dl >= 0 ? "text-rose" : "text-sky"}">${dl >= 0 ? "+" : ""}${fmtn(dl, d)}</td></tr>`; }).join("")}</table>
      <p class="text-xs text-mute mt-3">Model price with these values: <b class="text-ink">${inr(v.achieved_price)}</b>${off ? " (some features hit realistic limits, so it can't match the budget exactly)" : ""}.</p>`;
  } else {
    h += `<table class="t mt-6">${FEATS.map(([key, l, u, d]) => `<tr><td class="font-medium">${l}</td><td>${fmtn(r.avg_features[key], d)}${u}</td></tr>`).join("")}</table>`;
  }
  $("results").innerHTML = h;
}
init();