/* GeoAI Detect web client */
const $ = (s) => document.querySelector(s);
const PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
                 "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324"];
const DEFAULT_CLASSES = ["building", "road", "tree"];

const state = {
  image: null,               // current image info
  classes: [],               // [{name, on}]
  features: { type: "FeatureCollection", features: [] },
  tool: "pan",
  points: [],                // [[lng, lat, 1|0]]
  hidden: new Set(),
  busy: false,
};

function colorFor(name) {
  const all = Array.from(new Set([...DEFAULT_CLASSES, ...state.classes.map(c => c.name),
    ...state.features.features.map(f => f.properties.class)]));
  return PALETTE[all.indexOf(name) % PALETTE.length];
}
function status(msg, err = false) {
  const el = $("#status"); el.textContent = msg; el.style.color = err ? "var(--danger)" : "";
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
const jpost = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

/* ------------------------------------------------------------------ map */
const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256,
             attribution: "© OpenStreetMap contributors", maxzoom: 19 },
    },
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#888" } },
             { id: "osm", type: "raster", source: "osm" }],
  },
  center: [0, 20], zoom: 2, maxZoom: 23,
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }));

map.on("load", () => {
  map.addSource("features", { type: "geojson", data: state.features, promoteId: "fid" });
  map.addSource("preview", { type: "geojson", data: empty() });
  map.addSource("points", { type: "geojson", data: empty() });
  map.addSource("box", { type: "geojson", data: empty() });
  const colorExpr = ["coalesce", ["get", "color"], "#ff0"];
  map.addLayer({ id: "f-fill", type: "fill", source: "features",
    paint: { "fill-color": colorExpr, "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.55, 0.3] } });
  map.addLayer({ id: "f-line", type: "line", source: "features",
    paint: { "line-color": colorExpr, "line-width": ["case", ["==", ["geometry-type"], "LineString"], 3, 1.5] } });
  map.addLayer({ id: "p-fill", type: "fill", source: "preview", paint: { "fill-color": "#00e5ff", "fill-opacity": 0.35 } });
  map.addLayer({ id: "p-line", type: "line", source: "preview", paint: { "line-color": "#00e5ff", "line-width": 2, "line-dasharray": [2, 1] } });
  map.addLayer({ id: "box", type: "line", source: "box", paint: { "line-color": "#00e5ff", "line-width": 2 } });
  map.addLayer({ id: "pts", type: "circle", source: "points",
    paint: { "circle-radius": 6, "circle-stroke-width": 2, "circle-stroke-color": "#fff",
             "circle-color": ["case", ["==", ["get", "label"], 1], "#16a34a", "#dc2626"] } });
  init();
});

function empty() { return { type: "FeatureCollection", features: [] }; }

function setImageLayer(info) {
  if (map.getLayer("img")) map.removeLayer("img");
  if (map.getSource("img")) map.removeSource("img");
  map.addSource("img", { type: "raster", tiles: [`/api/images/${info.id}/tiles/{z}/{x}/{y}.png`],
    tileSize: 256, bounds: info.bounds_wgs84, minzoom: 0, maxzoom: info.max_zoom + 3 });
  map.addLayer({ id: "img", type: "raster", source: "img", paint: { "raster-opacity": +$("#opacity").value } }, "f-fill");
  map.fitBounds(info.bounds_wgs84, { padding: 30, duration: 0 });
}

/* --------------------------------------------------------------- images */
async function refreshImages(selectId) {
  const list = await api("/api/images");
  const sel = $("#imageSelect");
  sel.innerHTML = '<option value="">— or pick a loaded image —</option>' +
    list.map(i => `<option value="${i.id}">${i.name}</option>`).join("");
  if (selectId) sel.value = selectId;
}
async function openImage(id) {
  if (!id) return;
  const info = await api(`/api/images/${id}`);
  state.image = info;
  $("#imageSelect").value = id;
  $("#imgInfo").textContent =
    `${info.name}\n${info.width} × ${info.height} px, ${info.bands} band(s), ${info.dtype}\n` +
    `CRS ${info.crs}  ·  ${info.gsd_m < 1 ? (info.gsd_m * 100).toFixed(1) + " cm" : info.gsd_m.toFixed(2) + " m"}/px`;
  setImageLayer(info);
  await loadFeatures();
  status(`Opened ${info.name}`);
}
$("#imageSelect").onchange = (e) => openImage(e.target.value);
$("#upload").onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  const fd = new FormData(); fd.append("file", file);
  status(`Uploading ${file.name}…`);
  try {
    const info = await api("/api/images", { method: "POST", body: fd });
    await refreshImages(info.id); await openImage(info.id);
  } catch (err) { status(err.message, true); }
  e.target.value = "";
};
$("#svcType").onchange = (e) => $("#svcType").parentElement.classList.toggle("wms", e.target.value === "wms");
$("#svcFetch").onclick = async () => {
  const b = map.getBounds();
  const body = { type: $("#svcType").value, url: $("#svcUrl").value.trim(), layer: $("#svcLayer").value.trim(),
    crs: $("#svcCrs").value.trim(), gsd: +$("#svcGsd").value,
    bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()] };
  if (!body.url) return status("Enter a service URL", true);
  status("Downloading imagery from service…");
  try {
    const info = await jpost("/api/images/fetch", body);
    await refreshImages(info.id); await openImage(info.id);
  } catch (err) { status(err.message, true); }
};

/* ------------------------------------------------------------- classes */
function renderClasses() {
  $("#classChips").innerHTML = state.classes.map((c, i) =>
    `<span class="chip ${c.on ? "on" : ""}" data-i="${i}"><span class="dot" style="background:${colorFor(c.name)}"></span>${c.name}<span class="x" data-x="${i}">×</span></span>`).join("");
  const names = Array.from(new Set([...state.classes.map(c => c.name), ...state.features.features.map(f => f.properties.class)]));
  const cur = $("#drawClass").value;
  $("#drawClass").innerHTML = names.map(n => `<option>${n}</option>`).join("");
  if (names.includes(cur)) $("#drawClass").value = cur;
}
$("#classChips").onclick = (e) => {
  const x = e.target.dataset.x;
  if (x !== undefined) { state.classes.splice(+x, 1); return renderClasses(); }
  const chip = e.target.closest(".chip"); if (!chip) return;
  state.classes[+chip.dataset.i].on = !state.classes[+chip.dataset.i].on; renderClasses();
};
function addClass() {
  const v = $("#newClass").value.trim().toLowerCase();
  if (v && !state.classes.some(c => c.name === v)) state.classes.push({ name: v, on: true });
  $("#newClass").value = ""; renderClasses();
}
$("#addClass").onclick = addClass;
$("#newClass").onkeydown = (e) => { if (e.key === "Enter") addClass(); };
$("#thr").oninput = (e) => $("#thrVal").textContent = (+e.target.value).toFixed(2);

/* ----------------------------------------------------------- detection */
$("#detect").onclick = async () => {
  if (!state.image) return status("Open an image first", true);
  const classes = state.classes.filter(c => c.on).map(c => c.name);
  if (!classes.length) return status("Select at least one class", true);
  $("#detect").disabled = true;
  const bar = $("#progress"); bar.classList.remove("hidden");
  try {
    const job = await jpost(`/api/images/${state.image.id}/detect`, {
      classes, engine: $("#engine").value, threshold: +$("#thr").value, tile_size: +$("#tileSize").value,
      min_area: +$("#minArea").value, regularize: $("#regularize").checked, replace: $("#replace").checked });
    while (true) {
      await new Promise(r => setTimeout(r, 1000));
      const j = await api(`/api/jobs/${job.job_id}`);
      bar.firstElementChild.style.width = `${Math.round(j.progress * 100)}%`;
      bar.lastElementChild.textContent = j.message;
      if (j.status === "done") { status(j.message); break; }
      if (j.status === "error") throw new Error(j.message);
    }
    await loadFeatures();
  } catch (err) { status(err.message, true); }
  finally { $("#detect").disabled = false; setTimeout(() => bar.classList.add("hidden"), 1500); }
};

/* ------------------------------------------------------------ features */
async function loadFeatures() {
  if (!state.image) return;
  state.features = await api(`/api/images/${state.image.id}/features`);
  redraw();
}
function redraw() {
  for (const f of state.features.features) f.properties.color = colorFor(f.properties.class);
  const visible = { type: "FeatureCollection",
    features: state.features.features.filter(f => !state.hidden.has(f.properties.class)) };
  map.getSource("features")?.setData(visible);
  const counts = {};
  for (const f of state.features.features) counts[f.properties.class] = (counts[f.properties.class] || 0) + 1;
  $("#layers").innerHTML = Object.keys(counts).length ? Object.entries(counts).map(([k, n]) =>
    `<div class="layer"><input type="checkbox" data-k="${k}" ${state.hidden.has(k) ? "" : "checked"}>
     <span class="dot" style="background:${colorFor(k)}"></span>${k}<span class="n">${n}</span>
     <button data-clear="${k}" title="Remove all ${k}">clear</button></div>`).join("")
    : '<p class="hint">No objects yet.</p>';
  renderClasses();
}
$("#layers").onclick = async (e) => {
  if (e.target.dataset.k !== undefined) {
    const k = e.target.dataset.k; e.target.checked ? state.hidden.delete(k) : state.hidden.add(k); return redraw();
  }
  const k = e.target.dataset.clear;
  if (k && confirm(`Remove all "${k}" objects?`)) {
    await api(`/api/images/${state.image.id}/features?label=${encodeURIComponent(k)}`, { method: "DELETE" });
    await loadFeatures();
  }
};
$("#showBase").onchange = (e) => map.setLayoutProperty("osm", "visibility", e.target.checked ? "visible" : "none");
$("#opacity").oninput = (e) => map.getLayer("img") && map.setPaintProperty("img", "raster-opacity", +e.target.value);
$("#export").onclick = () => {
  if (!state.image) return status("Open an image first", true);
  if (!state.features.features.length) return status("Nothing to export", true);
  window.location = `/api/images/${state.image.id}/export.zip`;
};

/* --------------------------------------------------- interactive tools */
const HINTS = {
  pan: "Pan the map. Pick a tool to segment objects with SAM in real time.",
  point: "Click on an object to segment it. Shift+click excludes an area. Enter accepts, Esc clears.",
  box: "Drag a box around one object. It is segmented and added when you release.",
  select: "Click an object to delete it.",
};
function setTool(t) {
  state.tool = t;
  document.querySelectorAll("#tools button").forEach(b => b.classList.toggle("active", b.dataset.tool === t));
  $("#toolHint").textContent = HINTS[t];
  map.getCanvas().classList.toggle("crosshair", t !== "pan");
  if (t === "box") map.dragPan.disable(); else map.dragPan.enable();
  if (t !== "point") clearPoints();
}
$("#tools").onclick = (e) => { const t = e.target.closest("button")?.dataset.tool; if (t) setTool(t); };
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const k = e.key.toLowerCase();
  if (k === "v") setTool("pan"); else if (k === "p") setTool("point");
  else if (k === "b") setTool("box"); else if (k === "s") setTool("select");
  else if (e.key === "Enter" && state.points.length) acceptPoints();
  else if (e.key === "Escape") clearPoints();
});

function drawClass() {
  const c = $("#drawClass").value || state.classes.find(c => c.on)?.name || "object";
  return c;
}
async function segment(body, commit) {
  if (!state.image) { status("Open an image first", true); return null; }
  if (state.busy) return null;
  state.busy = true; status("Segmenting…");
  const t0 = performance.now();
  try {
    const res = await jpost(`/api/images/${state.image.id}/segment`, { ...body, label: drawClass(), commit });
    status(`Segmented in ${((performance.now() - t0) / 1000).toFixed(1)} s (score ${res.score.toFixed(2)})`);
    return res;
  } catch (err) { status(err.message, true); return null; }
  finally { state.busy = false; }
}

// point mode
async function updatePointPreview() {
  map.getSource("points").setData({ type: "FeatureCollection", features: state.points.map(p => ({
    type: "Feature", properties: { label: p[2] }, geometry: { type: "Point", coordinates: [p[0], p[1]] } })) });
  $("#pointActions").classList.toggle("hidden", !state.points.length);
  if (!state.points.length) return map.getSource("preview").setData(empty());
  const res = await segment({ points: state.points }, false);
  if (res) map.getSource("preview").setData(res.feature);
}
async function acceptPoints() {
  if (!state.points.length) return;
  const res = await segment({ points: state.points }, true);
  clearPoints();
  if (res) await loadFeatures();
}
function clearPoints() {
  state.points = [];
  map.getSource("points")?.setData(empty());
  map.getSource("preview")?.setData(empty());
  $("#pointActions").classList.add("hidden");
}
$("#acceptPts").onclick = acceptPoints;
$("#clearPts").onclick = clearPoints;

map.on("click", async (e) => {
  if (state.tool === "point") {
    state.points.push([e.lngLat.lng, e.lngLat.lat, e.originalEvent.shiftKey ? 0 : 1]);
    updatePointPreview();
  } else if (state.tool === "select") {
    const bb = [[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]];
    const f = map.queryRenderedFeatures(bb, { layers: ["f-line", "f-fill"] })[0];
    if (!f) return;
    await api(`/api/images/${state.image.id}/features/${f.properties.fid}`, { method: "DELETE" });
    await loadFeatures();
    status(`Deleted ${f.properties.class}`);
  }
});

// hover highlight in select mode
let hoverId = null;
map.on("mousemove", "f-fill", (e) => {
  if (state.tool !== "select") return;
  if (hoverId !== null) map.setFeatureState({ source: "features", id: hoverId }, { hover: false });
  hoverId = e.features[0].id; map.setFeatureState({ source: "features", id: hoverId }, { hover: true });
});
map.on("mouseleave", "f-fill", () => {
  if (hoverId !== null) map.setFeatureState({ source: "features", id: hoverId }, { hover: false }); hoverId = null;
});

// box mode
let boxStart = null;
map.on("mousedown", (e) => { if (state.tool === "box") boxStart = e.lngLat; });
map.on("mousemove", (e) => {
  if (state.tool !== "box" || !boxStart) return;
  const a = boxStart, b = e.lngLat;
  map.getSource("box").setData({ type: "Feature", geometry: { type: "LineString",
    coordinates: [[a.lng, a.lat], [b.lng, a.lat], [b.lng, b.lat], [a.lng, b.lat], [a.lng, a.lat]] } });
});
map.on("mouseup", async (e) => {
  if (state.tool !== "box" || !boxStart) return;
  const a = boxStart, b = e.lngLat; boxStart = null;
  const box = [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)];
  const pa = map.project(a), pb = map.project(b);
  if (Math.abs(pa.x - pb.x) < 5 || Math.abs(pa.y - pb.y) < 5) { map.getSource("box").setData(empty()); return; }
  const res = await segment({ box }, true);
  map.getSource("box").setData(empty());
  if (res) await loadFeatures();
});

/* ---------------------------------------------------------------- init */
async function init() {
  state.classes = DEFAULT_CLASSES.map(n => ({ name: n, on: true }));
  renderClasses();
  try {
    const cfg = await api("/api/config");
    $("#engine").innerHTML = cfg.engines.map(e => `<option value="${e.id}" ${e.available ? "" : "disabled"}>${e.name}${e.available ? "" : " (unavailable)"}</option>`).join("");
    $("#engine").value = cfg.default_engine;
    status(`Ready · device: ${cfg.device}`);
    await refreshImages();
    if (cfg.last_image) openImage(cfg.last_image);
  } catch (err) { status("Server not reachable: " + err.message, true); }
}
