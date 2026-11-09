/* GeoAI Detect: work area client */
"use strict";
const $ = (s) => document.querySelector(s);
const COLORS = {
  building: "#e6194b", road: "#f58231", road_centerline: "#ffd60a", sidewalk: "#b06cff",
  parking: "#8c8c8c", tree: "#2fbf4a", "solar panel": "#1e5bff", water: "#00b7ff",
};
const EXTRA = ["#911eb4", "#46f0f0", "#bcf60c", "#008080", "#9a6324", "#f032e6"];
const LABELS = { building: "Buildings", road: "Roads", road_centerline: "Road centerlines", sidewalk: "Sidewalks",
  parking: "Parking lots", tree: "Trees", "solar panel": "Solar panels" };
// same 7-parameter transformation pyproj/PROJ use for EPSG:2039 (0.5 m accuracy); matches pyproj to the cm
proj4.defs("EPSG:2039", "+proj=tmerc +lat_0=31.7343936111111 +lon_0=35.2045169444444 +k=1.0000067 " +
  "+x_0=219529.584 +y_0=626907.39 +ellps=GRS80 +towgs84=23.772,17.49,17.859,-0.3132,-1.85274,1.67299,-5.4262 +units=m +no_defs");

const state = {
  cfg: null, sources: [], active: null, view: {}, aoi: null, tool: "pan",
  features: { type: "FeatureCollection", features: [] }, hidden: new Set(), job: null,
  points: [], desktop: false, exportFolder: "",
};
const colorFor = (c) => COLORS[c] || EXTRA[[...c].reduce((a, ch) => a + ch.charCodeAt(0), 0) % EXTRA.length];
const nice = (c) => LABELS[c] || c.charAt(0).toUpperCase() + c.slice(1);
const empty = () => ({ type: "FeatureCollection", features: [] });

function status(msg, err = false) { const s = $("#status"); s.textContent = msg; s.classList.toggle("err", err); }
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch {} throw new Error(m); }
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}
const jpost = (p, b) => api(p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b) });
const store = { get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} } };

/* ================================================================== map */
const map = new maplibregl.Map({
  container: "map",
  style: { version: 8, sources: {
      osm: { type: "raster", tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"], tileSize: 256,
             maxzoom: 19, attribution: "© OpenStreetMap contributors" } },
    layers: [{ id: "bg", type: "background", paint: { "background-color": "#9aa3ad" } },
             { id: "osm", type: "raster", source: "osm", paint: { "raster-opacity": 0.85 } },
             { id: "imagery-top", type: "background", paint: { "background-opacity": 0 } }] },
  center: [35.0, 31.6], zoom: 7, maxZoom: 22, attributionControl: { compact: true },
});
map.addControl(new maplibregl.NavigationControl(), "top-right");
map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-right");

map.on("load", () => {
  for (const id of ["aoi", "features", "preview", "pts", "drag"]) map.addSource(id, { type: "geojson", data: empty(), promoteId: id === "features" ? "fid" : undefined });
  const col = ["coalesce", ["get", "color"], "#ff0"];
  map.addLayer({ id: "f-fill", type: "fill", source: "features", filter: ["==", ["geometry-type"], "Polygon"],
    paint: { "fill-color": col, "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.6, 0.35] } });
  map.addLayer({ id: "f-line", type: "line", source: "features",
    paint: { "line-color": col, "line-width": ["case", ["==", ["geometry-type"], "LineString"], 2.5, 1.2] } });
  map.addLayer({ id: "aoi-fill", type: "fill", source: "aoi", paint: { "fill-color": "#1f6feb", "fill-opacity": 0.06 } });
  map.addLayer({ id: "aoi-line", type: "line", source: "aoi", paint: { "line-color": "#1f6feb", "line-width": 2.5, "line-dasharray": [3, 2] } });
  map.addLayer({ id: "drag-line", type: "line", source: "drag", paint: { "line-color": "#00e5ff", "line-width": 2 } });
  map.addLayer({ id: "p-fill", type: "fill", source: "preview", paint: { "fill-color": "#00e5ff", "fill-opacity": 0.35 } });
  map.addLayer({ id: "p-line", type: "line", source: "preview", paint: { "line-color": "#00e5ff", "line-width": 2 } });
  map.addLayer({ id: "pts", type: "circle", source: "pts", paint: { "circle-radius": 6, "circle-stroke-width": 2,
    "circle-stroke-color": "#fff", "circle-color": ["case", ["==", ["get", "l"], 1], "#16a34a", "#dc2626"] } });
  init();
});

map.on("mousemove", (e) => {
  const { lng, lat } = e.lngLat;
  const [x, y] = proj4("EPSG:4326", "EPSG:2039", [lng, lat]);
  $("#coords").textContent = `ITM ${x.toFixed(1)}, ${y.toFixed(1)}   ·   ${lat.toFixed(6)}, ${lng.toFixed(6)}`;
});

/* ============================================================== sources */
function addSourceLayer(s) {
  const id = "src-" + s.id;
  if (map.getSource(id)) return;
  map.addSource(id, { type: "raster", tiles: [`/api/sources/${s.id}/tiles/{z}/{x}/{y}.png`], tileSize: 256,
    bounds: s.bounds_wgs84, minzoom: 0, maxzoom: Math.min(22, s.max_zoom || 22) });
  map.addLayer({ id, type: "raster", source: id, paint: { "raster-opacity": 1, "raster-fade-duration": 0 } }, "imagery-top");
}
function renderSources() {
  const el = $("#sourceList");
  if (!state.sources.length) { el.innerHTML = '<p class="hint">Open an image file or add a service.</p>'; return; }
  el.innerHTML = state.sources.map(s => {
    const v = state.view[s.id] || { on: true, op: 1 };
    const kind = s.kind === "file" ? `${s.size[0]}×${s.size[1]} px · ${(s.gsd_m * 100).toFixed(0)} cm` : s.kind.toUpperCase();
    return `<div class="item ${s.id === state.active ? "selected" : ""}" data-id="${s.id}">
      <input type="radio" name="active" ${s.id === state.active ? "checked" : ""} title="Use for detection">
      <input type="checkbox" class="vis" ${v.on ? "checked" : ""} title="Show">
      <span class="name" title="${s.name} — click to zoom">${s.name}<br><span class="hint">${kind} · ${s.crs || ""}</span></span>
      <input type="range" class="op" min="0" max="1" step="0.05" value="${v.op}" title="Opacity">
      <button class="x" title="Remove">✕</button></div>`;
  }).join("");
  $(".svc-only")?.classList.toggle("hidden", activeSource()?.kind === "file" || !activeSource());
}
const activeSource = () => state.sources.find(s => s.id === state.active);
function setActive(id, zoom = false) {
  state.active = id; store.set("geoai.active", id);
  renderSources(); updateAoiInfo();
  const s = activeSource();
  if (zoom && s) zoomToSource(s);
}
function zoomToSource(s) {
  const b = s.bounds_wgs84;
  if (s.kind === "file" || (b[2] - b[0]) < 5) map.fitBounds([[b[0], b[1]], [b[2], b[3]]], { padding: 30, duration: 400 });
}
$("#sourceList").addEventListener("click", async (e) => {
  const it = e.target.closest(".item"); if (!it) return;
  const id = it.dataset.id; const s = state.sources.find(x => x.id === id);
  if (e.target.type === "radio") setActive(id);
  else if (e.target.classList.contains("x")) {
    if (!confirm(`Remove "${s.name}" from the list? (Results are kept.)`)) return;
    await api(`/api/sources/${id}`, { method: "DELETE" });
    if (map.getLayer("src-" + id)) { map.removeLayer("src-" + id); map.removeSource("src-" + id); }
    state.sources = state.sources.filter(x => x.id !== id);
    if (state.active === id) state.active = state.sources[0]?.id || null;
    renderSources();
  } else if (e.target.classList.contains("name") || e.target.closest(".name")) { setActive(id); zoomToSource(s); }
});
$("#sourceList").addEventListener("input", (e) => {
  const it = e.target.closest(".item"); if (!it) return;
  const id = it.dataset.id; const v = state.view[id] = state.view[id] || { on: true, op: 1 };
  if (e.target.classList.contains("vis")) { v.on = e.target.checked; map.setLayoutProperty("src-" + id, "visibility", v.on ? "visible" : "none"); }
  if (e.target.classList.contains("op")) { v.op = +e.target.value; map.setPaintProperty("src-" + id, "raster-opacity", v.op); }
});
async function sourceAdded(s) {
  state.sources.push(s); addSourceLayer(s); setActive(s.id); zoomToSource(s);
  status(`Added ${s.name}. Draw a rectangle (R) where you want to detect.`);
}

/* open file */
$("#btnOpenFile").onclick = async () => {
  if (state.desktop && window.pywebview?.api) {
    const p = await window.pywebview.api.open_file();
    if (!p) return;
    status("Opening " + p + " …");
    try { await sourceAdded(await jpost("/api/sources/file", { path: p })); } catch (e) { status(e.message, true); }
  } else $("#dlgFile").showModal();
};
$("#fileOpen").onclick = async (ev) => {
  ev.preventDefault();
  const f = $("#fileUpload").files[0]; const p = $("#filePath").value.trim();
  $("#dlgFile").close();
  try {
    if (p) { status("Opening " + p + " …"); await sourceAdded(await jpost("/api/sources/file", { path: p })); }
    else if (f) {
      status(`Uploading ${f.name} …`); const fd = new FormData(); fd.append("file", f);
      await sourceAdded(await api("/api/sources/upload", { method: "POST", body: fd }));
    }
  } catch (e) { status(e.message, true); }
};

/* add service */
function renderSaved() {
  const saved = store.get("geoai.services", []);
  $("#svcSaved").innerHTML = saved.length ? "<p class='hint'>Recently used:</p>" + saved.map((s, i) =>
    `<div class="item" data-i="${i}"><span class="name">${s.name || s.layer || s.kind} <span class="badge">${s.kind}</span></span><button type="button" class="x">✕</button></div>`).join("") : "";
}
$("#btnAddService").onclick = () => { renderSaved(); svcKindChanged(); $("#dlgService").showModal(); };
function svcKindChanged() {
  const k = $("#svcKind").value;
  document.querySelectorAll(".wmts-only").forEach(e => e.classList.toggle("hidden", k !== "wmts"));
  document.querySelectorAll(".wms-only").forEach(e => e.classList.toggle("hidden", k !== "wms"));
}
$("#svcKind").onchange = svcKindChanged;
$("#svcConnect").onclick = async () => {
  const url = $("#svcUrl").value.trim(); if (!url) return;
  $("#svcLayer").innerHTML = '<option value="">Connecting…</option>';
  try {
    const r = await jpost("/api/services/inspect", { kind: "wmts", url });
    $("#svcLayer").innerHTML = r.layers.map(l => `<option value="${l.id}">${l.title}${l.levels_m.length ? " · best " + (Math.min(...l.levels_m) * 100).toFixed(0) + " cm" : ""}</option>`).join("");
  } catch (e) { $("#svcLayer").innerHTML = `<option value="">${e.message}</option>`; }
};
$("#svcSaved").addEventListener("click", (e) => {
  const it = e.target.closest(".item"); if (!it) return;
  const saved = store.get("geoai.services", []); const s = saved[+it.dataset.i];
  if (e.target.classList.contains("x")) { saved.splice(+it.dataset.i, 1); store.set("geoai.services", saved); return renderSaved(); }
  $("#svcKind").value = s.kind; $("#svcUrl").value = s.url; $("#svcName").value = s.name || "";
  $("#svcWmsLayer").value = s.layer || ""; svcKindChanged();
  if (s.kind === "wmts") { $("#svcLayer").innerHTML = `<option value="${s.layer}">${s.layer}</option>`; }
});
$("#svcAdd").onclick = async (ev) => {
  ev.preventDefault();
  const kind = $("#svcKind").value, url = $("#svcUrl").value.trim();
  if (!url) return;
  const layer = kind === "wmts" ? ($("#svcLayer").value || "") : $("#svcWmsLayer").value.trim();
  $("#dlgService").close(); status("Connecting to the service…");
  try {
    const s = await jpost("/api/sources/service", { kind, url, layer, name: $("#svcName").value });
    const saved = store.get("geoai.services", []).filter(x => x.url !== url || x.layer !== s.layer);
    saved.unshift({ kind, url, layer: s.layer, name: s.name }); store.set("geoai.services", saved.slice(0, 10));
    await sourceAdded(s);
  } catch (e) { status(e.message, true); }
};

/* ============================================================== tools */
const HINTS = {
  pan: "", rect: "Drag a rectangle around the area to detect",
  point: "Click an object to add it (Shift+click to exclude). Enter = add, Esc = cancel",
  box: "Drag a box around one object to add it", delete: "Click an object to delete it",
};
function setTool(t) {
  state.tool = t;
  document.querySelectorAll("#tools button").forEach(b => b.classList.toggle("active", b.dataset.tool === t));
  map.getCanvas().classList.toggle("crosshair", t !== "pan");
  (t === "rect" || t === "box") ? map.dragPan.disable() : map.dragPan.enable();
  $("#toolHint").textContent = HINTS[t]; $("#toolHint").classList.toggle("hidden", !HINTS[t]);
  if (t !== "point") clearPoints();
  $("#pointActions").classList.toggle("hidden", t !== "point");
}
$("#tools").onclick = (e) => { const t = e.target.closest("button")?.dataset.tool; if (t) setTool(t); };
document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName) || document.querySelector("dialog[open]")) return;
  const k = e.key.toLowerCase();
  if (k === "v") setTool("pan"); else if (k === "r") setTool("rect"); else if (k === "p") setTool("point");
  else if (k === "b") setTool("box"); else if (k === "d") setTool("delete");
  else if (e.key === "Enter" && state.points.length) acceptPoints();
  else if (e.key === "Escape") { clearPoints(); setTool("pan"); }
});

/* drag rectangle (used by "rect" and "box") */
let dragStart = null;
map.on("mousedown", (e) => { if (state.tool === "rect" || state.tool === "box") dragStart = e.lngLat; });
map.on("mousemove", (e) => {
  if (!dragStart) return;
  const a = dragStart, b = e.lngLat;
  map.getSource("drag").setData({ type: "Feature", geometry: { type: "LineString",
    coordinates: [[a.lng, a.lat], [b.lng, a.lat], [b.lng, b.lat], [a.lng, b.lat], [a.lng, a.lat]] } });
});
map.on("mouseup", async (e) => {
  if (!dragStart) return;
  const a = dragStart, b = e.lngLat; dragStart = null;
  map.getSource("drag").setData(empty());
  const pa = map.project(a), pb = map.project(b);
  if (Math.abs(pa.x - pb.x) < 6 || Math.abs(pa.y - pb.y) < 6) return;
  const bb = [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)];
  if (state.tool === "rect") { setAoi(bb); setTool("pan"); }
  else if (state.tool === "box") { const r = await segment({ box: bb }, true); if (r) await loadResults(); }
});

function setAoi(bb) {
  state.aoi = bb; store.set("geoai.aoi", bb);
  map.getSource("aoi").setData(bb ? { type: "Feature", geometry: { type: "Polygon", coordinates: [[
    [bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]], [bb[0], bb[1]]]] } } : empty());
  updateAoiInfo();
}
async function updateAoiInfo() {
  const el = $("#aoiInfo"), s = activeSource();
  if (!state.aoi) { el.innerHTML = 'Choose the <b>⬚ Rectangle</b> tool (R) and drag on the map.'; return; }
  if (!s) { el.textContent = "Rectangle set. Open an image or add a service."; return; }
  try {
    const [w, so, e, n] = state.aoi;
    const r = await api(`/api/estimate?source=${s.id}&w=${w}&s=${so}&e=${e}&n=${n}&gsd=${$("#gsd").value}`);
    el.innerHTML = `<b>${r.area_ha < 100 ? r.area_ha.toFixed(1) : Math.round(r.area_ha)} ha</b> · ${r.megapixels} Mpx<br>` +
      `Estimated time: ${r.minutes[0]}–${r.minutes[1]} min (${state.cfg.device.toUpperCase()})` +
      (r.too_big ? '<br><b style="color:var(--danger)">Very large area: consider a smaller rectangle or coarser resolution.</b>' : "");
  } catch (err) { el.textContent = err.message; }
}
$("#gsd").onchange = updateAoiInfo;
$("#btnZoomAoi").onclick = () => state.aoi && map.fitBounds([[state.aoi[0], state.aoi[1]], [state.aoi[2], state.aoi[3]]], { padding: 40 });

/* ============================================================= classes */
function renderClasses() {
  const saved = store.get("geoai.classes", null);
  $("#classList").innerHTML = state.cfg.classes.map(c => {
    const on = saved ? saved.includes(c.id) : c.default;
    const sub = (c.options || []).map(o => {
      const k = `${c.id}:${o.id}`; const so = saved ? saved.includes(k) : o.default;
      return `<label><input type="checkbox" data-opt="${o.id}" data-parent="${c.id}" ${so ? "checked" : ""}> ${o.name}</label>`;
    }).join("");
    return `<label><input type="checkbox" data-cls="${c.id}" ${on ? "checked" : ""}>
      <span class="swatch" style="background:${colorFor(c.id)}"></span> ${c.name}</label>${sub ? `<div class="sub">${sub}</div>` : ""}`;
  }).join("");
}
function chosenClasses() {
  const cls = [...document.querySelectorAll("[data-cls]:checked")].map(e => e.dataset.cls);
  const opts = {};
  document.querySelectorAll("[data-opt]").forEach(e => { opts[e.dataset.opt] = e.checked && cls.includes(e.dataset.parent); });
  store.set("geoai.classes", [...cls, ...[...document.querySelectorAll("[data-opt]:checked")].map(e => `${e.dataset.parent}:${e.dataset.opt}`)]);
  return { cls, opts };
}

/* ============================================================== run */
$("#btnRun").onclick = async () => {
  const s = activeSource();
  if (!s) return status("Open an image or add a service first", true);
  if (!state.aoi) { setTool("rect"); return status("Draw the detection rectangle first", true); }
  const { cls, opts } = chosenClasses();
  const custom = $("#customClass").value.split(",").map(x => x.trim()).filter(Boolean);
  if (!cls.length && !custom.length) return status("Choose at least one thing to detect", true);
  $("#btnRun").disabled = true; $("#btnCancel").classList.remove("hidden");
  const bar = $("#progress"); bar.classList.remove("hidden"); bar.firstElementChild.style.width = "0";
  try {
    const job = await jpost("/api/detect", { source: s.id, bbox: state.aoi, classes: cls, options: opts,
      gsd: +$("#gsd").value, custom });
    state.job = job.job_id;
    while (true) {
      await new Promise(r => setTimeout(r, 1000));
      const j = await api(`/api/jobs/${job.job_id}`);
      bar.firstElementChild.style.width = `${Math.round(j.progress * 100)}%`;
      bar.lastElementChild.textContent = j.message;
      status(j.message);
      if (j.status === "done") break;
      if (j.status === "cancelled") throw new Error("Detection cancelled");
      if (j.status === "error") throw new Error(j.message);
    }
    await loadResults();
  } catch (e) { status(e.message, true); }
  finally {
    state.job = null; $("#btnRun").disabled = false; $("#btnCancel").classList.add("hidden");
    setTimeout(() => bar.classList.add("hidden"), 2500);
  }
};
$("#btnCancel").onclick = () => state.job && jpost(`/api/jobs/${state.job}/cancel`, {});

/* ============================================================ results */
async function loadResults() {
  const [fc, pr] = await Promise.all([api("/api/features"), api("/api/project")]);
  for (const f of fc.features) f.properties.color = colorFor(f.properties.class);
  state.features = fc; state.project = pr;
  redraw(); renderLayers(); renderRuns();
}
function redraw() {
  map.getSource("features")?.setData({ type: "FeatureCollection",
    features: state.features.features.filter(f => !state.hidden.has(f.properties.class)) });
}
function sortedClasses(counts) {
  const order = ["building", "road", "road_centerline", "sidewalk", "parking", "tree", "solar panel"];
  return Object.keys(counts).sort((a, b) => ((order.indexOf(a) + 1) || 99) - ((order.indexOf(b) + 1) || 99));
}
function renderLayers() {
  const counts = state.project?.counts || {};
  const keys = sortedClasses(counts);
  $("#layerList").innerHTML = keys.length ? keys.map(k => `<div class="item" data-k="${k}">
    <input type="checkbox" ${state.hidden.has(k) ? "" : "checked"}><span class="swatch" style="background:${colorFor(k)}"></span>
    <span class="name">${nice(k)}</span><span class="n">${counts[k]}</span></div>`).join("") : '<p class="hint">No results yet.</p>';
  const cls = Object.keys(counts);
  $("#drawClass").innerHTML = ["building", "tree", "solar panel", "road", ...cls.filter(c => !["building", "tree", "solar panel", "road", "road_centerline"].includes(c))]
    .filter((v, i, a) => a.indexOf(v) === i).map(c => `<option value="${c}">${nice(c)}</option>`).join("");
}
function renderRuns() {
  const runs = state.project?.runs || [];
  $("#runList").innerHTML = runs.length ? "<p class='hint'>Detection runs:</p>" + runs.map(r => `<div class="item" data-r="${r.id}">
    <span class="name" title="${r.classes.join(", ")}">${r.id.slice(9, 11)}:${r.id.slice(11, 13)} · ${r.source_name} · ${r.count} objects · ${r.seconds}s</span>
    <button class="x" title="Delete this run's results">✕</button></div>`).join("") : "";
}
$("#layerList").addEventListener("change", (e) => {
  const k = e.target.closest(".item")?.dataset.k; if (!k) return;
  e.target.checked ? state.hidden.delete(k) : state.hidden.add(k); redraw();
});
$("#runList").addEventListener("click", async (e) => {
  const it = e.target.closest(".item"); if (!it) return;
  const r = state.project.runs.find(x => x.id === it.dataset.r);
  if (e.target.classList.contains("x")) {
    if (!confirm("Delete the results of this run?")) return;
    await api(`/api/runs/${r.id}`, { method: "DELETE" }); return loadResults();
  }
  map.fitBounds([[r.bbox[0], r.bbox[1]], [r.bbox[2], r.bbox[3]]], { padding: 40 });
});
$("#btnClear").onclick = async () => {
  if (!confirm("Delete ALL detection results in this work area?")) return;
  await jpost("/api/project/clear", {}); await loadResults(); status("Results cleared");
};

/* feature popup (pan tool) and delete tool */
let hoverId = null;
map.on("mousemove", "f-fill", (e) => {
  if (!["delete", "pan"].includes(state.tool)) return;
  if (hoverId !== null) map.setFeatureState({ source: "features", id: hoverId }, { hover: false });
  hoverId = e.features[0].id; map.setFeatureState({ source: "features", id: hoverId }, { hover: true });
});
map.on("mouseleave", "f-fill", () => { if (hoverId !== null) map.setFeatureState({ source: "features", id: hoverId }, { hover: false }); hoverId = null; });
const SKIP = new Set(["fid", "color", "run", "source"]);
map.on("click", async (e) => {
  if (state.tool === "point") {
    state.points.push([e.lngLat.lng, e.lngLat.lat, e.originalEvent.shiftKey ? 0 : 1]);
    return updatePoints();
  }
  if (!["pan", "delete"].includes(state.tool)) return;
  const bb = [[e.point.x - 4, e.point.y - 4], [e.point.x + 4, e.point.y + 4]];
  const f = map.queryRenderedFeatures(bb, { layers: ["f-line", "f-fill"] })[0];
  if (!f) return;
  if (state.tool === "delete") {
    await api(`/api/features/${f.properties.fid}`, { method: "DELETE" });
    await loadResults(); return status(`Deleted one ${nice(f.properties.class).toLowerCase()} object`);
  }
  const rows = Object.entries(f.properties).filter(([k]) => !SKIP.has(k))
    .map(([k, v]) => `<tr><td>${k}</td><td>${k === "class" ? nice(v) : v}</td></tr>`).join("");
  new maplibregl.Popup({ closeButton: true, maxWidth: "260px" }).setLngLat(e.lngLat).setHTML(`<table>${rows}</table>`).addTo(map);
});

/* click-add / box-add (real-time SAM) */
async function segment(body, commit) {
  status("Segmenting…"); const t0 = performance.now();
  try {
    const r = await jpost("/api/segment", { ...body, source: state.active || "", label: $("#drawClass").value || "building", commit });
    status(`${commit ? "Added" : "Preview"} in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    return r;
  } catch (e) { status(e.message, true); return null; }
}
async function updatePoints() {
  map.getSource("pts").setData({ type: "FeatureCollection", features: state.points.map(p => ({ type: "Feature",
    properties: { l: p[2] }, geometry: { type: "Point", coordinates: [p[0], p[1]] } })) });
  if (!state.points.length) return map.getSource("preview").setData(empty());
  const r = await segment({ points: state.points }, false);
  if (r) map.getSource("preview").setData(r.feature);
}
async function acceptPoints() {
  if (!state.points.length) return;
  const r = await segment({ points: state.points }, true); clearPoints();
  if (r) await loadResults();
}
function clearPoints() { state.points = []; map.getSource("pts")?.setData(empty()); map.getSource("preview")?.setData(empty()); }
$("#acceptPts").onclick = acceptPoints;
$("#clearPts").onclick = () => { clearPoints(); setTool("pan"); };

/* ============================================================== export */
$("#btnExport").onclick = () => {
  const counts = state.project?.counts || {};
  if (!Object.keys(counts).length) return status("Nothing to export yet. Run a detection first.", true);
  $("#expLayers").innerHTML = sortedClasses(counts).map(k => `<label><input type="checkbox" data-exp="${k}" checked>
    <span class="swatch" style="background:${colorFor(k)}"></span> ${nice(k)} (${counts[k]})</label>`).join("");
  $("#expName").value = $("#expName").value || `geoai_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}`;
  $("#expFolder").value = state.exportFolder || store.get("geoai.exportFolder", "");
  $("#expMsg").textContent = ""; $("#dlgExport").showModal();
};
$("#expCrs").onchange = () => $("#expCrsCustom").classList.toggle("hidden", $("#expCrs").value !== "custom");
$("#expBrowse").onclick = async () => {
  const p = await window.pywebview?.api.choose_folder();
  if (p) { $("#expFolder").value = p; state.exportFolder = p; store.set("geoai.exportFolder", p); }
};
$("#expGo").onclick = async () => {
  const formats = [$("#fmtGdb").checked && "gdb", $("#fmtShp").checked && "shp"].filter(Boolean);
  if (!formats.length) return ($("#expMsg").textContent = "Choose at least one format.");
  const crs = $("#expCrs").value === "custom" ? $("#expCrsCustom").value.trim() : $("#expCrs").value;
  const classes = [...document.querySelectorAll("[data-exp]:checked")].map(e => e.dataset.exp);
  const folder = state.desktop ? $("#expFolder").value.trim() : "";
  if (state.desktop && !folder) return ($("#expMsg").textContent = "Choose a folder (Browse…).");
  $("#expMsg").textContent = "Exporting…"; $("#expGo").disabled = true;
  try {
    const r = await jpost("/api/export", { formats, crs, folder, name: $("#expName").value, classes });
    if (r.download) { window.location = r.download; $("#expMsg").textContent = `Done (${r.crs}). Download started.`; }
    else {
      $("#expMsg").innerHTML = `Saved (${r.crs}):<br>${r.paths.map(p => `<code>${p}</code>`).join("<br>")}
        <br><button type="button" class="small" id="openOut">Open folder</button>`;
      $("#openOut").onclick = () => window.pywebview?.api.open_folder(r.paths[0]);
    }
    status("Export finished");
  } catch (e) { $("#expMsg").textContent = e.message; }
  finally { $("#expGo").disabled = false; }
};

/* =============================================================== init */
async function init() {
  try {
    state.cfg = await api("/api/config");
    state.desktop = state.cfg.desktop;
    document.querySelectorAll(".desktop-only").forEach(e => e.classList.toggle("hidden", !state.desktop));
    $("#device").textContent = state.cfg.device === "cpu" ? "CPU" : "GPU (" + state.cfg.device + ")";
    renderClasses();
    state.sources = await api("/api/sources");
    for (const s of state.sources) addSourceLayer(s);
    const act = store.get("geoai.active", null);
    state.active = state.sources.find(s => s.id === act) ? act : state.sources[0]?.id || null;
    renderSources();
    const aoi = store.get("geoai.aoi", null);
    if (aoi) setAoi(aoi);
    await loadResults();
    if (aoi) map.fitBounds([[aoi[0], aoi[1]], [aoi[2], aoi[3]]], { padding: 40, duration: 0 });
    else if (activeSource()) zoomToSource(activeSource());
    status(state.sources.length ? "Ready" : "Ready. Start with 📂 Open image or 🌐 Add service.");
  } catch (e) { status("Cannot reach the app server: " + e.message, true); }
}
