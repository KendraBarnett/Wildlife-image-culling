const state = {
  folder: null,
  images: [],
  filterRating: "",
  current: null,
  saveTimer: null,
  browse: { path: "", parent: null, dirs: [], isRoot: true },
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

async function jget(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}
async function jpost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

async function refreshHealth() {
  try {
    const h = await jget("/api/health");
    const el = $("#health");
    if (h.ok) {
      el.textContent = `ready · ${h.vision_model}`;
      el.className = "health ok";
    } else {
      el.textContent = h.ollama_issue || "issue";
      el.className = "health bad";
    }
    updateWorkerControls(!!h.worker_paused);
  } catch (e) {
    $("#health").textContent = "server unreachable";
    $("#health").className = "health bad";
  }
}

function updateWorkerControls(paused) {
  const pill = $("#worker-state");
  pill.textContent = paused ? "analysis: paused" : "analysis: running";
  pill.classList.toggle("paused", paused);
  $("#worker-pause").classList.toggle("hidden", paused);
  $("#worker-resume").classList.toggle("hidden", !paused);
}

async function workerAction(path) {
  try {
    const r = await jpost(path, {});
    updateWorkerControls(!!r.paused);
    return r;
  } catch (e) {
    alert("Action failed: " + e.message);
  }
}

async function refreshFolders() {
  const folders = await jget("/api/folders");
  const bar = $("#folders-bar");
  bar.innerHTML = "";
  for (const f of folders) {
    const b = document.createElement("button");
    b.className = "folder-chip" + (state.folder === f.folder ? " active" : "");
    const short = f.folder.split("/").slice(-2).join("/");
    b.textContent = `${short} (${f.n})`;
    b.title = f.folder;
    b.onclick = () => selectFolder(f.folder);
    bar.appendChild(b);
  }
}

async function refreshStats() {
  const s = await jget("/api/stats");
  $("#stats").textContent = `${s.total || 0} total · ${s.analyzed || 0} analyzed · ${s.pending || 0} pending · ${s.rated || 0} rated`;
}

async function selectFolder(folder) {
  state.folder = folder;
  await refreshFolders();
  await refreshGrid();
}

async function refreshGrid() {
  const params = new URLSearchParams();
  if (state.folder) params.set("folder", state.folder);
  const fr = state.filterRating;
  if (fr === "0") {
    params.set("rating_max", "0");
  } else if (fr) {
    params.set("rating_min", fr);
    if (fr === "5") params.set("rating_max", "5");
  }
  const data = await jget(`/api/images?${params}`);
  state.images = data.images;
  renderGrid();
}

function renderGrid() {
  const grid = $("#grid");
  grid.innerHTML = "";
  for (const img of state.images) {
    grid.appendChild(renderCard(img));
  }
}

function renderCard(img) {
  const card = document.createElement("div");
  card.className = "card";
  for (const t of img.user_tags || []) card.classList.add(`tag-${t}`);

  const im = document.createElement("img");
  im.loading = "lazy";
  im.src = `/api/thumb/${img.id}`;
  im.alt = img.filename;
  card.appendChild(im);

  const badges = document.createElement("div");
  badges.className = "badges";
  if (img.ai_status === "done") {
    if (img.ai_eye_focus === "sharp") badges.appendChild(badge("eye✓", "good"));
    else if (img.ai_eye_focus === "soft") badges.appendChild(badge("eye~", "warn"));
    if (img.ai_motion === "in_motion") badges.appendChild(badge("motion", "good"));
    else if (img.ai_motion === "blurred") badges.appendChild(badge("blur", "warn"));
    if (img.ai_is_silhouette) badges.appendChild(badge("silh", "silh"));
  }
  card.appendChild(badges);

  const overlay = document.createElement("div");
  overlay.className = "rating-overlay";
  const stars = document.createElement("span");
  stars.className = "stars-mini";
  stars.textContent = img.user_rating ? "★".repeat(img.user_rating) : "";
  overlay.appendChild(stars);
  const scores = document.createElement("span");
  scores.className = "scores";
  if (img.ai_status === "done") {
    scores.textContent = `art ${img.ai_artistic_score || "?"} · port ${img.ai_portfolio_score || "?"}`;
  } else if (img.ai_status === "pending") {
    scores.textContent = "analyzing…";
  } else if (img.ai_status === "error") {
    scores.textContent = "ai error";
  }
  overlay.appendChild(scores);
  card.appendChild(overlay);

  card.onclick = () => openModal(img.id);
  return card;
}

function badge(text, cls) {
  const b = document.createElement("span");
  b.className = "badge " + (cls || "");
  b.textContent = text;
  return b;
}

async function openModal(id) {
  const img = await jget(`/api/image/${id}`);
  state.current = img;
  $("#modal").classList.remove("hidden");
  $("#modal-img").src = `/api/preview/${id}`;
  $("#modal-filename").textContent = img.filename;
  renderAiBlock(img);
  setStars(img.user_rating || 0);
  setTags(img.user_tags || []);
  $("#notes-input").value = img.user_notes || "";
  $("#save-status").textContent = "";
}

function renderAiBlock(img) {
  const el = $("#modal-ai");
  if (img.ai_status === "pending") {
    el.innerHTML = `<div class="pending">AI is analyzing this image…</div>`;
    return;
  }
  if (img.ai_status === "error") {
    const errMsg = (img.ai && img.ai.error) ? img.ai.error : "(no detail)";
    el.innerHTML = `<div class="ai-error"><div class="ai-error-title">AI error</div><div class="ai-error-detail"></div></div>`;
    el.querySelector(".ai-error-detail").textContent = errMsg;
    return;
  }
  const a = img.ai || {};
  const issues = (a.technical_issues || []).join(", ") || "—";
  el.innerHTML = `
    <div class="scores-row">
      <div class="score-pill"><div class="label">artistic</div><div class="val">${a.artistic_score ?? "?"}</div></div>
      <div class="score-pill"><div class="label">portfolio</div><div class="val">${a.portfolio_potential ?? "?"}</div></div>
    </div>
    <div class="row"><span>subject</span><strong>${esc(a.subject) || "—"}</strong></div>
    <div class="row"><span>eye focus</span><strong>${esc(a.eye_focus) || "—"}</strong></div>
    <div class="row"><span>motion</span><strong>${esc(a.motion) || "—"}</strong></div>
    <div class="row"><span>composition</span><strong>${esc(a.composition) || "—"}</strong></div>
    <div class="row"><span>lighting</span><strong>${esc(a.lighting) || "—"}</strong></div>
    <div class="row"><span>silhouette</span><strong>${a.is_silhouette ? "yes" : "no"}</strong></div>
    <div class="row"><span>issues</span><strong>${esc(issues)}</strong></div>
    ${a.notes ? `<div class="notes-line">${esc(a.notes)}</div>` : ""}
  `;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
}

function setStars(v) {
  $("#stars").dataset.rating = v;
  $$("#stars span").forEach((el) => {
    el.classList.toggle("lit", parseInt(el.dataset.v) <= v);
  });
}

function setTags(tags) {
  $$("#tag-chips button").forEach((el) => {
    el.classList.toggle("active", tags.includes(el.dataset.tag));
  });
}

function getActiveTags() {
  return [...$$("#tag-chips button.active")].map((el) => el.dataset.tag);
}

function scheduleSave() {
  if (!state.current) return;
  if (state.saveTimer) clearTimeout(state.saveTimer);
  $("#save-status").textContent = "saving…";
  $("#save-status").className = "save-status";
  state.saveTimer = setTimeout(doSave, 350);
}

async function doSave() {
  if (!state.current) return;
  const id = state.current.id;
  const rating = parseInt($("#stars").dataset.rating || "0") || null;
  const tags = getActiveTags();
  const notes = $("#notes-input").value;
  try {
    const r = await jpost(`/api/image/${id}/rate`, { rating, tags, notes });
    $("#save-status").textContent = r.sidecar_written ? "saved · xmp written" : `saved (xmp error: ${r.sidecar_error})`;
    $("#save-status").className = "save-status " + (r.sidecar_written ? "ok" : "bad");
    state.current.user_rating = rating;
    state.current.user_tags = tags;
    state.current.user_notes = notes;
    const idx = state.images.findIndex((x) => x.id === id);
    if (idx >= 0) {
      state.images[idx].user_rating = rating;
      state.images[idx].user_tags = tags;
    }
    renderGrid();
  } catch (e) {
    $("#save-status").textContent = "save failed: " + e.message;
    $("#save-status").className = "save-status bad";
  }
}

function closeModal() {
  $("#modal").classList.add("hidden");
  state.current = null;
}

function step(delta) {
  if (!state.current) return;
  const idx = state.images.findIndex((x) => x.id === state.current.id);
  if (idx < 0) return;
  const next = state.images[idx + delta];
  if (next) openModal(next.id);
}

async function openBrowser(path) {
  const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : "/api/browse";
  let data;
  try {
    data = await jget(url);
  } catch (e) {
    alert("Could not list folder: " + e.message);
    return;
  }
  state.browse = {
    path: data.path || "",
    parent: data.parent_path,
    dirs: data.dirs || [],
    isRoot: !!data.is_root,
  };
  $("#browse-path").textContent = data.is_root ? "Choose a starting location" : data.path;
  $("#browse-up").disabled = !!data.is_root;
  $("#browse-select").disabled = !!data.is_root;

  const list = $("#browse-list");
  list.innerHTML = "";
  if (!data.dirs.length) {
    const empty = document.createElement("div");
    empty.className = "browse-empty";
    empty.textContent = data.is_root
      ? "No starting locations available."
      : "No sub-folders here. Click \"Use this folder\" to ingest this one.";
    list.appendChild(empty);
  } else {
    for (const d of data.dirs) {
      const row = document.createElement("button");
      row.className = "browse-row";
      row.title = d.path;
      row.innerHTML = `<span class="folder-icon">📁</span><span class="folder-name"></span>`;
      row.querySelector(".folder-name").textContent = d.name;
      row.onclick = () => openBrowser(d.path);
      list.appendChild(row);
    }
  }
  $("#browse-modal").classList.remove("hidden");
}

function closeBrowser() {
  $("#browse-modal").classList.add("hidden");
}

document.addEventListener("DOMContentLoaded", () => {
  $("#ingest-btn").onclick = async () => {
    const folder = $("#folder-input").value.trim();
    if (!folder) return;
    const recursive = $("#recursive").checked;
    $("#ingest-status").textContent = "ingesting…";
    try {
      const r = await jpost("/api/ingest", { folder, recursive });
      $("#ingest-status").textContent = `${r.new} new · ${r.skipped} skipped (${r.elapsed_sec}s)`;
      await refreshFolders();
      await selectFolder(r.folder);
    } catch (e) {
      $("#ingest-status").textContent = "error: " + e.message;
    }
  };

  $("#filter-rating").onchange = (e) => {
    state.filterRating = e.target.value;
    refreshGrid();
  };

  $("#stars").onclick = (e) => {
    const v = parseInt(e.target.dataset.v);
    if (!v) return;
    setStars(v);
    scheduleSave();
  };
  $("#clear-rating").onclick = () => {
    setStars(0);
    scheduleSave();
  };
  $("#tag-chips").onclick = (e) => {
    if (e.target.tagName !== "BUTTON") return;
    e.target.classList.toggle("active");
    scheduleSave();
  };
  $("#notes-input").oninput = scheduleSave;

  $("#modal-close").onclick = closeModal;
  $("#prev-btn").onclick = () => step(-1);
  $("#next-btn").onclick = () => step(1);

  $("#worker-pause").onclick = () => workerAction("/api/worker/pause");
  $("#worker-resume").onclick = () => workerAction("/api/worker/resume");
  $("#worker-cancel").onclick = async () => {
    if (!confirm("Cancel all pending AI analysis? Already-analyzed images and your ratings are kept; pending images are marked cancelled and won't be analyzed unless re-ingested.")) return;
    const r = await workerAction("/api/worker/cancel");
    if (r) {
      $("#ingest-status").textContent = `cancelled ${r.cancelled} pending`;
      refreshStats();
      if (state.folder) refreshGrid();
    }
  };

  $("#browse-btn").onclick = () => {
    const current = $("#folder-input").value.trim();
    openBrowser(current || null);
  };
  $("#browse-close").onclick = closeBrowser;
  $("#browse-cancel").onclick = closeBrowser;
  $("#browse-up").onclick = () => {
    if (state.browse.parent) openBrowser(state.browse.parent);
    else openBrowser(null);
  };
  $("#browse-select").onclick = () => {
    if (!state.browse.path) return;
    $("#folder-input").value = state.browse.path;
    closeBrowser();
  };

  document.addEventListener("keydown", (e) => {
    if (!$("#browse-modal").classList.contains("hidden")) {
      if (e.key === "Escape") closeBrowser();
      return;
    }
    if ($("#modal").classList.contains("hidden")) return;
    if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return;
    if (e.key === "Escape") closeModal();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else if (e.key >= "1" && e.key <= "5") {
      setStars(parseInt(e.key));
      scheduleSave();
    } else if (e.key === "0") {
      setStars(0);
      scheduleSave();
    }
  });

  refreshHealth();
  refreshFolders();
  refreshStats();
  setInterval(() => { refreshStats(); refreshHealth(); }, 5000);
  setInterval(() => { if (state.folder) refreshGrid(); }, 8000);
});
