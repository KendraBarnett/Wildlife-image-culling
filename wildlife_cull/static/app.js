const state = {
  folder: null,
  images: [],
  filterRating: "",
  current: null,
  saveTimer: null,
  browse: { path: "", parent: null, dirs: [], isRoot: true },
  aiVocab: null,
  judges: [],
  selectMode: false,
  selected: new Set(),
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
  const parts = [
    `${s.total || 0} total`,
    `${s.analyzed || 0} fully analyzed`,
  ];
  if (s.partial) parts.push(`${s.partial} partial`);
  parts.push(`${(s.pending || 0) - (s.partial || 0)} pending`);
  if (s.errored) parts.push(`${s.errored} errored`);
  parts.push(`${s.rated || 0} rated`);
  $("#stats").textContent = parts.join(" · ");
}

async function selectFolder(folder) {
  state.folder = folder;
  await refreshFolders();
  await refreshGrid();
}

async function refreshAiVocab() {
  if (state.aiVocab) return;
  try {
    state.aiVocab = await jget("/api/ai-vocab");
    state.judges = await jget("/api/judges");
  } catch (e) {
    state.aiVocab = {};
    state.judges = [];
    return;
  }
  for (const sel of document.querySelectorAll("select[data-vocab]")) {
    const key = sel.dataset.vocab;
    const values = state.aiVocab[key] || [];
    for (const v of values) {
      const opt = document.createElement("option");
      opt.value = v;
      opt.textContent = pretty(v);
      sel.appendChild(opt);
    }
  }
}

function readFilters() {
  const p = new URLSearchParams();
  if (state.folder) p.set("folder", state.folder);
  const fr = state.filterRating;
  if (fr === "0") {
    p.set("rating_max", "0");
  } else if (fr) {
    p.set("rating_min", fr);
    if (fr === "5") p.set("rating_max", "5");
  }
  const map = {
    "filter-ai-status": "ai_status",
    "filter-animal-type": "animal_type",
    "filter-species": "species_contains",
    "filter-eye-focus": "eye_focus",
    "filter-motion": "motion",
    "filter-composition": "composition",
    "filter-lighting": "lighting",
    "filter-silhouette": "silhouette",
    "filter-issue": "has_issue",
    "filter-feedback": "has_feedback",
    "filter-subject": "subject_contains",
    "filter-min-artistic": "min_artistic",
    "filter-min-portfolio": "min_portfolio",
  };
  for (const [id, name] of Object.entries(map)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const v = el.value.trim();
    if (v) p.set(name, v);
  }
  return p;
}

async function refreshGrid() {
  const params = readFilters();
  const data = await jget(`/api/images?${params}`);
  state.images = data.images;
  for (const id of [...state.selected]) {
    if (!state.images.find((x) => x.id === id)) state.selected.delete(id);
  }
  renderGrid();
  updateSelectInfo();
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
  if (img.ai_status === "error") card.classList.add("ai-status-error");
  if (img.ai_status === "cancelled") card.classList.add("ai-status-cancelled");
  if (img.ai_feedback) card.classList.add("has-feedback");
  if (state.selected.has(img.id)) card.classList.add("selected");

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
  const judgesDone = img.judges_done || 0;
  const judgesTotal = img.judges_total || 0;
  if (img.ai_status === "done" || judgesDone > 0) {
    const a = partialAvg(img, "artistic_score") ?? img.ai_artistic_score;
    const p = partialAvg(img, "portfolio_potential") ?? img.ai_portfolio_score;
    const stamp = (judgesTotal && judgesDone < judgesTotal) ? ` (${judgesDone}/${judgesTotal})` : "";
    scores.textContent = `art ${formatScore(a)} · port ${formatScore(p)}${stamp}`;
  } else if (img.ai_status === "pending") {
    scores.textContent = "analyzing…";
  } else if (img.ai_status === "error") {
    scores.textContent = "ai error";
  } else if (img.ai_status === "cancelled") {
    scores.textContent = "cancelled";
  }
  overlay.appendChild(scores);
  card.appendChild(overlay);

  card.onclick = () => {
    if (state.selectMode) {
      if (state.selected.has(img.id)) state.selected.delete(img.id);
      else state.selected.add(img.id);
      card.classList.toggle("selected");
      updateSelectInfo();
    } else {
      openModal(img.id);
    }
  };
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
  renderFeedback(img);
  setStars(img.user_rating || 0);
  setTags(img.user_tags || []);
  $("#notes-input").value = img.user_notes || "";
  $("#save-status").textContent = "";
}

function renderFeedback(img) {
  const fb = img.ai_feedback || null;
  const summary = $("#modal-feedback-summary");
  if (fb) {
    const bits = [];
    if (fb.marked_wrong) bits.push("<b>Marked wrong</b>");
    if (typeof fb.artistic === "number") bits.push(`Art ${formatScore(fb.artistic)}`);
    if (typeof fb.portfolio === "number") bits.push(`Port ${formatScore(fb.portfolio)}`);
    if (fb.animal_type) bits.push(`Type: ${esc(fb.animal_type)}`);
    if (fb.species) bits.push(`Species: ${esc(fb.species)}`);
    if (fb.eye_focus) bits.push(`Eye: ${esc(pretty(fb.eye_focus))}`);
    if (fb.motion) bits.push(`Motion: ${esc(pretty(fb.motion))}`);
    if (fb.composition) bits.push(`Composition: ${esc(pretty(fb.composition))}`);
    if (fb.lighting) bits.push(`Lighting: ${esc(pretty(fb.lighting))}`);
    if (typeof fb.is_silhouette === "boolean") bits.push(`Silhouette: ${fb.is_silhouette ? "Yes" : "No"}`);
    if (fb.technical_issues) bits.push(`Issues: ${(fb.technical_issues || []).map(pretty).join(", ") || "none"}`);
    const note = fb.note ? `<div class="fb-summary-note">${esc(fb.note)}</div>` : "";
    summary.innerHTML = `<div class="fb-summary-title">YOUR CORRECTIONS</div><div class="fb-summary-fields">${bits.join(" · ")}</div>${note}`;
    summary.classList.remove("hidden");
  } else {
    summary.classList.add("hidden");
    summary.innerHTML = "";
  }

  const setVal = (id, v) => { const el = $(id); if (el) el.value = (v === undefined || v === null) ? "" : v; };
  const setCheck = (id, v) => { const el = $(id); if (el) el.checked = !!v; };
  setCheck("#fb-marked-wrong", fb && fb.marked_wrong);
  setVal("#fb-artistic", fb && typeof fb.artistic === "number" ? fb.artistic : "");
  setVal("#fb-portfolio", fb && typeof fb.portfolio === "number" ? fb.portfolio : "");
  setVal("#fb-animal-type", fb && fb.animal_type);
  setVal("#fb-species", fb && fb.species);
  setVal("#fb-subject", fb && fb.subject);
  setVal("#fb-eye-focus", fb && fb.eye_focus);
  setVal("#fb-motion", fb && fb.motion);
  setVal("#fb-composition", fb && fb.composition);
  setVal("#fb-lighting", fb && fb.lighting);
  setVal("#fb-silhouette", fb && (fb.is_silhouette === true ? "yes" : fb.is_silhouette === false ? "no" : ""));
  setVal("#fb-note", fb && fb.note);
  populateIssueChips(fb);
  $("#fb-status").textContent = "";
}

function populateIssueChips(fb) {
  const wrap = $("#fb-issue-chips");
  if (!wrap) return;
  wrap.innerHTML = "";
  const vocab = (state.aiVocab && state.aiVocab.technical_issues) || [];
  const selected = new Set((fb && fb.technical_issues) || []);
  for (const v of vocab) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.value = v;
    b.textContent = pretty(v);
    if (selected.has(v)) b.classList.add("active");
    b.onclick = () => b.classList.toggle("active");
    wrap.appendChild(b);
  }
}

function renderAiBlock(img) {
  const el = $("#modal-ai");
  const judges = state.judges || [];
  const judgeData = img.ai_judges || {};

  if (img.ai_status === "pending" && Object.keys(judgeData).length === 0) {
    const order = (judges.length ? judges.map((j) => j.label).join(" → ") : "the judges");
    el.innerHTML = `<div class="pending">Analyzing this image. Each judge runs in turn (${esc(order)}).</div>`;
    return;
  }
  if (img.ai_status === "error" && Object.keys(judgeData).length === 0) {
    const errMsg = (img.ai && img.ai.error) ? img.ai.error : "(no detail)";
    el.innerHTML = `<div class="ai-error"><div class="ai-error-title">AI error</div><div class="ai-error-detail"></div></div>`;
    el.querySelector(".ai-error-detail").textContent = errMsg;
    return;
  }

  const cards = judges.map((j) => renderJudgeCard(j, judgeData[j.name])).join("");
  const a = (() => {
    const primary = judges.find((j) => j.primary) || judges[0];
    if (primary && judgeData[primary.name] && judgeData[primary.name].result) {
      return judgeData[primary.name].result;
    }
    return img.ai || {};
  })();
  const issues = (a.technical_issues || []).map(pretty).join(", ") || "—";

  el.innerHTML = `
    <div class="judges-row">${cards}</div>
    <div class="row"><span>Type</span><strong>${esc(a.animal_type) || "—"}</strong></div>
    <div class="row"><span>Species</span><strong>${esc(a.species) || "—"}</strong></div>
    <div class="row"><span>Subject</span><strong>${esc(a.subject) || "—"}</strong></div>
    <div class="row"><span>Eye focus</span><strong>${esc(pretty(a.eye_focus)) || "—"}</strong></div>
    <div class="row"><span>Motion</span><strong>${esc(pretty(a.motion)) || "—"}</strong></div>
    <div class="row"><span>Composition</span><strong>${esc(pretty(a.composition)) || "—"}</strong></div>
    <div class="row"><span>Lighting</span><strong>${esc(pretty(a.lighting)) || "—"}</strong></div>
    <div class="row"><span>Silhouette</span><strong>${a.is_silhouette ? "Yes" : "No"}</strong></div>
    <div class="row"><span>Issues</span><strong>${esc(issues)}</strong></div>
    ${a.notes ? `<div class="notes-line">${esc(a.notes)}</div>` : ""}
  `;
}

function renderJudgeCard(judge, slot) {
  if (!slot) {
    return `<div class="judge-card pending"><div class="judge-name">${esc(judge.label)}</div><div class="judge-state">Pending…</div><div class="judge-model">${esc(judge.model)}</div></div>`;
  }
  if (slot.status === "error") {
    const detail = slot.error || "(no detail)";
    return `<div class="judge-card error"><div class="judge-name">${esc(judge.label)}</div><div class="judge-state">Error</div><div class="judge-error-detail" title="${esc(detail)}">${esc(detail.slice(0, 120))}</div><div class="judge-model">${esc(judge.model)}</div></div>`;
  }
  const r = slot.result || {};
  const notes = r.notes ? `<div class="judge-notes">${esc(r.notes)}</div>` : "";
  const art = r.artistic_score ?? "?";
  const port = r.portfolio_potential ?? "?";
  return `<div class="judge-card done">
    <div class="judge-name">${esc(judge.label)}</div>
    <div class="judge-scores"><span class="score">Art <b>${art}</b><span class="denom">/10</span></span><span class="score">Port <b>${port}</b><span class="denom">/10</span></span></div>
    ${notes}
    <div class="judge-model">${esc(judge.model)}</div>
  </div>`;
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"})[c]);
}

function pretty(s) {
  if (s === null || s === undefined || s === "") return "";
  return String(s).replace(/_/g, " ").replace(/\b([a-z])/g, (_, c) => c.toUpperCase());
}

function formatScore(v) {
  if (v === null || v === undefined || v === "") return "?";
  const n = Number(v);
  if (Number.isNaN(n)) return "?";
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

function partialAvg(img, field) {
  const judges = img.ai_judges || {};
  const vals = [];
  for (const slot of Object.values(judges)) {
    if (slot && slot.status === "done" && slot.result) {
      const v = slot.result[field];
      if (typeof v === "number") vals.push(v);
    }
  }
  if (!vals.length) return null;
  return Math.round((vals.reduce((s, n) => s + n, 0) / vals.length) * 10) / 10;
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

function setSelectMode(on) {
  state.selectMode = on;
  if (!on) state.selected.clear();
  $("#select-toggle").textContent = on ? "Exit select mode" : "Select…";
  $("#select-toggle").classList.toggle("danger", on);
  document.body.classList.toggle("select-mode", on);
  for (const cls of ["select-info", "select-all-visible", "select-clear", "reanalyze-selected"]) {
    $("#" + cls).classList.toggle("hidden", !on);
  }
  renderGrid();
  updateSelectInfo();
}

function updateSelectInfo() {
  const n = state.selected.size;
  $("#select-info").textContent = `${n} selected`;
  $("#reanalyze-selected").disabled = n === 0;
}

async function reanalyzeIds(ids) {
  if (!ids.length) return;
  try {
    await jpost("/api/reanalyze", { ids });
    state.selected.clear();
    setSelectMode(false);
    refreshStats();
    refreshGrid();
  } catch (e) {
    alert("Re-analyze failed: " + e.message);
  }
}

async function reanalyzeByStatus(status) {
  try {
    const body = { status };
    if (state.folder) body.folder = state.folder;
    const r = await jpost("/api/reanalyze", body);
    $("#ingest-status").textContent = `requeued ${r.reset} for analysis`;
    refreshStats();
    refreshGrid();
  } catch (e) {
    alert("Re-analyze failed: " + e.message);
  }
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

  for (const id of [
    "filter-ai-status", "filter-animal-type", "filter-eye-focus", "filter-motion",
    "filter-composition", "filter-lighting", "filter-silhouette", "filter-issue",
    "filter-feedback",
  ]) {
    $("#" + id).onchange = refreshGrid;
  }
  for (const id of ["filter-species", "filter-subject", "filter-min-artistic", "filter-min-portfolio"]) {
    let t;
    $("#" + id).oninput = () => {
      clearTimeout(t);
      t = setTimeout(refreshGrid, 250);
    };
  }
  $("#filter-clear").onclick = () => {
    state.filterRating = "";
    $("#filter-rating").value = "";
    for (const id of [
      "filter-ai-status", "filter-animal-type", "filter-species",
      "filter-eye-focus", "filter-motion", "filter-composition",
      "filter-lighting", "filter-silhouette", "filter-issue", "filter-feedback",
      "filter-subject", "filter-min-artistic", "filter-min-portfolio",
    ]) {
      $("#" + id).value = "";
    }
    refreshGrid();
  };

  $("#select-toggle").onclick = () => setSelectMode(!state.selectMode);
  $("#select-clear").onclick = () => {
    state.selected.clear();
    renderGrid();
    updateSelectInfo();
  };
  $("#select-all-visible").onclick = () => {
    for (const img of state.images) state.selected.add(img.id);
    renderGrid();
    updateSelectInfo();
  };
  $("#reanalyze-selected").onclick = () => {
    const ids = [...state.selected];
    if (!ids.length) return;
    if (!confirm(`Re-analyze ${ids.length} image(s)? They will go back into the analysis queue.`)) return;
    reanalyzeIds(ids);
  };
  $("#reanalyze-errored").onclick = () => {
    const scope = state.folder ? "in this folder" : "across all folders";
    if (!confirm(`Re-analyze every errored image ${scope}?`)) return;
    reanalyzeByStatus("error");
  };

  $("#fb-save").onclick = async () => {
    if (!state.current) return;
    const id = state.current.id;
    const issues = [...document.querySelectorAll("#fb-issue-chips button.active")].map((b) => b.dataset.value);
    const numOrNull = (sel) => {
      const v = $(sel).value;
      return v === "" ? null : Number(v);
    };
    const strOrNull = (sel) => {
      const v = $(sel).value.trim();
      return v === "" ? null : v;
    };
    const silVal = $("#fb-silhouette").value;
    const body = {
      marked_wrong: $("#fb-marked-wrong").checked,
      artistic: numOrNull("#fb-artistic"),
      portfolio: numOrNull("#fb-portfolio"),
      animal_type: strOrNull("#fb-animal-type"),
      species: strOrNull("#fb-species"),
      subject: strOrNull("#fb-subject"),
      eye_focus: strOrNull("#fb-eye-focus"),
      motion: strOrNull("#fb-motion"),
      composition: strOrNull("#fb-composition"),
      lighting: strOrNull("#fb-lighting"),
      is_silhouette: silVal === "yes" ? true : silVal === "no" ? false : null,
      technical_issues: issues.length ? issues : null,
      note: strOrNull("#fb-note"),
    };
    Object.keys(body).forEach((k) => { if (body[k] === null) delete body[k]; });
    try {
      const r = await jpost(`/api/image/${id}/feedback`, body);
      $("#fb-status").textContent = "Saved.";
      $("#fb-status").className = "fb-status ok";
      state.current.ai_feedback = r.feedback;
      renderFeedback(state.current);
      const idx = state.images.findIndex((x) => x.id === id);
      if (idx >= 0) state.images[idx].ai_feedback = r.feedback;
      renderGrid();
      refreshStats();
    } catch (e) {
      $("#fb-status").textContent = "Save failed: " + e.message;
      $("#fb-status").className = "fb-status bad";
    }
  };
  $("#fb-clear").onclick = async () => {
    if (!state.current) return;
    if (!confirm("Clear your corrections for this image?")) return;
    const id = state.current.id;
    try {
      await fetch(`/api/image/${id}/feedback`, { method: "DELETE" });
      state.current.ai_feedback = null;
      renderFeedback(state.current);
      const idx = state.images.findIndex((x) => x.id === id);
      if (idx >= 0) state.images[idx].ai_feedback = null;
      renderGrid();
      refreshStats();
      $("#fb-status").textContent = "Cleared.";
      $("#fb-status").className = "fb-status";
    } catch (e) {
      $("#fb-status").textContent = "Clear failed: " + e.message;
      $("#fb-status").className = "fb-status bad";
    }
  };

  $("#modal-reanalyze").onclick = async () => {
    if (!state.current) return;
    const id = state.current.id;
    try {
      await jpost(`/api/image/${id}/reanalyze`);
    } catch (e) {
      alert("Re-analyze failed: " + e.message);
      return;
    }
    closeModal();
    refreshStats();
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

  refreshAiVocab();
  refreshHealth();
  refreshFolders();
  refreshStats();
  setInterval(() => { refreshStats(); refreshHealth(); }, 5000);
  setInterval(() => { if (state.folder) refreshGrid(); }, 8000);
});
