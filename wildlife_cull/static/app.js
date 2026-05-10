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

function renderBudgetPill(c) {
  const el = $("#budget");
  if (!el) return;
  if (!c) { el.textContent = ""; el.className = "budget"; return; }
  if (!c.api_key_set) {
    el.textContent = "Claude: API key not set";
    el.className = "budget budget-disabled";
    el.title = "Set ANTHROPIC_API_KEY in scripts/run.sh to enable Phase 2 scoring.";
    renderBulkProgress(c.bulk);
    return;
  }
  const label = `Claude: $${c.spent_usd.toFixed(4)} of $${c.budget_usd.toFixed(2)} · ${c.images} scored · ${c.pct}%`;
  el.textContent = label;
  el.className = "budget budget-" + c.state;
  el.title = c.state === "blocked"
    ? "Budget cap reached. Raise CLAUDE_BUDGET_USD in scripts/run.sh and restart."
    : c.state === "warning"
    ? "Approaching budget cap (≥80%). New scoring still allowed until 100%."
    : `Model: ${c.model}. Remaining: $${c.remaining_usd.toFixed(4)}.`;
  renderBulkProgress(c.bulk);
}

let _bulkPollTimer = null;
function startBulkPoll() {
  if (_bulkPollTimer) return;
  _bulkPollTimer = setInterval(async () => {
    await refreshHealth();
    const b = window._lastBulkState;
    if (b && b.state !== "running") {
      clearInterval(_bulkPollTimer);
      _bulkPollTimer = null;
      refreshGrid();
    }
  }, 2000);
}

function renderBulkProgress(b) {
  window._lastBulkState = b;
  if (b && b.state === "running" && !_bulkPollTimer) startBulkPoll();
  const startBtn = $("#score-keepers");
  const cancelBtn = $("#score-keepers-cancel");
  const progress = $("#score-keepers-progress");
  if (!startBtn || !progress) return;
  if (!b || b.state === "idle") {
    startBtn.classList.remove("hidden");
    cancelBtn.classList.add("hidden");
    progress.classList.add("hidden");
    return;
  }
  if (b.state === "running") {
    startBtn.classList.add("hidden");
    cancelBtn.classList.remove("hidden");
    progress.classList.remove("hidden");
    progress.textContent = `Scoring ${b.done}/${b.total}${b.failed ? ` (${b.failed} errored)` : ""} · $${b.spent_this_run.toFixed(4)} this run · ${b.last_filename || "starting…"}`;
    progress.className = "bulk-info";
    return;
  }
  // Terminal states: done, cancelled, budget_blocked, error
  startBtn.classList.remove("hidden");
  cancelBtn.classList.add("hidden");
  progress.classList.remove("hidden");
  const summary = `${b.done}/${b.total} scored${b.failed ? `, ${b.failed} errored` : ""} · $${b.spent_this_run.toFixed(4)}`;
  if (b.state === "done") {
    progress.textContent = `Done: ${summary}`;
    progress.className = "bulk-info bulk-done";
  } else if (b.state === "cancelled") {
    progress.textContent = `Stopped: ${summary}`;
    progress.className = "bulk-info";
  } else if (b.state === "budget_blocked") {
    progress.textContent = `Budget cap hit at ${summary}. Raise CLAUDE_BUDGET_USD in run.sh to continue.`;
    progress.className = "bulk-info bulk-blocked";
  } else {
    progress.textContent = `Stopped: ${summary}${b.last_error ? ` (${b.last_error.slice(0, 80)})` : ""}`;
    progress.className = "bulk-info bulk-blocked";
  }
}

async function refreshHealth() {
  try {
    const h = await jget("/api/health");
    const el = $("#health");
    if (h.ok) {
      const trained = h.feedback_active
        ? ` · trained on ${h.feedback_active} of your corrections`
        : (h.feedback_corrections === 0 ? " · no corrections yet" : "");
      const t = h.worker_last_timing;
      const timing = t
        ? ` · last: ${t.total_secs}s (enc ${t.image_encode_secs}s, prompt ${t.prompt_eval_secs ?? "?"}s, gen ${t.eval_secs ?? "?"}s)`
        : "";
      const inflight = h.worker_in_flight
        ? ` · working on ${h.worker_in_flight.filename} (${Math.round(Date.now()/1000 - h.worker_in_flight.started_at)}s)`
        : "";
      el.textContent = `ready · ${h.vision_model}${trained}${inflight}${timing}`;
      el.className = "health ok";
    } else {
      el.textContent = h.ollama_issue || "issue";
      el.className = "health bad";
    }
    state.workerPaused = !!h.worker_paused;
    updateWorkerControls(state.workerPaused);
    renderBudgetPill(h.claude);
    // If the pause state changed since last poll, re-render the grid so
    // pending images flip between 'analyzing…' and 'paused'.
    if (state._lastWorkerPaused !== state.workerPaused) {
      state._lastWorkerPaused = state.workerPaused;
      if (state.images && state.images.length) renderGrid();
    }
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
  state.folders = folders;
  const bar = $("#folders-bar");
  bar.innerHTML = "";

  // 'All' pseudo-tab — aggregates every folder. Active when state.folder
  // is null. Clicking it clears the focus on the backend so the worker
  // round-robins across unpaused folders.
  const allBtn = document.createElement("button");
  allBtn.className = "folder-chip" + (state.folder === null ? " active" : "");
  const totalCount = folders.reduce((s, f) => s + (f.n || 0), 0);
  allBtn.textContent = `All (${totalCount})`;
  allBtn.title = "All folders combined";
  allBtn.onclick = () => selectFolder(null);
  bar.appendChild(allBtn);

  for (const f of folders) {
    const wrap = document.createElement("span");
    wrap.className = "folder-chip-wrap";
    const b = document.createElement("button");
    const classes = ["folder-chip"];
    if (state.folder === f.folder) classes.push("active");
    if (f.paused) classes.push("paused");
    if (!f.last_online) classes.push("offline");
    b.className = classes.join(" ");
    const short = f.folder.split("/").slice(-2).join("/");
    const suffix = (!f.last_online ? " (offline)" : "") + (f.paused ? " ⏸" : "");
    b.textContent = `${short} (${f.n})${suffix}`;
    b.title = f.folder + (!f.last_online ? " — drive not currently reachable" : "");
    b.onclick = () => selectFolder(f.folder);
    wrap.appendChild(b);

    // Per-folder pause toggle. Clicking flips the folder's state on
    // the backend; the worker picks it up on its next tick.
    const pauseBtn = document.createElement("button");
    pauseBtn.className = "folder-pause";
    pauseBtn.textContent = f.paused ? "▶" : "⏸";
    pauseBtn.title = f.paused ? "Resume analysis on this folder" : "Pause analysis on this folder";
    pauseBtn.onclick = async (e) => {
      e.stopPropagation();
      try {
        await jpost("/api/folders/pause", { folder: f.folder, paused: !f.paused });
      } catch (err) {
        alert("Could not toggle: " + err.message);
        return;
      }
      refreshFolders();
      refreshHealth();
    };
    wrap.appendChild(pauseBtn);

    bar.appendChild(wrap);
  }

  // Tell the backend which folder the user is looking at so the worker
  // prioritizes it. Null = 'All' tab → no preference.
  try {
    await jpost("/api/folders/focus", { folder: state.folder });
  } catch (e) { /* non-fatal */ }
}

async function refreshStats() {
  const url = state.folder
    ? `/api/stats?folder=${encodeURIComponent(state.folder)}`
    : "/api/stats";
  const s = await jget(url);
  const parts = [
    `${s.total || 0} total`,
    `${s.analyzed || 0} fully analyzed`,
  ];
  if (s.partial) parts.push(`${s.partial} partial`);
  parts.push(`${(s.pending || 0) - (s.partial || 0)} pending`);
  if (s.errored) parts.push(`${s.errored} errored`);
  if (s.unreachable) parts.push(`${s.unreachable} unreachable`);
  parts.push(`${s.rated || 0} rated`);
  const prefix = state.folder ? "" : "All folders · ";
  $("#stats").textContent = prefix + parts.join(" · ");
}

async function selectFolder(folder) {
  state.folder = folder;
  await refreshFolders();
  await refreshGrid();
  await refreshStats();
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
    "filter-keep": "keep",
    "filter-in-focus": "in_focus",
    "filter-scored": "scored",
    "filter-burst": "burst_only",
    "filter-subject": "subject_contains",
    "filter-min-technical": "min_technical",
    "filter-min-aesthetic": "min_aesthetic",
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
  if (img.ai_keep === "yes") badges.appendChild(badge("KEEP", "good"));
  else if (img.ai_keep === "no") badges.appendChild(badge("cull", "bad"));
  if (img.ai_status === "done") {
    if (img.ai_in_focus === "no") badges.appendChild(badge("OOF", "bad"));
    if (img.ai_motion === "in_motion") badges.appendChild(badge("motion", "good"));
    else if (img.ai_motion === "blurred") badges.appendChild(badge("blur", "warn"));
    if (img.ai_is_silhouette) badges.appendChild(badge("silh", "silh"));
  }
  if (img.burst_role === "best") badges.appendChild(badge(`★ best of #${img.burst_id}`, "burst-best"));
  else if (img.burst_role === "alt") badges.appendChild(badge(`burst #${img.burst_id}`, "burst-alt"));
  card.appendChild(badges);

  const overlay = document.createElement("div");
  overlay.className = "rating-overlay";
  const stars = document.createElement("span");
  stars.className = "stars-mini";
  stars.textContent = img.user_rating ? "★".repeat(img.user_rating) : "";
  overlay.appendChild(stars);
  const scores = document.createElement("span");
  scores.className = "scores";
  if (img.claude_technical_score != null || img.claude_aesthetic_score != null) {
    scores.textContent = `Technical ${formatScore(img.claude_technical_score)} · Aesthetic ${formatScore(img.claude_aesthetic_score)}`;
    scores.classList.add("claude-scores");
  } else if (img.ai_status === "done") {
    scores.textContent = img.ai_keep === "yes" ? "ready to score" : "";
  } else if (img.ai_status === "pending") {
    // Honest status — pending doesn't mean analyzing if the worker is
    // paused (globally or for this image's folder).
    const folderPaused = (state.folders || []).some(
      (f) => f.folder === img.folder && f.paused
    );
    if (state.workerPaused || folderPaused) {
      scores.textContent = "paused";
      scores.classList.add("paused");
    } else {
      scores.textContent = "analyzing…";
    }
  } else if (img.ai_status === "error") {
    scores.textContent = "ai error";
  } else if (img.ai_status === "cancelled") {
    scores.textContent = "cancelled";
  } else if (img.ai_status === "unreachable") {
    scores.textContent = "offline";
    scores.classList.add("offline");
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

async function refreshPreviewStats() {
  const el = $("#preview-stats");
  if (!el) return;
  try {
    const s = await jget("/api/admin/preview-stats");
    el.textContent = s.file_count > 0 ? `${s.gb} GB · ${s.file_count} files` : "";
  } catch (e) { /* ignore */ }
}

function openBulkTagModal() {
  const ids = [...state.selected];
  if (!ids.length) return;
  $("#bulk-tag-summary").textContent = `Applying tags to ${ids.length} selected image(s).`;
  $("#bulk-tag-status").textContent = "";
  $("#bulk-tag-status").className = "fb-status";
  $("#bulk-tag-custom").value = "";
  document.querySelectorAll("#bulk-tag-chips button.active").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll('input[name="bulk-tag-mode"]').forEach((r) => { r.checked = r.value === "add"; });
  $("#bulk-tag-modal").classList.remove("hidden");
}

async function applyBulkTags() {
  const ids = [...state.selected];
  if (!ids.length) return;
  const chipTags = [...document.querySelectorAll("#bulk-tag-chips button.active")].map((b) => b.dataset.tag);
  const customRaw = $("#bulk-tag-custom").value.trim();
  const customTags = customRaw ? customRaw.split(",").map((t) => t.trim()).filter(Boolean) : [];
  const tags = [...new Set([...chipTags, ...customTags])];
  const mode = document.querySelector('input[name="bulk-tag-mode"]:checked').value;
  if (!tags.length && mode === "add") {
    $("#bulk-tag-status").textContent = "Pick at least one tag or type a custom one.";
    $("#bulk-tag-status").className = "fb-status bad";
    return;
  }
  const status = $("#bulk-tag-status");
  status.textContent = `Applying to ${ids.length}…`;
  status.className = "fb-status";
  try {
    const r = await jpost("/api/bulk/tags", { image_ids: ids, tags, mode });
    status.textContent = `Tagged ${r.updated} image(s).${r.sidecar_errors && r.sidecar_errors.length ? ` ${r.sidecar_errors.length} XMP write error(s).` : ""}`;
    status.className = "fb-status ok";
    setTimeout(() => $("#bulk-tag-modal").classList.add("hidden"), 800);
    refreshGrid();
  } catch (e) {
    status.textContent = "Failed: " + e.message;
    status.className = "fb-status bad";
  }
}

function getPrimaryAi(img) {
  // Returns the structured AI result for this image — Phase 1 triage's
  // primary judge result, or the legacy ai_json fallback. This is the
  // baseline we prefill the correction form with.
  const judges = (state.judges || []);
  const judgeData = img.ai_judges || {};
  const primary = judges.find((j) => j.primary) || judges[0];
  if (primary && judgeData[primary.name] && judgeData[primary.name].result) {
    return judgeData[primary.name].result;
  }
  return img.ai || {};
}

function renderFeedback(img) {
  const fb = img.ai_feedback || null;
  const ai = getPrimaryAi(img);
  const summary = $("#modal-feedback-summary");
  if (fb) {
    const bits = [];
    if (fb.marked_wrong) bits.push("<b>Marked wrong</b>");
    if (fb.keep) bits.push(`Keep: ${fb.keep === "yes" ? "Yes" : "No"}`);
    if (fb.animal_type) bits.push(`Type: ${esc(fb.animal_type)}`);
    if (fb.species) bits.push(`Species: ${esc(fb.species)}`);
    if (fb.in_focus) bits.push(`In focus: ${fb.in_focus === "yes" ? "Yes" : "No"}`);
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

  // Prefill: user correction wins if present, otherwise show the AI's
  // value. This way the user can see at a glance what the AI said and
  // simply change any field that's wrong.
  const setVal = (id, v) => { const el = $(id); if (el) el.value = (v === undefined || v === null) ? "" : v; };
  const setCheck = (id, v) => { const el = $(id); if (el) el.checked = !!v; };
  const pick = (fbVal, aiVal) => (fbVal !== undefined && fbVal !== null && fbVal !== "") ? fbVal : aiVal;

  setCheck("#fb-marked-wrong", fb && fb.marked_wrong);
  setVal("#fb-keep",         pick(fb && fb.keep,         ai.keep));
  setVal("#fb-animal-type",  pick(fb && fb.animal_type,  ai.animal_type));
  setVal("#fb-species",      pick(fb && fb.species,      ai.species));
  setVal("#fb-subject",      pick(fb && fb.subject,      ai.subject));
  setVal("#fb-in-focus",     pick(fb && fb.in_focus,     ai.in_focus));
  setVal("#fb-eye-focus",    pick(fb && fb.eye_focus,    ai.eye_focus));
  setVal("#fb-motion",       pick(fb && fb.motion,       ai.motion));
  setVal("#fb-composition",  pick(fb && fb.composition,  ai.composition));
  setVal("#fb-lighting",     pick(fb && fb.lighting,     ai.lighting));
  const silFb = (fb && fb.is_silhouette === true) ? "yes" : (fb && fb.is_silhouette === false) ? "no" : null;
  const silAi = ai.is_silhouette ? "yes" : "no";
  setVal("#fb-silhouette",   silFb || silAi);
  setVal("#fb-note",         fb && fb.note);
  populateIssueChips(fb, ai);
  $("#fb-status").textContent = "";
}

function populateIssueChips(fb, ai) {
  const wrap = $("#fb-issue-chips");
  if (!wrap) return;
  wrap.innerHTML = "";
  const vocab = (state.aiVocab && state.aiVocab.technical_issues) || [];
  // Prefill: user correction's issue list if present, otherwise the AI's.
  const baseline = (fb && Array.isArray(fb.technical_issues))
    ? fb.technical_issues
    : ((ai && Array.isArray(ai.technical_issues)) ? ai.technical_issues : []);
  const selected = new Set(baseline);
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
    el.innerHTML = `<div class="pending">Triage in progress…</div>`;
    return;
  }
  if (img.ai_status === "error" && Object.keys(judgeData).length === 0) {
    const errMsg = (img.ai && img.ai.error) ? img.ai.error : "(no detail)";
    el.innerHTML = `<div class="ai-error"><div class="ai-error-title">Triage error</div><div class="ai-error-detail"></div></div>`;
    el.querySelector(".ai-error-detail").textContent = errMsg;
    return;
  }

  const a = (() => {
    const primary = judges.find((j) => j.primary) || judges[0];
    if (primary && judgeData[primary.name] && judgeData[primary.name].result) {
      return judgeData[primary.name].result;
    }
    return img.ai || {};
  })();
  const issues = (a.technical_issues || []).map(pretty).join(", ") || "—";
  const triageModel = (judges[0] && judges[0].model) || "qwen2.5vl:7b";

  const keepBadge = a.keep === "yes"
    ? `<span class="keep-badge keep-yes">KEEP</span>`
    : a.keep === "no"
    ? `<span class="keep-badge keep-no">DON'T KEEP</span>`
    : `<span class="keep-badge keep-unknown">—</span>`;

  const claudeBlock = (img.claude_technical_score != null || img.claude_aesthetic_score != null)
    ? `<div class="claude-scores-block">
        <div class="claude-scores-title">CLAUDE SCORES</div>
        <div class="claude-scores-row">
          <div class="claude-score"><span class="claude-score-label">Technical</span><strong>${formatScore(img.claude_technical_score)}</strong><span class="denom">/10</span></div>
          <div class="claude-score"><span class="claude-score-label">Aesthetic</span><strong>${formatScore(img.claude_aesthetic_score)}</strong><span class="denom">/10</span></div>
        </div>
        ${img.claude_reasoning ? `<div class="claude-reasoning">${esc(img.claude_reasoning)}</div>` : ""}
      </div>`
    : `<div class="claude-scores-block claude-not-scored">
        <button id="modal-score-claude" class="primary">Score with Claude</button>
        <span class="claude-cost-hint">~1¢ per image</span>
      </div>`;

  el.innerHTML = `
    <div class="triage-header">
      <div class="triage-keep">${keepBadge}</div>
      <div class="triage-model muted">${esc(triageModel)}</div>
    </div>
    ${claudeBlock}
    ${img.burst_id ? `<div class="row"><span>Burst</span><strong>#${img.burst_id} · ${esc(pretty(img.burst_role || ""))}</strong></div>` : ""}
    <div class="row"><span>Type</span><strong>${esc(a.animal_type) || "—"}</strong></div>
    <div class="row"><span>Species</span><strong>${esc(a.species) || "—"}</strong></div>
    <div class="row"><span>Subject</span><strong>${esc(a.subject) || "—"}</strong></div>
    <div class="row"><span>In focus</span><strong>${a.in_focus === "yes" ? "Yes" : a.in_focus === "no" ? "No" : "—"}</strong></div>
    <div class="row"><span>Eye focus</span><strong>${esc(pretty(a.eye_focus)) || "—"}</strong></div>
    <div class="row"><span>Motion</span><strong>${esc(pretty(a.motion)) || "—"}</strong></div>
    <div class="row"><span>Composition</span><strong>${esc(pretty(a.composition)) || "—"}</strong></div>
    <div class="row"><span>Lighting</span><strong>${esc(pretty(a.lighting)) || "—"}</strong></div>
    <div class="row"><span>Silhouette</span><strong>${a.is_silhouette ? "Yes" : "No"}</strong></div>
    <div class="row"><span>Issues</span><strong>${esc(issues)}</strong></div>
    ${a.notes ? `<div class="notes-line">${esc(a.notes)}</div>` : ""}
  `;
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
  for (const cls of ["select-info", "select-all-visible", "select-clear", "reanalyze-selected", "bulk-tag-selected"]) {
    const el = $("#" + cls);
    if (el) el.classList.toggle("hidden", !on);
  }
  renderGrid();
  updateSelectInfo();
}

function updateSelectInfo() {
  const n = state.selected.size;
  $("#select-info").textContent = `${n} selected`;
  $("#reanalyze-selected").disabled = n === 0;
  const bt = $("#bulk-tag-selected");
  if (bt) bt.disabled = n === 0;
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
    "filter-feedback", "filter-keep", "filter-in-focus", "filter-scored", "filter-burst",
  ]) {
    const el = $("#" + id);
    if (el) el.onchange = refreshGrid;
  }
  for (const id of ["filter-species", "filter-subject", "filter-min-technical", "filter-min-aesthetic"]) {
    let t;
    $("#" + id).oninput = () => {
      clearTimeout(t);
      t = setTimeout(refreshGrid, 250);
    };
  }
  $("#recompute-bursts").onclick = async () => {
    if (!state.folder) {
      alert("Pick a folder first (top of the page) — burst grouping operates on one folder at a time.");
      return;
    }
    const gap = parseFloat($("#burst-gap").value) || 120;
    const sim = parseFloat($("#burst-sim").value) || 0.85;
    $("#ingest-status").textContent = `Finding bursts (gap ${gap}s, sim ${sim})…`;
    try {
      const r = await jpost("/api/bursts/recompute", {
        folder: state.folder,
        time_gap_sec: gap,
        sim_threshold: sim,
      });
      let msg = `${r.bursts} burst(s) · ${r.images_in_bursts} frames`;
      if (r.bursts === 0 && r.reason) {
        msg = `0 bursts. ${r.reason}`;
        if (r.with_capture_time === 0 && r.total_in_folder > 0) {
          msg += " Or run 'Backfill capture times'.";
        }
      } else {
        msg += ` (gap ${r.time_gap_sec}s, sim ${r.sim_threshold}, ${r.with_embedding}/${r.total_in_folder} embedded, ${r.elapsed_sec}s)`;
      }
      $("#ingest-status").textContent = msg;
      refreshGrid();
    } catch (e) {
      $("#ingest-status").textContent = "Burst find failed: " + e.message;
    }
  };

  $("#filter-clear").onclick = () => {
    state.filterRating = "";
    $("#filter-rating").value = "";
    for (const id of [
      "filter-ai-status", "filter-animal-type", "filter-species",
      "filter-eye-focus", "filter-motion", "filter-composition",
      "filter-lighting", "filter-silhouette", "filter-issue", "filter-feedback",
      "filter-keep", "filter-in-focus", "filter-scored", "filter-burst",
      "filter-subject", "filter-min-technical", "filter-min-aesthetic",
    ]) {
      const el = $("#" + id);
      if (el) el.value = "";
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
  $("#bulk-tag-selected").onclick = () => openBulkTagModal();
  $("#bulk-tag-close").onclick = () => $("#bulk-tag-modal").classList.add("hidden");
  $("#bulk-tag-chips").onclick = (e) => {
    if (e.target.tagName !== "BUTTON") return;
    e.target.classList.toggle("active");
  };
  $("#bulk-tag-apply").onclick = applyBulkTags;
  $("#recheck-folders").onclick = async () => {
    try {
      const r = await jpost("/api/folders/recheck");
      const back = (r.online_again || []).length;
      const gone = (r.now_offline || []).length;
      let msg = "Folder probe complete.";
      if (back) msg += ` ${back} now online (unreachable images requeued).`;
      if (gone) msg += ` ${gone} just went offline.`;
      if (!back && !gone) msg += " No state changes.";
      alert(msg);
      refreshFolders();
      refreshStats();
      refreshGrid();
    } catch (e) {
      alert("Recheck failed: " + e.message);
    }
  };
  $("#backfill-capture-times").onclick = async () => {
    if (!confirm(
      "Read EXIF DateTimeOriginal for every image that doesn't have a "
      + "capture time yet?\n\n"
      + "Burst detection uses real capture times when available — file "
      + "modification times are often wrong on copied archives. Slow on "
      + "first run (one exiftool call per image) but only has to happen once."
    )) return;
    const btn = $("#backfill-capture-times");
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = "Reading EXIF…";
    try {
      const r = await jpost("/api/admin/backfill-capture-times");
      alert(`Scanned ${r.scanned} images, updated ${r.updated}, ${r.skipped} had no EXIF (${r.elapsed_sec}s).`);
    } catch (e) {
      alert("Backfill failed: " + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  };
  $("#refresh-sidecars").onclick = async () => {
    if (!confirm(
      "Rewrite every image's XMP sidecar from current AI / Claude state?\n\n"
      + "Fast — no re-analysis. Use this once to backfill the new "
      + "WC:* AI keywords into existing sidecars so Lightroom can see them."
    )) return;
    const btn = $("#refresh-sidecars");
    btn.disabled = true;
    const orig = btn.textContent;
    btn.textContent = "Refreshing…";
    try {
      const r = await jpost("/api/admin/refresh-sidecars");
      let msg = `Wrote ${r.written} of ${r.scanned} sidecars.`;
      if (r.errors && r.errors.length) {
        msg += `\n\n${r.errors.length} error(s):\n${r.errors.slice(0, 5).join("\n")}`;
        if (r.errors.length > 5) msg += `\n…and ${r.errors.length - 5} more.`;
      }
      alert(msg);
    } catch (e) {
      alert("Refresh failed: " + e.message);
    } finally {
      btn.disabled = false;
      btn.textContent = orig;
    }
  };
  $("#drop-folder-previews").onclick = async () => {
    if (!state.folder) {
      alert("Select a specific folder tab first — this drops previews for one folder at a time.");
      return;
    }
    const short = state.folder.split("/").slice(-2).join("/");
    if (!confirm(
      `Delete cached previews for "${short}"?\n\n`
      + "AI analysis, Claude scores, ratings, tags, and XMP sidecars "
      + "all stay intact — only the local JPEG cache for this folder "
      + "goes. Previews re-extract on demand when you next open one."
    )) return;
    try {
      const r = await jpost("/api/folders/drop-previews", { folder: state.folder });
      alert(`Removed ${r.files_removed} files, freed ${r.mb_freed} MB from ${short}.`);
      refreshPreviewStats();
      refreshGrid();
    } catch (e) {
      alert("Drop failed: " + e.message);
    }
  };
  $("#clear-previews").onclick = async () => {
    const stats = await jget("/api/admin/preview-stats").catch(() => null);
    const sizeNote = stats ? ` (${stats.file_count} files, ${stats.gb} GB)` : "";
    if (!confirm(
      `Delete all cached preview JPEGs${sizeNote}?\n\n`
      + "Previews will re-extract on demand the next time you reopen "
      + "a folder. RAW re-extraction is slow (minutes per hundred images)."
    )) return;
    try {
      const r = await jpost("/api/admin/clear-previews");
      alert(`Cleared ${r.files_removed} files, freed ${r.mb_freed} MB.`);
      refreshPreviewStats();
    } catch (e) {
      alert("Clear failed: " + e.message);
    }
  };
  $("#score-keepers").onclick = async () => {
    if (!confirm(
      "Send every keeper not yet scored to Claude. This costs real money "
      + "(roughly 1¢ per image) and stops automatically at the budget cap.\n\n"
      + "Continue?"
    )) return;
    try {
      await jpost("/api/claude/score-keepers");
    } catch (e) {
      alert("Could not start: " + e.message);
      return;
    }
    refreshHealth();
    startBulkPoll();
  };
  $("#score-keepers-cancel").onclick = async () => {
    try { await jpost("/api/claude/score-keepers/cancel"); } catch (e) {}
    refreshHealth();
  };

  $("#reanalyze-errored").onclick = () => {
    const scope = state.folder ? "in this folder" : "across all folders";
    if (!confirm(`Re-analyze every errored image ${scope}?`)) return;
    reanalyzeByStatus("error");
  };

  $("#fb-save").onclick = async () => {
    if (!state.current) return;
    const id = state.current.id;
    const ai = getPrimaryAi(state.current);
    const issues = [...document.querySelectorAll("#fb-issue-chips button.active")].map((b) => b.dataset.value);
    const strOrNull = (sel) => {
      const v = $(sel).value.trim();
      return v === "" ? null : v;
    };
    const silVal = $("#fb-silhouette").value;
    const silAiAsString = ai.is_silhouette ? "yes" : "no";
    const aiIssues = Array.isArray(ai.technical_issues) ? [...ai.technical_issues].sort() : [];
    const userIssues = [...issues].sort();
    const issuesChanged = JSON.stringify(aiIssues) !== JSON.stringify(userIssues);
    const note = strOrNull("#fb-note");

    // Form value vs AI value — only persist what the user actually changed.
    // The user's correction list becomes the training signal; same value
    // as the AI doesn't carry signal so we don't store it.
    const formVals = {
      keep:         strOrNull("#fb-keep"),
      animal_type:  strOrNull("#fb-animal-type"),
      species:      strOrNull("#fb-species"),
      subject:      strOrNull("#fb-subject"),
      in_focus:     strOrNull("#fb-in-focus"),
      eye_focus:    strOrNull("#fb-eye-focus"),
      motion:       strOrNull("#fb-motion"),
      composition:  strOrNull("#fb-composition"),
      lighting:     strOrNull("#fb-lighting"),
    };
    const body = { marked_wrong: $("#fb-marked-wrong").checked };
    for (const [k, v] of Object.entries(formVals)) {
      const aiVal = ai[k] === undefined || ai[k] === null ? null : String(ai[k]);
      const formVal = v === null ? null : String(v);
      if (formVal !== null && formVal !== aiVal) body[k] = v;
    }
    if (silVal && silVal !== silAiAsString) {
      body.is_silhouette = silVal === "yes";
    }
    if (issuesChanged) body.technical_issues = issues;
    if (note) body.note = note;
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

  document.addEventListener("click", async (e) => {
    if (e.target && e.target.id === "modal-score-claude") {
      if (!state.current) return;
      const id = state.current.id;
      const btn = e.target;
      btn.disabled = true;
      btn.textContent = "Scoring…";
      try {
        const r = await jpost(`/api/image/${id}/score-with-claude`);
        const idx = state.images.findIndex((x) => x.id === id);
        if (idx >= 0) {
          state.images[idx].claude_technical_score = r.score.technical_score;
          state.images[idx].claude_aesthetic_score = r.score.aesthetic_score;
          state.images[idx].claude_reasoning = r.score.reasoning;
          state.images[idx].claude_cost_usd = r.score.cost_usd;
          state.images[idx].claude_scored_at = Date.now() / 1000;
          state.current = state.images[idx];
        }
        renderAiBlock(state.current);
        renderGrid();
        refreshHealth();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = "Score with Claude";
        alert("Score failed: " + err.message);
      }
    }
  });

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
  refreshPreviewStats();
  setInterval(() => { refreshStats(); refreshHealth(); }, 5000);
  setInterval(() => { if (state.folder) refreshGrid(); }, 8000);
  setInterval(refreshPreviewStats, 30000);
});
