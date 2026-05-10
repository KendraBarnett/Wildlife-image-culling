# Roadmap

What's built and what's coming. Each phase is intended to be useful on
its own, so the tool keeps working while we add to it.

## ✅ Phase 1 — Local triage (shipped)

The hard cull on every image. Free, local, runs over every ingested photo.

- Folder ingest (RAW + JPEG, recursive optional)
- Fast preview extraction via `exiftool` (embedded JPEGs from RAW)
- Local triage via Qwen2.5-VL through Ollama, producing for each image:
  type, species, subject, in-focus yes/no, eye-focus state, motion,
  composition, lighting, silhouette, technical issues, **keep / don't
  keep**, and a notes paragraph
- CLIP embedding stored for every image (used for burst grouping)
- Browser UI: grid + detail view, 1–5 stars, tags, free-text notes
- Filter bar on every AI field plus keep / Claude-scored / Technical ≥ /
  Aesthetic ≥
- Auto-save to SQLite **and** to XMP sidecar files (reads existing XMP
  ratings on ingest too)
- Correction loop: when you flag the AI as wrong on an image, the model
  sees your most recent corrections as examples on every future call —
  in-context learning, no fine-tune required
- Watchdog: any single image stuck past 90s idle or 7 min total is
  killed and skipped, so one bad file doesn't take down the queue

## ✅ Phase 2 — Claude scoring on demand (shipped)

The keeper rank. Sends *only the images Phase 1 marked keep=yes* to
Claude Sonnet 4.6 for two real scores — Technical (1–10) and Aesthetic
(1–10) — plus a one-sentence rationale.

- Per-image **"Score with Claude"** button in the detail view
- Bulk **"Score all keepers with Claude"** button that walks every
  un-scored keeper, with live progress in the bulk bar
- Prompt caching enabled — the scoring rubric is identical per call, so
  it serves from cache at ~10% of input price after the first call
- Per-call usage written to a `claude_usage` table (every call's tokens
  + cost are auditable)
- Header budget pill shows running spend: green / yellow at ≥80% /
  red at ≥100%
- Hard budget cap via `CLAUDE_BUDGET_USD` env var — at 100% the
  per-image button refuses and the bulk run stops gracefully

## Phase 3 — Grouping & sorting

Make burst culling fast.

- Cluster near-duplicate frames using the CLIP embeddings (already
  computed in Phase 1)
- "Best of group" view: see a burst of 30 frames, with the AI's top
  picks highlighted, pick 1–2 keepers in seconds
- Per-folder rules: "move 1–2 stars to `Rejects/`, 4+ to `Picks/`,
  Portfolio-tagged to `Portfolio/`". Move vs. copy is your choice.
- Smarter burst tiebreakers: combine sharpness + Claude technical score
  when available

## Phase 4 — Personal classifier

The local triage starts learning your taste, not generic taste.

- Once you've corrected ~50 images, train a tiny classifier on top of
  the CLIP embeddings predicting **your keep / don't-keep decision**
- Surface "review the disagreements" — images where the model and you
  would call it differently. These are the highest-value training
  examples.
- Re-train automatically every N new corrections, in the background

## Phase 5 — Downstream artifacts

Once the cull and rank are dialed, do something with the keepers.

- Stock-image metadata writing (titles, keywords, descriptions) for the
  top frames, using Claude
- Portfolio curation: "build me a diverse set of 50 from these 800
  keepers, no near-duplicates, with one-sentence justifications"
- Contact-sheet PDF export of Picks / Portfolio for clients

## Phase 6 — Quality of life

- Native Mac app wrapper (a real icon in your Dock, not a Terminal
  window). Internals stay the same.
- Watched-folder mode: drop a card on the desk, app ingests
  automatically.
- Side-by-side compare for two images.

---

Anything you want pulled forward, dropped, or reshuffled, tell me — this
is your tool.
