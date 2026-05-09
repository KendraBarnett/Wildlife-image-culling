# Roadmap

What's built today and what's coming. Each phase is intended to be useful on
its own, so the tool keeps working while we add to it.

## Phase 1 — MVP (in progress)

The minimum needed to be useful as a culling tool today.

- Folder ingest (RAW + JPEG, recursive optional)
- Fast preview extraction via `exiftool` (embedded JPEGs from RAW)
- Per-image AI judgment via local Qwen2.5-VL through Ollama
- CLIP embedding stored for every image (saved now, used in Phase 2+)
- Browser UI: grid + detail view, 1–5 stars, tag chips, free-text notes
- Auto-save to SQLite **and** to XMP sidecar files
- Reads existing XMP ratings on ingest

## Phase 2 — Grouping & sorting

Make burst culling fast.

- Cluster near-duplicate images using the CLIP embeddings + a perceptual hash.
- "Best of group" view: see a burst of 30 frames, with the AI's top picks
  highlighted, pick 1–2 keepers in seconds.
- Per-folder rules: "move 1–2 stars to `Rejects/`, 4+ to `Picks/`,
  Portfolio-tagged to `Portfolio/`". Move vs. copy is your choice.
- Quick filters in the UI: by rating, by tag, by AI score range, by group.

## Phase 3 — Personal classifier

The AI starts learning your taste, not generic taste.

- Once you've rated ~200 images, train a small classifier (a few seconds on
  an M4) on top of the CLIP embeddings, predicting **your** 1–5 rating.
- The grid starts showing two scores: the generic AI's suggestion *and*
  "Kendra-likely-rating".
- Re-train automatically every N new ratings, in the background.
- A "review the disagreements" view: images where the model and you would
  rate very differently — these are the highest-value training examples.

## Phase 4 — Active learning + small fine-tunes

The vision model itself gets better at *your* photography.

- Periodically fine-tune a small adapter (LoRA) on the vision model using
  your highest-confidence Portfolio / Reject examples.
- Shoot-aware judgments: "you tend to keep silhouettes only when the subject
  shape is clean against sky" — start surfacing reasons, not just scores.
- Optional: a second model for **species ID** trained on your library.

## Phase 5 — Quality of life

Once the core is solid.

- Native Mac app wrapper (so it's a real icon in your Dock, not a Terminal
  window). The internals stay the same.
- Watched-folder mode: drop a card on the desk, the app ingests automatically.
- Side-by-side compare for two images.
- Export a contact sheet PDF of your Picks / Portfolio for clients.

---

Anything you want pulled forward, or dropped, tell me — this is your tool.
