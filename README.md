# Wildlife Image Culling

A local AI-assisted culling tool for wildlife photographers. The hard cull
runs entirely on your Mac — no cloud, no subscription, your photos never
leave your machine. An optional second pass calls Claude to score the
keepers, with a hard budget cap you control.

## What it does

**Phase 1 — local triage on every image (free, automatic).**
A small local vision model (Qwen2.5-VL via Ollama) looks at each image and
produces: type, species, subject, in-focus yes/no, eye-focus state, motion,
composition, lighting, silhouette, technical issues, and a **keep / don't
keep** decision. The triage call is the cull — its job is to filter out
the obviously unusable so you don't spend time (or money) on them.

**Phase 2 — Claude scoring of keepers (on demand, costs money).**
When you're ready, click "Score with Claude" on a single image, or "Score
all keepers with Claude" in the bulk bar. The keepers get two real scores
from Claude Sonnet 4.6 — **Technical** (1–10) and **Aesthetic** (1–10) —
plus a one-sentence rationale. Every call is recorded; the header shows
your running spend; a budget cap stops the run before you overshoot.

**Around both phases:**
- CLIP embedding stored for every image (used for burst grouping and
  near-duplicate detection).
- Browser UI: grid + detail view, 1–5 stars, tags, free-text notes,
  filters on every AI field.
- Correction loop: tell the local triage model when it's wrong and your
  corrections flow into its prompt as examples on future runs.
- Auto-save to SQLite **and** to XMP sidecar files Lightroom, Capture One,
  Photo Mechanic, and Bridge all read.

## Cost expectations

| Action | Cost | Time |
|---|---|---|
| Triage on every image (Phase 1) | Free | ~1 min per image on a 16 GB Mac mini |
| Score one keeper with Claude (Phase 2) | ~$0.005–0.011 | ~3–6 sec |
| Score 100 keepers with Claude | ~$0.50–1.10 | ~5–10 min |

A $5 cap buys you roughly 500–1000 Claude scores.

## Getting started

If you've never written code: read [SETUP.md](SETUP.md). It walks you
through every step on a Mac mini.

If you have: `./scripts/setup.sh && ./scripts/run.sh`.

## Day-to-day use

See [USER_GUIDE.md](USER_GUIDE.md).
