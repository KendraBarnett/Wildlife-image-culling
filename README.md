# Wildlife Image Culling

A local AI-assisted culling tool for wildlife photographers. Runs entirely on
your Mac — no cloud, no subscription, your photos never leave your machine.

**Status:** Phase 1 (MVP). See [ROADMAP.md](ROADMAP.md) for what's coming.

## What it does today

- Ingests a folder of photos (RAW or JPEG).
- Runs a local vision model on each image to score:
  subject, eye focus, motion, composition, silhouette intent, artistic score,
  portfolio potential.
- Computes a CLIP "fingerprint" of every image (used in later phases for burst
  grouping and learning your personal taste).
- Gives you a browser-based grid where you rate 1–5 and tag images.
- Writes ratings and keywords as **XMP sidecar files** that Lightroom, Capture
  One, Photo Mechanic, and Bridge all understand.

## Getting started

If you've never written code: read [SETUP.md](SETUP.md). It walks you through
every step on a Mac mini, line by line.

If you have: `./scripts/setup.sh && ./scripts/run.sh`.

## Day-to-day use

See [USER_GUIDE.md](USER_GUIDE.md).
