# Day-to-day use

This is the flow once setup is done. The tool runs in two phases — Phase 1
triages every image locally (free, automatic); Phase 2 scores keepers with
Claude when you tell it to (real money, you control the cap).

## 1. Start the app

Open Terminal, then:

```
cd ~/Documents/wildlife-image-culling
caffeinate -dimsu ./scripts/run.sh
```

`caffeinate` keeps macOS from throttling background work — without it the
analysis pauses every time you switch away from the terminal.

Your browser opens to the app.

## 2. Ingest a folder

Top of the page: a text box, a **Browse…** button, a **recursive**
checkbox, and a blue **Ingest** button.

The easiest way: click **Browse…**. A folder picker opens at your home
folder; mounted volumes (SD cards) show up under "Volume:". Click
through, click **Use this folder**, then **Ingest**.

You can also paste a path directly. From Finder, right-click the folder,
hold **Option (⌥)**, choose **"Copy [folder name] as Pathname"**, paste
into the text box.

Click **Ingest**. The app:

1. Scans for image files (CR3, NEF, ARW, RAF, ORF, DNG, JPG, JPEG, HEIC).
2. Extracts a preview from each one (fast — embedded JPEG from RAW).
3. Queues each image for Phase 1 triage.

You'll immediately see a grid of every image. The triage results fill in
over the next ~30–60 seconds per image as the local model works through
them. **You don't have to wait** — start culling the ones already done.

## 3. Phase 1 — read the triage verdicts

Each thumbnail shows:

- The image
- A **KEEP** badge (green) or **cull** badge (red), once triaged
- Smaller badges for in-focus / motion / silhouette state
- Your star rating (empty until you rate)

Click any thumbnail to open the **detail view**. You'll see:

- Larger image
- The **KEEP / DON'T KEEP** verdict at the top, with the triage model name
- A **"Score with Claude"** button (more on this in step 4)
- Structured fields: type, species, subject, in-focus, eye focus, motion,
  composition, lighting, silhouette, issues
- The model's notes — what it sees and why it called keep or cull
- Your 1–5 stars, tag chips, free-text notes
- A **"Train your eye"** dropdown where you can correct the AI

Use **← →** to move between images without leaving detail view.

## 4. Phase 2 — score keepers with Claude (optional, costs money)

Once Phase 1 has triaged the shoot, you have a stack of keepers worth a
closer look. Phase 2 is where Claude Sonnet 4.6 ranks them.

There are two ways to trigger it:

**Per-image:** open a keeper in detail view, click the **"Score with
Claude"** button. About 3–6 seconds later you'll see two scores:

- **Technical** (1–10) — how well-executed: eye sharpness, exposure,
  noise, framing, motion handling.
- **Aesthetic** (1–10) — how distinctive: light, behavior, moment,
  composition, story. These two scores measure different things and
  usually differ.
- A one-sentence rationale.

Cost is ~1¢ per image. Use this when you want to spot-check a few before
committing to a bulk run.

**Bulk:** in the bulk bar above the grid, click **"Score all keepers with
Claude"**. This walks every image marked `keep=yes` that isn't scored yet,
calling Claude on each one. Live progress shows up next to the button:

```
Scoring 23/87 · $0.18 this run · IMG_1234.JPG
```

A **Stop scoring** button appears while it's running. You can leave the
page running; refreshing or closing the browser tab doesn't stop the
backend job.

## 5. The budget pill — your spending guard

In the header you'll see a budget pill:

```
Claude: $0.0247 of $5.00 · 7 scored · 0.5%
```

It's **green** under 80%, **yellow** at 80–99%, **red** at 100%.

- **At any point under 100%:** scoring works normally.
- **At 100%:** the "Score with Claude" buttons stop firing. Per-image
  attempts get an error; bulk runs stop gracefully with a "Budget cap
  hit" message.

To lift the cap: open `scripts/run.sh`, raise `CLAUDE_BUDGET_USD`, restart.

If you see **"Claude: API key not set"** in gray instead, edit
`scripts/run.sh` and uncomment / set your `ANTHROPIC_API_KEY` line.

## 6. Filter to find your portfolio picks

The filter bar above the grid has all the usual fields plus:

- **Keep** — show only keepers (or only culls).
- **Claude scored** — show only what's been through Phase 2.
- **Technical ≥** / **Aesthetic ≥** — numeric cutoffs on the Claude
  scores.

A typical workflow: filter `Keep = yes` and `Aesthetic ≥ 7` to surface
the strongest frames in the shoot.

## 7. Star ratings and tags

You don't need to click save — every rating, tag, and note change is
auto-saved to the local database **and** to an XMP sidecar file next to
the original.

Example: rating `IMG_1234.CR3` writes `IMG_1234.CR3.xmp` into the same
folder. When you later open that folder in Lightroom or Capture One, the
ratings and keywords are already there.

## 8. Train the triage model on your taste

The local triage model is generic — it knows what "in focus" and "good
composition" mean, but it doesn't know you. When it gets a call wrong,
correct it: open the image, expand **"Train your eye on this image —
correct the AI"**, set the right values, write one sentence about what
it got wrong, hit **Save corrections**.

After ~5 corrections, the triage model starts seeing your most recent
corrections as examples on every future analysis. The header pill shows
`trained on N of your corrections` when this is live.

This loop is for Phase 1 only — Claude (Phase 2) uses a fixed scoring
rubric. The way to influence Claude scores is to be selective about which
images you mark keep before kicking off a bulk run.

## 9. Stop the app

Switch to the Terminal window where `run.sh` is running, press
**Control + C**. Done.

---

## Tips

- **Big imports first:** the first time you run this on an old archive of
  thousands of photos, ingest one folder at a time so the queue stays
  manageable. Progress is saved between sessions — closing doesn't lose
  work.
- **Spot-check before bulk:** before clicking "Score all keepers," try
  per-image Claude scoring on 3–5 representative images. If the scores
  feel right, run the bulk; if they don't, that's free signal before
  you've spent any real money.
- **The cull is the cost saver:** the more selective Phase 1 is about
  what counts as a keeper, the cheaper Phase 2 runs are. If you're
  finding too many marginal frames going to Claude, raise the bar in
  the triage corrections.
- **Already culled in Lightroom?** Existing XMP sidecar ratings are read
  on ingest and shown as your starting rating.
- **Where's the data?** A single file at
  `~/Documents/wildlife-image-culling/data/cull.db`. Back it up like a
  Lightroom catalog.
