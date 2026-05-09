# Day-to-day use

This is the flow once setup is done.

## 1. Start the app

Open Terminal, then:

```
cd ~/Documents/wildlife-image-culling
./scripts/run.sh
```

Your browser opens to the app.

## 2. Ingest a folder

Click **Open folder**, paste the full path to the folder you want to cull.
Examples:

- An SD card mount: `/Volumes/EOS_R5/DCIM/100EOSR5`
- A shoot folder you've copied to disk: `/Users/kendra/Photos/2026-05-09 Owls`
- An old archive: `/Users/kendra/Photos/Archive/2019`

Click **Ingest**. The app:

1. Scans the folder for image files (CR3, NEF, ARW, RAF, ORF, DNG, JPG, JPEG,
   HEIC).
2. Extracts a fast preview for every file.
3. Queues each image for AI analysis.

You'll immediately see a grid of every image. AI scores fill in over the next
few minutes as the vision model works through them. **You don't have to wait
— start culling the ones that already have scores.**

> The AI processes about 4–8 images per minute on an M4 Pro. A 500-image shoot
> takes ~75 minutes of background time. You can leave it running and come back.

## 3. Review and rate

Each thumbnail shows:

- The image
- AI's suggested **artistic score** and **portfolio score** (1–10)
- A small badge for eye focus and motion
- Your current rating (empty until you rate it)

Click any thumbnail to open the **detail view**:

- Larger image
- Full AI judgment (subject, composition, lighting, technical issues, notes)
- **Star rating** 1–5 (click the stars, or press 1–5 on your keyboard)
- **Tag chips**: Keep / Portfolio / Reject / Silhouette-Intentional /
  Eyes-Sharp / Soft-Focus
- A free-text **notes** field for your own comments

Use the arrow keys (← →) to move between images without leaving the detail view.

## 4. Save / sidecars

You don't need to click save — every rating change is auto-saved to the local
database **and** to an XMP sidecar file next to the original image.

Example: rating `IMG_1234.CR3` writes `IMG_1234.CR3.xmp` into the same folder.
When you later open the folder in Lightroom or Capture One, your rating and
keywords are already there.

## 5. Use the AI's hint, but trust your eye

The Phase 1 AI is generic — it knows what "eye sharpness" and "good
composition" mean, but it doesn't know your taste yet. **The AI's score is a
hint, not a verdict.** Use it to spot things to look at carefully (like
"high portfolio score on a frame I would have skipped"), but you always make
the final call.

In Phase 3 we'll start training a personal model on your ratings, and the
AI's scores will start matching your taste.

## 6. Stop the app

Switch to Terminal, press **Control + C**. Done.

---

## Tips

- **Big imports first:** the first time you run this on an old archive of
  thousands of photos, ingest one folder at a time so the queue stays
  manageable. Once a folder is ingested, the work is saved — closing the app
  doesn't lose progress.
- **Already culled in Lightroom?** Existing XMP sidecar ratings are read on
  ingest and shown as your starting rating. The AI fills in tags and notes
  around them.
- **Where's the database?** A single file at
  `~/Documents/wildlife-image-culling/data/cull.db`. Back it up the same way
  you back up Lightroom catalogs.
