# Setup — first time on your Mac mini

This guide assumes you have **never used the Terminal before**. We're going to
set things up once, and then day-to-day you'll just double-click an icon.

Everything runs locally. Nothing gets uploaded.

**Estimated time:** 30–60 minutes, mostly waiting on downloads.

> **Important — where to do this:** every step in this guide is run on the
> **Mac mini**, in the Mac mini's Terminal. Not your MacBook Pro. Right now
> nothing is installed on the Mac mini and the code only lives on GitHub —
> that's expected. This guide downloads the code, the AI models, and all the
> tools to the Mac mini for the first time. By the end, the Mac mini will be
> fully self-sufficient and you can put the MacBook away.

---

## Before you start

You need:

- A Mac mini with Apple Silicon (M1, M2, M3, or M4). Intel Macs won't work well.
- At least 16 GB of RAM (you confirmed M4 / M4 Pro 16 GB+ — you're set).
- About **40 GB of free disk space** (three vision models at 5–8 GB each, plus a CLIP model, plus your photo previews).
- A reasonably fast internet connection for the one-time downloads.
- The Mac mini powered on, signed in, and connected to the same Wi-Fi /
  network as your MacBook (so you can copy-paste commands between them if
  you like — but you can also just type them on the Mac mini directly).

## Step 1 — Open Terminal

1. Press **⌘ + Space** to open Spotlight.
2. Type **Terminal** and press Return.
3. A window with a blinking cursor opens. This is where we'll type commands.

> Whenever this guide shows a line that looks like `this`, you copy that line,
> paste it into Terminal, and press Return. To paste, use **⌘ + V**.

## Step 2 — Install Homebrew

Homebrew is a "package manager" — a way to install other tools cleanly. Paste
this single line into Terminal and press Return:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your Mac password. Type it (you won't see any dots — that's
normal) and press Return. Wait until you see "Installation successful!"

When it finishes, it will print **two lines** starting with `echo` and `eval`
under a "Next steps" heading. Copy and paste those two lines exactly as shown,
one at a time, pressing Return after each. (This is Homebrew's way of telling
your terminal where to find it.)

Verify it worked:

```
brew --version
```

You should see something like `Homebrew 4.x.x`.

## Step 3 — Install the tools we need

Paste this whole block into Terminal at once (it's one command split across
lines for readability):

```
brew install python@3.12 git exiftool ollama
```

This installs:
- **Python 3.12** — the language the app is written in.
- **git** — to download the project.
- **exiftool** — reads RAW files and extracts the embedded preview JPEG very
  fast. (This is the same tool Photo Mechanic uses internally.)
- **ollama** — runs the local AI vision model.

Wait until it finishes (a few minutes).

## Step 4 — Start Ollama and download the three vision models

Start the Ollama background service:

```
brew services start ollama
```

The app uses **three different vision models** so it can score each photo from
three different perspectives (a "judges" panel — Editor / NatGeo / Stock).
Each judge has its own model and persona, and the app combines their scores.
You only download these once.

Total download: about **18 GB**. Plan on 30–60 minutes depending on your
internet speed. Run all three in the same Terminal window — each one will
print a progress bar:

```
ollama pull qwen2.5vl:7b
ollama pull llama3.2-vision:11b
ollama pull minicpm-v
```

What each one is for:

- **`qwen2.5vl:7b`** (~5 GB) — the **Editor** judge. Harsh technical critic:
  focus, light, composition, exposure.
- **`llama3.2-vision:11b`** (~8 GB) — the **NatGeo** judge. Story-driven:
  behavior, conservation context, sense of place, rarity.
- **`minicpm-v`** (~5 GB) — the **Stock** judge. Commercial appeal: clean
  isolation, broad marketability, room for layout.

When all three are done, sanity-check that the first one works:

```
ollama run qwen2.5vl:7b "Say hello in one word."
```

You should see a one-word reply. Press **Control + D** to exit. If you got a
reply, Ollama is healthy and the models are loaded.

> **If any of these won't pull**, your Ollama is probably old. Run
> `brew upgrade ollama` then try again. The app's health indicator (top right
> of the browser UI) will tell you exactly which model is missing if it can't
> find one when you start the server.

> **About RAM:** your Mac mini has 16 GB and each of these models is 5–8 GB.
> Only **one** model is loaded into RAM at a time; the app processes all your
> photos with the Editor first, then swaps to NatGeo and processes them all
> again, then Stock. This is why analysis takes a while — it's three passes,
> not three parallel runs. You leave it running.

## Step 5 — Download the project from GitHub onto the Mac mini

Right now, the code only lives on GitHub. This step pulls it down to the Mac
mini for the first time. (The actual app, your photos, your database — none
of that exists on the Mac mini yet. After this step, the code will. After
Step 6 the Python libraries will. After Step 7 you'll have a running app.)

Pick where you want the project to live. The Documents folder is a fine
default. In the Mac mini's Terminal:

```
cd ~/Documents
git clone https://github.com/KendraBarnett/wildlife-image-culling.git
cd wildlife-image-culling
git checkout claude/wildlife-photo-metadata-tool-F1apY
```

What each line does:
- `cd ~/Documents` — moves into your Documents folder.
- `git clone …` — copies the GitHub repo into a new `wildlife-image-culling`
  folder inside Documents. This is the actual download from GitHub.
- `cd wildlife-image-culling` — moves into the folder we just downloaded.
- `git checkout claude/...` — switches to the development branch where the
  current MVP lives.

### What if `git clone` asks for a username and password?

That means the repo is **private** and GitHub wants you to log in. The browser
password no longer works for the command line — you need a **Personal Access
Token (PAT)**. One-time setup, ~3 minutes:

1. On the **MacBook**, open a browser and sign in to github.com as the same
   user that owns the repo (KendraBarnett).
2. Go to **Settings → Developer settings → Personal access tokens →
   Tokens (classic) → Generate new token (classic)**.
   Direct link: <https://github.com/settings/tokens/new>
3. **Note**: name it something like "mac mini wildlife cull".
4. **Expiration**: 90 days is fine.
5. **Scopes**: check the box next to **`repo`** (the whole `repo` group).
6. Click **Generate token** at the bottom. GitHub shows you a long string
   that starts with `ghp_…`. **Copy it now** — once you leave the page you
   can never see it again.
7. Back on the **Mac mini Terminal**, re-run `git clone`. When it asks:
   - Username: your GitHub username (e.g. `KendraBarnett`)
   - Password: paste the `ghp_…` token you just copied (it won't show — that's
     normal). Press Return.

That's it. Your Mac mini will remember the token in its Keychain so you
won't be asked again on future `git pull` or `git push`.

### How do I know if the repo is public or private?

Easy check: on your MacBook, open a private/incognito browser window (so you
aren't logged in) and visit:
<https://github.com/KendraBarnett/wildlife-image-culling>

- If the page loads and shows the files — **public**, no token needed.
- If GitHub shows a 404 or "page not found" — **private**, follow the PAT
  steps above.

## Step 6 — Run the one-time setup script

```
./scripts/setup.sh
```

This creates a private Python environment inside the project folder and
installs the libraries we need (FastAPI for the web server, sentence-transformers
for CLIP embeddings, etc.). First time it'll take 5–10 minutes. The first run
of CLIP will download another ~600 MB model file too.

When it finishes, you'll see `Setup complete.`

## Step 7 — Launch the app for the first time

```
./scripts/run.sh
```

Two things happen:

1. Your default browser opens to `http://localhost:8765`.
2. Terminal shows log lines as the server runs.

You'll see a near-empty page asking you to "Open a folder". Don't click
anything yet — first let me show you what to do day-to-day.

Continue to [USER_GUIDE.md](USER_GUIDE.md).

---

## Stopping the app

In the Terminal window where you ran `run.sh`, press **Control + C**. The
server stops. Closing the Terminal window also stops it.

## Updating the app later

When I push improvements, you'll update like this:

```
cd ~/Documents/wildlife-image-culling
git pull
./scripts/setup.sh
```

## Troubleshooting

**"command not found: brew"** — Step 2 didn't finish. Re-read the "Next steps"
output from the Homebrew installer.

**"command not found: ollama"** — Step 3 didn't finish, or your terminal
session is stale. Close Terminal, reopen it, try again.

**The browser shows "This site can't be reached"** — the server hasn't started
yet. Wait 5 seconds and reload. If that doesn't work, look at the Terminal
window for an error message and send it to me.

**"Ollama: connection refused"** — start Ollama with
`brew services start ollama`.

**Anything else** — copy the error from Terminal and send it to me. I'll fix.
