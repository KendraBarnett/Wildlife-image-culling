# Setup — first time on your Mac mini

This guide assumes you have **never used the Terminal before**. We're going to
set things up once, and then day-to-day you'll just double-click an icon.

Everything runs locally. Nothing gets uploaded.

**Estimated time:** 30–60 minutes, mostly waiting on downloads.

---

## Before you start

You need:

- A Mac mini with Apple Silicon (M1, M2, M3, or M4). Intel Macs won't work well.
- At least 16 GB of RAM (you confirmed M4 / M4 Pro 16 GB+ — you're set).
- About 20 GB of free disk space (the AI models are ~5 GB each).
- A reasonably fast internet connection for the one-time downloads.

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

## Step 4 — Start Ollama and download the vision model

Start the Ollama background service:

```
brew services start ollama
```

Now download the vision model we'll use. This is a ~5 GB download — go grab a
coffee:

```
ollama pull qwen2.5vl:7b
```

When that finishes, test that it works:

```
ollama run qwen2.5vl:7b "Say hello in one word."
```

You should see a one-word reply. Press **Control + D** to exit. If you got a
reply, the AI is working.

> **If `qwen2.5vl:7b` won't pull**, your Ollama is older. Run
> `brew upgrade ollama` and try again. As a fallback, `llava:7b` works too —
> tell me and I'll switch the default.

## Step 5 — Download the project

Pick where you want the project to live. The Documents folder is a fine
default. In Terminal:

```
cd ~/Documents
git clone https://github.com/KendraBarnett/wildlife-image-culling.git
cd wildlife-image-culling
git checkout claude/wildlife-photo-metadata-tool-F1apY
```

> If `git clone` asks for a username/password, that means the repo is private.
> Tell me and I'll guide you through making a personal access token. If it's
> already public, this Just Works.

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
