# Setup — first time on your Mac mini

This guide assumes you have **never used the Terminal before**. We set
things up once, and then day-to-day you'll just open Terminal and run one
command.

The local triage runs entirely on your Mac — nothing gets uploaded.
The optional Phase 2 (Claude scoring) sends *only the keepers* to
Anthropic, and only when you click the button. You control whether to
turn that on.

**Estimated time:** 30–60 minutes, mostly waiting on downloads.

> **Important — where to do this:** every step is run on the **Mac mini**,
> in the Mac mini's Terminal. Right now nothing is installed there and
> the code only lives on GitHub — that's expected. This guide downloads
> the code, the AI model, and all the tools to the Mac mini for the
> first time. By the end the Mac mini is self-sufficient.

---

## Before you start

You need:

- A Mac mini with Apple Silicon (M1–M4). Intel Macs won't work well.
- At least 16 GB of RAM.
- About **15 GB of free disk space** (one vision model at ~6 GB, plus a
  CLIP embedding model, plus your photo previews).
- A reasonably fast internet connection for the one-time downloads.
- *(Optional for Phase 2)* an **Anthropic API key** if you want Claude
  scoring on keepers. You can skip this and add it later — Phase 1 works
  without it. See Step 8 for how to get one.

## Step 1 — Open Terminal

1. Press **⌘ + Space** to open Spotlight.
2. Type **Terminal** and press Return.
3. A window with a blinking cursor opens. This is where we'll paste
   commands.

> Whenever this guide shows a line that looks like `this`, copy it,
> paste into Terminal with **⌘ + V**, and press Return.

## Step 2 — Install Homebrew

Homebrew is a package manager that lets us install other tools cleanly.
Paste this single line and press Return:

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

It will ask for your Mac password. Type it (you won't see any dots —
that's normal) and press Return. Wait for "Installation successful!"

When it finishes it prints **two lines** under a "Next steps" heading,
starting with `echo` and `eval`. Copy and paste each one into Terminal
(this tells your shell where to find Homebrew).

Verify:

```
brew --version
```

You should see `Homebrew 4.x.x`.

## Step 3 — Install the tools we need

Paste this single command:

```
brew install python@3.12 git exiftool ollama
```

This installs:
- **Python 3.12** — the language the app is written in.
- **git** — to download the project from GitHub.
- **exiftool** — extracts the embedded JPEG preview from RAW files
  quickly (same tool Photo Mechanic uses internally).
- **ollama** — runs the local vision model.

Wait until it finishes (a few minutes).

## Step 4 — Start Ollama and download the vision model

Start the Ollama background service:

```
brew services start ollama
```

The app uses **one local vision model** — `qwen2.5vl:7b`, ~6 GB. It
handles the Phase 1 triage on every image. Download it once:

```
ollama pull qwen2.5vl:7b
```

Plan on 10–20 minutes depending on your internet speed.

When it finishes, sanity-check:

```
ollama run qwen2.5vl:7b "Say hello in one word."
```

You should see a one-word reply. Press **Control + D** to exit. If you
got a reply, Ollama is healthy and the model is loaded.

> **If the pull fails**, your Ollama is probably out of date. Run
> `brew upgrade ollama` and try again. The app's health indicator (top
> right of the browser UI) will tell you exactly which model is missing
> if it can't find one when you start the server.

> **About RAM:** the model uses ~8 GB while loaded. Together with
> macOS and your browser, you'll be using most of your 16 GB while
> analysis is running. Close Chrome tabs you don't need and quit
> Creative Cloud if it's running — both eat memory and there's no
> reason to have them open while culling.

## Step 5 — Download the project from GitHub

The code only lives on GitHub right now. Pull it down once:

```
cd ~/Documents
git clone https://github.com/KendraBarnett/wildlife-image-culling.git
cd wildlife-image-culling
git checkout claude/wildlife-photo-metadata-tool-F1apY
```

What each line does:
- `cd ~/Documents` — moves into your Documents folder.
- `git clone …` — copies the repo into a new `wildlife-image-culling`
  folder.
- `cd wildlife-image-culling` — moves into that folder.
- `git checkout claude/...` — switches to the development branch.

### If `git clone` asks for a username and password

The repo is **private** and GitHub wants you to log in. Browser passwords
no longer work for the command line — you need a **Personal Access
Token (PAT)**. One-time setup, ~3 minutes:

1. On the **MacBook**, sign in to github.com as the same user that owns
   the repo (KendraBarnett).
2. Go to **Settings → Developer settings → Personal access tokens →
   Tokens (classic) → Generate new token (classic)**.
   Direct link: <https://github.com/settings/tokens/new>
3. **Note**: name it `mac mini wildlife cull`.
4. **Expiration**: 90 days is fine.
5. **Scopes**: check the box next to **`repo`**.
6. Click **Generate token**. GitHub shows you a `ghp_…` string.
   **Copy it now** — once you leave the page you can never see it again.
7. Back on the **Mac mini Terminal**, re-run `git clone`. When prompted:
   - Username: `KendraBarnett`
   - Password: paste the `ghp_…` token (it won't show — that's normal).

The Mac mini will remember the token in Keychain.

## Step 6 — Run the one-time setup script

```
./scripts/setup.sh
```

This creates a private Python environment inside the project folder and
installs the libraries (FastAPI, the Anthropic SDK, sentence-transformers
for CLIP, etc.). Takes 5–10 minutes. The first run also downloads a
~600 MB CLIP model file.

When it finishes, you'll see `Setup complete.`

## Step 7 — *(Optional)* Enable Phase 2 — Claude scoring

If you want to use the "Score with Claude" buttons on keepers, you need
an Anthropic API key. **Skip this step if you only want the free local
triage** — you can come back later.

### Get an API key

1. Go to <https://console.anthropic.com/>.
2. Sign in (or sign up) with the account you want to bill against.
3. Add a payment method under **Settings → Billing**.
4. Go to **API Keys**, click **Create Key**, name it `mac mini culling`,
   and copy the `sk-ant-…` string. You can never see it again after
   leaving the page.

### Add the key to your run script

Open `scripts/run.sh` in TextEdit:

```
open -e scripts/run.sh
```

Near the top you'll see this block:

```sh
# export ANTHROPIC_API_KEY="sk-ant-..."
export CLAUDE_BUDGET_USD="${CLAUDE_BUDGET_USD:-5.00}"
```

Uncomment the first line (remove the `#`) and paste in your key:

```sh
export ANTHROPIC_API_KEY="sk-ant-paste-your-key-here"
export CLAUDE_BUDGET_USD="5.00"
```

Save and close TextEdit.

### About the budget cap

`CLAUDE_BUDGET_USD` is a **hard ceiling** on what Claude scoring will
spend, in dollars. The UI shows your running total in the header. At
80% it warns; at 100% it stops accepting new scoring requests.

`$5.00` is a reasonable starting point — it buys you roughly 500–1000
image scores. Raise it any time by editing this file and restarting the
app. Lower it if you want a tighter leash.

## Step 8 — Launch the app for the first time

```
caffeinate -dimsu ./scripts/run.sh
```

`caffeinate` keeps macOS from throttling background work — without it,
analysis pauses when you switch to another app. (Plain `./scripts/run.sh`
works too, but you'll see it slow down when you're not looking.)

Two things happen:

1. Your default browser opens to `http://localhost:8765`.
2. Terminal shows log lines as the server runs.

You'll see a mostly-empty page. Click **Browse…** to pick a folder, then
follow [USER_GUIDE.md](USER_GUIDE.md).

The top-right of the browser shows two pills:

- **Health** — should turn green after a few seconds, reading something
  like `ready · qwen2.5vl:7b`.
- **Budget** — shows your Claude spend. If you skipped Step 7, it reads
  "Claude: API key not set" in gray, which is fine.

---

## Stopping the app

In the Terminal window where you ran `run.sh`, press **Control + C**.
Closing the Terminal window also stops it.

## Updating the app later

When new improvements ship:

```
cd ~/Documents/wildlife-image-culling
git pull
./scripts/setup.sh
```

The `setup.sh` re-run picks up any new Python dependencies (like the
Anthropic SDK). If nothing changed, it's a no-op.

## Troubleshooting

**"command not found: brew"** — Step 2 didn't finish. Re-read the "Next
steps" output from the Homebrew installer.

**"command not found: ollama"** — Step 3 didn't finish, or your terminal
session is stale. Close Terminal, reopen, try again.

**Browser shows "This site can't be reached"** — the server hasn't
started yet. Wait 5 seconds and reload. If that doesn't work, look at
the Terminal window for an error message.

**"Ollama: connection refused"** — start it with
`brew services start ollama`.

**"server unreachable" in red top-right** — the whole Python process
died. Restart with `caffeinate -dimsu ./scripts/run.sh`. The DB is
persistent, so analysis picks up where it left off.

**"Claude: API key not set"** — Step 7 was skipped or your key is in
the wrong file. Open `scripts/run.sh`, confirm the
`export ANTHROPIC_API_KEY=...` line is uncommented and has your key.

**"Budget cap reached" when trying to score** — Claude spending hit
your `CLAUDE_BUDGET_USD` ceiling. Edit `scripts/run.sh`, raise the
number, restart.

**Analysis freezes mid-shoot** — usually App Nap (macOS pausing
background work because the terminal isn't focused). Restart with
`caffeinate -dimsu ./scripts/run.sh` instead of plain `run.sh`.

**Anything else** — copy the error from Terminal and send it to me.
