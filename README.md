# ModelDL

Downloader for large model files from services that hand out **expiring signed URLs** —
HuggingFace and Civitai in particular.

The problem it solves: those services answer the download URL with a 302 to a CDN link
signed for a few minutes. When the connection drops, a browser retries the *expired* link,
gets a 403, and starts the whole file over. On a 20 GB checkpoint that is the difference
between a hiccup and an evening.

## The core idea

**File identity and transport URL are separate things.**

`(repo, revision, path)` or `(modelVersionId, fileId)` is stable and gets persisted.
The signed CDN URL is a disposable that is re-minted on every reconnect. Everything else
follows from that split:

- `sfd/providers/` — knows how to mint a fresh URL for a service. Knows nothing about chunks.
- `sfd/core/` — knows how to move bytes safely. Knows nothing about HuggingFace or Civitai.

## What the transfer layer guarantees

| | |
|---|---|
| **Never corrupts a partial file** | A resume request answered `200` instead of `206` is refused without writing a byte. `Content-Range` is parsed and its start offset checked against what was asked. |
| **Never silently restarts** | If the server stops honouring ranges while a partial download exists, the transfer fails loudly rather than starting from zero. |
| **Catches trickles, not just silence** | A throughput floor over a rolling window kills connections that keep the socket alive while moving nothing. No read timeout will ever fire on those. |
| **Verifies before publishing** | SHA256 is computed by a background task chasing the contiguous completed prefix, so verification costs no extra pass. Nothing is renamed into place until it matches; a mismatch is quarantined as `.part.corrupt`, not deleted. |
| **Resumes across restarts** | Progress lives in `<file>.part.json` next to the data, so a `.part` stays resumable even if the queue database is lost. |
| **Never waits on one slow connection** | A worker with nothing left to claim splits the largest in-flight chunk and takes half of it. Without this the tail of a download collapses onto whichever connection is holding the last chunk — which may well be the worst one. |
| **Refuses login pages** | An HTML body where a model was expected is an error, not a 12 KB `.safetensors`. |
| **Skips what you already have** | A file already in place is hashed and skipped. Hashing costs a read; re-fetching costs the whole transfer. |

## What a model *is* beats what a site *calls* it

Placement is decided by reading the file's own header over a Range request — the first
megabyte of a 24 GB file is enough — and only then by the service's label.

That ordering is not the obvious one, and it comes from a real case. Civitai lists Sick
Ollie Krea2 as a **Checkpoint**, but the file holds only `model.diffusion_model.*` weights:
no VAE, no text encoder. In ComfyUI it loads through *Load Diffusion Model*, not *Load
Checkpoint*. Filed under the site's label it would land in a folder whose loader cannot open
it. A site's category is what an uploader picked from a dropdown; the header is what the
weights are, and the weights decide which loader works.

The same reading separates `flux1-dev-Q4_K_S.gguf` from `Qwen3-1.7B-Q4_K_M.gguf` — identical
extension, `general.architecture` of `flux` versus `qwen3`, and two different folders.

When the two sources disagree the verdict says so and asks rather than filing silently. The
service still wins where the header is unreadable (a `.pth` upscaler), and remains the only
source of base model and trigger words.

## Five traps worth knowing about

All were found by pointing the thing at real services, and none is guessable from the docs.
Each has a regression test.

**HuggingFace's Xet bridge binds the signature to the probe's byte range.** Resolve with
`Range: bytes=0-0` to sniff metadata and the URL you get back is valid *for that one byte*.
Every real chunk request then returns `403 Auth failed: invalid range`, and no amount of
retrying or re-resolving helps, because each re-resolve poisons itself the same way. Probe
with `HEAD`, which carries no Range — the resulting URL serves arbitrary ranges and can be
reused across connections.

**`file.truncate()` on Windows zero-fills.** It goes through the CRT's `_chsize_s`, which
extends a file by *writing* zero buffers in a loop: 75–97 seconds and 18.5 GB of pointless
writes before the download starts. Marking the file sparse first does not help on its own —
`truncate` materialises the holes anyway. Setting the sparse flag and then calling
`SetEndOfFile` directly does the same job in **37 milliseconds**.

**Civitai serves from two CDNs and one of them rejects HEAD.** Backblaze (`b2.civitai.com`)
answers HEAD normally; Cloudflare R2 hands out URLs presigned for GET and returns 403 to a
HEAD on the same URL — a presigned signature covers the method. The URL is still perfectly
good for ranged reads, so a probe that treats that 403 as an access failure blocks every
R2-hosted download. Only a refusal from `civitai.com` itself, before any redirect, is a real
permission problem.

**Civitai returns one filename for every variant of a version.** A version carrying bf16,
mxfp8, fp8, int8 and nf4 reports the identical `name` for all five, so writing two of them
into the same folder leaves one file and no way to tell which precision it is. The provider
appends the distinguishing detail — `model.bf16.safetensors` — but only where a name is
actually shared, so ordinary single-file versions are untouched. Downloads need an API key
(a clean 401 without one); metadata does not, so variants can be listed before anyone is
asked for credentials.

**The CDN's ETag is not the file's SHA256.** On a Xet-backed repo it is the Xet content id:
64 hex characters, identical in shape to a digest, and a completely different value (it is
also what appears in the CDN path). Only `x-linked-etag`, set by huggingface.co on its
redirect, is the hash of the bytes. Confusing the two means computing a perfectly correct
digest, comparing it against something unrelated, and quarantining a flawless download.
An ETag we cannot identify leaves the hash unset, so verification is skipped rather than
failed — a missing hash must never produce a verdict.

## Two engines for HuggingFace

**native** is the transfer described above: resume we control, a hash we verify, per-chunk
progress, placement exactly where the layout says. It is the default.

**hf_hub** runs HuggingFace's own client — in a **subprocess**, never as a library call in
the server. That is not fussiness; it follows from one line in their documentation:

> All environment variables are read at import time of `huggingface_hub`. Any modification
> made afterwards will not be taken into account.

Half of what tunes that client is environment variables — `HF_HUB_DISABLE_XET`,
`HF_XET_HIGH_PERFORMANCE`, `HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY`. A long-running server
cannot change any of them without a fresh interpreter. A subprocess also gets a clean
connection pool each time, which is the documented shape of "the second repository
downloaded at zero bytes per second", and it can actually be killed — a wedged thread inside
someone else's library cannot.

Progress comes back as JSON lines from a `tqdm` subclass, so it is real byte counts rather
than scraped console output, and the token travels in the environment rather than on a
command line every process on the machine can read.

Worth switching to it for Xet chunk deduplication when re-fetching a repo that changed.
Either engine can fall back to the other once when it fails outright; they speak different
protocols, so one sometimes succeeds where the other is refused. **Verification is ours
either way** — a file fetched by `hf_hub` is still hashed against what the Hub advertised
before it counts as done, because a guarantee that lapses when a setting changes is not one.

`huggingface_hub` writes its bookkeeping into a hidden `.cache/huggingface/` inside the
destination. That is what makes its resume work; it stays contained.

## On connection count

`scripts/bench_connections.py` measures throughput at several concurrency levels and
reports client CPU alongside it, which distinguishes a per-connection cap from an aggregate
one from a client-side bottleneck.

Measured against HuggingFace's CDN, extra connections buy **consistency, not headroom**:

| connections | observed range | worst |
|---|---|---|
| 1 | 2.4 – 17.4 MiB/s | 2.4 |
| 6 | 6.8 – 24.0 MiB/s | 6.8 |
| 16 | 19.2 – 27.3 MiB/s | 19.2 |

The peak barely moves, but a single connection can draw a bad edge and crawl at a tenth of
the achievable rate. Sixteen connections average that lottery out, which is why that is the
default — and also what `hf-xet` uses (`HF_XET_NUM_CONCURRENT_RANGE_GETS`).

The ~26 MiB/s ceiling in that run was the route, not the machine: client CPU never exceeded
0.34 of a core. Later runs against the same Hub reached close to a gigabit, and Civitai
regularly does 40–100 MiB/s on the same connection. So treat those numbers as one
afternoon's weather, not a limit — and rerun the benchmark rather than trusting them. The
part that holds is the shape: **when CPU stays low, adding threads or processes cannot help,
because the client was never the thing in the way.**

An aside that costs people an evening: Task Manager reports **bits** per second and
downloaders report **bytes**. 43 MB/s and "344 Mbit/s" are the same number. The UI has a
one-click toggle for exactly this reason.

## Windows specifics

- Each connection gets its own file handle (independent file positions, no write locking),
  opened unbuffered so the hasher's separate read handle sees the data.
- Connection count is clamped to the destination's media type — parallel out-of-order writes
  are a win on NVMe and a seek storm on a spinning disk. Note that a lot of SATA hardware
  reports `MediaType: Unspecified` with `SpindleSpeed: 0`, which is genuinely
  indistinguishable; those fall back to a conservative default and can be pinned manually.

## Getting started

Needs Python 3.12 or newer. On Windows:

```bash
run.cmd
```

First run creates `.venv`, installs the dependencies and opens the interface; every run after
that just starts it. `run.ps1` does the same from PowerShell.

By hand, or on Linux and macOS:

```bash
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[hf,desktop]"
.venv\Scripts\python.exe scripts\serve.py --open
```

`desktop` is the standalone window (pywebview). Leave it out and everything still works —
the same server opens in your browser instead, which is what a headless box wants anyway.

**`python scripts/serve.py` will not work.** That runs whatever Python is on PATH, which is
not the one holding the dependencies — always the interpreter inside `.venv`. The scripts
detect that mistake and print the command you meant, rather than a bare
`ModuleNotFoundError` naming a package that has nothing to do with the actual problem.

Nothing needs configuring to start: with no settings at all, files land in `downloads/`
unsorted. Tokens and a library path go in the Settings panel — or in `$HF_TOKEN` and
`$CIVITAI_TOKEN`, which take priority and keep them out of the settings file.

## The web interface

```bash
python scripts/serve.py --open
```

A single page on `http://127.0.0.1:7788`: paste a link, watch the queue, adjust settings.
It binds to localhost only and has no authentication, because it holds your tokens.

The queue lives in SQLite and survives being closed, crashed or rebooted — a task caught
mid-download comes back as pending with its bytes intact, since the `.part` file carries its
own resume state. Pausing is a real disconnect rather than a held-open socket; there is
nothing to lose by pausing for an hour.

Uncertain placements arrive **blocked** rather than filed. Adding one Civitai link that
expands to five quantisations queues 67 GB that will not move until you accept or correct
each one, with the reason for the guess on the card.

*Elsewhere…* answers the same question with your library instead of our category names. The
list is the folders that are really there, best guesses first: the one the layout would have
used (base-model grouping included, spelled out rather than implied), the runners-up for that
kind, any folder whose name appears in the filename, then everything else by what is actually
in it, and finally the canonical homes that have no folder yet. That last group matters
because the file may be the first LoRA a library has ever had; the folder-name match matters
because the categories deliberately do not claim `sams`, `insightface`, `reactor` and the
rest — a bare `.pt` gives nothing to tell them apart — so until now the right answer for a
SAM checkpoint was not on the list at all.

Search narrows it, and typing a folder that does not exist offers it as a choice. Nothing is
created at that moment: the folder appears when the file lands, so a download that fails
leaves no empty folders behind. The one thing that does create a directory on request is
*Move to…* below, and only ever inside the library root. What you pick is taken exactly as
given — nothing is appended underneath you, which is how a
successful download goes missing. Tick **send this kind of model here from now on** and the
choice becomes the mapping for that kind; a base-model folder like `checkpoints/Krea 2` names
no kind, so it files the one file and leaves the mapping alone.

*Move to…* on a finished download is the same dialog asked after the fact, for when the
guess was accepted and turned out wrong anyway. It moves the model **and everything named
after it** — the `.json` record, `.civitai.info`, the trigger-word `.txt` and
`.preview.png` — because a model manager that finds a checkpoint without its preview shows
a blank card, and dragging one file in Explorer is exactly how that happens. Sidecars
collected elsewhere with **keep sidecars in** follow the mirror of the library tree rather
than being dumped beside the model they were deliberately kept away from.

Nothing is overwritten: if the destination already holds a file of that name the move is
refused before anything is touched. The model goes first and the companions follow, so a
companion that cannot move leaves the model where you asked for it and names the ones that
stayed behind — reporting the model as stuck when it moved perfectly well would send you
hunting in the wrong folder.

The list in that dialog is the library and nothing else, so another drive is unreachable
from it by construction. **Another drive…** opens the system's own folder dialog instead,
and accepts anywhere on the machine. What makes that safe is that the request never names a
destination: the path is chosen in a modal window the OS put in front of whoever is at the
keyboard, and is never sent by the page. Something reaching this unauthenticated API that
is not a person — a web page that pointed its own hostname at 127.0.0.1, say — can make a
folder picker appear and nothing else. It cannot answer one, and it cannot say where a file
should land.

Across a drive boundary a rename cannot exist, so the move becomes what it really is: every
byte copied, then the original deleted. That takes as long as downloading the file did, so
it reports its progress and the queue keeps running throughout. The copy lands under a
`.moving` name and is renamed into place only once it is whole — the destination never holds
a half-written model that looks finished — and if it fails partway the fragment goes and the
original is still sitting where it was, untouched.

**Stop the move** appears beside the progress while that copy runs, and the same applies to
giving up as to failing: the fragment is deleted and the original has not been touched, so
there is nothing to put back. The stop is a request rather than a kill, because a thread
copying a file cannot be interrupted — only asked between blocks — so expect it to take
until the current 4 MB is written. It follows the event stream rather than the tab that
started the move, so the button is there after a reload and in a second window. Once the
last byte is across there is nothing left to stop: the sidecars that follow are far too
small to wait for, and stopping between them would strand the model away from them.

Turn off **start downloads on add** and links pile up paused instead, so an afternoon of
collecting them costs no bandwidth until you press *Start all*. That releases what is merely
waiting; blocked tasks stay blocked, because they are waiting on a decision rather than on
permission, and answering it is the point.

**Add new downloads to** decides which end of the queue a pasted link joins: the bottom, so
it waits its turn, or the top, so it is what runs next — the setting to flip when the queue
is a long backlog and the thing you just found is the thing you actually want. A link that
expands into several files keeps its own order either way. Anything already downloading
keeps going; the queue only decides what is picked up next, and a card can still be dragged
by its grip afterwards.

A failure that a wait might fix is picked back up on its own — after 30 seconds, then two
minutes, then ten. A router rebooting at 3am costs minutes rather than the rest of the
night. Failures no wait can fix are never retried: a missing token, a refused licence, a
hash that did not match and a disk with no room left are all answered by a person, and
asking the service again every thirty seconds is how a temporary refusal becomes a ban.
The switch is **retry failures on their own**; pressing *Retry* by hand also forgives the
attempts already spent.

**Speed limit** caps the whole queue rather than each connection, and takes effect while
downloads are running — which is when you actually reach for it. It applies to the native
transfer; the `huggingface_hub` engine downloads in a subprocess of its own and is not
capped. (Neither is the stall watchdog fooled by it: the floor it uses drops with the
ceiling, so a connection being held back on purpose is not mistaken for a dead one.)

The header says what the queue as a whole is doing — fetched of total, current speed, ETA —
and warns when what is left does not fit on the disk. That warning is worth more before
the queue runs than after: preallocation is sparse, so nothing is reserved up front and a
full disk otherwise turns up forty gigabytes into a download. A file that plainly cannot
fit is refused before it starts, with the numbers in the message.

The filter box and the state dropdown narrow a long list; finished downloads collapse to
one line each until opened. Dragging is disabled while a filter is on, because the reorder
would only see the rows on screen and would shuffle them around the ones it cannot.

**Info** on a finished task opens its stored record in the card — page link, hash, why it
was filed where it was, and the trigger words with a copy button. Worth having once records
are collected into their own directory, where they are tidy and hard to find.

Progress streams over Server-Sent Events; the page patches rows in place rather than
re-rendering, so a list updating four times a second does not fight with your scrolling.

The server answers only to `127.0.0.1` and `localhost` by name, not merely by address. It
has no authentication — it holds your tokens and is not meant to be reachable — and a page
on any website can point a hostname it owns at the loopback address and talk to a local
server as same-origin. Checking the name it was asked for is what closes that.

## The command line

All of it works without the browser. Use the interpreter from `.venv`, written here as
`python` for brevity:

```bash
python scripts/download.py https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/blob/main/Qwen3-1.7B-Q4_K_M.gguf
```

Kill it mid-download, run the same command again — it picks up where it stopped.

Paste whatever you have. A Hub `/blob/` page, a `/resolve/` link, a `/tree/` folder, a bare
`org/name`, a Civitai model page, a Civitai download button link, an AIR urn, or a plain URL
to anything else. Anything holding several files expands into them:

```bash
python scripts/download.py unsloth/Qwen3-1.7B-GGUF --include '*Q4_K_M*' --dry-run
python scripts/download.py 'https://civitai.com/models/2676616?modelVersionId=3154408' --dry-run
```

A link that expands to more than one file will not start downloading until you narrow it
with `--include` or confirm with `--all` — one Civitai version can be five quantisations of
the same checkpoint, which is 67 GB when you wanted 12.

`scripts/probe.py <url>` inspects a link without downloading it — size, hash, range
support, whether a token is needed. `scripts/bench_connections.py` measures throughput.

Tokens come from `$HF_TOKEN` / `$CIVITAI_TOKEN`; prefer those over `--hf-token`, since a
command line is visible to every process on the machine.

## Tests

```bash
python -m pytest -q
```

The integration tests run against a local server that misbehaves on demand: rotating
signatures, dropped connections, ignored `Range` headers, login pages, lying checksums.

## Sorting into an existing library

```bash
python scripts/download.py <url> --library 'F:\ComfyUI\Shared' --dry-run
```

The layout is **adopted, not imposed**. An install with forty folders and firm habits about
them does not need a downloader inventing a structure alongside. Adoption also has to settle
synonyms, since ComfyUI accepts several names for the same thing and installs collect both:
when `unet` and `diffusion_models` both exist, the one already holding files wins. Folders
belonging to custom nodes are left alone rather than claimed.

Detector folders are deliberately *not* unified. `sams`, `sam2` and `facedetection` hold
detectors too, but segmenters, face detectors and YOLO models are read by different nodes and
a bare `.pt` gives nothing to tell them apart — so they stay unclaimed instead of being filed
confidently in the wrong place.

Placement is shown before anything downloads, and an uncertain one stops the run:

```
    23.9 GB  sickOllie_krea2.bf16.safetensors
           ? -> F:\ComfyUI\Shared\diffusion_models\Krea 2\sickOllie_krea2.bf16.safetensors
             diffusion_model (high) — contains denoiser weights and nothing else; the
             service lists it as checkpoint, but the file's contents decide which loader
             can open it
```

Each model gets `<name>.json` beside it with source, hash, base model, why it was filed where
it was — and **trigger words**, which Civitai publishes, every plain download discards, and
nothing can recover afterwards. Set `sidecar_dir` to collect those records somewhere else
instead; the library's folder structure is mirrored underneath, because two
`model.safetensors` in different categories are different files and flattening them would
leave one overwritten.

Civitai downloads also get `<name>.civitai.info` and `<name>.preview.png`. Those are never
moved: the A1111 and ComfyUI model managers look for them beside the model and nowhere else.
They can be turned off entirely, but not relocated.

A LoRA additionally gets `<name>.txt` holding **only** its trigger words. That filename has a
specific meaning — A1111 extensions and several ComfyUI loader nodes read it as activation
text and paste the contents straight into the prompt — so the source link deliberately stays
in the JSON record. Putting it in the `.txt` would put it in your generations. Civitai
usually returns every word inside one comma-joined string, which is re-split and deduplicated
on the way in.

`scripts/backfill_triggers.py <library> [--apply]` writes those files for anything downloaded
before the feature existed, reading the words back out of the JSON records.

### Sample images

The pictures a model is published with are shown in the queue: a thumbnail on every row,
the full set behind it, and under each one the prompt, sampler, steps, cfg and seed that
produced it. Trigger words say which tokens wake a LoRA up and nothing about the prompt
around them; the samples were made by someone who knew, and those settings are published by
the service and lost by every plain download. They are written into the `.json` record too.

Nothing is fetched ahead of time. A picture is requested when the page first draws it,
kept in `preview_dir` — a cache, deletable at any time — and served from there afterwards,
including for every other file of the same model version, which shares the images. Rows ask
for a 320-pixel copy from the CDN rather than shrinking the original, so a thumbnail costs
about forty kilobytes rather than three megabytes.

Samples the service marks as adult are covered until clicked. The URLs come from an API
response, which is remote data, so only `https` and only the hosts we download from are
ever requested.

Base-model subfolders come from the service, not the file — the opposite of the category
rule, and deliberately. A Pony LoRA records `sdxl_base_v1-0` in its training metadata:
true, and useless for filing, because Pony LoRAs do not work on plain SDXL.

## Desktop Application & Standalone Executable

ModelDL runs as a standalone desktop window on Windows (powered by Microsoft Edge WebView2 via `pywebview`), macOS (WKWebView), and Linux (WebKitGTK).

### Running locally
```cmd
run.cmd
```
or PowerShell:
```powershell
.\run.ps1
```
Optional flags:
- `--browser`: open in default web browser instead of standalone desktop window.
- `--no-gui`: run headless backend server without opening a window or browser.
- `--port 7788`: change port. If that port is unavailable another is picked automatically
  and printed on startup — on Windows, Hyper-V and WSL reserve blocks of ports at boot
  (`netsh interface ipv4 show excludedportrange protocol=tcp`), and a reserved port refuses
  the bind rather than reporting itself as in use.

### Building standalone ModelDL.exe
To build a single-file executable that does not require Python or any dependencies:
```cmd
build.cmd
```
or
```cmd
.venv\Scripts\python scripts\build_exe.py
```
The resulting `ModelDL.exe` will be located in `dist/`.

## Status

Working, and verified against the live services end to end: transfer core, the HuggingFace,
Civitai and generic HTTP providers, model classification, library placement with sidecars,
the persistent queue, and the desktop UI.

