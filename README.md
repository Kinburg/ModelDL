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
.venv\Scripts\python.exe -m pip install -e ".[hf]"
.venv\Scripts\python.exe scripts\serve.py --open
```

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
each one — with the category dropdown right there.

Turn off **start downloads on add** and links pile up paused instead, so an afternoon of
collecting them costs no bandwidth until you press *Start all*. That releases what is merely
waiting; blocked tasks stay blocked, because they are waiting on a decision rather than on
permission, and answering it is the point.

**Info** on a finished task opens its stored record in the card — page link, hash, why it
was filed where it was, and the trigger words with a copy button. Worth having once records
are collected into their own directory, where they are tidy and hard to find.

Progress streams over Server-Sent Events; the page patches rows in place rather than
re-rendering, so a list updating four times a second does not fight with your scrolling.

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

Base-model subfolders come from the service, not the file — the opposite of the category
rule, and deliberately. A Pony LoRA records `sdxl_base_v1-0` in its training metadata:
true, and useless for filing, because Pony LoRAs do not work on plain SDXL.

## Status

Working, and verified against the live services end to end: transfer core, the HuggingFace,
Civitai and generic HTTP providers, model classification, library placement with sidecars,
the persistent queue, and the local web UI.

Still to come: the `hf_hub` engine — running HuggingFace's own client in a subprocess, for
whole-repo transfers and as a fallback when the native path is refused.
