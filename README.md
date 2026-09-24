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

A single page on `http://127.0.0.1:7788`, in three panes: the views and the library's folders
on the left, a list in the middle, and everything about whatever is selected on the right.
The dividers between them are dragged to taste, and a double-click puts one back. It binds
to localhost only and has no authentication, because it holds your tokens.

How the window was left — the pane widths, the folders open in the tree, the view, the sort
order — is kept in `settings.json`, under `ui`, rather than in the page's own storage. The
desktop window runs WebView2 in private mode, and private mode forgets `localStorage` every
time the app is closed; a layout that reset itself on every start would be one nobody
bothered to arrange.

The page is plain ES modules — `app.js` and the files in `js/` — loaded as they are, with
no build step, because the whole thing is served by the process that downloads and has to
keep working from inside a PyInstaller bundle. They are served as `text/javascript` whatever
the Windows registry thinks a `.js` file is (on more machines than one would like, some
installer has told it `text/plain`, and a browser will not run a module served as text), and
revalidated on every load, so an updated app never runs yesterday's page.

### The library

The queue says what is downloading; the library says what you have and where it is. It is
every model file in the library's folders, downloaded here or not, kept as a table that is a
cache of the disk — rebuilt by walking the folders, which happens when the app starts and
again every time the window comes back to the front. That is cheap enough to do without a
second thought: a real library of three hundred models and a terabyte, in five hundred
folders, is read in a fifth of a second. Only a listing is read; a file's header is read
once per version of the file, in the background, and its hash only when someone asks.

The tree on the left is that library as it is on disk: every folder holding a model, with
how many and how much, plus the folders named for a kind of model — `hypernetworks` with
nothing in it yet is still where hypernetworks go. A folder is opened in the middle with
what is in it and in the folders under it; *Subfolders* turns the second part off. A
download shows up in the folder it is going to land in, with its progress, before a byte of
it is there.

ComfyUI reads models from several folders and drives, and so does this: *Settings →
Library folders* lists them, and *Add a folder…* adds one through the system's own dialog.
Downloads are filed into the main one, using the folders it already has; any other can be
made the main one, and any can be taken off the list, which touches nothing on disk. A
folder inside one of them that is not part of the library — a llama.cpp checkout with a
dozen vocabulary GGUFs, a custom node's test data — can be hidden from the tree by its
right-click menu, and shown again from Settings.

Records collected into `sidecar_dir` mirror the main folder at the top, as they always did;
every other folder of the library gets a mirror of its own under `@roots/`, named after its
whole path, and a file in no folder of the library — moved to another drive — is mirrored by
the folder it is in, under `@outside/`. `D:\models\loras\a.safetensors` and
`E:\models\loras\a.safetensors` are two files, and one mirror for both would give them one
record — and one note. Making another folder the main one moves the records to match.

A record written under an older arrangement — beside the model before `sidecar_dir` was set,
in the mirror of whichever folder was the main one then, flat — is still found, and a move
takes it to where it would be written today. Found only when nothing else can claim it,
though: a record in one of those old places is taken only if it names this model's file and
is not the proper record of another model that exists, since the old top-level mirror of one
folder is exactly where another folder's model keeps its record now.

A model split into shards — `model-00001-of-00004.safetensors` — is one entry, moved and
deleted as one with its index. It is not renamed here: every part and the index would have
to change together, and a loader that finds three of four is worse off than one that finds
none.

### Models that went missing

A model deleted or moved outside the app stays in the library, grey, where it was last seen,
and in *Missing* with a count on it. Most of those were moved, not deleted, and the walk that
notices one is gone is the walk that notices where it went: a file of the same name and the
same size turning up somewhere else, while exactly one model of that name and size went
missing, is the same file dragged in Explorer. It is linked to its new place on its own and
the page says so. On one real library that recovered twelve models of thirty-six the moment
it first ran. Two candidates is a question rather than a guess: the missing model offers
both, and *This is it* makes one of them the model.

Its record goes with it when the record is in `sidecar_dir`, which is this app's own
directory. What stayed behind in the folder it was dragged out of — the preview, the trigger
words, a record kept beside the model — is someone's library, and is offered rather than
moved: *Bring them here* renames each to the model's current name on the way.

*Download again…* is for one that was deleted. Everything needed was kept — which file, from
where, its hash — so nothing is looked up to fetch it: the same question as any download is
asked, where it goes, with the folder it was in first. Whichever folder it goes to, the file
that arrives is that model back — the same entry, its history now two downloads long, its
note carried into the new record. *Download all again…* does the whole *Missing* view at
once, each back into the folder it was in, with the total against the free space first.

*Find online…* is for one whose link is gone, or never was: a model Civitai took down, a
file from a browser. Civitai is asked by the hash the library kept — the whole file's, which
is proof, or the AutoV1, which with the size is very nearly — and the Hub by the name and
the size, proven by the hash when it is known. The file itself is not needed, which is the
point: it is missing. What is found is listed with how sure it is, and downloads into the
same entry, the same way. A file renamed on its way here is found by nothing but a person,
so the dialog links the two sites' own searches as well.

*Find the file…* is for the rest: the system's own file dialog, for a model that was renamed
as well as moved. The rule is the one *Another drive…* follows — the request says which
model, the operating system asks the person at the keyboard which file, and the page never
names a path. A file of another size than the model had is a question, held on the server
under a token while the page asks it, so that the path still never travels through the page.

*Forget…* takes a missing model out of the library for good, and offers to delete what it
left on disk: its record, and whatever was named after it where it used to be. A model
downloaded here keeps its line in the history, with its note. One found on disk has no
history to keep a note in, and the dialog says so. A model found on disk that disappears
with nothing of anyone's attached to it — no note, no record, nothing identified — goes
quietly: keeping every file that ever passed through a folder, grey, forever, is how a
library fills up with rubbish.

### Models from somewhere else

Most of a library usually did not come through this app — on the one above, two thirds. Those
are described from the files themselves, which say more than one might expect. The header
gives the kind of model (the same classifier as a download), what precision the weights are
really stored in, and how many parameters there are; a GGUF adds its architecture, its
quantisation and its context length. kohya writes its whole training configuration into a
LoRA — the base model, the rank, and how often each tag appeared in the captions, which for
most LoRAs is as near to the trigger words as a file gets, and is shown as *Most frequent
training tags*. `modelspec.*` can add a title, an author, a description and a thumbnail.

What other tools left beside it is read too. A `.civitai.info` from Civitai Helper is the
whole Civitai record: the model is then as well described as one downloaded here, with its
samples and their prompts. A1111's `<model>.json` gives its activation text, preferred
weight and notes; LoRA Manager's `.metadata.json` and Stability Matrix's `.cm-info.json`
their names and base models; any picture named after the model is its preview. Move, rename
and delete know those files too, since a rename that left another tool's sidecar under the
old name would be the half-rename this app exists to prevent.

The folder is a witness as well. For a file the header cannot read — a `.pth`, an `.onnx` —
the folder it is in is the best evidence there is, and says so. Between a language model, a
text encoder and a vision tower the folder wins outright, because those are one file used by
different loaders: Qwen in `text_encoders` is a text encoder by its job. Anywhere else a
header that is sure is believed over the folder, and the difference is shown — a LoRA sitting
in `checkpoints` is a LoRA the checkpoint loader will not open, and *Move to…* is right
there.

*Identify* is the rest, and it is a button, never something done behind anyone's back. Two
services are asked, each proven by the file's SHA256. Civitai looks a file up by its hash.
The Hub cannot — nothing on it finds a file by its contents — but it can be asked by name:
which model cards mention the file (a card often links its real home), which repositories
are called like it; and for any repository it publishes the exact size and SHA256 of every
file. So on the Hub a file is found by its name, narrowed by its size, and proven by its
hash. On one real library, nineteen of twenty text encoders, VAEs and detectors that Civitai
had never heard of were on the Hub under their own names.

Reading a whole file is what takes the time — a few seconds for a LoRA, minutes for a large
checkpoint on a mechanical drive — so a big one is only read when there is something to
prove. First the quick look: Civitai by the AutoV1 hash, the one A1111 used to show, which
is sixty-four kilobytes of the file and which Civitai still answers to; and the Hub by name
and size. When neither has anything the file could be, that is the answer, in a second or
two instead of a 27 GB read. When either does, the file is read and the hash settles it.
One at a time, in the background, with its progress and a *Stop* in the status bar.

A model found on Civitai gets the files a download of it would have left, by the same
settings — its record, and the `.civitai.info`, preview and trigger `.txt` — but never over
one that is already there, which may be another tool's or somebody's own. One found on the
Hub gets its record: the repository and the path in it, the commit, its licence and base
model from the repository's tags — and so a way to be downloaded again, and checked for a
newer commit. A download's hash was checked against the bytes as they landed, so looking one
up needs no second read.

### The history

*Clear finished* no longer deletes anything: it takes finished downloads off the Downloads
list, and they stay in *History* — every download that finished, newest first, by day. A
renamed model shows what it is called now and the name it arrived under, since the name the
service gave it is what anyone searching that service for it again will type. One whose
files were deleted, or that went missing and was forgotten, stays too, marked as such, with
*Download it again*: which file, from where and where it was is all still known, and the one
question left — where it goes now — is asked with the folder it was in first. Taking a line
out of the history is its own button, and touches no model.

A link to a file that is still in the library is not queued again; the page says where it
already is. A file that was deleted, or went missing, is — that is what pasting its link
again is for.

### Duplicates, leftovers and newer versions

*Duplicates* lists the same file kept in more than one place. A size shared by several files
is only a reason to look closer: every LoRA trained at one rank for one architecture comes
out the same number of bytes — eleven different Krea 2 LoRAs of exactly 228,588,904 bytes
on one library — and so does every fine-tune saved at one precision. What tells them apart
is a fingerprint: five pieces of 64 KB each, from the start, the quarters and the end,
hashed. Pieces that differ prove the files differ, and those files never appear here. It is
taken in the background, once per version of a file, alongside the header — a third of a
megabyte per file, and on that library 106 files in a fifth of a second. Of 33 sets of files
the size alone called duplicates, 18 turned out to be different models.

Pieces that agree are not proof, though: two merges that left the text encoder untouched
can agree wherever they are sampled. So a set is *very likely identical* until the hashes of
the whole files are known, and *Confirm by hash* reads only those files; after that it is
*identical*, and only then can the copies you do not keep be deleted. The star marks the one
kept — the library suggests the one it knows most about, in the main library folder — and a
click on another moves it. When the copies sit in folders different nodes read, say
`insightface/` and `simswap/`, the set says so: each node may need its own copy. A workflow
names the file it loads, folder included, so one that used a deleted copy needs pointing at
the kept one.

That is what *Link the copies* is for. The other copies become other names of the one kept —
NTFS hard links — so every path keeps working, every node and workflow finds the same files
where it always did, and the room of each copy is freed. There is no original and no link
afterwards: each name is the file, equally, and a loader sees an ordinary file. Only copies
the hashes prove the same are linked, only on the drive the kept one is on (a hard link
cannot cross a volume), and never one that changed since the library last read it. Each is
swapped in with one rename, so a name is never missing, and one a program holds open is left
as it was and said so.

A file under several names is marked *linked* wherever it is shown, and the inspector lists
every name it has, the ones outside the library too. Deleting one name frees nothing while
another is left, and the delete dialog says so, with the other names. A program that saves
by writing a new file and renaming it over the old one quietly gives that name a copy of its
own again; nothing breaks, the room is simply taken again. *Make separate copies* is the way
back on purpose: every byte copied beside the name, with its dates — so its hash still holds
— and swapped in, after the free space has been checked, with its progress and a *Stop* in
the status bar. The Duplicates view lists linked files in a section of their own, with the
room they save.

*Cleanup* lists what belongs to nothing — the `.part` of a download nobody is coming back
for (the 40 GB fragment from March), the `.part.corrupt` of one that failed its checksum, the
`.moving` of an interrupted copy, a preview or a `.civitai.info` named after a model that is
not there, a record in `sidecar_dir` for a model that is gone. A download still in the queue
keeps its `.part`; a model the library still remembers, missing or not, keeps what it left,
since forgetting it is where that is offered; a record whose model is somewhere else in the
library has lost track of it rather than outlived it, and is left alone. Everything else is
deleted from there, with the list in front of you first.

*Check for a newer version* asks each model's service. Civitai lists a model's versions
newest first, so anything ahead of this one is newer, and *Download it* queues the new
version's primary file rather than every quantisation it carries. On the Hub a file keeps its
name when it changes, so the question there is whether the same path on the same branch now
hashes differently.

### Changing a model

*Move to…* is the question "where does this go?" asked after the fact: the same ranked list
of the library's real folders that a new download is offered, across every folder of the
library, without the one it is in now. Selecting several and dragging them onto a folder in the tree does the same;
either way the move takes the model and everything named after it, refuses to overwrite
anything, and can be undone from the message that says it happened. *Another drive…* opens
the system's own folder dialog for anywhere else. Across a drive boundary a move is a copy,
with its progress and a *Stop* in the status bar; the fragment goes if it is stopped, and the
original has not been touched.

*Rename* happens in place — click the name, or press F2 — with the extension outside the
box, because every loader dispatches on it. Everything named after the model takes the new
name, the record's own `filename` is rewritten, and so is every line of the history about
it.

*Delete from disk…* shows the actual list of files, with sizes, before any of it goes,
because this is permanent: there is no recycle bin behind it. The model goes first and on its
own terms — if it will not go, which on Windows means a loader has the weights open, nothing
else is touched either. The history keeps a line saying it was deleted.

### Your note

*Your note* is for the thing about a model that only you know: the weight past which it
burns, the LoRA it fights, why you kept this quantisation and not the other. It is written
straight into the inspector and saved when you click away or press Ctrl+Enter — and saved,
too, if you select something else while it is half written, since that is not the moment
anyone meant to throw away what they wrote.

It lives in the model's `.json` record, not in the database, and that is the whole of the
design: the record follows the file through a move and a rename, it goes when *Delete* goes,
and the library only carries a copy — the same relationship `downloaded` has with the
`.part.json` beside a half-finished file. A model with no record — a file from elsewhere, or
one downloaded with records turned off — is given one to hold the note, and the page says
so, because a new file appearing beside a model is not something to find out about later.
Only the record, though: the `.civitai.info` and the trigger `.txt` are a separate choice.

A note can be written while the file is still downloading, which is when what you know about
a model is in your head. It waits with the download, and the write that lands the file is
the write that puts it in the record. The search box looks through notes, since *which of
these was the one that did hands properly* is a question about a note.

### Downloads

Paste a link into the box at the top — or anywhere, with Ctrl+V — or drag one from the
browser onto the window. Only the first line with anything on it is taken: a copied
paragraph with a link in it would otherwise arrive as one unparseable line.

The queue lives in SQLite and survives being closed, crashed or rebooted — a task caught
mid-download comes back as pending with its bytes intact, since the `.part` file carries its
own resume state. Pausing is a real disconnect rather than a held-open socket; there is
nothing to lose by pausing for an hour.

Nothing is queued until you have said where it goes. The link is resolved first — the files
it names, and what each of them is, read from its header over a range request — and then
*Add download* asks. Its list is the library's real folders, across every library folder,
ranked by what is in them rather than by what they are called: where the version you already
have is; where your other models of the same kind for the same base model are ("55 Krea 2
LoRAs here", with `Krea 2` and `Krea2` taken for the same base); where your last one went;
where the layout would have filed it, base-model grouping spelled out; then the folders that
hold that kind, any folder whose name is a word of the filename, everything else, and the
canonical homes that have no folder yet — the file may be the first LoRA a library has ever
had. The top row is marked, so Enter is the whole answer most of the time; the arrows move
it, a click marks another, typing searches or names a folder that does not exist yet
(nothing is created until the file lands), and *Another drive…* opens the system's own
dialog. Tick **send this kind of model here from now on** and the choice becomes the mapping
for that kind; a base-model folder like `checkpoints/Krea 2` names no kind, so it files the
one download and leaves the mapping alone.

A link that names several files lists them, ticked the way a person would. A Civitai version
ticks the file its own download button gives, not every quantisation it carries. A Hub
repository of one model — its weights, in shards or not, and the configs that go with them —
ticks all of it, and offers to keep the repository's folders under the one chosen, since a
transformers model is a folder that only works whole; a repository of several — twenty
quantisations, a pack of files for different nodes — ticks nothing, because which of them is
wanted is the question. A file already in the library is shown where it is, unticked. Twenty
quantisations of one model are one kind of file, so one header is read for all of them.

**Smart download placement**, off by default, files each download on its own by what it is
instead, and asks only when that is uncertain: such a download arrives as *needs a decision*,
waiting with the reason for the guess beside it, and *Elsewhere…* answers with the same
ranked list.

Turn off **start downloads on add** and links pile up paused instead, so an afternoon of
collecting them costs no bandwidth until you press *Start all*. That releases what is merely
waiting; downloads that need a decision stay where they are, because they are waiting on an
answer rather than on permission. **Add new downloads to** decides which end of the queue a
pasted link joins; a download can still be dragged by its grip afterwards.

A failure that a wait might fix is picked back up on its own — after 30 seconds, then two
minutes, then ten. Failures no wait can fix are never retried: a missing token, a refused
licence, a hash that did not match and a disk with no room left are all answered by a
person, and asking the service again every thirty seconds is how a temporary refusal becomes
a ban. **Speed limit** caps the whole queue rather than each connection, and takes effect
while downloads are running (the `huggingface_hub` engine downloads in a subprocess of its
own and is not capped).

The status bar says what the queue as a whole is doing — fetched of total, current speed,
ETA — and warns when what is left does not fit on the disk, which is worth more before the
queue runs than after: preallocation is sparse, so nothing is reserved up front.

Every download says which site its file came off: `civitai.com`, `huggingface.co`, or the host
of a plain link — the domain rather than our provider name, so a mirror like `civitai.red`
reads as itself.

A LoRA's trigger words sit in the inspector with a **Copy** button, comma-joined exactly as
the `.txt` beside the model holds them. The pictures a model is published with are there
too, and under each one the prompt, sampler, steps, cfg and seed that produced it. Samples the
service marks as adult are covered until clicked.

Progress streams over Server-Sent Events; the page patches rows in place rather than
re-rendering, so a list updating four times a second does not fight with your scrolling.

### Keyboard and mouse

Every list selects with a click, Ctrl+click and Shift+click, and with the arrow keys; Ctrl+A
takes the lot. Right-click anything for what can be done to it. F2 renames, Delete deletes
(or forgets, for a model that is missing), Enter opens a model's samples, Ctrl+F goes to the
search box and Escape backs out of whatever is open. The search box searches the whole
library from a folder, and narrows the list in every other view.

The server answers only to `127.0.0.1` and `localhost` by name, not merely by address. It
has no authentication — it holds your tokens and is not meant to be reachable — and a page
on any website can point a hostname it owns at the loopback address and talk to a local
server as same-origin. Checking the name it was asked for is what closes that.

It does not close the other way in. A page on any site can send a request straight to
`127.0.0.1` without renaming anything: a POST with no body, or with a body of no declared
type, needs nobody's permission to be sent, and a server that did not look would carry it
out — a move, a delete, a folder taken off the list. The browser says where every such
request comes from, in `Origin` and in fetch metadata, and anything that changes something
is refused unless it comes from the app's own page.

For the same reason no request that moves, renames, writes or deletes a file names a path:
a model is named by its id, a folder by which folder of the library and a place inside it,
and anywhere else is chosen in a dialog the operating system put in front of whoever is at
the keyboard.

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

Right-clicking a text field opens the usual editing menu — undo, cut, copy, paste, select
all. pywebview ties WebView2's context menus to debug mode, so the setting is turned on by
itself at startup; the page's own menu, the browser one with reload and save-as in it, stays
off, because this window is not a browser. Keyboard copy and paste work regardless.

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

### Releases

The newest build is always at
<https://github.com/Kinburg/ModelDL/releases/latest/download/ModelDL.exe>. Give it a folder
of its own: `queue.db`, `settings.json`, `downloads/` and `previews/` are kept in the folder
it is started from. The exe is not code-signed, so SmartScreen warns the first time it runs
— *More info*, then *Run anyway*.

A release is made by pushing a version tag:

```bash
git tag v0.2.0
git push origin v0.2.0
```

The [release workflow](.github/workflows/release.yml) then runs the tests on Windows, builds
the exe with `scripts/build_exe.py`, starts it headless to see that the bundle came out whole
— one missing a module still builds, then dies on launch without a word — and attaches it to
a release named after the tag. The version is written down in one place, `__version__` in
`sfd/__init__.py`, and a tag that disagrees with it stops the run before anything is built:
raise it and commit first. Started by hand instead, from *Actions → Release → Run workflow*,
the workflow builds and tests without publishing anything, and the exe stays with the run as
an artifact.

## Status

Working, and verified against the live services end to end: transfer core, the HuggingFace,
Civitai and generic HTTP providers, model classification, library placement with sidecars,
the persistent queue, and the desktop UI.

## License

[MIT](LICENSE).

