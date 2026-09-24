// Everything the page can do to a model, a download or the library, in one place — the
// buttons, the right-click menu, the keyboard and a drag onto the tree all end up here, so
// that each thing is done one way and says the same things when it is done.

import { del, get, post, put } from "./api.js";
import { baseName, debounce, esc, fmtBytes, kindLabel, plural } from "./util.js";
import { icon } from "./icons.js";
import {
  state, invalidate, upsertModel, removeModel, upsertTask, remember, folderLabel,
} from "./store.js";
import { toast, toastError } from "./toasts.js";
import { addDialog, confirmDialog, filesDialog, modal, pickFolder, promptDialog } from "./dialogs.js";
import { forgetPreviews } from "./lightbox.js";

// --- loading ----------------------------------------------------------------------------

export async function loadLibrary() {
  const body = await get("/api/library");
  state.roots = body.roots;
  state.folders = body.folders;
  state.hidden = body.hidden || [];
  state.sep = body.sep || "\\";
  state.syncedAt = body.synced_at;
  state.jobs = body.job || state.jobs;
  state.models = new Map(body.models.map((m) => [m.id, m]));
  state.details.clear();
  invalidate("tree", "list", "inspector", "status", "header");
}

export async function loadTasks() {
  const { tasks } = await get("/api/tasks");
  state.tasks = new Map(tasks.map((t) => [t.id, t]));
  invalidate("tree", "list", "inspector", "status", "header");
}

export async function loadSettings() {
  const { settings, error } = await get("/api/settings");
  state.settings = settings;
  if (error) toast(error, { level: "error" });
  invalidate("list", "inspector");
  return settings;
}

export async function loadDetails(id, force = false) {
  if (!force && state.details.has(id)) return state.details.get(id);
  const details = await get(`/api/models/${id}`);
  state.details.set(id, details);
  return details;
}

// The server has walked the disk again; the page is told what changed over the event
// stream, so all this has to do is ask.
let lastRescan = 0;
export async function rescan({ quiet = true } = {}) {
  const now = Date.now();
  if (quiet && now - lastRescan < 4000) return;
  lastRescan = now;
  try {
    const { report } = await post("/api/library/rescan");
    if (!quiet && report) {
      toast(`Library read again in ${report.elapsed.toFixed(1)}s — ${plural(report.models, "model")}`);
    }
  } catch (error) { if (!quiet) toastError(error); }
}

// --- navigating ---------------------------------------------------------------------------

export function go(view, { keepSelection = false } = {}) {
  if (view.kind !== "settings" && state.leaving && !state.leaving()) return;
  state.view = view;
  if (!keepSelection) { state.selection = new Set(); state.anchor = null; state.focus = null; }
  if (view.kind === "folder") {
    // Opening a folder opens the path to it in the tree, so it can be seen where it is.
    const parts = (view.relative || "").split("/").filter(Boolean);
    for (let depth = 0; depth < parts.length; depth++) {
      state.expanded.add(`${view.root}:${parts.slice(0, depth).join("/")}`);
    }
    state.expanded.add(`${view.root}:`);
    remember({ expanded: [...state.expanded] });
  }
  if (view.kind !== "settings") remember({ view });
  if (view.kind === "cleanup") loadCleanup();
  if (view.kind === "duplicates") loadDuplicates();
  invalidate("tree", "list", "inspector", "header");
}

export function showModel(id) {
  const model = state.models.get(id);
  if (!model) return;
  if (model.root !== null && model.root !== undefined) {
    go({ kind: "folder", root: model.root, relative: model.relative });
  } else {
    go({ kind: "folder", root: "outside", relative: "" });
  }
  state.selection = new Set([`m${id}`]);
  state.anchor = state.focus = `m${id}`;
  invalidate("list", "inspector");
  requestAnimationFrame(() => {
    const row = document.querySelector(`[data-key="m${id}"]`);
    if (row) row.scrollIntoView({ block: "nearest" });
  });
}

// --- adding downloads ---------------------------------------------------------------------

// A pasted link is resolved, then placed: the page asks where it goes — unless smart
// placement is on, when the server files it by what it is and asks only when unsure.
export async function addSource(text) {
  const source = String(text || "").trim();
  if (!source) return false;
  // Without a library there is nothing to choose between: everything goes to the one
  // downloads folder, as it always has.
  if (state.settings.smart_placement || !state.settings.library_root) return addFiled(source);
  const closing = toast("Resolving the link…", { timeout: 0 });
  let resolved;
  try {
    resolved = await post("/api/resolve", { source });
  } catch (error) {
    closing();
    toastError(error);
    return false;
  }
  closing();
  return placeResolved(resolved);
}

// Ask where a resolved link goes, and queue it there. A system folder dialog cancelled is
// not the end of the question — the add dialog comes back, with its answer still to give.
export async function placeResolved(resolved) {
  for (;;) {
    const answer = await addDialog(resolved);
    if (!answer) {
      del(`/api/resolve/${encodeURIComponent(resolved.token)}`).catch(() => {});
      return false;
    }
    let body;
    try {
      body = answer.anywhere
        ? await post(`/api/resolve/${encodeURIComponent(resolved.token)}/queue-anywhere`, {
          files: answer.files, keep_structure: answer.keep_structure, remember: answer.remember,
        })
        : await post(`/api/resolve/${encodeURIComponent(resolved.token)}/queue`, {
          files: answer.files, root: answer.root, folder: answer.relative,
          keep_structure: answer.keep_structure, remember: answer.remember,
        });
    } catch (error) {
      toastError(error);
      return false;
    }
    if (body.cancelled) continue;
    body.tasks.forEach(upsertTask);
    const where = answer.anywhere ? body.folder : folderLabel(answer.root, answer.relative);
    const busy = body.queued.length ? ` — ${plural(body.queued.length, "file")} already downloading` : "";
    const kind = body.remembered ? ` — every ${kindLabel(body.remembered)} goes here now` : "";
    toast(body.tasks.length ? `Added ${plural(body.tasks.length, "file")} → ${where}${busy}${kind}`
      : `Nothing added${busy}`, {
      level: body.tasks.length ? "ok" : "info",
      actions: body.tasks.length ? [{ label: "Show", run: () => go({ kind: "downloads" }) }] : [],
    });
    if (body.remembered) loadSettings();
    invalidate("tree", "list", "status", "header");
    return body.tasks.length > 0;
  }
}

// Smart placement: queued at once, filed by what the files turn out to be.
async function addFiled(source) {
  const closing = toast("Resolving the link…", { timeout: 0 });
  try {
    const { tasks, skipped } = await post("/api/tasks", { source });
    tasks.forEach(upsertTask);
    closing();
    const extra = skipped && skipped.length ? ` — ${skipped.length} already in your library` : "";
    toast(`Added ${plural(tasks.length, "file")}${extra}`, {
      level: "ok",
      actions: [{ label: "Show", run: () => go({ kind: "downloads" }) }],
    });
    invalidate("tree", "list", "status", "header");
    return true;
  } catch (error) {
    closing();
    if (error.status === 409 && /library/.test(error.message)) {
      toast(error.message, { level: "info" });
    } else {
      toastError(error);
    }
    return false;
  }
}

// --- downloads ------------------------------------------------------------------------------

export async function taskCommand(id, command) {
  try {
    if (command === "remove") await del(`/api/tasks/${id}`);
    else if (command === "confirm") await post(`/api/tasks/${id}/confirm`, {});
    else await post(`/api/tasks/${id}/${command}`);
  } catch (error) { toastError(error); }
}

export async function startAll() {
  try {
    const { released } = await post("/api/tasks/start-all");
    toast(`Started ${plural(released, "download")}`);
    await loadTasks();
  } catch (error) { toastError(error); }
}

export async function pauseAll() {
  try {
    const { paused } = await post("/api/tasks/pause-all");
    toast(`Paused ${plural(paused, "download")}`);
  } catch (error) { toastError(error); }
}

export async function clearFinished() {
  try {
    const { removed } = await post("/api/tasks/clear");
    toast(`${plural(removed, "finished download")} moved to the history`, {
      actions: [{ label: "Open history", run: () => go({ kind: "history" }) }],
    });
    await loadTasks();
  } catch (error) { toastError(error); }
}

// "Where does this go?" for a download the classifier would not file on its own.
export async function placeTask(id) {
  const task = state.tasks.get(id);
  if (!task) return;
  let body;
  try { body = await get(`/api/tasks/${id}/folders`); }
  catch (error) { toastError(error); return; }
  if (!body.root) {
    toast("No library folder is set, so everything goes to one folder — add one in Settings.");
    return;
  }
  const chosen = await pickFolder({
    title: "Where does this go?",
    subtitle: `${esc(task.filename || task.source)} → a folder of the library`,
    folders: body.folders,
    created: "created when the file lands",
  });
  if (!chosen) return;
  const label = folderLabel(chosen.root, chosen.relative);
  try {
    await post(`/api/tasks/${id}/confirm`, {
      root: chosen.root, folder: chosen.relative, remember: chosen.remember && !!chosen.kind,
    });
    toast(chosen.remember && chosen.kind
      ? `Filing into ${label} — every ${kindLabel(chosen.kind)} goes here now`
      : `Filing into ${label}`);
    if (chosen.remember && chosen.kind) loadSettings();
  } catch (error) { toastError(error); }
}

export async function saveTaskNote(id, text) {
  const { note } = await post(`/api/tasks/${id}/note`, { note: text });
  const task = state.tasks.get(id);
  if (task) { task.note = note; invalidate("list"); }
  return note;
}

export async function deleteTaskFiles(id) {
  const task = state.tasks.get(id);
  if (!task) return;
  let found;
  try { found = await get(`/api/tasks/${id}/files`); }
  catch (error) { toastError(error); return; }
  if (!found.files.length) {
    toast("Nothing of this download is on the disk — Remove takes it off the list");
    return;
  }
  const answer = await filesDialog({
    title: "Delete from disk",
    intro: `${esc(task.filename || task.source)} and everything named after it:`,
    groups: [{ files: found.files, modelFirst: !!found.model }],
    warning: "Deleted for good — nothing here goes to a recycle bin.",
  });
  if (!answer) return;
  try {
    const result = await del(`/api/tasks/${id}/files`);
    reportDeleted(result);
    await loadTasks();
  } catch (error) { toastError(error); }
}

// --- the history --------------------------------------------------------------------------

// "Download it again" from the history: the same question as any download — where — with
// the folder it was in first.
export async function redownload(taskId) {
  let resolved;
  try { resolved = await post(`/api/history/${taskId}/again`); }
  catch (error) { toastError(error); return; }
  await placeResolved(resolved);
}

// A missing model downloaded again, into the library entry it left: its history and its
// note stay with it, whichever folder it goes to now.
export async function downloadAgain(modelId) {
  let resolved;
  try { resolved = await post(`/api/models/${modelId}/again`); }
  catch (error) { toastError(error); return; }
  await placeResolved(resolved);
}

// Where a missing model could be downloaded from, looked for by what the library kept of
// it: Civitai by its hash, HuggingFace by its name and size.
export async function findOnline(id) {
  const model = state.models.get(id);
  if (!model) return;
  const closing = toast(`Looking for ${model.filename} on Civitai and HuggingFace…`, { timeout: 0 });
  let answer;
  try { answer = await post(`/api/models/${id}/find-online`); }
  catch (error) { closing(); toastError(error); return; }
  closing();
  const index = await foundDialog(model, answer);
  if (index === null) return;
  let resolved;
  try { resolved = await post(`/api/found/${encodeURIComponent(answer.token)}/${index}`); }
  catch (error) { toastError(error); return; }
  await placeResolved(resolved);
}

function foundDialog(model, answer) {
  return new Promise((resolve) => {
    const proof = (hit) => (hit.proven ? `<span class="ok">The same file — its SHA256 matches</span>`
      : hit.source === "civitai" ? `<span class="warn">The same quick hash and size — very likely the same file</span>`
        : hit.same_name ? `<span class="warn">The same name and size — not proven by a hash</span>`
          : `<span class="warn">The same size under another name — not proven by a hash</span>`);
    const rows = answer.found.map((hit, index) => `
      <div class="found-row">
        <div class="found-main">
          <div class="found-title"><span class="chip quiet">${esc(hit.host)}</span><span>${esc(hit.title)}</span></div>
          ${hit.detail ? `<div class="small muted mono">${esc(hit.detail)}</div>` : ""}
          <div class="small">${proof(hit)}${hit.size ? ` · ${fmtBytes(hit.size)}` : ""}</div>
        </div>
        <div class="found-side">
          ${hit.page ? `<a class="button" href="${esc(hit.page)}" target="_blank" rel="noopener noreferrer">${icon("external")}Page</a>` : ""}
          <button class="${index === 0 ? "primary" : ""}" data-pick="${index}">${icon("download")}Download from here…</button>
        </div>
      </div>`).join("");
    const how = answer.searched.sha256 ? "by its hash or by its name" : "by its name and size";
    const none = `<div class="dialog-message">Neither Civitai nor HuggingFace has a file like <b>${esc(model.filename)}</b> — looked for ${how}.
      A file renamed on its way here is found by nothing but a person.</div>`;
    const problems = answer.problems.length
      ? `<div class="warn small" style="margin-top:10px">${answer.problems.map((p) => esc(p)).join("<br>")}</div>` : "";
    const handle = modal({
      title: answer.found.length ? "Found online" : "Not found online",
      wide: true,
      body: (answer.found.length
        ? `<div class="small muted" style="margin-bottom:10px">Where ${esc(model.filename)} could be downloaded from. It comes back into the same library entry, with its history and its note.</div>${rows}`
        : none) + problems,
      footer: `<span class="small muted">Look for yourself:</span>
        <a class="button" href="${esc(answer.search.civitai)}" target="_blank" rel="noopener noreferrer">${icon("search")}Civitai</a>
        <a class="button" href="${esc(answer.search.huggingface)}" target="_blank" rel="noopener noreferrer">${icon("search")}HuggingFace</a>
        <span class="grow"></span><button data-role="close">Close</button>`,
      onClose: (value) => resolve(value ?? null),
    });
    handle.dialog.querySelectorAll("[data-pick]").forEach((button) => {
      button.onclick = () => handle.close(Number(button.dataset.pick));
    });
    handle.dialog.querySelector('[data-role="close"]').onclick = () => handle.close(null);
  });
}

// Several at once, each back into the folder it was in — asked once, for all of them.
export async function downloadAllAgain(ids) {
  const models = [].concat(ids).map((id) => state.models.get(id))
    .filter((m) => m && m.state === "missing" && m.source);
  if (!models.length) { toast("None of these has a link to download it from"); return; }
  const total = models.reduce((sum, m) => sum + (m.size || 0), 0);
  const roots = [...new Set(models.map((m) => m.root).filter((r) => r !== null && r !== undefined))]
    .map((index) => state.roots[index]).filter(Boolean);
  const free = roots.filter((r) => r.free != null).map((r) => `${fmtBytes(r.free)} free on ${esc(r.name)}`).join(", ");
  const room = Math.max(...roots.map((r) => r.free ?? Infinity), 0);
  const ok = await confirmDialog({
    title: models.length === 1 ? "Download again" : `Download ${models.length} models again`,
    message: `${models.length === 1 ? esc(models[0].filename) : plural(models.length, "model")} — each back into the folder it was in,`
      + ` with its history and its note. <b>${fmtBytes(total)}</b> to download${free ? ` · ${free}` : ""}.`
      + (total > room ? `<div class="warn small" style="margin-top:8px">More than there is room for: a download that does not fit waits with an error instead of filling the disk.</div>` : ""),
    confirm: "Download",
  });
  if (!ok) return;
  try {
    const { tasks, failed } = await post("/api/models/redownload", { ids: models.map((m) => m.id) });
    tasks.forEach(upsertTask);
    toast(`Queued ${plural(tasks.length, "model")} again`, {
      level: tasks.length ? "ok" : "info", actions: [{ label: "Show", run: () => go({ kind: "downloads" }) }],
    });
    if (failed.length) {
      toast(failed.map((f) => `${state.models.get(f.id)?.filename || f.id}: ${f.error}`).join("; "), { level: "error" });
    }
    invalidate("tree", "list", "status");
  } catch (error) { toastError(error); }
}

export async function removeFromHistory(taskIds) {
  const ids = [].concat(taskIds);
  const ok = await confirmDialog({
    title: "Remove from history",
    message: ids.length === 1
      ? "The download is taken out of the history. The model itself, if it is still in your library, stays where it is."
      : `${plural(ids.length, "download")} are taken out of the history. The models themselves stay where they are.`,
    confirm: "Remove",
  });
  if (!ok) return;
  for (const id of ids) {
    try { await del(`/api/history/${id}`); state.tasks.delete(id); }
    catch (error) { toastError(error); }
  }
  state.selection = new Set();
  invalidate("list", "inspector");
}

// --- models ------------------------------------------------------------------------------

export async function renameModel(id, name) {
  try {
    const result = await post(`/api/models/${id}/rename`, { name });
    if (result.unchanged) return result;
    if (result.failed.length) {
      toast(`Renamed to ${result.filename}, but ${plural(result.failed.length, "file")} kept the old name: `
        + result.failed.map((f) => `${baseName(f.path)} (${f.reason})`).join("; "), { level: "error" });
    } else {
      toast(`Renamed to ${result.filename}${result.renamed ? ` with ${plural(result.renamed, "file")} named after it` : ""}`, { level: "ok" });
    }
    return result;
  } catch (error) { toastError(error); return null; }
}

export async function saveModelNote(id, text) {
  const result = await post(`/api/models/${id}/note`, { note: text });
  const model = state.models.get(id);
  if (model) model.note = result.note;
  if (result.record_written) {
    toast("Note saved — a .json record was written to hold it");
  }
  invalidate("list");
  return result.note;
}

export async function revealModel(id) {
  try { await post(`/api/models/${id}/reveal`); } catch (error) { toastError(error); }
}

export async function revealFolder(root, relative) {
  try { await post("/api/folders/reveal", { root, relative }); } catch (error) { toastError(error); }
}

export async function moveModelsDialog(ids) {
  ids = [].concat(ids).filter((id) => state.models.get(id)?.state === "present");
  if (!ids.length) return;
  let body;
  try { body = await get(`/api/models/${ids[0]}/folders`); }
  catch (error) { toastError(error); return; }
  const first = state.models.get(ids[0]);
  const chosen = await pickFolder({
    title: ids.length > 1 ? `Move ${ids.length} models` : "Move to",
    subtitle: ids.length > 1
      ? `${plural(ids.length, "model")} and everything named after them`
      : `${esc(first.filename)} and everything named after it`,
    folders: body.folders,
    anywhere: true,
  });
  if (!chosen) return;
  if (chosen.anywhere) { await moveAnywhere(ids, chosen.remember); return; }
  await moveModels(ids, chosen.root, chosen.relative, { remember: chosen.remember });
}

export async function moveModels(ids, root, relative, { remember: rememberIt = false, undo = true } = {}) {
  ids = [].concat(ids);
  const before = new Map(ids.map((id) => {
    const model = state.models.get(id);
    return [id, model ? { root: model.root, relative: model.relative } : null];
  }));
  if (ids.every((id) => {
    const at = before.get(id);
    return at && at.root === root && at.relative === relative;
  })) return;
  const label = folderLabel(root, relative);
  const closing = ids.length > 1 ? toast(`Moving ${plural(ids.length, "model")} to ${label}…`, { timeout: 0 }) : null;
  let body;
  try {
    body = ids.length === 1
      ? { results: [{ id: ids[0], ...(await post(`/api/models/${ids[0]}/move`, { root, folder: relative, remember: rememberIt })) }] }
      : await post("/api/models/move", { ids, root, folder: relative, remember: rememberIt });
  } catch (error) { closing && closing(); toastError(error); return; }
  closing && closing();
  // One model's reply is the move itself; several come back as a list with the mapping
  // beside it. Either way the answer about the mapping is the server's.
  const remembered = body.remembered ?? body.results?.[0]?.remembered ?? null;
  reportMoves(body.results, label, remembered, undo ? before : null);
  if (remembered) loadSettings();
}

async function moveAnywhere(ids, rememberIt) {
  let body;
  try {
    body = ids.length === 1
      ? { results: [{ id: ids[0], ...(await post(`/api/models/${ids[0]}/move-anywhere`, { remember: rememberIt })) }] }
      : await post("/api/models/move-anywhere", { ids, remember: rememberIt });
  } catch (error) { toastError(error); return; }
  if (body.cancelled || body.results?.[0]?.cancelled) return;
  const remembered = body.remembered ?? body.results?.[0]?.remembered ?? null;
  reportMoves(body.results, body.folder || body.results?.[0]?.folder || "the chosen folder", remembered, null);
  if (remembered) loadSettings();
}

function reportMoves(results, label, remembered, before) {
  const moved = results.filter((r) => r.ok && !r.unchanged);
  const stopped = results.find((r) => r.stopped);
  const failed = results.filter((r) => !r.ok && !r.stopped);
  const stranded = results.flatMap((r) => r.failed || []);
  if (stopped) toast("The move was stopped — the file has not been touched");
  if (failed.length) {
    toast(`${plural(failed.length, "model")} could not be moved: ${failed.map((f) => f.error).join("; ")}`, { level: "error" });
  }
  if (stranded.length) {
    toast(`Moved, but ${plural(stranded.length, "file")} stayed behind: `
      + stranded.map((f) => `${baseName(f.path)} (${f.reason})`).join("; "), { level: "error" });
  }
  if (!moved.length) return;
  const actions = [];
  if (before) {
    actions.push({
      label: "Undo",
      run: async () => {
        // Back where each one was, which for a selection gathered from several folders is
        // several moves.
        const groups = new Map();
        for (const result of moved) {
          const at = before.get(result.id);
          if (!at || at.root === null || at.root === undefined) continue;
          const key = `${at.root}:${at.relative}`;
          if (!groups.has(key)) groups.set(key, { ...at, ids: [] });
          groups.get(key).ids.push(result.id);
        }
        for (const group of groups.values()) {
          await moveModels(group.ids, group.root, group.relative, { undo: false });
        }
      },
    });
  }
  const kind = remembered ? ` — every ${kindLabel(remembered)} goes here now` : "";
  toast(`Moved ${moved.length === 1 ? baseName(moved[0].dest) : plural(moved.length, "model")} to ${label}${kind}`,
    { level: "ok", actions });
}

export async function deleteModels(ids, { intro = "", after = null, titleOf = null } = {}) {
  ids = [].concat(ids);
  let groups;
  try {
    if (ids.length === 1) {
      const found = await get(`/api/models/${ids[0]}/files`);
      groups = [{ id: ids[0], files: found.files, others: found.others }];
    } else {
      ({ groups } = await post("/api/models/files", { ids }));
    }
  } catch (error) { toastError(error); return; }
  groups = groups.filter((g) => g.files.length);
  if (!groups.length) { toast("Nothing of these models is on the disk any more"); return; }
  const single = groups.length === 1 ? state.models.get(groups[0].id) : null;
  // A name of a file that has others: deleting it frees nothing, and the others keep working.
  const shared = groups.filter((g) => g.others && g.others.count);
  const linkedNote = shared.length
    ? `<div class="banner banner-info" style="margin-top:10px"><div class="banner-text">${shared.map((g) => {
      const name = state.models.get(g.id)?.filename || "";
      const where = g.others.paths.slice(0, 3).map((p) => `<div class="mono small">${esc(p)}</div>`).join("");
      return `<b>${esc(name)}</b> is one file with ${plural(g.others.count, "other name")} — deleting it here frees no room while ${g.others.count === 1 ? "that one is" : "they are"} left:${where}`;
    }).join("")}</div></div>`
    : "";
  const answer = await filesDialog({
    title: "Delete from disk",
    intro: (intro || (single
      ? `${esc(single.filename)} and everything named after it:`
      : `${plural(groups.length, "model")} and everything named after them:`)) + linkedNote,
    groups: groups.map((g) => {
      const model = state.models.get(g.id);
      return {
        title: groups.length > 1 && model ? (titleOf ? titleOf(model) : model.filename) : "",
        files: g.files, modelFirst: true,
      };
    }),
    warning: "Deleted for good — nothing here goes to a recycle bin. The history keeps a line saying it was deleted.",
  });
  if (!answer) return;
  try {
    if (ids.length === 1) {
      const result = await del(`/api/models/${ids[0]}/files`);
      removeModel(ids[0]);
      reportDeleted(result);
    } else {
      const { results } = await post("/api/models/delete", { ids: groups.map((g) => g.id) });
      const done = results.filter((r) => r.ok);
      done.forEach((r) => removeModel(r.id));
      const failed = results.filter((r) => !r.ok);
      toast(`Deleted ${plural(done.length, "model")}`, { level: done.length ? "ok" : "info" });
      if (failed.length) toast(failed.map((f) => f.error).join("; "), { level: "error" });
    }
    invalidate("tree", "list", "inspector");
  } catch (error) { toastError(error); }
  if (after) after();
}

function reportDeleted(result) {
  if (result.failed && result.failed.length) {
    toast(`The model is gone, but ${plural(result.failed.length, "file")} stayed behind: `
      + result.failed.map((f) => `${baseName(f.path)} (${f.reason})`).join("; "), { level: "error" });
    return;
  }
  const count = result.deleted.length;
  toast(`Deleted ${plural(count, "file")}${result.missing ? " — the model itself was already gone" : ""}`, { level: "ok" });
}

export async function forgetModels(ids) {
  ids = [].concat(ids).filter((id) => state.models.get(id)?.state === "missing");
  if (!ids.length) return;
  const groups = [];
  for (const id of ids) {
    try {
      const found = await get(`/api/models/${id}/leftovers`);
      if (found.files.length) groups.push({ title: state.models.get(id)?.filename, files: found.files });
    } catch { /* one that cannot be read has nothing to offer */ }
  }
  // A note on a download survives in the history, which keeps its own copy with the line
  // for that download. A note on a model found on disk has nowhere else to be.
  const lost = ids.filter((id) => {
    const model = state.models.get(id);
    return model?.note && model.origin !== "downloaded";
  }).length;
  const names = ids.length === 1 ? esc(state.models.get(ids[0]).filename) : plural(ids.length, "missing model");
  const intro = `${names} will be taken out of the library. A download keeps its line in the history, with its note.`
    + (lost ? ` <span class="warn">${lost === 1 && ids.length === 1 ? "Its note" : plural(lost, "note")} will be lost — ${lost === 1 ? "it was" : "they were"} not downloaded here, so there is no history to keep ${lost === 1 ? "it" : "them"} in.</span>` : "");
  let cleanup = false;
  if (groups.length) {
    const answer = await filesDialog({
      title: ids.length === 1 ? "Forget this model" : `Forget ${ids.length} models`,
      intro,
      groups,
      confirm: "Forget",
      checkbox: { label: "Also delete these leftover files for good", checked: true },
    });
    if (!answer) return;
    cleanup = !!answer.checked;
  } else {
    const ok = await confirmDialog({ title: "Forget", message: intro, confirm: "Forget", danger: true });
    if (!ok) return;
  }
  try {
    if (ids.length === 1) {
      await post(`/api/models/${ids[0]}/forget`, { cleanup });
      removeModel(ids[0]);
    } else {
      const { results } = await post("/api/models/forget", { ids, cleanup });
      results.filter((r) => r.ok).forEach((r) => removeModel(r.id));
    }
    toast(`Forgot ${plural(ids.length, "model")}${cleanup ? " and deleted what it left behind" : ""}`, { level: "ok" });
    invalidate("tree", "list", "inspector");
  } catch (error) { toastError(error); }
}

export async function locateModel(id) {
  try {
    const answer = await post(`/api/models/${id}/locate`);
    if (answer.cancelled) return;
    if (answer.confirm) {
      const c = answer.confirm;
      const ok = await confirmDialog({
        title: "A file of another size",
        message: `<b>${esc(baseName(c.path))}</b> is ${fmtBytes(c.actual)}; the model was ${fmtBytes(c.expected)}. `
          + "It may be a different file with the same purpose, or another version of it.",
        confirm: "Link it anyway",
      });
      if (!ok) return;
      await post(`/api/models/${id}/locate/confirm`, { token: c.token });
    }
    toast("Found it — the model is back in the library", { level: "ok" });
  } catch (error) { toastError(error); }
}

export async function relinkModel(id, candidate) {
  try {
    await post(`/api/models/${id}/relink`, { candidate });
    toast("Linked to the file found on disk", { level: "ok" });
  } catch (error) { toastError(error); }
}

export async function bringBack(id) {
  try {
    const result = await post(`/api/models/${id}/bring-back`);
    if (result.failed.length) {
      toast(`${plural(result.failed.length, "file")} could not be brought over: `
        + result.failed.map((f) => `${baseName(f.path)} (${f.reason})`).join("; "), { level: "error" });
    } else {
      toast(`Brought ${plural(result.moved, "file")} over to the model`, { level: "ok" });
    }
  } catch (error) { toastError(error); }
}

export async function identify(ids) {
  ids = [].concat(ids);
  const models = ids.map((id) => state.models.get(id)).filter(Boolean);
  const total = models.reduce((sum, m) => sum + (m.sha256 ? 0 : m.size || 0), 0);
  if (models.length > 1 || total > 8 * 1024 ** 3) {
    const ok = await confirmDialog({
      title: "Identify",
      message: `Each file is looked up on Civitai by its hash and on HuggingFace by its name, the answer proven`
        + ` by the whole file's SHA256. A big file is read in full only when one of them has something it could be;`
        + ` at most ${fmtBytes(total)} to read${models.length > 1 ? ` across ${plural(models.length, "model")}` : ""}.`
        + ` It can be stopped from the status bar.`,
      confirm: "Start",
    });
    if (!ok) return;
  }
  try {
    const { queued } = await post("/api/models/identify", { ids });
    if (!queued) toast("Those are already being identified");
  } catch (error) { toastError(error); }
}

export async function verify(ids) {
  try { await post("/api/models/verify", { ids: [].concat(ids) }); } catch (error) { toastError(error); }
}

export async function hashModels(ids) {
  try { await post("/api/models/hash", { ids: [].concat(ids) }); } catch (error) { toastError(error); }
}

export async function stopJobs() {
  try { await post("/api/jobs/stop"); } catch (error) { toastError(error); }
}

export async function checkUpdates(ids) {
  ids = [].concat(ids).filter((id) => {
    const m = state.models.get(id);
    return m && (m.provider === "civitai" || m.provider === "huggingface");
  });
  if (!ids.length) {
    toast("Nothing here came from Civitai or HuggingFace, so there is nothing to check");
    return;
  }
  const closing = toast(`Checking ${plural(ids.length, "model")} for newer versions…`, { timeout: 0 });
  try {
    const result = await post("/api/models/check-updates", { ids });
    closing();
    toast(result.updates
      ? `${plural(result.updates, "model")} ${result.updates === 1 ? "has" : "have"} a newer version`
      : `Checked ${plural(result.checked, "model")} — nothing newer`,
    { level: result.updates ? "ok" : "info" });
    if (result.failed) toast(`${plural(result.failed, "check")} could not be made`, { level: "error" });
  } catch (error) { closing(); toastError(error); }
}

export async function downloadUpdate(model) {
  const update = model.update || {};
  if (!update.version_id) { window.open(update.page, "_blank", "noopener"); return; }
  const host = (model.host || "civitai.com").replace(/^www\./, "");
  // The primary file of the new version: the same one a download button on its page
  // would give, rather than every quantisation the version carries.
  await addSource(`https://${host}/api/download/models/${update.version_id}`);
}

export function modelLabel(model) {
  return model.title && model.title !== model.filename ? model.title : model.filename;
}

// --- the library's folders ------------------------------------------------------------------

export async function newFolder(root, parent) {
  const name = await promptDialog({
    title: "New folder",
    label: `Inside ${folderLabel(root, parent)}`,
    confirm: "Create",
  });
  if (!name) return;
  try {
    await post("/api/folders", { root, parent, name });
    const relative = parent ? `${parent}/${name}` : name;
    state.expanded.add(`${root}:${parent}`);
    go({ kind: "folder", root, relative });
  } catch (error) { toastError(error); }
}

export async function hideFolder(root, relative) {
  const ok = await confirmDialog({
    title: "Hide this folder",
    message: `<b>${esc(folderLabel(root, relative))}</b> and everything under it is left out of the library.`
      + " Nothing on disk is touched; it can be shown again from Settings.",
    confirm: "Hide it",
  });
  if (!ok) return;
  try {
    const { hidden } = await post("/api/folders/hide", { root, relative });
    state.hidden = hidden;
    if (state.view.kind === "folder" && state.view.root === root
        && (state.view.relative === relative || state.view.relative.startsWith(`${relative}/`))) {
      go({ kind: "folder", root, relative: relative.split("/").slice(0, -1).join("/") });
    }
    toast("Hidden from the library", { actions: [{ label: "Undo", run: () => unhideFolder(hidden.length - 1) }] });
    await loadLibrary();
  } catch (error) { toastError(error); }
}

export async function unhideFolder(index) {
  try {
    const { hidden } = await post("/api/folders/unhide", { index });
    state.hidden = hidden;
    await loadLibrary();
    invalidate("list");
  } catch (error) { toastError(error); }
}

export async function addRoot() {
  try {
    const answer = await post("/api/roots/add");
    if (answer.cancelled) return;
    if (answer.unchanged) { toast("That folder is already in the library"); return; }
    toast("Added to the library", { level: "ok" });
    await loadSettings();
    await loadLibrary();
    // A new folder goes to the end of the list, so nothing already open moves — unless it
    // became the main one, which only happens when there was none.
    if (state.roots.length === 1) afterRootsChanged(state.view.kind === "settings");
  } catch (error) { toastError(error); }
}

// The tree names its folders by their place in the list of library folders, so a change to
// that list leaves the open folders and the current view pointing at the wrong ones. Both
// start again from the main folder.
function afterRootsChanged(stay) {
  state.expanded = new Set(state.roots.map((r) => `${r.index}:`));
  remember({ expanded: [...state.expanded] });
  if (!stay) go({ kind: "folder", root: 0, relative: "" });
  else if (state.prefs.view && state.prefs.view.kind === "folder") {
    remember({ view: { kind: "folder", root: 0, relative: "" } });
  }
}

export async function removeRoot(index, { stay = false } = {}) {
  const root = state.roots[index];
  if (!root) return;
  const primary = index === 0;
  const next = state.roots[1];
  const ok = await confirmDialog({
    title: "Take this folder off the list",
    message: `<b>${esc(root.path)}</b> is no longer read into the library. Nothing on disk is touched.`
      + (primary
        ? next ? ` Downloads will be filed into <b>${esc(next.path)}</b> from now on.`
               : " With no library folder left, downloads go to the plain downloads folder."
        : ""),
    confirm: "Remove",
  });
  if (!ok) return;
  try {
    await post(`/api/roots/${index}/remove`);
    await loadSettings();
    await loadLibrary();
    afterRootsChanged(stay);
  } catch (error) { toastError(error); }
}

export async function makePrimary(index, { stay = false } = {}) {
  const root = state.roots[index];
  if (!root) return;
  const ok = await confirmDialog({
    title: "File downloads here",
    message: `New downloads will be sorted into <b>${esc(root.path)}</b>, using the folders it already has.`,
    confirm: "Make it the main folder",
  });
  if (!ok) return;
  try {
    await post(`/api/roots/${index}/primary`);
    await loadSettings();
    await loadLibrary();
    afterRootsChanged(stay);
  } catch (error) { toastError(error); }
}

// --- cleanup and duplicates ---------------------------------------------------------------

export async function loadCleanup() {
  state.cleanup = { loading: true, items: [] };
  invalidate("list");
  try {
    const body = await get("/api/cleanup");
    state.cleanup = { loading: false, ...body };
  } catch (error) {
    state.cleanup = { loading: false, items: [], error: error.message };
  }
  invalidate("list", "inspector", "tree");
}

export async function deleteCleanup(ids) {
  const items = (state.cleanup?.items || []).filter((i) => ids.includes(i.id));
  if (!items.length) return;
  const answer = await filesDialog({
    title: "Delete leftovers",
    intro: `${plural(items.length, "item")} that belong to no model in the library:`,
    groups: items.map((i) => ({ title: `${i.name} — ${i.why}`, files: i.files })),
    warning: "Deleted for good — nothing here goes to a recycle bin.",
  });
  if (!answer) return;
  try {
    // Named by the scan they were shown from: another window looking again renumbers the
    // items, and this list's numbers would then mean other files.
    const result = await post("/api/cleanup/delete", { ids: items.map((i) => i.id), token: state.cleanup.token });
    reportDeleted(result);
    state.selection = new Set();
    await loadCleanup();
  } catch (error) {
    toastError(error);
    if (error.status === 409) loadCleanup();
  }
}

// Looked at again whenever the view opens, and after anything that changes the answer — a
// hash worked out, a copy deleted. What is on screen stays there until the new answer comes:
// a list that blinks to "Looking…" every time a hash lands is one nobody can read.
let duplicatesAsked = 0;
export async function loadDuplicates() {
  const asked = ++duplicatesAsked;
  if (!state.duplicates || state.duplicates.error) {
    state.duplicates = { loading: true, groups: [] };
    invalidate("list");
  }
  let answer;
  try {
    answer = { loading: false, ...(await get("/api/duplicates")) };
  } catch (error) {
    answer = { loading: false, groups: [], error: error.message };
  }
  if (asked !== duplicatesAsked) return;
  state.duplicates = answer;
  invalidate("list", "tree", "inspector");
}

const reloadDuplicates = debounce(() => loadDuplicates(), 700);

// Whether something that just changed is a copy the Duplicates view is showing: its hash was
// worked out, or it went.
export function duplicatesTouched(models = [], removed = null) {
  if (state.view.kind !== "duplicates" || !state.duplicates?.groups) return;
  const listed = new Map(state.duplicates.groups.flatMap((g) => g.models.map((m) => [m.id, m])));
  const changed = models.some((m) => listed.has(m.id) && (listed.get(m.id).sha256 || null) !== (m.sha256 || null));
  if (changed || (removed !== null && listed.has(removed))) reloadDuplicates();
}

// Which copy of a set stays when the others are deleted: the one picked here, or the one the
// library suggests.
export function keeperOf(group) {
  const chosen = state.keepers.get(group.key);
  return group.models.some((m) => m.id === chosen) ? chosen : group.keep;
}

export function chooseKeeper(key, id) {
  state.keepers.set(key, id);
  invalidate("list");
}

// The copies of a set that could become names of the one kept: on its drive, which is as
// far as a hard link reaches, and not already the same file.
export function linkable(group) {
  const kept = group.models.find((m) => m.id === keeperOf(group));
  if (!kept || !kept.volume) return [];
  return group.models.filter((m) => m.id !== kept.id && m.volume === kept.volume && m.file_id !== kept.file_id);
}

const placeOf = (m) => (m.root === null || m.root === undefined ? m.folder : folderLabel(m.root, m.relative));

export async function linkCopies(group) {
  const kept = group.models.find((m) => m.id === keeperOf(group));
  const others = linkable(group);
  if (!kept || !others.length) return;
  const away = group.models.filter((m) => m.id !== kept.id && !others.includes(m) && m.file_id !== kept.file_id);
  const ok = await confirmDialog({
    title: "Link the copies",
    message: `${others.length === 1 ? "The copy" : "The copies"} in ${others.map((m) => `<b>${esc(placeOf(m))}</b>`).join(", ")}`
      + ` become other names of the one in <b>${esc(placeOf(kept))}</b>. Every path keeps working — nodes and workflows find`
      + ` the same files where they always did — and <b>${fmtBytes((group.size || 0) * others.length)}</b> is freed.`
      + `<div class="small muted" style="margin-top:8px">From then on it is one file under several names: deleting one of them`
      + ` frees nothing while another is left, and a program that saves by rewriting a file gives that name a copy of its own`
      + ` again. <i>Make separate copies</i> undoes it.</div>`
      + (away.length ? `<div class="small warn" style="margin-top:6px">${plural(away.length, "copy", "copies")} on another drive stay as they are — a link cannot reach there.</div>` : ""),
    confirm: "Link them",
  });
  if (!ok) return;
  try {
    const body = await post("/api/duplicates/link", { keep: kept.id, ids: others.map((m) => m.id) });
    const done = body.results.filter((r) => r.ok && !r.unchanged);
    const failed = body.results.filter((r) => !r.ok);
    if (done.length) toast(`Linked ${plural(done.length, "copy", "copies")} — ${fmtBytes(body.freed)} freed`, { level: "ok" });
    if (failed.length) {
      toast(failed.map((f) => `${state.models.get(f.id)?.filename || f.id}: ${f.error}`).join("; "), { level: "error" });
    }
  } catch (error) { toastError(error); }
  loadDuplicates();
}

// Give names of a shared file copies of their own again: the way back from linking. It
// takes the room again, so the room is shown before anything is copied.
export async function separateModels(ids, { intro = "" } = {}) {
  const models = [].concat(ids).map((id) => state.models.get(id)).filter(Boolean);
  if (!models.length) return;
  const total = models.reduce((sum, m) => sum + (m.size || 0), 0);
  const root = state.roots[models[0].root];
  const free = root && root.free != null ? root.free : null;
  const ok = await confirmDialog({
    title: models.length === 1 ? "Make a separate copy" : "Make separate copies",
    message: (intro || (models.length === 1
      ? `<b>${esc(models[0].filename)}</b> gets a copy of its own; its other names stay one file.`
      : `${plural(models.length, "name")} each get a copy of their own.`))
      + ` Every path keeps working. <b>${fmtBytes(total)}</b> more on the disk${free != null ? ` · ${fmtBytes(free)} free` : ""}.`
      + (free != null && total > free ? `<div class="warn small" style="margin-top:8px">More than there is room for.</div>` : ""),
    confirm: models.length === 1 ? "Make the copy" : "Make the copies",
  });
  if (!ok) return;
  let results;
  try {
    results = models.length === 1
      ? [{ id: models[0].id, ...(await post(`/api/models/${models[0].id}/separate`)) }]
      : (await post("/api/models/separate", { ids: models.map((m) => m.id) })).results;
  } catch (error) { toastError(error); results = []; }
  const done = results.filter((r) => r.ok && !r.unchanged);
  const failed = results.filter((r) => !r.ok && !r.stopped);
  if (done.length) toast(`Made ${plural(done.length, "separate copy", "separate copies")}`, { level: "ok" });
  if (results.some((r) => r.stopped)) toast("Stopped — that name is still one file with the others");
  if (failed.length) {
    toast(failed.map((f) => `${state.models.get(f.id)?.filename || f.id}: ${f.error}`).join("; "), { level: "error" });
  }
  if (state.view.kind === "duplicates") loadDuplicates();
}

export async function duplicateCommand(what, key) {
  const group = [...(state.duplicates?.groups || []), ...(state.duplicates?.linked || [])]
    .find((g) => g.key === key);
  if (!group) return;
  if (what === "link") return linkCopies(group);
  if (what === "separate") {
    // One name keeps the file; every other name gets a copy of its own.
    const [first, ...rest] = group.models;
    return separateModels(rest.map((m) => m.id), {
      intro: `The file stays under <b>${esc(placeOf(first))}</b>; ${plural(rest.length, "other name")} each get a copy of their own.`,
    });
  }
  if (what === "confirm") {
    await hashModels(group.to_hash);
    toast(`Hashing ${plural(group.to_hash.length, "file")} — ${fmtBytes(group.to_read)} to read. The list updates when it is done.`);
    return;
  }
  if (what === "dedupe") {
    const keep = keeperOf(group);
    const kept = state.models.get(keep) || group.models.find((m) => m.id === keep);
    const others = group.models.map((m) => m.id).filter((id) => id !== keep);
    // The copies share a name more often than not, so each is told apart by where it is.
    const where = (m) => (m.root === null || m.root === undefined ? m.folder : folderLabel(m.root, m.relative));
    await deleteModels(others, {
      intro: `The copy in <b>${esc(kept ? where(kept) : "")}</b> stays. These go, with everything named after them:`,
      titleOf: where,
      after: () => loadDuplicates(),
    });
  }
}

// --- settings --------------------------------------------------------------------------------

export async function saveSettings(patch) {
  const { settings } = await put("/api/settings", patch);
  state.settings = settings;
  invalidate("list", "inspector");
  return settings;
}

export function afterModelChange(id) {
  state.details.delete(id);
  forgetPreviews({ kind: "model", id });
}
