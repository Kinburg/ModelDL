// Everything the page can do to a model, a download or the library, in one place — the
// buttons, the right-click menu, the keyboard and a drag onto the tree all end up here, so
// that each thing is done one way and says the same things when it is done.

import { del, get, post, put } from "./api.js";
import { baseName, esc, fmtBytes, kindLabel, plural } from "./util.js";
import {
  state, invalidate, upsertModel, removeModel, upsertTask, remember, folderLabel,
} from "./store.js";
import { toast, toastError } from "./toasts.js";
import { confirmDialog, filesDialog, modal, pickFolder, promptDialog } from "./dialogs.js";
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

export async function addSource(text) {
  const source = String(text || "").trim();
  if (!source) return false;
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
    subtitle: `${esc(task.filename || task.source)} → somewhere under ${esc(body.root)}`,
    folders: body.folders.map((f) => ({ ...f, root: 0 })),
    rootsAllowed: false,
    created: "created when the file lands",
  });
  if (!chosen) return;
  try {
    await post(`/api/tasks/${id}/confirm`, { folder: chosen.relative, remember: chosen.remember && !!chosen.kind });
    toast(chosen.remember && chosen.kind
      ? `Filing into ${chosen.relative} — every ${chosen.kind} goes here now`
      : `Filing into ${chosen.relative}`);
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

export async function redownload(taskId) {
  try {
    const { task } = await post(`/api/history/${taskId}/redownload`);
    upsertTask(task);
    toast(`${task.filename} is queued again`, {
      level: "ok", actions: [{ label: "Show", run: () => go({ kind: "downloads" }) }],
    });
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

export async function deleteModels(ids) {
  ids = [].concat(ids);
  let groups;
  try {
    if (ids.length === 1) {
      const found = await get(`/api/models/${ids[0]}/files`);
      groups = [{ id: ids[0], files: found.files }];
    } else {
      ({ groups } = await post("/api/models/files", { ids }));
    }
  } catch (error) { toastError(error); return; }
  groups = groups.filter((g) => g.files.length);
  if (!groups.length) { toast("Nothing of these models is on the disk any more"); return; }
  const single = groups.length === 1 ? state.models.get(groups[0].id) : null;
  const answer = await filesDialog({
    title: "Delete from disk",
    intro: single
      ? `${esc(single.filename)} and everything named after it:`
      : `${plural(groups.length, "model")} and everything named after them:`,
    groups: groups.map((g) => ({
      title: groups.length > 1 ? state.models.get(g.id)?.filename : "",
      files: g.files, modelFirst: true,
    })),
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
      title: "Identify on Civitai",
      message: `Each file is read in full to work out its SHA256, then looked up on Civitai.`
        + ` That is ${fmtBytes(total)} to read${models.length > 1 ? ` across ${plural(models.length, "model")}` : ""}`
        + ` — on a mechanical drive, a while. It can be stopped from the status bar.`,
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

export async function loadDuplicates() {
  state.duplicates = { loading: true, exact: [], possible: [] };
  invalidate("list");
  try {
    state.duplicates = { loading: false, ...(await get("/api/duplicates")) };
  } catch (error) {
    state.duplicates = { loading: false, exact: [], possible: [], error: error.message };
  }
  invalidate("list", "tree");
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
