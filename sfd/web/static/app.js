// The page: three panes over one library, kept current by an event stream.
//
// No framework and no build step — the whole thing is served by the same process that does
// the downloading, and has to keep working from inside a PyInstaller bundle where there is
// nothing to build with. The modules in ./js/ are plain ES modules the WebView loads as-is.

import { $, debounce, setBits } from "./js/util.js";
import { icon } from "./js/icons.js";
import {
  state, invalidate, onRender, upsertTask, upsertModel, removeModel, select, selectedModels,
  flushPrefs, remember,
} from "./js/store.js";
import * as act from "./js/actions.js";
import { toast } from "./js/toasts.js";
import { closeMenu, menuOpen } from "./js/menu.js";
import { closeTopDialog, dialogOpen } from "./js/dialogs.js";
import { openLightbox, forgetPreviews } from "./js/lightbox.js";
import { wireSidebar } from "./js/sidebar.js";
import { wireList, patchProgress, moveSelection, selectAll } from "./js/list.js";
import { patchInspectorProgress, startRename, flushNoteOnExit } from "./js/inspector.js";
import { wireStatus, refreshSpace } from "./js/statusbar.js";

// --- the header ---------------------------------------------------------------------------

function renderHeader() {
  const search = $("search");
  const placeholder = {
    folder: "Search the library",
    downloads: "Filter the downloads",
    history: "Search the history",
    missing: "Filter the missing",
    unidentified: "Filter these",
    duplicates: "Search the library",
    cleanup: "Filter the leftovers",
    settings: "Search the library",
  }[state.view.kind] || "Search";
  if (search.placeholder !== placeholder) search.placeholder = placeholder;
  $("open-settings").classList.toggle("on", state.view.kind === "settings");
}
onRender("header", renderHeader);

function wireHeader() {
  document.querySelectorAll("[data-icon]").forEach((node) => {
    node.insertAdjacentHTML("afterbegin", icon(node.dataset.icon));
  });
  const source = $("source");
  const add = async () => {
    if (!source.value.trim()) return;
    $("add").disabled = true;
    try { if (await act.addSource(source.value)) source.value = ""; }
    finally { $("add").disabled = false; }
  };
  $("add").onclick = add;
  source.addEventListener("keydown", (event) => { if (event.key === "Enter") add(); });
  // A copied block of text can hold the link and a paragraph around it, and the box is one
  // line: whatever the newlines are, an <input> drops them and glues the rest together into
  // something no provider can parse. The first line with anything on it is the one meant.
  source.addEventListener("paste", (event) => {
    const text = event.clipboardData && event.clipboardData.getData("text");
    if (!text || !text.includes("\n")) return;
    const line = text.split("\n").map((s) => s.trim()).find(Boolean) || "";
    event.preventDefault();
    if (!document.execCommand("insertText", false, line)) {
      source.setRangeText(line, source.selectionStart, source.selectionEnd, "end");
    }
  });

  const search = $("search");
  const apply = debounce(() => {
    state.search = search.value;
    if (state.view.kind === "settings") act.go({ kind: "folder", root: 0, relative: "" });
    invalidate("list");
  }, 120);
  search.addEventListener("input", apply);
  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && search.value) {
      event.stopPropagation();
      search.value = "";
      state.search = "";
      invalidate("list");
    }
    if (event.key === "ArrowDown") {
      // Handled here, all of it: the window's own arrow handling would move the selection
      // a second time once the focus had left the box.
      event.preventDefault();
      event.stopPropagation();
      $("rows")?.focus();
      moveSelection(1);
    }
  });
  $("open-settings").onclick = () => {
    if (state.view.kind === "settings") act.go(state.prefs.view || { kind: "folder", root: 0, relative: "" });
    else act.go({ kind: "settings" });
  };
}

// --- the panes ----------------------------------------------------------------------------

const LIMITS = { left: [170, 520], right: [260, 640] };
const DEFAULTS = { left: 250, right: 360 };

function applyWidths() {
  const wanted = {};
  for (const side of ["left", "right"]) {
    const width = Number(state.prefs[side]) || DEFAULTS[side];
    wanted[side] = Math.max(LIMITS[side][0], Math.min(LIMITS[side][1], width));
  }
  // A narrow window gives the side panes less rather than squeezing the list out of
  // existence: what was chosen is kept, and comes back when the window is wide again.
  const room = window.innerWidth - 10 - 340;
  const sides = wanted.left + wanted.right;
  if (sides > room) {
    const scale = Math.max(0, room) / sides;
    for (const side of ["left", "right"]) {
      wanted[side] = Math.max(LIMITS[side][0] * 0.8, Math.round(wanted[side] * scale));
    }
  }
  $("sidebar").style.width = `${wanted.left}px`;
  $("inspector").style.width = `${wanted.right}px`;
}
window.addEventListener("resize", debounce(applyWidths, 60));

function wireSplitters() {
  document.querySelectorAll(".splitter").forEach((handle) => {
    const side = handle.dataset.side;
    const pane = side === "left" ? $("sidebar") : $("inspector");
    handle.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      handle.setPointerCapture(event.pointerId);
      handle.classList.add("dragging");
      document.body.classList.add("resizing");
      const start = event.clientX;
      const width = pane.offsetWidth;
      const move = (e) => {
        const delta = side === "left" ? e.clientX - start : start - e.clientX;
        const other = side === "left" ? $("inspector").offsetWidth : $("sidebar").offsetWidth;
        // The list in the middle keeps room enough to be a list.
        const room = window.innerWidth - other - 340;
        const next = Math.max(LIMITS[side][0], Math.min(LIMITS[side][1], room, width + delta));
        pane.style.width = `${next}px`;
      };
      const up = () => {
        handle.removeEventListener("pointermove", move);
        handle.removeEventListener("pointerup", up);
        handle.classList.remove("dragging");
        document.body.classList.remove("resizing");
        remember({ [side]: pane.offsetWidth });
      };
      handle.addEventListener("pointermove", move);
      handle.addEventListener("pointerup", up);
    });
    handle.addEventListener("dblclick", () => {
      pane.style.width = `${DEFAULTS[side]}px`;
      remember({ [side]: DEFAULTS[side] });
    });
  });
}

// --- the keyboard -------------------------------------------------------------------------

function deleteSelection() {
  const keys = [...state.selection];
  if (!keys.length) return;
  const kinds = new Set(keys.map((k) => k[0]));
  if (kinds.has("m")) {
    const models = selectedModels();
    const present = models.filter((m) => m.state === "present").map((m) => m.id);
    const missing = models.filter((m) => m.state === "missing").map((m) => m.id);
    if (present.length) act.deleteModels(present);
    else if (missing.length) act.forgetModels(missing);
    return;
  }
  if (kinds.has("c")) { act.deleteCleanup(keys.filter((k) => k[0] === "c").map((k) => Number(k.slice(1)))); return; }
  const tasks = keys.filter((k) => k[0] === "t").map((k) => Number(k.slice(1)));
  if (state.view.kind === "history") act.removeFromHistory(tasks);
  else if (state.view.kind === "downloads") tasks.forEach((id) => act.taskCommand(id, "remove"));
}

function wireKeys() {
  window.addEventListener("keydown", (event) => {
    const active = document.activeElement;
    const typing = active && (/^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName) || active.isContentEditable);
    const mod = event.ctrlKey || event.metaKey;
    // A key something else already answered — a note saved with Ctrl+Enter, a dialog that
    // closed on Enter and took its field with it — is not the window's to answer again.
    if (event.defaultPrevented || !event.target.isConnected) return;
    if (event.key === "Escape") {
      if (menuOpen()) { closeMenu(); return; }
      if (closeTopDialog()) { event.preventDefault(); return; }
      if (!typing && state.selection.size) { select([]); return; }
      if (typing && active.blur) active.blur();
      return;
    }
    if (dialogOpen() || menuOpen()) return;
    if (mod && event.key.toLowerCase() === "f") { event.preventDefault(); $("search").focus(); $("search").select(); return; }
    // Paste anywhere and the link lands in the box it was meant for — unless something is
    // already taking the keystroke.
    if (mod && event.key.toLowerCase() === "v" && !typing) { $("source").focus(); return; }
    // Settings is a page, not a list: its arrow keys scroll it, and there is nothing in it
    // to select or delete.
    if (typing || state.view.kind === "settings") return;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      moveSelection(event.key === "ArrowDown" ? 1 : -1, event.shiftKey);
    } else if (mod && event.key.toLowerCase() === "a") {
      event.preventDefault();
      selectAll();
    } else if (event.key === "F2") {
      const models = selectedModels();
      if (models.length === 1) { event.preventDefault(); startRename(models[0].id); }
    } else if (event.key === "Delete") {
      event.preventDefault();
      deleteSelection();
    } else if (event.key === "Enter") {
      const models = selectedModels();
      if (models.length === 1 && models[0].previews && models[0].state === "present") {
        openLightbox({ kind: "model", id: models[0].id }, 0, models[0].filename);
      }
    }
  });
}

// --- the event stream ---------------------------------------------------------------------

const concerns = (key) => state.selection.has(key);

function onEvent(data) {
  switch (data.type) {
    case "task": {
      if (!upsertTask(data.task)) return;
      invalidate("tree", "list", "status");
      const model = data.task.model_id ? `m${data.task.model_id}` : "";
      if (concerns(`t${data.task.id}`) || (model && concerns(model))) invalidate("inspector");
      refreshSpace();
      return;
    }
    case "removed":
      state.tasks.delete(data.id);
      if (state.selection.delete(`t${data.id}`)) invalidate("inspector");
      invalidate("tree", "list", "status");
      return;
    case "reload":
      act.loadTasks();
      return;
    case "progress": {
      const task = state.tasks.get(data.id);
      if (!task) return;
      task.downloaded = data.downloaded;
      task.size = data.total ?? task.size;
      task.fraction = task.size ? data.downloaded / task.size : 0;
      task.speed = data.speed;
      task.eta = data.eta;
      task.connections = data.connections;
      patchProgress(task);
      patchInspectorProgress(task);
      invalidate("status");
      return;
    }
    case "moving":
      // Driven by the stream rather than by whoever started the move, so the Stop button
      // is there after a reload and in a second window — the copy outlives both.
      state.moving = { ...(state.moving || {}), ...data };
      invalidate("status");
      return;
    case "moved":
      state.moving = null;
      invalidate("status");
      return;
    case "models": {
      act.duplicatesTouched(data.models);
      let selected = false;
      for (const model of data.models) {
        upsertModel(model);
        forgetPreviews({ kind: "model", id: model.id });
        if (concerns(`m${model.id}`)) selected = true;
      }
      if (!selected) {
        for (const key of state.selection) {
          const task = key[0] === "t" ? state.tasks.get(Number(key.slice(1))) : null;
          if (task && data.models.some((m) => m.id === task.model_id)) selected = true;
        }
      }
      invalidate("tree", "list", "header");
      if (selected) invalidate("inspector");
      return;
    }
    case "model_removed":
      act.duplicatesTouched([], data.id);
      removeModel(data.id);
      invalidate("tree", "list", "inspector");
      return;
    case "library":
      act.loadLibrary();
      return;
    case "folders":
      state.roots = data.roots;
      state.folders = data.folders;
      state.syncedAt = data.synced_at;
      invalidate("tree", "list", "status");
      if (!state.selection.size) invalidate("inspector");
      return;
    case "jobs":
      state.jobs = { current: data.current, queued: data.queued || [] };
      invalidate("status");
      return;
    case "toast": {
      const models = data.models || [];
      toast(data.message, {
        level: data.level === "error" ? "error" : "info",
        actions: models.length === 1 ? [{ label: "Show", run: () => act.showModel(models[0]) }]
          : models.length > 1 ? [{ label: "Show", run: () => act.showModel(models[0]) }] : [],
      });
      return;
    }
    case "note":
      toast(data.message);
      return;
    default:
  }
}

// Whether the stream has been open before: a stream that comes back after a sleep or a
// restart of the server has missed whatever happened meanwhile.
let streamOpened = false;

function connect() {
  const stream = new EventSource("/api/events");
  stream.onopen = () => {
    // Anything that happened while the stream was down is only in a fresh snapshot.
    if (streamOpened) {
      state.moving = null;
      act.loadTasks();
      act.loadLibrary();
    }
    streamOpened = true;
  };
  stream.onmessage = (event) => {
    try { onEvent(JSON.parse(event.data)); } catch (error) { console.error(error); }
  };
  stream.onerror = () => {
    stream.close();
    setTimeout(connect, 2000);
  };
}

// --- dropping a link onto the window ----------------------------------------------------------

function wireDrops() {
  // A drag that started on this page — models on their way to a folder, a download being
  // reordered — is not a link being dropped in from outside.
  const isModels = (event) => {
    const types = event.dataTransfer.types;
    return types.includes("application/x-modeldl-models") || types.includes("application/x-modeldl-reorder");
  };
  document.addEventListener("dragover", (event) => {
    if (isModels(event)) return;
    const types = event.dataTransfer.types;
    if (types.includes("text/uri-list") || types.includes("text/plain")) {
      event.preventDefault();
      event.dataTransfer.dropEffect = "copy";
      document.body.classList.add("drop-link");
    }
  });
  document.addEventListener("dragleave", (event) => {
    if (!event.relatedTarget) document.body.classList.remove("drop-link");
  });
  document.addEventListener("drop", (event) => {
    document.body.classList.remove("drop-link");
    if (isModels(event)) return;
    const text = event.dataTransfer.getData("text/uri-list") || event.dataTransfer.getData("text/plain");
    const line = String(text || "").split("\n").map((s) => s.trim()).find((s) => s && !s.startsWith("#"));
    if (!line || !/^https?:\/\//i.test(line)) return;
    event.preventDefault();
    // A picture of this page's own, dragged a little by a shaky click, is not a download.
    try { if (new URL(line).origin === location.origin) return; } catch { return; }
    act.addSource(line);
  });
}

// --- starting up ------------------------------------------------------------------------------

function restore(settings) {
  const prefs = settings.ui || {};
  state.prefs = { ...prefs };
  setBits(prefs.units === "bits");
  if (Array.isArray(prefs.expanded)) state.expanded = new Set(prefs.expanded);
  if (prefs.sort && prefs.sort.key) state.sort = prefs.sort;
  if (typeof prefs.deep === "boolean") state.deep = prefs.deep;
  applyWidths();
}

// The window opens where it was left. Only the very first time, with nothing to go back
// to, does it choose: the downloads if there are any, the library otherwise.
function firstView() {
  const saved = state.prefs.view;
  if (saved && saved.kind) {
    if (saved.kind !== "folder") return saved;
    if (saved.root === "outside" || (typeof saved.root === "number" && saved.root < state.roots.length)) return saved;
  }
  const unfinished = [...state.tasks.values()].some((t) => t.state !== "done" && !t.archived);
  return unfinished ? { kind: "downloads" } : { kind: "folder", root: 0, relative: "" };
}

async function start() {
  wireHeader();
  wireSplitters();
  wireSidebar();
  wireList();
  wireStatus();
  wireKeys();
  wireDrops();
  const settings = await act.loadSettings();
  restore(settings);
  await Promise.all([act.loadLibrary(), act.loadTasks()]);
  state.view = firstView();
  if (state.view.kind === "cleanup") act.loadCleanup();
  if (state.view.kind === "duplicates") act.loadDuplicates();
  invalidate();
  connect();
  refreshSpace();
  // Whatever happened to the folders while the window was in the background — a model
  // dragged in Explorer, a download finished by a browser — is read in when it comes back.
  window.addEventListener("focus", () => act.rescan());
  window.addEventListener("beforeunload", () => { flushNoteOnExit(); flushPrefs(); });
}

start().catch((error) => toast(error.message || String(error), { level: "error" }));
