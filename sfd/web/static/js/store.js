// What the page knows, in one place, and the one way of asking for it to be redrawn.
//
// The server is the source of truth and sends snapshots; this module keeps the latest of
// each and works out what the page needs from them — the folder tree with its counts, which
// models are in a folder, where a download is going to land.

import { put } from "./api.js";
import { debounce } from "./util.js";

export const state = {
  settings: {},
  roots: [],
  folders: [],
  hidden: [],
  sep: "\\",
  syncedAt: null,
  models: new Map(),
  tasks: new Map(),
  jobs: { current: null, queued: [] },
  moving: null,
  // Which list the middle pane shows. `folder` carries a root and a place inside it.
  view: { kind: "folder", root: 0, relative: "" },
  search: "",
  // Selection keys: `m12` a model, `t5` a download or a history entry, `c3` a cleanup item.
  selection: new Set(),
  anchor: null,
  focus: null,
  expanded: new Set(),
  sort: { key: "name", dir: 1 },
  deep: true,
  details: new Map(),
  revealed: new Set(),
  prefs: {},
  cleanup: null,
  duplicates: null,
};

// --- redrawing ------------------------------------------------------------------------

const renderers = new Map();
const dirty = new Set();
let frame = 0;
let fallback = 0;

export function onRender(region, fn) { renderers.set(region, fn); }

// Everything that changes something on screen asks for the regions it touched, and they
// are redrawn once, on the next frame — a burst of forty events is one redraw, not forty.
// A minimised window draws no frames at all, so a timer stands in for the frame there:
// the page is then current the moment it is shown again, instead of one frame behind.
export function invalidate(...regions) {
  for (const region of regions.length ? regions : renderers.keys()) dirty.add(region);
  if (!frame) {
    frame = requestAnimationFrame(flush);
    fallback = setTimeout(flush, 150);
  }
}

function flush() {
  cancelAnimationFrame(frame);
  clearTimeout(fallback);
  frame = 0;
  fallback = 0;
  const regions = [...dirty];
  dirty.clear();
  if (regions.includes("tree") || regions.includes("list")) rebuildTree();
  for (const region of ["tree", "list", "inspector", "status", "header"]) {
    if (!regions.includes(region)) continue;
    const fn = renderers.get(region);
    try { fn && fn(); } catch (error) { console.error(region, error); }
  }
}

// --- remembering how the window was left ---------------------------------------------

let pendingPrefs = {};
const savePrefs = debounce(() => {
  const patch = pendingPrefs;
  pendingPrefs = {};
  put("/api/ui", { prefs: patch }).catch(() => {});
}, 800);

export function remember(patch) {
  Object.assign(state.prefs, patch);
  Object.assign(pendingPrefs, patch);
  savePrefs();
}

// On the way out, whatever is still waiting goes with `keepalive`, which lets the request
// outlive the page that made it.
export function flushPrefs() {
  if (!Object.keys(pendingPrefs).length) return;
  const patch = pendingPrefs;
  pendingPrefs = {};
  try {
    fetch("/api/ui", {
      method: "PUT", keepalive: true, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prefs: patch }),
    });
  } catch { /* the window is closing either way */ }
}

// --- where things are -----------------------------------------------------------------

// The server says how its file system compares paths; the page compares them the same way,
// so a download's destination can be matched to a folder of the tree.
export function pathKey(path) {
  const text = String(path || "");
  return state.sep === "\\" ? text.replace(/\//g, "\\").toLowerCase() : text;
}

export function placementOf(path) {
  const key = pathKey(path);
  let best = null;
  for (const root of state.roots) {
    const base = root.key.replace(/[\\/]+$/, "");
    if (key.startsWith(base + state.sep) && (!best || base.length > best.base.length)) {
      best = { index: root.index, base };
    }
  }
  if (!best) return { root: null, relative: "" };
  const rest = String(path).slice(best.base.length + 1).replace(/\\/g, "/");
  const parts = rest.split("/");
  parts.pop();
  return { root: best.index, relative: parts.join("/") };
}

export const nodeKey = (root, relative) => `${root}:${relative}`;

// --- the tree ---------------------------------------------------------------------------

export const tree = { nodes: new Map(), roots: [], outside: { count: 0, size: 0, missing: 0 } };

function node(root, relative) {
  const key = nodeKey(root, relative);
  let found = tree.nodes.get(key);
  if (!found) {
    found = {
      key, root, relative,
      name: relative ? relative.split("/").pop() : "",
      children: [], exists: true, count: 0, size: 0, missing: 0, downloading: 0,
    };
    tree.nodes.set(key, found);
    if (relative) {
      const parent = node(root, relative.split("/").slice(0, -1).join("/"));
      parent.children.push(found);
    }
  }
  return found;
}

export function rebuildTree() {
  tree.nodes = new Map();
  tree.outside = { count: 0, size: 0, missing: 0 };
  tree.roots = state.roots.map((root) => {
    const top = node(root.index, "");
    top.name = root.name;
    top.exists = root.exists;
    return top;
  });
  for (const folder of state.folders) {
    if (folder.root >= state.roots.length) continue;
    node(folder.root, folder.relative).exists = folder.exists;
  }
  const add = (root, relative, size, missing, downloading) => {
    const parts = relative ? relative.split("/") : [];
    for (let depth = parts.length; depth >= 0; depth--) {
      const at = node(root, parts.slice(0, depth).join("/"));
      if (downloading) at.downloading += 1;
      else if (missing) at.missing += 1;
      else { at.count += 1; at.size += size || 0; }
    }
  };
  for (const model of state.models.values()) {
    if (model.root === null || model.root === undefined || model.root >= state.roots.length) {
      if (model.state === "missing") tree.outside.missing += 1;
      else { tree.outside.count += 1; tree.outside.size += model.size || 0; }
      continue;
    }
    add(model.root, model.relative, model.size, model.state === "missing", false);
  }
  for (const task of state.tasks.values()) {
    if (task.state === "done" || !task.dest) continue;
    const place = placementOf(task.dest);
    if (place.root !== null) add(place.root, place.relative, 0, false, true);
  }
  for (const item of tree.nodes.values()) {
    item.children.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }));
  }
}

// --- lists ---------------------------------------------------------------------------------

export function modelsIn(root, relative, deep = state.deep) {
  const found = [];
  const prefix = relative ? `${relative}/` : "";
  for (const model of state.models.values()) {
    if (root === "outside") {
      if (model.root === null || model.root === undefined) found.push(model);
      continue;
    }
    if (model.root !== root) continue;
    if (model.relative === relative || (deep && (relative === "" || model.relative.startsWith(prefix)))) {
      found.push(model);
    }
  }
  return found;
}

export function downloadsIn(root, relative, deep = state.deep) {
  const prefix = relative ? `${relative}/` : "";
  return [...state.tasks.values()].filter((task) => {
    if (task.state === "done" || !task.dest) return false;
    const place = placementOf(task.dest);
    if (root === "outside") return place.root === null;
    if (place.root !== root) return false;
    return place.relative === relative || (deep && (relative === "" || place.relative.startsWith(prefix)));
  });
}

export const activeTasks = () => [...state.tasks.values()].filter((t) => !t.archived);
export const missingModels = () => [...state.models.values()].filter((m) => m.state === "missing");
export const unidentifiedModels = () => [...state.models.values()].filter(
  (m) => m.origin === "found" && !m.identified && m.state === "present");

export function selectedModels() {
  return [...state.selection].filter((k) => k[0] === "m")
    .map((k) => state.models.get(Number(k.slice(1)))).filter(Boolean);
}

export function select(keys, { anchor = null, focus = null } = {}) {
  state.selection = new Set(keys);
  state.anchor = anchor ?? (keys.length ? keys[keys.length - 1] : null);
  state.focus = focus ?? state.anchor;
  invalidate("list", "inspector");
}

export function upsertTask(task) {
  const known = state.tasks.get(task.id);
  // Snapshots arrive from two places — the event stream and the reply to whatever request
  // just made the change — and they can overtake each other.
  if (known && known.updated_at > task.updated_at) return false;
  state.tasks.set(task.id, { ...known, ...task });
  return true;
}

export function upsertModel(model) {
  state.models.set(model.id, model);
  state.details.delete(model.id);
}

export function removeModel(id) {
  state.models.delete(id);
  state.details.delete(id);
  if (state.selection.delete(`m${id}`)) invalidate("inspector");
}

export const rootLabel = (index) => {
  const root = state.roots[index];
  return root ? root.name : "?";
};

export function folderLabel(root, relative) {
  if (root === "outside") return "Outside library";
  const name = rootLabel(root);
  return relative ? `${name} › ${relative.split("/").join(" › ")}` : name;
}
