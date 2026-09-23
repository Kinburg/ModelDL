// The left pane: the views, and the tree of the library's folders.
//
// The tree is the library as it is on disk — every folder that holds a model, with how many
// and how much — plus the folders a missing model was last seen in, drawn grey. Dropping
// models on a folder moves them there.

import { esc, fmtBytes, plural } from "./util.js";
import { icon } from "./icons.js";
import {
  state, tree, invalidate, remember, onRender, activeTasks, missingModels, unidentifiedModels,
  nodeKey,
} from "./store.js";
import { showMenu } from "./menu.js";
import {
  go, moveModels, newFolder, hideFolder, revealFolder, addRoot, removeRoot, makePrimary, rescan,
} from "./actions.js";

const VIEWS = [
  { kind: "downloads", label: "Downloads", icon: "download" },
  { kind: "history", label: "History", icon: "history" },
  { kind: "missing", label: "Missing", icon: "alert" },
  { kind: "unidentified", label: "Unidentified", icon: "help" },
  { kind: "duplicates", label: "Duplicates", icon: "layers" },
  { kind: "cleanup", label: "Cleanup", icon: "broom" },
];

function badge(kind) {
  if (kind === "downloads") {
    const tasks = activeTasks();
    const open = tasks.filter((t) => t.state !== "done").length;
    const blocked = tasks.filter((t) => t.state === "blocked").length;
    if (blocked) return `<span class="badge warn" title="${plural(blocked, "download")} need a decision">${open}</span>`;
    return open ? `<span class="badge">${open}</span>` : "";
  }
  if (kind === "missing") {
    const count = missingModels().length;
    return count ? `<span class="badge warn">${count}</span>` : "";
  }
  if (kind === "unidentified") {
    const count = unidentifiedModels().length;
    return count ? `<span class="badge quiet">${count}</span>` : "";
  }
  if (kind === "duplicates" && state.duplicates && !state.duplicates.loading) {
    const count = (state.duplicates.exact || []).length;
    return count ? `<span class="badge quiet">${count}</span>` : "";
  }
  if (kind === "cleanup" && state.cleanup && !state.cleanup.loading && state.cleanup.total) {
    return `<span class="badge quiet">${fmtBytes(state.cleanup.total)}</span>`;
  }
  return "";
}

function renderSidebar() {
  const holder = document.getElementById("sidebar");
  const current = state.view;
  const views = VIEWS.map((v) => `
    <button class="side-item ${current.kind === v.kind ? "on" : ""}" data-view="${v.kind}">
      ${icon(v.icon)}<span class="side-label">${v.label}</span>${badge(v.kind)}
    </button>`).join("");

  const rows = [];
  for (const top of tree.roots) walk(top, 0, rows);
  if (tree.outside.count || tree.outside.missing) {
    const on = current.kind === "folder" && current.root === "outside";
    rows.push(`
      <div class="tree-row ${on ? "on" : ""}" data-node="outside" style="--depth:0">
        <span class="twisty"></span>${icon("folder", "tree-icon")}
        <span class="tree-name">Outside the library</span>
        <span class="tree-count">${tree.outside.count || ""}${tree.outside.missing ? `<span class="dot warn" title="${plural(tree.outside.missing, "missing model")}"></span>` : ""}</span>
      </div>`);
  }

  const scrolled = holder.querySelector(".tree")?.scrollTop || 0;
  holder.innerHTML = `
    <div class="side-section">${views}</div>
    <div class="side-section grow">
      <div class="side-title">
        <span>Library</span>
        <span class="grow"></span>
        <button class="icon-button" data-action="rescan" title="Read the folders again">${icon("refresh")}</button>
        <button class="icon-button" data-action="add-root" title="Add a folder to the library">${icon("plus")}</button>
      </div>
      <div class="tree" role="tree">${rows.join("") || `<div class="side-empty">No library folders yet.
        <button class="link-button" data-action="add-root">Add one</button></div>`}</div>
    </div>`;
  holder.querySelector(".tree").scrollTop = scrolled;
}

function walk(item, depth, rows) {
  const current = state.view;
  const on = current.kind === "folder" && current.root === item.root && current.relative === item.relative;
  const open = state.expanded.has(item.key);
  const top = item.relative === "";
  const root = top ? state.roots[item.root] : null;
  const ghost = !item.exists || (!item.count && !item.downloading && item.missing);
  const counts = `${item.count ? item.count : ""}${item.downloading ? `<span class="dot accent" title="${plural(item.downloading, "download")} landing here"></span>` : ""}${item.missing ? `<span class="dot warn" title="${plural(item.missing, "missing model")}"></span>` : ""}`;
  const title = top
    ? `${root.path}${root.primary ? " — downloads are filed here" : ""}${root.free != null ? ` · ${fmtBytes(root.free)} free` : ""}`
    : `${item.relative}${item.size ? ` · ${fmtBytes(item.size)}` : ""}`;
  rows.push(`
    <div class="tree-row ${on ? "on" : ""} ${ghost ? "ghost" : ""} ${top ? "top" : ""}" data-node="${esc(item.key)}"
         style="--depth:${depth}" title="${esc(title)}" role="treeitem" aria-expanded="${open}">
      <button class="twisty" data-toggle ${item.children.length ? "" : "disabled"} aria-label="Expand">
        ${item.children.length ? icon(open ? "chevron-down" : "chevron-right") : ""}</button>
      ${icon(top ? "drive" : open ? "folder-open" : "folder", "tree-icon")}
      <span class="tree-name">${esc(item.name)}${top && root.primary && state.roots.length > 1 ? ` <span class="tree-tag" title="Downloads are filed into this folder">${icon("download")}</span>` : ""}</span>
      <span class="tree-count">${counts}</span>
    </div>`);
  if (open) for (const child of item.children) walk(child, depth + 1, rows);
}

// --- behaviour -----------------------------------------------------------------------------

function parseNode(key) {
  if (key === "outside") return { root: "outside", relative: "" };
  const at = key.indexOf(":");
  return { root: Number(key.slice(0, at)), relative: key.slice(at + 1) };
}

function toggle(key) {
  if (state.expanded.has(key)) state.expanded.delete(key);
  else state.expanded.add(key);
  remember({ expanded: [...state.expanded] });
  invalidate("tree");
}

export function wireSidebar() {
  const holder = document.getElementById("sidebar");

  holder.addEventListener("click", (event) => {
    const view = event.target.closest("[data-view]");
    if (view) { go({ kind: view.dataset.view }); return; }
    const action = event.target.closest("[data-action]");
    if (action) {
      if (action.dataset.action === "add-root") addRoot();
      if (action.dataset.action === "rescan") rescan({ quiet: false });
      return;
    }
    const row = event.target.closest(".tree-row");
    if (!row) return;
    if (event.target.closest("[data-toggle]")) { toggle(row.dataset.node); return; }
    const { root, relative } = parseNode(row.dataset.node);
    go({ kind: "folder", root, relative });
  });

  holder.addEventListener("dblclick", (event) => {
    const row = event.target.closest(".tree-row");
    if (row && row.dataset.node !== "outside" && !event.target.closest("[data-toggle]")) toggle(row.dataset.node);
  });

  holder.addEventListener("contextmenu", (event) => {
    const row = event.target.closest(".tree-row");
    if (!row || row.dataset.node === "outside") return;
    event.preventDefault();
    const { root, relative } = parseNode(row.dataset.node);
    const top = relative === "";
    const info = state.roots[root];
    showMenu(event.clientX, event.clientY, [
      { label: "Show in Explorer", icon: "external", run: () => revealFolder(root, relative) },
      { label: "New folder…", icon: "plus", run: () => newFolder(root, relative) },
      "-",
      !top && { label: "Hide from the library", icon: "eye-off", run: () => hideFolder(root, relative) },
      top && !info.primary && { label: "File downloads here", icon: "download", run: () => makePrimary(root) },
      top && { label: "Take off the list", icon: "x", run: () => removeRoot(root), danger: true, disabled: info.downloads },
      "-",
      { label: "Read the folders again", icon: "refresh", run: () => rescan({ quiet: false }) },
    ]);
  });

  // Dropping models onto a folder moves them there, with every file named after them.
  let over = null;
  const clear = () => {
    if (over) over.classList.remove("drop");
    over = null;
    clearTimeout(holder._opener);
  };
  holder.addEventListener("dragover", (event) => {
    if (!event.dataTransfer.types.includes("application/x-modeldl-models")) return;
    const row = event.target.closest(".tree-row");
    if (!row || row.dataset.node === "outside") { clear(); return; }
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    if (over === row) return;
    clear();
    over = row;
    row.classList.add("drop");
    // Hovering over a closed folder for a moment opens it, so a deep folder can be reached.
    // Started once per folder rather than on every dragover: those arrive many times a
    // second, and a timer restarted by each of them would never run out.
    clearTimeout(holder._opener);
    const node = row.dataset.node;
    if (!state.expanded.has(node)) {
      holder._opener = setTimeout(() => {
        if (over && over.dataset.node === node && !state.expanded.has(node)) toggle(node);
      }, 700);
    }
  });
  holder.addEventListener("dragleave", (event) => {
    if (!holder.contains(event.relatedTarget)) clear();
  });
  holder.addEventListener("drop", async (event) => {
    const row = event.target.closest(".tree-row");
    clear();
    const data = event.dataTransfer.getData("application/x-modeldl-models");
    if (!row || !data) return;
    event.preventDefault();
    const { root, relative } = parseNode(row.dataset.node);
    let ids;
    try { ids = JSON.parse(data); } catch { return; }
    await moveModels(ids, root, relative);
  });
}

onRender("tree", renderSidebar);

export { nodeKey };
