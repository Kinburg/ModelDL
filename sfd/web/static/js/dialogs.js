// Dialogs: the questions the page has to ask before it does something it cannot take back,
// and the folder picker. Each returns a promise of the answer, so the code asking reads
// top to bottom: ask, then act on what was said.

import { esc, fmtBytes, plural } from "./util.js";
import { icon } from "./icons.js";
import { state } from "./store.js";

const stack = [];

export const dialogOpen = () => stack.length > 0;

export function modal({ title, body = "", footer = "", wide = false, className = "", onClose, focus }) {
  const backdrop = document.createElement("div");
  backdrop.className = "modal-backdrop open";
  backdrop.innerHTML = `
    <div class="modal ${wide ? "wide" : ""} ${className}" role="dialog" aria-modal="true">
      <div class="modal-header">
        <strong class="modal-title">${esc(title)}</strong>
        <button class="icon-button modal-close" aria-label="Close">${icon("x")}</button>
      </div>
      <div class="modal-body">${body}</div>
      ${footer ? `<div class="modal-footer">${footer}</div>` : ""}
    </div>`;
  document.body.appendChild(backdrop);
  const dialog = backdrop.querySelector(".modal");
  let closed = false;
  const close = (value) => {
    if (closed) return;
    closed = true;
    backdrop.remove();
    const index = stack.indexOf(handle);
    if (index >= 0) stack.splice(index, 1);
    onClose && onClose(value);
  };
  const handle = { backdrop, dialog, close };
  stack.push(handle);
  backdrop.querySelector(".modal-close").onclick = () => close(null);
  backdrop.addEventListener("mousedown", (event) => {
    if (event.target === backdrop) close(null);
  });
  if (focus) {
    const target = dialog.querySelector(focus);
    if (target) setTimeout(() => target.focus(), 0);
  }
  return handle;
}

// Innermost first: the lightbox can be opened from inside the folder picker, and one press
// of Escape closing both would throw away the question that was being answered.
export function closeTopDialog() {
  const top = stack[stack.length - 1];
  if (!top) return false;
  top.close(null);
  return true;
}

export function confirmDialog({ title, message, confirm = "OK", danger = false, detail = "" }) {
  return new Promise((resolve) => {
    const handle = modal({
      title,
      body: `<div class="dialog-message">${message}</div>${detail}`,
      footer: `<button class="${danger ? "danger-button" : "primary"}" data-role="yes">${esc(confirm)}</button>
               <button data-role="no">Cancel</button>`,
      onClose: (value) => resolve(!!value),
    });
    handle.dialog.querySelector('[data-role="yes"]').onclick = () => handle.close(true);
    handle.dialog.querySelector('[data-role="no"]').onclick = () => handle.close(false);
    // Nothing is focused on purpose when the answer destroys something: a dialog that
    // opens with that button under the cursor's Enter is a dialog that deletes by reflex.
    if (!danger) handle.dialog.querySelector('[data-role="yes"]').focus();
  });
}

export function promptDialog({ title, label = "", value = "", suffix = "", confirm = "OK", hint = "" }) {
  return new Promise((resolve) => {
    const handle = modal({
      title,
      body: `${label ? `<div class="dialog-message">${esc(label)}</div>` : ""}
        <div class="input-with-suffix">
          <input data-role="value" spellcheck="false" autocomplete="off" value="${esc(value)}">
          ${suffix ? `<span class="suffix">${esc(suffix)}</span>` : ""}
        </div>
        ${hint ? `<div class="small muted" style="margin-top:8px">${esc(hint)}</div>` : ""}`,
      footer: `<button class="primary" data-role="yes">${esc(confirm)}</button><button data-role="no">Cancel</button>`,
      onClose: (result) => resolve(result ?? null),
    });
    const input = handle.dialog.querySelector('[data-role="value"]');
    setTimeout(() => { input.focus(); input.select(); }, 0);
    const accept = () => handle.close(input.value.trim() || null);
    input.addEventListener("keydown", (event) => {
      if (event.key !== "Enter") return;
      // Answered here, and nowhere else: by the time the key reached the window the
      // dialog would be gone, and Enter there opens a model's samples.
      event.preventDefault();
      event.stopPropagation();
      accept();
    });
    handle.dialog.querySelector('[data-role="yes"]').onclick = accept;
    handle.dialog.querySelector('[data-role="no"]').onclick = () => handle.close(null);
  });
}

// A permanent delete of a set of files is a question nobody should have to answer from
// memory, so the set is on screen while it is asked.
export function filesDialog({ title, intro, groups, confirm, warning, checkbox }) {
  return new Promise((resolve) => {
    const total = groups.reduce((sum, g) => sum + g.files.reduce((s, f) => s + (f.size || 0), 0), 0);
    const count = groups.reduce((sum, g) => sum + g.files.length, 0);
    const list = groups.map((group) => `
      ${group.title ? `<div class="filelist-title">${esc(group.title)}</div>` : ""}
      <ul class="filelist">${group.files.map((file, index) => `
        <li class="${index === 0 && group.modelFirst ? "model" : ""}" title="${esc(file.path)}">
          <span class="path">${esc(file.name)}</span>
          <span class="size">${fmtBytes(file.size)}</span>
        </li>`).join("")}</ul>`).join("");
    const handle = modal({
      title,
      wide: groups.length > 1,
      body: `<div class="dialog-message">${intro}</div>${list}
        ${checkbox ? `<label class="check-row"><input type="checkbox" data-role="check" ${checkbox.checked ? "checked" : ""}> <span>${esc(checkbox.label)}</span></label>` : ""}
        ${warning ? `<div class="warn small" style="margin-top:10px">${esc(warning)}</div>` : ""}`,
      footer: `<button class="danger-button" data-role="yes">${esc(confirm || `Delete ${plural(count, "file")}`)}</button>
        <button data-role="no">Cancel</button><span class="grow"></span>
        <span class="small muted">${plural(count, "file")} · ${fmtBytes(total)}</span>`,
      onClose: (value) => resolve(value ?? null),
    });
    const check = handle.dialog.querySelector('[data-role="check"]');
    handle.dialog.querySelector('[data-role="yes"]').onclick = () =>
      handle.close({ ok: true, checked: check ? check.checked : undefined });
    handle.dialog.querySelector('[data-role="no"]').onclick = () => handle.close(null);
  });
}

// --- the folder picker ------------------------------------------------------------------

// A typed path is offered as a row of its own: the folder a file needs may not exist yet.
const typeable = (text) =>
  text && !text.startsWith("/") && !text.startsWith("\\")
  && !/^[a-z]:/i.test(text) && !text.split(/[\\/]/).includes("..");

// Resolves to {root, relative, remember}, {anywhere, remember} or null.
export function pickFolder({
  title = "Move to", subtitle = "", folders = [], preview = "", anywhere = false,
  rememberable = true, rootsAllowed = true, created = "created by the move",
}) {
  return new Promise((resolve) => {
    const multi = rootsAllowed && state.roots.length > 1;
    const handle = modal({
      title,
      body: `${preview ? `<div class="picker-preview">${preview}</div>` : ""}
        ${subtitle ? `<div class="small muted picker-subtitle">${subtitle}</div>` : ""}
        <div class="picker-search">
          ${multi ? `<select data-role="root" aria-label="Library folder for a new folder">
            ${state.roots.map((r) => `<option value="${r.index}">${esc(r.name)}</option>`).join("")}</select>` : ""}
          <input data-role="search" placeholder="Search, or type a folder to create"
                 spellcheck="false" autocomplete="off" aria-label="Search the library, or type a folder to create">
        </div>
        <div class="picker-list" data-role="list"></div>`,
      footer: `${rememberable ? `<label class="check-row small muted" data-role="remember-row"
          title="Only applies to a folder that names a kind of model. A base-model folder like checkpoints/Krea 2 files this one file and leaves the mapping alone.">
          <input type="checkbox" data-role="remember"> <span>Send this kind of model here from now on</span></label>` : ""}
        <span class="grow"></span>
        ${anywhere ? `<button data-role="anywhere" title="Choose any folder on this machine, including one on another drive">Another drive…</button>` : ""}`,
      wide: true,
      onClose: (value) => resolve(value ?? null),
      focus: '[data-role="search"]',
    });
    const $ = (role) => handle.dialog.querySelector(`[data-role="${role}"]`);
    const remember = () => ($("remember") ? $("remember").checked : false);

    const render = () => {
      const typed = $("search").value.trim().replace(/\\/g, "/").replace(/\/+$/, "");
      const wanted = typed.toLowerCase();
      const newRoot = multi ? Number($("root").value) : 0;
      const label = (f) => (multi || f.root ? `${state.roots[f.root]?.name ?? "?"} › ` : "") + f.relative;
      const rows = folders.filter((f) => label(f).toLowerCase().includes(wanted)
        || f.relative.toLowerCase().includes(wanted));
      const html = [];
      if (typed && typeable(typed) && !rows.some((f) => f.root === newRoot && f.relative.toLowerCase() === wanted)) {
        html.push(row({ root: newRoot, relative: typed, reason: created, category: null }, "new", label));
      }
      html.push(...rows.slice(0, 300).map((f) => row(f, f.exists === false ? "absent" : "", label)));
      $("list").innerHTML = html.join("")
        || `<div class="small muted" style="padding:10px">Nothing matches. Type a folder to use it anyway.</div>`;
      $("list").querySelectorAll(".folder-row").forEach((button) => {
        button.onclick = () => handle.close({
          root: Number(button.dataset.root), relative: button.dataset.folder,
          kind: button.dataset.kind || null, remember: remember(),
        });
      });
    };
    $("search").oninput = render;
    if (multi) $("root").onchange = render;
    $("search").addEventListener("keydown", (event) => {
      // Type enough to identify the folder, press Enter, done — the list is ranked, so the
      // top row is the answer far more often than not.
      if (event.key !== "Enter") return;
      event.preventDefault();
      event.stopPropagation();
      const first = $("list").querySelector(".folder-row");
      if (first) first.click();
    });
    if (anywhere) $("anywhere").onclick = () => handle.close({ anywhere: true, remember: remember() });
    render();
  });
}

function row(folder, extra, label) {
  const note = [folder.reason, folder.models ? plural(folder.models, "model") : ""]
    .filter(Boolean).join(" · ");
  return `
    <button class="folder-row ${extra}" data-root="${folder.root ?? 0}" data-folder="${esc(folder.relative)}"
            data-kind="${esc(folder.category || "")}">
      ${icon(extra === "new" ? "plus" : "folder")}
      <span class="path">${esc(label(folder))}</span>
      <span class="note">${esc(note)}</span>
    </button>`;
}
