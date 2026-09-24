// Dialogs: the questions the page has to ask before it does something it cannot take back,
// and the folder picker. Each returns a promise of the answer, so the code asking reads
// top to bottom: ask, then act on what was said.

import { esc, fmtBytes, kindLabel, plural } from "./util.js";
import { icon, kindIcon } from "./icons.js";
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

// --- choosing a folder ------------------------------------------------------------------

// A typed path is offered as a row of its own: the folder a file needs may not exist yet.
const typeable = (text) =>
  text && !text.startsWith("/") && !text.startsWith("\\")
  && !/^[a-z]:/i.test(text) && !text.split(/[\\/]/).includes("..");

const chooserHtml = (multi) => `
  <div class="picker-search">
    ${multi ? `<select data-role="root" aria-label="Library folder for a new folder">
      ${state.roots.map((r) => `<option value="${r.index}">${esc(r.name)}</option>`).join("")}</select>` : ""}
    <input data-role="search" placeholder="Search, or type a folder to create"
           spellcheck="false" autocomplete="off" aria-label="Search the library, or type a folder to create">
  </div>
  <div class="picker-list" data-role="list"></div>`;

// The ranked list of folders both the picker and the add dialog ask with: a box that
// searches it or names a folder that does not exist yet, arrows to move through it, Enter
// for the marked row. What a click means is the asker's: the picker takes the folder there
// and then; the add dialog only marks it, having more than one thing to ask.
function folderChooser(dialog, { folders, multi, created, onPick, pickOnClick = true, onMove = () => {} }) {
  const $ = (role) => dialog.querySelector(`[data-role="${role}"]`);
  const label = (f) => (multi || f.root ? `${state.roots[f.root]?.name ?? "?"} › ` : "") + f.relative;
  let shown = [];
  let current = 0;
  let query = null;

  const mark = () => {
    $("list").querySelectorAll(".folder-row").forEach((button, index) => {
      button.classList.toggle("current", index === current);
    });
    onMove(shown[current] || null);
  };

  const render = () => {
    const typed = $("search").value.trim().replace(/\\/g, "/").replace(/\/+$/, "");
    if (typed !== query) { current = 0; query = typed; }
    const wanted = typed.toLowerCase();
    const newRoot = multi ? Number($("root").value) : 0;
    const rows = folders.filter((f) => label(f).toLowerCase().includes(wanted)
      || f.relative.toLowerCase().includes(wanted));
    shown = [];
    if (typed && typeable(typed) && !rows.some((f) => f.root === newRoot && f.relative.toLowerCase() === wanted)) {
      shown.push({ root: newRoot, relative: typed, reason: created, category: null, typed: true });
    }
    shown.push(...rows.slice(0, 300));
    current = Math.min(current, Math.max(0, shown.length - 1));
    $("list").innerHTML = shown.map((f) => row(f, f.typed ? "new" : f.exists === false ? "absent" : "", label)).join("")
      || `<div class="small muted" style="padding:10px">Nothing matches. Type a folder to use it anyway.</div>`;
    $("list").querySelectorAll(".folder-row").forEach((button, index) => {
      button.onclick = () => {
        current = index;
        mark();
        if (pickOnClick) onPick(shown[index]);
      };
      button.ondblclick = () => onPick(shown[index]);
    });
    mark();
  };

  const move = (step) => {
    if (!shown.length) return;
    current = Math.max(0, Math.min(shown.length - 1, current + step));
    mark();
    $("list").querySelectorAll(".folder-row")[current]?.scrollIntoView({ block: "nearest" });
  };

  $("search").oninput = render;
  if (multi) $("root").onchange = render;
  $("search").addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      event.stopPropagation();
      move(event.key === "ArrowDown" ? 1 : -1);
      return;
    }
    // Type enough to identify the folder, press Enter, done — the list is ranked, so the
    // top row is the answer far more often than not.
    if (event.key !== "Enter") return;
    event.preventDefault();
    event.stopPropagation();
    if (shown[current]) onPick(shown[current]);
  });
  render();
  return {
    current: () => shown[current] || null,
    setFolders(next) { folders = next; render(); },
  };
}

const REMEMBER_HINT = "Only applies to a folder that names a kind of model. A base-model folder like"
  + " checkpoints/Krea 2 files this one download and leaves the mapping alone.";

// Resolves to {root, relative, kind, remember}, {anywhere, remember} or null.
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
        ${chooserHtml(multi)}`,
      footer: `${rememberable ? `<label class="check-row small muted" data-role="remember-row" title="${esc(REMEMBER_HINT)}">
          <input type="checkbox" data-role="remember"> <span>Send this kind of model here from now on</span></label>` : ""}
        <span class="grow"></span>
        ${anywhere ? `<button data-role="anywhere" title="Choose any folder on this machine, including one on another drive">Another drive…</button>` : ""}`,
      wide: true,
      onClose: (value) => resolve(value ?? null),
      focus: '[data-role="search"]',
    });
    const $ = (role) => handle.dialog.querySelector(`[data-role="${role}"]`);
    const remember = () => ($("remember") ? $("remember").checked : false);
    folderChooser(handle.dialog, {
      folders, multi, created,
      onPick: (f) => handle.close({ root: f.root, relative: f.relative, kind: f.category || null, remember: remember() }),
    });
    if (anywhere) $("anywhere").onclick = () => handle.close({ anywhere: true, remember: remember() });
  });
}

function row(folder, extra, label) {
  const note = folder.reason || (folder.models ? plural(folder.models, "model") : "");
  return `
    <button class="folder-row ${extra}" data-root="${folder.root ?? 0}" data-folder="${esc(folder.relative)}"
            data-kind="${esc(folder.category || "")}">
      ${icon(extra === "new" ? "plus" : "folder")}
      <span class="path">${esc(label(folder))}</span>
      <span class="note">${esc(note)}</span>
    </button>`;
}

// --- adding a download ------------------------------------------------------------------

function placeName(place) {
  const root = state.roots[place.root];
  const name = root ? root.name : "?";
  return place.relative ? `${name} › ${place.relative.split("/").join(" › ")}` : name;
}

// "Where does it go?", asked of a link that has been resolved but not queued: which of its
// files, and into which folder — the likeliest first, marked, so that Enter is the whole
// answer most of the time. Resolves to {files, root, relative, keep_structure, remember},
// {files, anywhere, keep_structure, remember} or null.
export function addDialog(resolved) {
  return new Promise((resolve) => {
    const items = resolved.items;
    const main = items[resolved.main] || items[0];
    const several = items.length > 1;
    const multi = state.roots.length > 1;
    const heading = main.model_name
      ? [main.model_name, main.version_name].filter(Boolean).join(" / ")
      : resolved.label || main.filename;
    const cover = !!(resolved.nsfw && state.settings.blur_nsfw);
    const picture = resolved.previews
      ? `<div class="add-thumb ${cover ? "covered" : ""}" data-role="thumb" title="${cover ? "Marked adult — click to uncover" : ""}">
           <img src="/api/resolve/${encodeURIComponent(resolved.token)}/preview?w=160" alt="" draggable="false"></div>`
      : `<div class="add-thumb empty">${icon(kindIcon(main.category))}</div>`;
    const chips = [
      main.category && `<span class="chip kind">${esc(kindLabel(main.category))}</span>`,
      main.base_model && `<span class="chip">${esc(main.base_model)}</span>`,
      !several && main.size && `<span class="chip quiet">${esc(fmtBytes(main.size))}</span>`,
      resolved.host && `<span class="chip quiet">${esc(resolved.host)}</span>`,
    ].filter(Boolean).join("");
    const files = several ? `
      <div class="add-files">
        <div class="add-files-head">
          <span data-role="count"></span><span class="grow"></span>
          <button class="mini" data-role="all">All</button><button class="mini" data-role="none">None</button>
        </div>
        <div class="add-file-list">${items.map((item) => `
          <label class="add-file ${item.have ? "have" : ""}" title="${esc(item.relative || item.filename)}">
            <input type="checkbox" data-file="${item.index}" ${item.checked ? "checked" : ""}>
            <span class="add-file-name">${esc(item.relative || item.filename)}</span>
            ${item.model_file ? `<span class="chip kind">${esc(kindLabel(item.category))}</span>` : ""}
            ${item.primary ? `<span class="chip quiet" title="The file the download button on its page gives">main</span>` : ""}
            ${item.have ? `<span class="small warn" title="${esc(item.have.path)}">in your library</span>` : ""}
            <span class="add-file-size">${item.size ? esc(fmtBytes(item.size)) : ""}</span>
          </label>`).join("")}</div>
        ${resolved.folder_name ? `<label class="check-row small add-structure"><input type="checkbox" data-role="structure" ${resolved.structure ? "checked" : ""}>
          <span>Keep the repository's folders, inside <b class="mono">${esc(resolved.folder_name)}/</b></span></label>` : ""}
      </div>` : "";
    const have = main.have;
    const already = !several && have
      ? `<div class="banner banner-warn"><div class="banner-text">Already in your library, in <b>${esc(have.root !== null && have.root !== undefined ? placeName(have) : have.path)}</b>. Downloading it again makes a second copy.</div></div>`
      : "";
    const handle = modal({
      title: "Add download",
      wide: true,
      className: "add-dialog",
      body: `
        <div class="add-head">${picture}
          <div class="add-what">
            <div class="add-title">${esc(heading)}</div>
            ${chips ? `<div class="chips">${chips}</div>` : ""}
            ${main.reason ? `<div class="add-why">${esc(main.reason)}</div>` : ""}
          </div>
        </div>
        ${already}${files}
        <div class="add-where">Where ${several ? "they go" : "it goes"}</div>
        ${chooserHtml(multi)}`,
      footer: `
        <label class="check-row small muted" data-role="remember-row" title="${esc(REMEMBER_HINT)}">
          <input type="checkbox" data-role="remember"> <span>Send this kind of model here from now on</span></label>
        <span class="grow"></span>
        <span class="small muted" data-role="summary"></span>
        <button data-role="anywhere" title="Choose any folder on this machine, including one on another drive">Another drive…</button>
        <button data-role="cancel">Cancel</button>
        <button class="primary" data-role="go">${icon("download")}Download</button>`,
      onClose: (value) => resolve(value ?? null),
      focus: '[data-role="search"]',
    });
    const $ = (role) => handle.dialog.querySelector(`[data-role="${role}"]`);
    const boxes = () => [...handle.dialog.querySelectorAll("[data-file]")];
    const picked = () => (several ? boxes().filter((b) => b.checked).map((b) => Number(b.dataset.file)) : [main.index]);
    const structure = () => !!($("structure") && $("structure").checked);
    // The folders offered follow the files ticked: a VAE ticked alone is asked about as a VAE.
    const rankingNow = () => {
      const chosen = picked().map((i) => items[i]);
      const lead = chosen.find((item) => item.model_file) || chosen[0] || main;
      return lead.ranking;
    };
    let ranking = rankingNow();

    const summarize = (folder) => {
      const chosen = picked().map((i) => items[i]);
      const known = chosen.reduce((sum, item) => sum + (item.size || 0), 0);
      const unknown = chosen.some((item) => !item.size);
      const root = folder ? state.roots[folder.root] : null;
      $("summary").textContent = [
        several ? plural(chosen.length, "file") : "",
        known ? `${fmtBytes(known)}${unknown ? " +" : ""}` : "",
        root && root.free != null ? `${fmtBytes(root.free)} free on ${root.name}` : "",
      ].filter(Boolean).join(" · ");
      // Remembering only means anything for a folder that names a kind.
      $("remember-row").hidden = !(folder && folder.category);
    };
    const go = (folder, anywhere = false) => {
      const chosen = picked();
      if (!chosen.length) {
        $("count").innerHTML = `<span class="err">Tick at least one file</span>`;
        return;
      }
      const remember = !!$("remember").checked;
      if (anywhere) {
        handle.close({ files: chosen, anywhere: true, keep_structure: structure(), remember });
      } else if (folder) {
        handle.close({
          files: chosen, root: folder.root, relative: folder.relative,
          keep_structure: structure(), remember: remember && !$("remember-row").hidden,
        });
      }
    };
    const chooser = folderChooser(handle.dialog, {
      folders: resolved.rankings[ranking] || [],
      multi,
      created: "created when the file lands",
      pickOnClick: false,
      onPick: (folder) => go(folder),
      onMove: summarize,
    });
    const refresh = () => {
      if (several) $("count").innerHTML = `${plural(items.length, "file")} · <b>${picked().length}</b> ticked`;
      const next = rankingNow();
      if (next !== ranking && resolved.rankings[next]) {
        ranking = next;
        chooser.setFolders(resolved.rankings[next]);
      } else {
        summarize(chooser.current());
      }
    };

    if (several) {
      handle.dialog.querySelector(".add-file-list").addEventListener("change", refresh);
      $("all").onclick = () => { boxes().forEach((b) => { b.checked = true; }); refresh(); };
      $("none").onclick = () => { boxes().forEach((b) => { b.checked = false; }); refresh(); };
    }
    const thumb = $("thumb");
    if (thumb) {
      thumb.onclick = () => thumb.classList.remove("covered");
      thumb.querySelector("img").onerror = () => { thumb.classList.add("empty"); thumb.innerHTML = icon(kindIcon(main.category)); };
    }
    $("go").onclick = () => go(chooser.current());
    $("anywhere").onclick = () => go(null, true);
    $("cancel").onclick = () => handle.close(null);
    refresh();
  });
}

