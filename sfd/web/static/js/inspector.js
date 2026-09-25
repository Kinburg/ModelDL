// The right pane: everything about what is selected, and what can be done with it.
//
// For a model, that is the whole story: what it is and where it came from, the words that
// wake it up, what you wrote about it, and every download that produced it. For nothing
// selected, it is the folder or the view itself.

import { post } from "./api.js";
import {
  esc, fmtBytes, fmtSpeed, fmtEta, fmtWhen, fmtDate, fmtDuration, fmtAgo, kindLabel, plural,
  copyText, stemOf, suffixOf, baseName, accessChip, paidNow, paidSentence,
} from "./util.js";
import { icon, kindIcon } from "./icons.js";
import {
  state, invalidate, onRender, modelsIn, folderLabel, placementOf, activeTasks, selectedModels,
  standingOf, versionsHere,
} from "./store.js";
import { menuFrom } from "./menu.js";
import { covered, openLightbox, previewSrc, previewsOf, flash } from "./lightbox.js";
import { toast, toastError } from "./toasts.js";
import { chips, historyStatus, modelMenu, canIdentify } from "./list.js";
import * as act from "./actions.js";

let note = null;   // {key, id, kind, value, dirty}
let renaming = null;

// --- a note being written -------------------------------------------------------------------

async function saveNote() {
  if (!note || !note.dirty) return;
  const pending = note;
  pending.dirty = false;
  try {
    if (pending.kind === "model") await act.saveModelNote(pending.id, pending.value);
    else await act.saveTaskNote(pending.id, pending.value);
    const hint = document.querySelector(`[data-note-state="${pending.key}"]`);
    if (hint) { hint.textContent = "Saved"; hint.className = "note-state ok"; }
  } catch (error) {
    pending.dirty = true;
    toastError(error);
  }
}

// Called before the pane is redrawn: a note half typed when the selection moved on is saved
// rather than lost — the moment someone clicks the next model is not the moment they meant
// to throw away what they wrote about this one.
export function flushNote() { if (note && note.dirty) saveNote(); }

// The same, as the window closes: an ordinary request dies with the page, and `keepalive`
// is what lets this one outlive it.
export function flushNoteOnExit() {
  if (!note || !note.dirty) return;
  const url = note.kind === "model" ? `/api/models/${note.id}/note` : `/api/tasks/${note.id}/note`;
  note.dirty = false;
  try {
    fetch(url, {
      method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ note: note.value }),
    });
  } catch { /* the window is closing either way */ }
}

function noteBlock(key, kind, id, value, placeholder) {
  if (!note || note.key !== key || !note.dirty) note = { key, kind, id, value: value || "", dirty: false };
  return `
    <section class="insp-section">
      <div class="insp-label">Your note <span class="note-state" data-note-state="${esc(key)}"></span></div>
      <textarea class="note-box" data-note="${esc(key)}" rows="3" maxlength="4000" spellcheck="false"
        placeholder="${esc(placeholder)}">${esc(note.value)}</textarea>
    </section>`;
}

function wireNote(holder) {
  const box = holder.querySelector("[data-note]");
  if (!box || !note) return;
  const hint = holder.querySelector("[data-note-state]");
  box.addEventListener("input", () => {
    note.value = box.value;
    note.dirty = true;
    if (hint) { hint.textContent = "Unsaved — saved when you click away, or Ctrl+Enter"; hint.className = "note-state"; }
  });
  box.addEventListener("blur", saveNote);
  box.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) { event.preventDefault(); saveNote(); box.blur(); }
  });
}

// --- pieces ------------------------------------------------------------------------------------

function preview(source, count, nsfw, title) {
  if (!count) return "";
  const hide = covered(source, nsfw);
  return `
    <button class="insp-preview ${hide ? "covered" : ""}" data-open="0" title="${hide ? "Marked adult — click to uncover" : count > 1 ? `${count} samples` : "Full size"}">
      <img src="${previewSrc(source, 0, 640)}" alt="" draggable="false" onerror="this.parentElement.remove()">
      ${count > 1 ? `<span class="preview-count">${count}</span>` : ""}
    </button>`;
}

function row(label, value, extra = "") {
  if (value === undefined || value === null || value === "") return "";
  return `<dt>${esc(label)}</dt><dd>${value}${extra}</dd>`;
}

const copyButton = (text, label = "Copy") =>
  `<button class="mini" data-copy-text="${esc(text)}">${esc(label)}</button>`;

function words(list, label) {
  if (!list || !list.length) return "";
  return `
    <section class="insp-section">
      <div class="insp-label">${esc(label)}</div>
      <div class="tags">${list.map((w) => `<span class="tag">${esc(w)}</span>`).join("")}
        ${copyButton(list.join(", "))}</div>
    </section>`;
}

function banner(kind, html, buttons = "") {
  return `<div class="banner banner-${kind}"><div class="banner-text">${html}</div>${buttons ? `<div class="banner-actions">${buttons}</div>` : ""}</div>`;
}

// --- a model ------------------------------------------------------------------------------------

const capital = (text) => (text ? text[0].toUpperCase() + text.slice(1) : text);

// What buying a newer version comes to: said before Download is pressed, not learnt from
// Civitai's refusal afterwards.
function paidLine(access) {
  if (!paidNow(access)) return "";
  return `<div class="paid-line ${access.owned === true ? "ok" : "warn"}">${accessChip(access)}${esc(paidSentence(access))}</div>`;
}
const pageLink = (url) => (url
  ? `<a class="button" href="${esc(url)}" target="_blank" rel="noopener noreferrer">${icon("external")}Open the page</a>` : "");

// What the last check for newer versions said about it.
function updateBanners(model) {
  const standing = standingOf(model);
  const record = model.update || {};
  const pick = record.update || {};
  if (standing === "update") {
    const when = pick.published_at ? `, ${esc(fmtDate(Date.parse(pick.published_at) / 1000))}` : "";
    const file = pick.file?.name
      ? `<div class="small muted mono">${esc(pick.file.name)}${pick.file.size ? ` · ${esc(fmtBytes(pick.file.size))}` : ""}</div>` : "";
    return [banner("accent",
      pick.id
        ? `A newer version: <b>${esc(pick.name || "")}</b>${pick.base_model ? ` for ${esc(pick.base_model)}` : ""}${when}${record.count > 1 ? ` (${record.count} newer in all)` : ""}.${file}${paidLine(pick.access)}`
        : `The file has changed on HuggingFace since it came here — ${esc(pick.name || "a newer commit")}. Fetching it would replace this one, so that is left to its page.`,
      `${pick.id && pick.file?.id ? `<button data-do="download-update">${icon("download")}Download…</button>` : ""}
       <button data-do="skip-update" title="Stop counting it — until a version newer than it comes out">${icon("eye-off")}Skip this version</button>
       ${pageLink(pick.page)}`)];
  }
  if (standing === "other") {
    const here = versionsHere();
    const others = (record.others || []).filter((o) => !here.has(o.id));
    const names = others.slice(0, 4).map((o) => `<b>${esc(o.name)}</b>${o.base_model && o.base_model !== model.base_model ? ` for ${esc(o.base_model)}` : ""}${paidNow(o.access) ? ` (${o.access.permanent ? "paid" : "early access"})` : ""}`);
    const more = Math.max(0, (record.others_count || others.length) - names.length);
    const text = [
      record.skipped ? `${esc(record.skipped.name || "The newer version")} is skipped: it is counted again if a newer one comes out.` : "",
      names.length ? `Newer on its page, but not an update of this one: ${names.join(", ")}${more ? ` and ${more} more` : ""}.` : "",
    ].filter(Boolean).join(" ");
    return [banner("quiet", text,
      `${record.skipped ? `<button data-do="unskip-update">${icon("eye")}Count it again</button>` : ""}${pageLink(record.page)}`)];
  }
  if (standing === "gone") {
    return [banner("info", `${esc(capital(record.error || "gone from its site"))} — the copy here may be the last one.`)];
  }
  if (standing === "error") {
    return [banner("quiet", `Its newer versions could not be asked about: ${esc(record.error || "no answer")}.`)];
  }
  return [];
}

function modelPanel(id) {
  const model = state.models.get(id);
  if (!model) return `<div class="insp-empty">That model is not in the library any more.</div>`;
  const details = state.details.get(id);
  if (!details) {
    act.loadDetails(id).then(() => {
      if (state.selection.has(`m${id}`) || currentHistoryModel() === id) invalidate("inspector");
    }).catch(() => {});
  }
  const d = details || {};
  const header = d.header || {};
  const extras = d.extras || {};
  const meta = d.meta || {};
  const missing = model.state === "missing";
  const source = { kind: "model", id };
  const name = model.filename;

  const banners = [];
  if (missing) {
    const candidates = (d.candidates || []).map((c) => `
      <div class="candidate">
        <span class="mono" title="${esc(c.path)}">${esc(c.root !== null && c.root !== undefined ? folderLabel(c.root, c.relative) : c.folder)}</span>
        <button class="mini primary" data-relink="${c.id}">This is it</button>
      </div>`).join("");
    banners.push(banner("warn",
      `Not where the library last saw it — missing since ${esc(fmtDate(model.missing_since))}.
       <div class="mono small">${esc(model.path)}</div>
       ${model.source ? `<div class="small">${model.origin === "downloaded" ? "Downloaded" : "Identified"}${model.host ? ` from ${esc(model.host)}` : ""} — the link is kept, and can be used again.</div>` : ""}
       ${candidates ? `<div class="candidates"><div class="small">Found on disk under the same name and size:</div>${candidates}</div>` : ""}`,
      `${model.source ? `<button class="primary" data-do="again">${icon("download")}Download again…</button>` : ""}
       <button data-do="online" title="Look for it on Civitai by its hash and on HuggingFace by its name">${icon("globe")}Find online…</button>
       <button data-do="locate">${icon("search")}Find the file…</button>
       <button class="danger" data-do="forget">${icon("x")}Forget…</button>`));
  }
  const names = d.names || [];
  if (!missing && names.length > 1) {
    const list = names.map((n) => `
      <div class="link-name">
        <span class="mono small" title="${esc(n.path)}">${esc(n.path)}</span>
        ${n.this ? `<span class="chip quiet">this one</span>`
          : n.model_id ? `<button class="mini" data-show="${n.model_id}">Show</button>`
            : `<span class="small muted">outside the library</span>`}
      </div>`).join("");
    banners.push(banner("info",
      `One file under ${names.length} names — the same data, taking the room of one. Deleting this name frees nothing while another is left.
       <div class="link-names">${list}</div>`,
      `<button data-do="separate" title="Give this name a copy of its own again">${icon("copy")}Make a separate copy…</button>`));
  } else if (!missing && model.links > 1) {
    banners.push(banner("info", `One file under ${model.links} names — deleting this name frees nothing while another is left.`,
      `<button data-do="separate">${icon("copy")}Make a separate copy…</button>`));
  }
  if (!missing && (d.left_behind_files || []).length) {
    const list = d.left_behind_files.map((p) => `<div class="mono small">${esc(baseName(p.from))}</div>`).join("");
    banners.push(banner("info",
      `${plural(d.left_behind_files.length, "file")} named after it stayed in the folder it was moved out of:${list}`,
      `<button data-do="bring-back">${icon("back")}Bring ${d.left_behind_files.length === 1 ? "it" : "them"} here</button>`));
  }
  banners.push(...updateBanners(model));
  if (header.folder_says && !missing) {
    banners.push(banner("info",
      `The file itself says <b>${esc(kindLabel(model.category))}</b>, but it is in a folder for ${esc(kindLabel(header.folder_says).toLowerCase())}s — the loader that folder is read by may not open it.`,
      `<button data-do="move">${icon("move")}Move to…</button>`));
  }
  const verified = (d.lookup_info || {}).verified;
  if (verified && verified.ok === false) {
    banners.push(banner("error", "This file does not match the hash it was downloaded with. It may be damaged, or replaced by another file of the same name."));
  }
  const also = d.lookup_info || {};
  if (model.lookup === "not_found" && model.origin === "found") {
    const where = (also.searched || []).includes("huggingface") ? "Civitai or HuggingFace" : "Civitai";
    banners.push(banner("quiet", `Not on ${where} — looked up ${esc(fmtAgo(also.at))}`
      + (also.quick ? ", without reading the whole file: Civitai does not know its quick hash, and nothing on HuggingFace has its name and size." : ".")));
  }
  if (model.origin === "downloaded" && also.result === "found" && also.page) {
    const hub = also.source === "huggingface";
    banners.push(banner("quiet",
      hub ? `Also on HuggingFace, in <b>${esc(also.repo_id || "")}</b>.`
        : `Also on Civitai, as <b>${esc([also.model_name, also.version_name].filter(Boolean).join(" / ") || "a model")}</b>.`,
      `<a class="button" href="${esc(also.page)}" target="_blank" rel="noopener noreferrer">${icon("external")}Open the page</a>`));
  }

  const identified = model.identified;
  const title = model.title && model.title !== name
    ? `<div class="insp-title">${esc(model.title)}${model.version_name ? ` <span class="muted">/ ${esc(model.version_name)}</span>` : ""}</div>` : "";
  const nameBlock = renaming === id
    ? `<div class="insp-name editing"><input data-role="rename" value="${esc(stemOf(name))}" spellcheck="false"><span class="suffix">${esc(suffixOf(name))}</span></div>`
    : `<div class="insp-name"><span class="name-text ${missing ? "muted" : ""}" data-do="${missing ? "" : "rename"}" title="${missing ? "" : "Click or press F2 to rename"}">${esc(name)}</span></div>`;

  const history = d.history || [];
  const first = history[history.length - 1];
  const latest = history[0];
  const facts = [
    row("Model", identified && (model.title || meta.model_name)
      ? `${esc([meta.model_name || model.title, meta.version_name].filter(Boolean).join(" / "))}` : ""),
    row("Page", d.page ? `<a href="${esc(d.page)}" target="_blank" rel="noopener noreferrer">${esc(d.page.replace(/^https?:\/\//, ""))}</a>` : ""),
    row("Kind", model.category ? `${esc(kindLabel(model.category))}${model.confidence ? ` <span class="muted">(${esc(model.confidence)})</span>` : ""}` : ""),
    row("Why", d.reason ? esc(d.reason) : ""),
    row("Base model", model.base_model ? esc(model.base_model) : ""),
    row("Downloaded", latest ? `${esc(fmtWhen(latest.finished_at || latest.created_at))}${latest.duration ? ` · took ${esc(fmtDuration(latest.duration))}` : ""}${latest.average_speed ? ` · ${esc(fmtSpeed(latest.average_speed))}` : ""}` : ""),
    row("From", latest ? esc(latest.origin || latest.provider || "") : model.origin === "found" ? `<span class="muted">not downloaded by ModelDL — first seen ${esc(fmtDate(model.first_seen))}</span>` : ""),
    row("Arrived as", first && first.original_filename && first.original_filename !== name ? `<span class="mono">${esc(first.original_filename)}</span>` : ""),
    row("Size", `${esc(fmtBytes(model.size))}${model.parts ? ` in ${model.parts} parts` : ""}`),
    row("Parameters", header.params ? esc(humanCount(header.params)) : ""),
    row("Precision", model.precision ? esc(model.precision) : ""),
    row("Format", header.format ? esc(header.format) + (header.tensors ? ` <span class="muted">· ${header.tensors} tensors</span>` : "") : ""),
    row("Architecture", header.architecture ? esc(header.architecture) + (header.size_label ? ` · ${esc(header.size_label)}` : "") : ""),
    row("Context", header.context ? esc(header.context.toLocaleString("en-GB")) : ""),
    row("Trained", header.kohya ? esc([
      header.kohya.network && `${header.kohya.network}`,
      header.kohya.dim && `rank ${header.kohya.dim}`,
      header.kohya.alpha && `alpha ${header.kohya.alpha}`,
      header.kohya.base_model && `on ${header.kohya.base_model}`,
      header.kohya.images && `${header.kohya.images} images`,
    ].filter(Boolean).join(" · ")) : ""),
    row("Author", (header.modelspec || {}).author ? esc(header.modelspec.author) : ""),
    row("SHA256", model.sha256
      ? `<span class="mono">${esc(model.sha256)}</span>${verified ? (verified.ok ? ` <span class="ok">✓ matches</span>` : verified.ok === false ? ` <span class="err">does not match</span>` : "") : ""}`
      : missing || model.parts ? "" : `<button class="mini" data-do="hash">Work it out</button>`),
    row("Path", `<span class="mono">${esc(model.path)}</span> ${copyButton(model.path)}`),
    row("Record", d.record_path ? `<span class="mono">${esc(d.record_path)}</span>` : ""),
    row("Also by", extras.sources ? esc(extras.sources.join(", ")) : ""),
  ].join("");

  const description = header.description || extras.description;
  const theirNotes = extras.their_notes;
  const triggers = model.trigger_words || [];
  const tags = (header.tags || []).map((t) => t[0]).slice(0, 12);
  const samples = model.previews > 1 && !missing ? `<div class="strip inline" data-role="strip"></div>` : "";

  return `
    <div class="insp">
      ${missing ? `<div class="insp-preview empty">${icon("file-off")}</div>` : preview(source, model.previews, model.nsfw, name) || `<div class="insp-preview empty">${icon(kindIcon(model.category))}</div>`}
      ${nameBlock}
      ${title}
      <div class="chips">${chips(model)}</div>
      ${missing ? "" : `<div class="insp-actions">
        <button data-do="reveal" title="Show in Explorer">${icon("external")}Show</button>
        <button data-do="move">${icon("move")}Move to…</button>
        <button data-do="rename">${icon("edit")}Rename</button>
        ${canIdentify(model) ? `<button data-do="identify" title="Look it up on Civitai by its hash and on HuggingFace by its name">${icon("hash")}Identify</button>` : ""}
        <button class="icon-button" data-do="more" aria-label="More">${icon("dots")}</button>
      </div>`}
      ${banners.join("")}
      ${words(triggers, "Trigger words")}
      ${!triggers.length && tags.length ? words(tags, "Most frequent training tags") : ""}
      ${extras.preferred_weight ? `<div class="small muted insp-line">Preferred weight: ${esc(extras.preferred_weight)}</div>` : ""}
      ${noteBlock(`m${id}`, "model", id, model.note, "What you will want to know the next time you find this file — the weight that works, what it clashes with, why you kept it.")}
      ${theirNotes ? `<section class="insp-section"><div class="insp-label">Notes another tool kept</div><div class="quote">${esc(theirNotes)}</div></section>` : ""}
      ${description ? `<section class="insp-section"><div class="insp-label">Description</div><div class="quote clamp">${esc(description)}</div></section>` : ""}
      ${samples}
      <section class="insp-section"><dl class="facts">${facts}</dl></section>
      ${history.length > 1 ? `<section class="insp-section"><div class="insp-label">Downloaded ${history.length} times</div>
        ${history.map((t) => `<div class="small">${esc(fmtWhen(t.finished_at || t.created_at))} · ${esc(t.origin || "")}</div>`).join("")}</section>` : ""}
      ${missing ? "" : `<div class="insp-danger"><button class="danger" data-do="delete">${icon("trash")}Delete from disk…</button></div>`}
    </div>`;
}

function humanCount(n) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 1e10 ? 0 : 1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

function currentHistoryModel() {
  const [key] = [...state.selection];
  if (!key || key[0] !== "t") return null;
  const task = state.tasks.get(Number(key.slice(1)));
  return task?.model_id || null;
}

async function wireModel(holder, id) {
  const model = state.models.get(id);
  if (!model) return;
  holder.onclick = async (event) => {
    const relink = event.target.closest("[data-relink]");
    if (relink) { await act.relinkModel(id, Number(relink.dataset.relink)); return; }
    const show = event.target.closest("[data-show]");
    if (show) { act.showModel(Number(show.dataset.show)); return; }
    const opener = event.target.closest("[data-open]");
    if (opener) {
      if (opener.classList.contains("covered")) {
        state.revealed.add(`model${id}`);
        invalidate("inspector", "list");
        return;
      }
      openLightbox({ kind: "model", id }, Number(opener.dataset.open), model.filename);
      return;
    }
    const target = event.target.closest("[data-do]");
    if (!target || !target.dataset.do) return;
    const what = target.dataset.do;
    if (what === "reveal") act.revealModel(id);
    else if (what === "move") act.moveModelsDialog([id]);
    else if (what === "rename") startRename(id);
    else if (what === "identify") act.identify([id]);
    else if (what === "hash") act.hashModels([id]);
    else if (what === "delete") act.deleteModels([id]);
    else if (what === "locate") act.locateModel(id);
    else if (what === "again") act.downloadAgain(id);
    else if (what === "online") act.findOnline(id);
    else if (what === "separate") act.separateModels([id]);
    else if (what === "forget") act.forgetModels([id]);
    else if (what === "bring-back") act.bringBack(id);
    else if (what === "download-update") act.downloadUpdate([id]);
    else if (what === "skip-update") act.skipUpdate([id]);
    else if (what === "unskip-update") act.skipUpdate([id], false);
    else if (what === "more") menuFrom(target, modelMenu([model]));
  };
  const input = holder.querySelector('[data-role="rename"]');
  if (input) {
    input.focus();
    input.select();
    const finish = async (commit) => {
      if (renaming !== id) return;
      const value = input.value.trim();
      renaming = null;
      if (commit && value && value !== stemOf(model.filename)) await act.renameModel(id, value);
      invalidate("inspector");
    };
    input.addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); finish(true); }
      if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
  }
  const strip = holder.querySelector('[data-role="strip"]');
  if (strip) {
    try {
      const { previews } = await previewsOf({ kind: "model", id });
      strip.innerHTML = previews.map((p, i) => `
        <img class="shot ${covered({ kind: "model", id }, p.nsfw) ? "covered" : ""}" src="${previewSrc({ kind: "model", id }, i, 240)}"
             data-open="${i}" loading="lazy" alt="" draggable="false">`).join("");
    } catch { strip.remove(); }
  }
}

export function startRename(id) {
  const model = state.models.get(id);
  if (!model || model.state !== "present") return;
  if (model.parts) { toast("A model split into several files cannot be renamed here"); return; }
  if (![...state.selection].includes(`m${id}`)) {
    state.selection = new Set([`m${id}`]);
    invalidate("list");
  }
  renaming = id;
  invalidate("inspector");
}

document.addEventListener("rename-model", (event) => startRename(event.detail));

// --- a download ---------------------------------------------------------------------------------

function taskPanel(id) {
  const task = state.tasks.get(id);
  if (!task) return `<div class="insp-empty">That download is not on the list any more.</div>`;
  if (task.state === "done" && task.model_id && state.models.has(task.model_id)) return modelPanel(task.model_id);
  const source = { kind: "task", id };
  const pct = task.fraction ? Math.round(task.fraction * 100) : 0;
  const where = task.dest ? placementOf(task.dest) : { root: null };
  const buttons = [];
  if (task.state === "blocked") {
    buttons.push(`<button class="primary" data-task="confirm">${icon("check")}Accept</button>`);
    buttons.push(`<button data-task="where">${icon("folder")}Elsewhere…</button>`);
  }
  if (task.state === "running" || task.state === "pending") buttons.push(`<button data-task="pause">${icon("pause")}Pause</button>`);
  if (task.state === "paused") buttons.push(`<button class="primary" data-task="resume">${icon("play")}Resume</button>`);
  if (task.state === "failed") buttons.push(`<button class="primary" data-task="retry">${icon("retry")}Retry</button>`);
  if (task.dest) buttons.push(`<button data-task="reveal">${icon("external")}Show</button>`);
  const timing = [];
  if (task.finished_at) timing.push(row("Finished", esc(fmtWhen(task.finished_at))));
  else if (task.started_at && task.state === "running") timing.push(row("Started", esc(fmtWhen(task.started_at))));
  timing.push(row("Added", esc(fmtWhen(task.created_at))));
  if (task.state === "failed" && task.retry_at) timing.push(row("Retrying", esc(fmtWhen(task.retry_at))));
  else if (task.attempts) timing.push(row("Attempt", String(task.attempts + (task.state === "done" ? 0 : 1))));

  return `
    <div class="insp">
      ${preview(source, task.previews, task.nsfw, task.filename) || `<div class="insp-preview empty">${icon("download")}</div>`}
      <div class="insp-name"><span class="name-text">${esc(task.filename || task.source)}</span></div>
      ${task.label ? `<div class="insp-title">${esc(task.label)}</div>` : ""}
      <div class="chips"><span class="state ${task.state}">${task.state === "blocked" ? "needs a decision" : task.state}</span>
        ${task.category ? `<span class="chip kind">${esc(kindLabel(task.category))}</span>` : ""}
        ${task.base_model ? `<span class="chip">${esc(task.base_model)}</span>` : ""}
        ${task.origin ? `<span class="chip quiet">${esc(task.origin)}</span>` : ""}</div>
      ${task.state !== "done" ? `<div class="track big"><div class="fill ${task.state}" style="width:${pct}%"></div></div>
        <div class="small muted" data-role="insp-stats">${taskStats(task)}</div>` : ""}
      <div class="insp-actions">${buttons.join("")}</div>
      ${task.state === "blocked" && task.reason ? banner("warn", `${esc(kindLabel(task.category))}${task.confidence ? ` (${esc(task.confidence)})` : ""} — ${esc(task.reason)}${task.disagreement ? `<div class="small">The service lists it as ${esc(kindLabel(task.disagreement))}.</div>` : ""}`) : ""}
      ${task.error ? banner("error", esc(task.error)) : ""}
      ${words(task.trigger_words, "Trigger words")}
      ${noteBlock(`t${id}`, "task", id, task.note, "What you know about it now, while it downloads. It goes into the model's record when the file lands.")}
      <section class="insp-section"><dl class="facts">
        ${row("Goes to", where.root !== null ? esc(folderLabel(where.root, where.relative)) : task.dest ? `<span class="mono">${esc(task.dest)}</span>` : "")}
        ${row("Why there", task.reason && task.state !== "blocked" ? esc(task.reason) : "")}
        ${row("Size", esc(fmtBytes(task.size)))}
        ${row("Link", `<span class="mono">${esc(task.source)}</span>`)}
        ${timing.join("")}
      </dl></section>
      <div class="insp-danger">
        ${task.dest && task.state !== "done" ? `<button class="danger" data-task="delete-files">${icon("trash")}Delete its files…</button>` : ""}
        <button class="danger" data-task="remove" title="Take it off the list. Files stay on the disk">${icon("x")}Remove</button>
      </div>
    </div>`;
}

function taskStats(task) {
  if (task.state === "running") {
    return `${fmtBytes(task.downloaded)} of ${fmtBytes(task.size)} · ${fmtSpeed(task.speed)}`
      + (task.connections ? ` · ${task.connections} connections` : "")
      + (task.eta ? ` · ETA ${fmtEta(task.eta)}` : "");
  }
  return `${fmtBytes(task.downloaded)} of ${fmtBytes(task.size)}`;
}

export function patchInspectorProgress(task) {
  const holder = document.getElementById("inspector");
  const [key] = [...state.selection];
  if (key !== `t${task.id}`) return;
  const fill = holder.querySelector(".track.big .fill");
  if (fill) fill.style.width = `${Math.round((task.fraction || 0) * 100)}%`;
  const numbers = holder.querySelector('[data-role="insp-stats"]');
  if (numbers) numbers.textContent = taskStats(task);
}

function wireTask(holder, id) {
  holder.onclick = (event) => {
    const opener = event.target.closest("[data-open]");
    if (opener) {
      if (opener.classList.contains("covered")) { state.revealed.add(`task${id}`); invalidate("inspector", "list"); return; }
      openLightbox({ kind: "task", id }, 0, state.tasks.get(id)?.filename);
      return;
    }
    const target = event.target.closest("[data-task]");
    if (!target) return;
    const what = target.dataset.task;
    if (what === "where") act.placeTask(id);
    else if (what === "reveal") post(`/api/tasks/${id}/reveal`).catch(toastError);
    else if (what === "delete-files") act.deleteTaskFiles(id);
    else act.taskCommand(id, what);
  };
}

// --- a history entry ------------------------------------------------------------------------------

function historyPanel(id) {
  const task = state.tasks.get(id);
  if (!task) return `<div class="insp-empty">Not in the history any more.</div>`;
  const status = historyStatus(task);
  if (status.model) return modelPanel(status.model.id);
  const page = task.model_name ? null : null;
  return `
    <div class="insp">
      ${preview({ kind: "task", id }, task.previews, task.nsfw, task.filename) || `<div class="insp-preview empty">${icon("history")}</div>`}
      <div class="insp-name"><span class="name-text muted">${esc(task.filename)}</span></div>
      ${task.model_name ? `<div class="insp-title">${esc([task.model_name, task.version_name].filter(Boolean).join(" / "))}</div>` : ""}
      <div class="chips"><span class="state muted">${esc(status.label)}</span>${task.origin ? `<span class="chip quiet">${esc(task.origin)}</span>` : ""}</div>
      ${banner("quiet", task.fate === "forgotten"
        ? "It went missing and was taken out of the library. Everything needed to fetch it again was kept."
        : "Its files were deleted. Everything needed to fetch it again was kept.",
        `<button class="primary" data-hist="again">${icon("download")}Download it again</button>`)}
      ${task.note ? `<section class="insp-section"><div class="insp-label">Your note</div><div class="quote">${esc(task.note)}</div></section>` : ""}
      ${words(task.trigger_words, "Trigger words")}
      <section class="insp-section"><dl class="facts">
        ${row("Downloaded", esc(fmtWhen(task.finished_at || task.created_at)))}
        ${row("Arrived as", task.original_filename && task.original_filename !== task.filename ? `<span class="mono">${esc(task.original_filename)}</span>` : "")}
        ${row("Size", esc(fmtBytes(task.size)))}
        ${row("Was at", `<span class="mono">${esc(task.dest)}</span>`)}
        ${row("Link", `<span class="mono">${esc(task.source)}</span>`)}
      </dl></section>
      <div class="insp-danger"><button class="danger" data-hist="remove">${icon("x")}Remove from the history</button></div>
    </div>`;
}

function wireHistory(holder, id) {
  holder.onclick = (event) => {
    const opener = event.target.closest("[data-open]");
    if (opener) { openLightbox({ kind: "task", id }, 0, state.tasks.get(id)?.filename); return; }
    const target = event.target.closest("[data-hist]");
    if (!target) return;
    if (target.dataset.hist === "again") act.redownload(id);
    if (target.dataset.hist === "remove") act.removeFromHistory([id]);
  };
}

// --- a cleanup item -------------------------------------------------------------------------------

function cleanupPanel(id) {
  const item = (state.cleanup?.items || []).find((i) => i.id === id);
  if (!item) return `<div class="insp-empty">Gone already.</div>`;
  return `
    <div class="insp">
      <div class="insp-name"><span class="name-text">${esc(item.name)}</span></div>
      <div class="chips"><span class="chip quiet">${esc(item.why)}</span></div>
      <section class="insp-section"><dl class="facts">
        ${row("Folder", `<span class="mono">${esc(item.folder)}</span>`)}
        ${row("Size", esc(fmtBytes(item.size)))}
      </dl></section>
      <section class="insp-section"><div class="insp-label">Files</div>
        <ul class="filelist">${item.files.map((f) => `<li><span class="path">${esc(f.name)}</span><span class="size">${esc(fmtBytes(f.size))}</span></li>`).join("")}</ul>
      </section>
      <div class="insp-danger"><button class="danger" data-clean="delete">${icon("trash")}Delete…</button></div>
    </div>`;
}

// --- several at once ------------------------------------------------------------------------------

function multiPanel(keys) {
  const models = selectedModels();
  const tasks = keys.filter((k) => k[0] === "t").map((k) => state.tasks.get(Number(k.slice(1)))).filter(Boolean);
  const cleanups = keys.filter((k) => k[0] === "c");
  const size = models.reduce((s, m) => s + (m.size || 0), 0) + tasks.reduce((s, t) => s + (t.size || 0), 0);
  const names = [...models.map((m) => m.filename), ...tasks.map((t) => t.filename)].slice(0, 8);
  const present = models.filter((m) => m.state === "present");
  const missing = models.filter((m) => m.state === "missing");
  const buttons = [];
  if (present.length) {
    buttons.push(`<button data-multi="move">${icon("move")}Move ${present.length} to…</button>`);
    buttons.push(`<button data-multi="identify">${icon("hash")}Identify</button>`);
    buttons.push(`<button data-multi="updates">${icon("update")}Check for newer versions</button>`);
  }
  if (tasks.length && state.view.kind === "history") buttons.push(`<button class="danger" data-multi="unhistory">${icon("x")}Remove from the history</button>`);
  if (tasks.length && state.view.kind === "downloads") buttons.push(`<button class="danger" data-multi="remove">${icon("x")}Remove from the list</button>`);
  const danger = [];
  if (present.length) danger.push(`<button class="danger" data-multi="delete">${icon("trash")}Delete ${present.length} from disk…</button>`);
  const again = missing.filter((m) => m.source);
  if (again.length) buttons.push(`<button data-multi="again">${icon("download")}Download ${again.length} again…</button>`);
  if (missing.length) danger.push(`<button class="danger" data-multi="forget">${icon("x")}Forget ${missing.length} missing…</button>`);
  if (cleanups.length) danger.push(`<button class="danger" data-multi="clean">${icon("trash")}Delete ${cleanups.length} leftovers…</button>`);
  return `
    <div class="insp">
      <div class="insp-name"><span class="name-text">${plural(keys.length, "item")} selected</span></div>
      <div class="small muted">${fmtBytes(size)}${models.length ? ` · drag them onto a folder to move them` : ""}</div>
      <section class="insp-section">${names.map((n) => `<div class="small mono ellipsis">${esc(n)}</div>`).join("")}
        ${keys.length > names.length ? `<div class="small muted">and ${keys.length - names.length} more</div>` : ""}</section>
      <div class="insp-actions column">${buttons.join("")}</div>
      <div class="insp-danger column">${danger.join("")}</div>
    </div>`;
}

function wireMulti(holder) {
  holder.onclick = (event) => {
    const target = event.target.closest("[data-multi]");
    if (!target) return;
    const models = selectedModels();
    const present = models.filter((m) => m.state === "present").map((m) => m.id);
    const taskIds = [...state.selection].filter((k) => k[0] === "t").map((k) => Number(k.slice(1)));
    const what = target.dataset.multi;
    if (what === "move") act.moveModelsDialog(present);
    if (what === "identify") act.identify(models.filter(canIdentify).map((m) => m.id));
    if (what === "updates") act.checkUpdates(models.map((m) => m.id));
    if (what === "delete") act.deleteModels(present);
    if (what === "forget") act.forgetModels(models.filter((m) => m.state === "missing").map((m) => m.id));
    if (what === "again") act.downloadAllAgain(models.filter((m) => m.state === "missing").map((m) => m.id));
    if (what === "unhistory") act.removeFromHistory(taskIds);
    if (what === "remove") taskIds.forEach((id) => act.taskCommand(id, "remove"));
    if (what === "clean") act.deleteCleanup([...state.selection].filter((k) => k[0] === "c").map((k) => Number(k.slice(1))));
  };
}

// --- nothing selected ------------------------------------------------------------------------------

function viewPanel() {
  const view = state.view;
  if (view.kind === "folder" && view.root !== "outside") {
    const root = state.roots[view.root];
    if (!root) return `<div class="insp-empty">Add a folder to the library to begin — the one ComfyUI reads its models from, and any others it reads too.
      <div style="margin-top:12px"><button class="primary" data-view-do="add-root">${icon("plus")}Add a folder</button></div></div>`;
    const models = modelsIn(view.root, view.relative, true);
    const present = models.filter((m) => m.state === "present");
    const kinds = new Map();
    for (const m of present) kinds.set(m.category || "other", (kinds.get(m.category || "other") || 0) + 1);
    const size = present.reduce((s, m) => s + (m.size || 0), 0);
    const missing = models.length - present.length;
    return `
      <div class="insp">
        <div class="insp-preview empty">${icon(view.relative ? "folder-open" : "drive")}</div>
        <div class="insp-name"><span class="name-text">${esc(view.relative ? view.relative.split("/").pop() : root.name)}</span></div>
        <div class="small muted mono">${esc(view.relative ? `${root.path}${state.sep}${view.relative.replace(/\//g, state.sep)}` : root.path)}</div>
        <div class="insp-actions">
          <button data-view-do="reveal">${icon("external")}Show</button>
          <button data-view-do="new-folder">${icon("plus")}New folder…</button>
        </div>
        <section class="insp-section"><dl class="facts">
          ${row("Models", `${present.length}${missing ? ` <span class="warn">+ ${missing} missing</span>` : ""}`)}
          ${row("Size", esc(fmtBytes(size)))}
          ${!view.relative && root.free != null ? row("Free on disk", esc(fmtBytes(root.free))) : ""}
          ${!view.relative ? row("Downloads", root.primary ? "are filed into this folder" : "go elsewhere") : ""}
        </dl></section>
        ${kinds.size ? `<section class="insp-section"><div class="insp-label">What is in it</div>
          <div class="tags">${[...kinds.entries()].sort((a, b) => b[1] - a[1]).map(([k, n]) => `<span class="tag">${esc(kindLabel(k))} · ${n}</span>`).join("")}</div></section>` : ""}
        <div class="small muted insp-line">Select a model to see everything about it. Drag models onto a folder in the tree to move them.</div>
      </div>`;
  }
  if (view.kind === "downloads") {
    const tasks = activeTasks().filter((t) => t.state !== "done");
    const left = tasks.reduce((s, t) => s + Math.max(0, (t.size || 0) - (t.downloaded || 0)), 0);
    const speed = tasks.reduce((s, t) => s + (t.state === "running" ? t.speed || 0 : 0), 0);
    return `
      <div class="insp">
        <div class="insp-name"><span class="name-text">Downloads</span></div>
        <section class="insp-section"><dl class="facts">
          ${row("In the list", String(activeTasks().length))}
          ${row("Still to fetch", esc(fmtBytes(left)))}
          ${speed ? row("Speed", esc(fmtSpeed(speed))) : ""}
          ${speed ? row("Finishing in", esc(fmtEta(left / speed))) : ""}
        </dl></section>
        <div class="small muted insp-line">Paste a HuggingFace or Civitai link into the box at the top — or press Ctrl+V anywhere, or drop it on the window. ${state.settings.smart_placement
          ? "Each file is filed by what it is; one whose placement is uncertain waits for you here instead of being filed by a guess."
          : "Before anything downloads, you are asked where it goes — the folder that already holds its kind first."}</div>
      </div>`;
  }
  const texts = {
    history: "Every download that finished, newest first. A renamed model shows its name now and the one it arrived under; a deleted one can be downloaded again from here.",
    updates: "Newer versions of the models here, by the version they update to. Download asks where it goes, the folder of the old version first; the old version stays where it is. Skip this version stops one being counted until a newer one comes out.",
    missing: "Models the library remembers that are no longer where it last saw them. Select one to find its file or to forget it.",
    unidentified: "Models that came from somewhere else. What they are is read from the files; identifying one looks it up on Civitai by its hash and on HuggingFace by its name, and fills in the rest.",
    duplicates: "The same file kept in more than one place. In each set, mark the copy to keep — the star — and delete the others once the hashes have confirmed they are the same.",
    cleanup: "Leftovers: unfinished downloads nobody is coming back for, and files named after models that are gone. Select to see what is in each.",
  };
  return `<div class="insp-empty">${esc(texts[view.kind] || "")}</div>`;
}

function wireView(holder) {
  holder.onclick = (event) => {
    const target = event.target.closest("[data-view-do]");
    if (!target) return;
    const view = state.view;
    if (target.dataset.viewDo === "reveal") act.revealFolder(view.root, view.relative);
    if (target.dataset.viewDo === "new-folder") act.newFolder(view.root, view.relative);
    if (target.dataset.viewDo === "add-root") act.addRoot();
  };
}

// --- drawing ---------------------------------------------------------------------------------------

let shownFor = "";
let held = false;

function renderInspector() {
  const holder = document.getElementById("inspector");
  const keys = [...state.selection];
  const signature = `${state.view.kind}|${keys.join(",")}|${renaming}`;
  // Someone is typing in here — a note, a new name. Redrawing now, because a download
  // somewhere moved on a percent, would take the caret out from under them. The redraw
  // waits until they are done.
  const typing = holder.contains(document.activeElement)
    && /^(TEXTAREA|INPUT)$/.test(document.activeElement.tagName);
  if (typing && signature === shownFor) {
    if (!held) {
      held = true;
      document.activeElement.addEventListener("blur", () => {
        held = false;
        setTimeout(() => invalidate("inspector"), 0);
      }, { once: true });
    }
    return;
  }
  // The same thing redrawn keeps its place; something else selected starts at the top.
  const scroll = signature === shownFor ? holder.scrollTop : 0;
  shownFor = signature;
  flushNote();
  let html;
  let wire = () => {};
  if (state.view.kind === "settings") {
    html = `<div class="insp-empty">Changes are saved when you press Save at the bottom of the page. Tokens are never sent back to the page once saved.</div>`;
  } else if (!keys.length) {
    html = viewPanel();
    wire = () => wireView(holder);
  } else if (keys.length > 1) {
    html = multiPanel(keys);
    wire = () => wireMulti(holder);
  } else {
    const [key] = keys;
    const id = Number(key.slice(1));
    if (key[0] === "m") { html = modelPanel(id); wire = () => wireModel(holder, id); }
    else if (key[0] === "t") {
      const task = state.tasks.get(id);
      const isHistory = state.view.kind === "history" || (task && task.state === "done");
      const shownModel = task && task.state === "done" && task.model_id && state.models.has(task.model_id) ? task.model_id : null;
      if (shownModel) { html = modelPanel(shownModel); wire = () => wireModel(holder, shownModel); }
      else if (isHistory) { html = historyPanel(id); wire = () => wireHistory(holder, id); }
      else { html = taskPanel(id); wire = () => wireTask(holder, id); }
    } else if (key[0] === "c") {
      html = cleanupPanel(id);
      wire = () => {
        holder.onclick = (event) => {
          if (event.target.closest('[data-clean="delete"]')) act.deleteCleanup([id]);
        };
      };
    }
  }
  holder.innerHTML = html || "";
  holder.scrollTop = scroll;
  holder.querySelectorAll("[data-copy-text]").forEach((button) => {
    button.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (await copyText(button.dataset.copyText)) flash(button);
      else toast("The clipboard would not take it", { level: "error" });
    });
  });
  wire();
  wireNote(holder);
}

onRender("inspector", renderInspector);
