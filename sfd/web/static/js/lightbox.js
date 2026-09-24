// Sample images at full size, with the prompt and settings that produced each one. A LoRA's
// trigger words say which tokens wake it up and nothing about the prompt around them; the
// samples were made by someone who knew. A picture made in ComfyUI usually still carries the
// whole workflow that made it, and that is offered too.

import { get, post } from "./api.js";
import { copyText, esc, plural } from "./util.js";
import { icon } from "./icons.js";
import { modal } from "./dialogs.js";
import { state, invalidate } from "./store.js";
import { toast, toastError } from "./toasts.js";
import * as act from "./actions.js";

const cache = new Map();

// `source` is {kind: "model" | "task", id}.
export const previewBase = (source) =>
  source.kind === "task" ? `/api/tasks/${source.id}` : `/api/models/${source.id}`;

export async function previewsOf(source) {
  const key = `${source.kind}${source.id}`;
  if (!cache.has(key)) cache.set(key, await get(`${previewBase(source)}/previews`));
  return cache.get(key);
}

export function forgetPreviews(source) {
  cache.delete(`${source.kind}${source.id}`);
}

export const previewSrc = (source, index, width) =>
  `${previewBase(source)}/preview/${index}${width ? `?w=${width}` : ""}`;

export const coverKey = (source) => `${source.kind}${source.id}`;
export const covered = (source, nsfw) =>
  !!nsfw && !!state.settings.blur_nsfw && !state.revealed.has(coverKey(source));

export async function openLightbox(source, index = 0, title = "") {
  let body;
  try { body = await previewsOf(source); }
  catch (error) { toast(error.message, { level: "error" }); return; }
  const items = body.previews || [];
  if (!items.length) { toast("No sample images came with this one"); return; }

  let at = Math.min(index, items.length - 1);
  // What each sample carries, once the server has looked inside them all: null until then.
  let found = null;
  // The workflows themselves, fetched as soon as one is on screen, so that Copy puts it on
  // the clipboard straight from the click.
  const texts = new Map();
  const handle = modal({
    title: title || "Samples",
    wide: true,
    className: "lightbox",
    body: `<div class="stage" data-role="stage"></div>
           <div class="strip" data-role="strip"></div>
           <dl class="record" data-role="meta"></dl>`,
  });
  const $ = (role) => handle.dialog.querySelector(`[data-role="${role}"]`);
  const open = () => document.body.contains(handle.backdrop);

  const drawStage = () => {
    const item = items[at];
    const hidden = covered(source, item.nsfw);
    const media = item.type === "video"
      ? `<video src="${previewSrc(source, at)}" autoplay loop muted playsinline controls></video>`
      : `<img src="${previewSrc(source, at, 1024)}" alt="" draggable="false">`;
    $("stage").innerHTML =
      (items.length > 1 ? `<button class="nav" data-step="-1" aria-label="Previous">${icon("chevron-right", "flip")}</button>` : "")
      + `<div class="frame ${hidden ? "covered" : ""}">${media}`
      + (hidden ? `<button class="uncover">Marked adult — show it</button>` : "")
      + `</div>`
      + (items.length > 1 ? `<button class="nav" data-step="1" aria-label="Next">${icon("chevron-right")}</button>` : "");
    $("stage").querySelectorAll("[data-step]").forEach((b) => {
      b.onclick = () => { at = (at + Number(b.dataset.step) + items.length) % items.length; draw(); };
    });
    const uncover = $("stage").querySelector(".uncover");
    if (uncover) uncover.onclick = () => { state.revealed.add(coverKey(source)); draw(); invalidate("list", "inspector"); };
  };

  const drawStrip = () => {
    $("strip").hidden = items.length < 2;
    $("strip").innerHTML = items.map((p, i) => `
      <span class="shot-wrap">
        <img class="shot ${i === at ? "current" : ""} ${covered(source, p.nsfw) ? "covered" : ""}"
             src="${previewSrc(source, i, 160)}" data-index="${i}" loading="lazy" alt="" draggable="false">
        ${markHtml(found && found[i])}
      </span>`).join("");
    $("strip").querySelectorAll(".shot").forEach((img) => {
      img.onclick = () => { at = Number(img.dataset.index); draw(); };
    });
  };

  const textOf = (index) => {
    if (!texts.has(index)) {
      const pending = get(`${previewBase(source)}/workflows/${index}`).then((answer) => answer.text);
      pending.catch(() => texts.delete(index));
      texts.set(index, pending);
    }
    return texts.get(index);
  };

  const drawMeta = () => {
    const index = at;
    $("meta").innerHTML = metaHtml(items[index], found ? found[index] || { kind: "skipped" } : null);
    wireCopies($("meta"));
    const copy = $("meta").querySelector('[data-wf="copy"]');
    if (copy) {
      textOf(index).catch(() => {});
      copy.onclick = async () => {
        let text;
        try { text = await textOf(index); } catch (error) { toastError(error); return; }
        if (await copyText(text)) flash(copy);
        else toast("The clipboard would not take it", { level: "error" });
      };
    }
    const save = $("meta").querySelector('[data-wf="save"]');
    if (save) save.onclick = () => saveWorkflow(source, index, save);
    const retry = $("meta").querySelector('[data-wf="retry"]');
    if (retry) retry.onclick = () => look();
  };

  const draw = () => {
    handle.dialog.querySelector(".modal-title").textContent =
      `${title || "Samples"}${items.length > 1 ? ` — ${at + 1} of ${items.length}` : ""}`;
    drawStage();
    drawStrip();
    drawMeta();
  };

  // All the samples at once, not only the one on screen: the marks on the strip are what
  // say which of eight pictures is worth opening. Only the meta and the strip are redrawn
  // when the answer comes, so a video that is playing keeps playing.
  const look = async () => {
    found = null;
    drawMeta();
    let answer;
    try { answer = (await get(`${previewBase(source)}/workflows`)).workflows; }
    catch (error) { answer = items.map(() => ({ kind: "error", error: error.message })); }
    if (!open()) return;
    found = answer;
    drawStrip();
    drawMeta();
  };

  const keys = (event) => {
    if (!open()) { window.removeEventListener("keydown", keys, true); return; }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      event.stopPropagation();
      at = (at + (event.key === "ArrowLeft" ? -1 : 1) + items.length) % items.length;
      draw();
    }
  };
  window.addEventListener("keydown", keys, true);
  draw();
  look();
}

const markHtml = (wf) => wf && (wf.kind === "graph" || wf.kind === "api")
  ? `<span class="shot-mark" title="${wf.kind === "graph" ? "Carries its ComfyUI workflow" : "Carries its ComfyUI prompt, in the API format"}">${icon("workflow")}</span>`
  : "";

function metaHtml(item, wf) {
  const meta = item.meta || {};
  const rows = [];
  const add = (label, value, copyable) => {
    if (!value) return;
    rows.push(`<dt>${esc(label)}</dt><dd><span data-role="${copyable || ""}">${esc(String(value))}</span>`
      + (copyable ? ` <button class="mini" data-copy="${copyable}">Copy</button>` : "") + `</dd>`);
  };
  add("Prompt", meta.prompt, "prompt");
  add("Negative", meta.negative_prompt, "negative");
  add("Settings", [
    meta.model, meta.sampler,
    meta.steps && `${meta.steps} steps`,
    meta.cfg_scale && `cfg ${meta.cfg_scale}`,
    meta.seed && `seed ${meta.seed}`,
    meta.clip_skip && `clip skip ${meta.clip_skip}`,
    meta.size,
  ].filter(Boolean).join(" · "));
  const flow = workflowHtml(item, wf);
  if (flow) rows.push(flow);
  // rel=noreferrer: the service has no business learning which of someone's downloads
  // they were looking at.
  if (item.url) {
    rows.push(`<dt>Original</dt><dd><a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">open it on the service</a></dd>`);
  }
  if (item.local) rows.push(`<dt>Picture</dt><dd class="muted">kept beside the model</dd>`);
  return rows.length ? rows.join("")
    : `<dt></dt><dd class="muted">No generation settings were published with this one.</dd>`;
}

// The graph goes onto the clipboard as the JSON ComfyUI's canvas takes from a paste; the API
// format is not taken that way, only from a file dropped onto the canvas, so it is only saved.
function workflowHtml(item, wf) {
  if (item.type === "video") return "";
  const row = (html) => `<dt>Workflow</dt><dd class="wf-cell">${html}</dd>`;
  if (!wf) return row(`<span class="muted">Looking inside the picture…</span>`);
  const save = `<button class="mini" data-wf="save" title="${esc(saveTitle())}">${icon("save")}Save</button>`;
  if (wf.kind === "graph") {
    return row(`<div class="wf">ComfyUI · ${esc(plural(wf.nodes, "node"))}`
      + ` <button class="mini" data-wf="copy" title="Put the workflow on the clipboard">${icon("copy")}Copy</button> ${save}</div>`
      + `<div class="small muted">Ctrl+V on the ComfyUI canvas opens it in a new tab</div>`);
  }
  if (wf.kind === "api") {
    return row(`<div class="wf">ComfyUI, API format only · ${esc(plural(wf.nodes, "node"))} ${save}</div>`
      + `<div class="small muted">ComfyUI opens this one when the saved file is dropped onto its canvas</div>`);
  }
  if (wf.kind === "none") return row(`<span class="muted">None in this picture</span>`);
  if (wf.kind === "error") {
    return row(`<span class="muted">The picture could not be read${wf.error ? ` — ${esc(wf.error)}` : ""}</span>`
      + ` <button class="mini" data-wf="retry">Try again</button>`);
  }
  return "";
}

function saveTitle() {
  const folder = state.settings.workflow_dir || state.settings.workflow_dir_found;
  return folder ? `Save it into ${folder}, named after the model` : "Save it — you will be asked where, once";
}

async function saveWorkflow(source, index, button) {
  const url = `${previewBase(source)}/workflows/${index}/save`;
  try {
    let saved = await post(url);
    let chosen = null;
    if (saved.needs_folder) {
      // Asked once: the answer becomes the setting, and every later Save goes straight there.
      chosen = await act.pickSystemFolder(state.settings.library_root || "");
      if (!chosen) return;
      await act.saveSettings({ workflow_dir: chosen });
      saved = await post(url);
      if (!saved.ok) return;
    }
    flash(button, saved.existed ? "Already saved" : "Saved");
    const message = `${saved.existed ? "Already saved as" : "Saved as"} ${saved.name}`
      + (chosen ? ` in ${saved.folder} — Settings can change where` : "")
      + (saved.kind === "api" ? ". Drop the file onto the ComfyUI canvas to open it" : "");
    toast(message, {
      level: "ok",
      actions: [{ label: "Show", run: () => post("/api/workflows/reveal", { name: saved.name }).catch(toastError) }],
    });
  } catch (error) { toastError(error); }
}

export function wireCopies(holder) {
  holder.querySelectorAll("[data-copy]").forEach((button) => {
    button.onclick = async () => {
      const source = holder.querySelector(`[data-role="${button.dataset.copy}"]`);
      if (!source) return;
      if (await copyText(source.textContent)) flash(button);
      else toast("The clipboard would not take it", { level: "error" });
    };
  });
}

// Said on the button that was pressed: with several copy buttons on screen, the one that
// answers is the answer to which of them landed.
export function flash(button, said = "Copied") {
  if (!button) return;
  const was = button.innerHTML;
  button.textContent = said;
  button.classList.add("flashed");
  setTimeout(() => { button.innerHTML = was; button.classList.remove("flashed"); }, 1400);
}
