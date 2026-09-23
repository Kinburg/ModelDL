// Sample images at full size, with the prompt and settings that produced each one. A LoRA's
// trigger words say which tokens wake it up and nothing about the prompt around them; the
// samples were made by someone who knew.

import { get } from "./api.js";
import { copyText, esc } from "./util.js";
import { icon } from "./icons.js";
import { modal } from "./dialogs.js";
import { state, invalidate } from "./store.js";
import { toast } from "./toasts.js";

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
  const handle = modal({
    title: title || "Samples",
    wide: true,
    className: "lightbox",
    body: `<div class="stage" data-role="stage"></div>
           <div class="strip" data-role="strip"></div>
           <dl class="record" data-role="meta"></dl>`,
  });
  const $ = (role) => handle.dialog.querySelector(`[data-role="${role}"]`);

  const draw = () => {
    const item = items[at];
    const hidden = covered(source, item.nsfw);
    handle.dialog.querySelector(".modal-title").textContent =
      `${title || "Samples"}${items.length > 1 ? ` — ${at + 1} of ${items.length}` : ""}`;
    const media = item.type === "video"
      ? `<video src="${previewSrc(source, at)}" autoplay loop muted playsinline controls></video>`
      : `<img src="${previewSrc(source, at, 1024)}" alt="" draggable="false">`;
    $("stage").innerHTML =
      (items.length > 1 ? `<button class="nav" data-step="-1" aria-label="Previous">${icon("chevron-right", "flip")}</button>` : "")
      + `<div class="frame ${hidden ? "covered" : ""}">${media}`
      + (hidden ? `<button class="uncover">Marked adult — show it</button>` : "")
      + `</div>`
      + (items.length > 1 ? `<button class="nav" data-step="1" aria-label="Next">${icon("chevron-right")}</button>` : "");
    $("strip").hidden = items.length < 2;
    $("strip").innerHTML = items.map((p, i) => `
      <img class="shot ${i === at ? "current" : ""} ${covered(source, p.nsfw) ? "covered" : ""}"
           src="${previewSrc(source, i, 160)}" data-index="${i}" loading="lazy" alt="" draggable="false">`).join("");
    $("meta").innerHTML = metaHtml(item);
    $("stage").querySelectorAll("[data-step]").forEach((b) => {
      b.onclick = () => { at = (at + Number(b.dataset.step) + items.length) % items.length; draw(); };
    });
    const uncover = $("stage").querySelector(".uncover");
    if (uncover) uncover.onclick = () => { state.revealed.add(coverKey(source)); draw(); invalidate("list", "inspector"); };
    $("strip").querySelectorAll(".shot").forEach((img) => {
      img.onclick = () => { at = Number(img.dataset.index); draw(); };
    });
    wireCopies($("meta"));
  };

  const keys = (event) => {
    if (!document.body.contains(handle.backdrop)) { window.removeEventListener("keydown", keys, true); return; }
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      event.stopPropagation();
      at = (at + (event.key === "ArrowLeft" ? -1 : 1) + items.length) % items.length;
      draw();
    }
  };
  window.addEventListener("keydown", keys, true);
  draw();
}

function metaHtml(item) {
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
  // rel=noreferrer: the service has no business learning which of someone's downloads
  // they were looking at.
  if (item.url) {
    rows.push(`<dt>Original</dt><dd><a href="${esc(item.url)}" target="_blank" rel="noopener noreferrer">open it on the service</a></dd>`);
  }
  if (item.local) rows.push(`<dt>Picture</dt><dd class="muted">kept beside the model</dd>`);
  return rows.length ? rows.join("")
    : `<dt></dt><dd class="muted">No generation settings were published with this one.</dd>`;
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
