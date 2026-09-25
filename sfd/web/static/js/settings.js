// Settings, as a page of its own rather than a dialog: the library's folders come first,
// because they decide what everything else in the window shows.

import { get, post } from "./api.js";
import { esc, fmtBytes } from "./util.js";
import { icon } from "./icons.js";
import { state } from "./store.js";
import { toastError } from "./toasts.js";
import * as act from "./actions.js";

const FIELDS = ["profile", "connections", "concurrent_downloads", "disk_kind", "sidecar_dir",
  "queue_position", "max_speed_kb", "download_dir", "workflow_dir"];
const CHECKS = ["group_by_base_model", "smart_placement", "verify_hash", "write_sidecars", "write_compat_files",
  "write_trigger_txt", "auto_start", "auto_retry", "fetch_previews", "blur_nsfw"];

let dirty = false;

const field = (id, label, input, help = "") => `
  <label for="set-${id}">${esc(label)}${help ? `<span class="help" title="${esc(help)}">?</span>` : ""}</label>
  <div>${input}</div>`;
const text = (id, placeholder = "") => `<input id="set-${id}" data-field="${id}" placeholder="${esc(placeholder)}" spellcheck="false">`;
const number = (id, min, max, step = 1, placeholder = "") =>
  `<input id="set-${id}" data-field="${id}" type="number" min="${min}" ${max ? `max="${max}"` : ""} step="${step}" placeholder="${esc(placeholder)}">`;
const check = (id) => `<input id="set-${id}" data-check="${id}" type="checkbox">`;
const select = (id, options) => `<select id="set-${id}" data-field="${id}">${options.map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")}</select>`;
const browse = (id, placeholder) => `<div class="input-with-button">${text(id, placeholder)}
  <button type="button" data-browse="${id}" title="Choose in the system's folder dialog">Browse…</button></div>`;

function rootsHtml() {
  if (!state.roots.length) return `<div class="small muted">No folders yet.</div>`;
  return state.roots.map((root) => `
    <div class="root-row">
      ${icon("drive")}
      <div class="root-main">
        <div class="mono">${esc(root.path)}</div>
        <div class="small muted">${root.primary ? (root.downloads ? "the downloads folder — used while no library folder is set" : "downloads are filed into this folder") : "read into the library"}
          ${root.free != null ? ` · ${fmtBytes(root.free)} free` : ""}${root.exists ? "" : ` · <span class="warn">not found</span>`}</div>
      </div>
      ${root.primary ? `<span class="chip kind">main</span>` : `<button class="mini" data-root-do="primary" data-index="${root.index}">File downloads here</button>`}
      <button class="icon-button" data-root-do="reveal" data-index="${root.index}" title="Show in Explorer">${icon("external")}</button>
      ${root.downloads ? "" : `<button class="icon-button" data-root-do="remove" data-index="${root.index}" title="Take off the list">${icon("x")}</button>`}
    </div>`).join("");
}

// What an empty field means right now: ComfyUI's own folder if the library is inside an
// install, otherwise a question the first time a workflow is saved.
function workflowPlaceholder() {
  const found = state.settings.workflow_dir_found;
  return found ? `empty = ComfyUI's own, ${found}` : "empty = asked the first time you save one";
}

function hiddenHtml() {
  if (!state.hidden.length) return "";
  return `<div class="settings-sub">Hidden from the library</div>` + state.hidden.map((path, index) => `
    <div class="root-row quiet">${icon("eye-off")}<div class="root-main mono">${esc(path)}</div>
      <button class="mini" data-unhide="${index}">Show again</button></div>`).join("");
}

export function renderSettings() {
  const holder = document.getElementById("center");
  const existing = holder.querySelector("#settings-form");
  if (existing) {
    existing.querySelector("[data-role=roots]").innerHTML = rootsHtml();
    existing.querySelector("[data-role=hidden]").innerHTML = hiddenHtml();
    return;
  }
  dirty = false;
  holder.innerHTML = `
    <div class="list-head"><div class="list-title">Settings</div><span class="grow"></span>
      <span class="small muted" data-role="saved"></span>
      <button class="primary" data-role="save">${icon("check")}Save</button></div>
    <div class="settings" id="settings-form">
      <section>
        <h3>Library folders</h3>
        <p class="small muted">Every folder models are kept in — ComfyUI can read from several, on several drives, and so can this. Downloads are sorted into the main one, using the folders it already has.</p>
        <div data-role="roots">${rootsHtml()}</div>
        <button data-root-do="add">${icon("plus")}Add a folder…</button>
        <div data-role="hidden">${hiddenHtml()}</div>
        <div class="grid">
          ${field("profile", "Folder names", select("profile", [["comfyui", "ComfyUI"], ["a1111", "A1111"]]), "The names used for a kind of model that has no folder yet")}
          ${field("group_by_base_model", "Group by base model", check("group_by_base_model"), "loras/Pony, checkpoints/Flux.1 D — a subfolder per base model")}
          ${field("smart_placement", "Smart download placement", check("smart_placement"), "On: each new download is filed into the folder that suits it, and asks only when that is uncertain. Off: every download asks where it goes, the likeliest folder first")}
          ${field("download_dir", "Downloads folder", browse("download_dir", "downloads"), "Where files go while no library folder is set")}
        </div>
        <button data-role="layout">Check folder mapping</button>
        <pre class="small muted" data-role="layout-out" hidden></pre>
      </section>
      <section>
        <h3>Downloads</h3>
        <div class="grid">
          ${field("connections", "Connections per file", number("connections", 1, 64))}
          ${field("concurrent_downloads", "Files at once", number("concurrent_downloads", 1, 8))}
          ${field("auto_start", "Start downloads on add", check("auto_start"), "Off: added links wait paused until you press Start all")}
          ${field("queue_position", "Add new downloads to", select("queue_position", [["bottom", "the bottom of the list"], ["top", "the top of the list"]]))}
          ${field("max_speed_kb", "Speed limit, KB/s", number("max_speed_kb", 0, null, 64, "0 = unlimited"), "Shared by every connection of every file. Takes effect immediately")}
          ${field("auto_retry", "Retry failures on their own", check("auto_retry"), "After 30 s, 2 min and 10 min. A missing token or a full disk is never retried")}
          ${field("disk_kind", "Target disk", select("disk_kind", [["", "detect automatically"], ["ssd", "SSD / NVMe"], ["hdd", "mechanical"]]))}
          ${field("verify_hash", "Verify checksums", check("verify_hash"))}
        </div>
      </section>
      <section>
        <h3>Beside each model</h3>
        <div class="grid">
          ${field("write_sidecars", "Write a .json record", check("write_sidecars"), "Where it came from, its hash, its trigger words — and your note")}
          ${field("sidecar_dir", "Keep records in", browse("sidecar_dir", "empty = beside each model"), "A folder collects them, mirroring the library's folders")}
          ${field("write_compat_files", "Also .civitai.info and a preview", check("write_compat_files"), "Read by the A1111 and ComfyUI model managers, beside the model")}
          ${field("write_trigger_txt", "Trigger words as .txt", check("write_trigger_txt"), "Loaders paste this file into the prompt, so it holds the trigger words and nothing else")}
          ${field("fetch_previews", "Sample images", check("fetch_previews"), "The pictures a model is published with, and the prompts that made them")}
          ${field("blur_nsfw", "Cover adult samples", check("blur_nsfw"), "Uncovered by a click, until the app is restarted")}
          ${field("workflow_dir", "Save workflows to", browse("workflow_dir", workflowPlaceholder()), "Where Save in the sample viewer puts the ComfyUI workflow a picture carries, named after its model: lenovo_qwen21 - sample 2.json. Empty: ComfyUI's own workflows folder, when a library folder is the models folder of a ComfyUI install")}
        </div>
      </section>
      <section>
        <h3>Accounts</h3>
        <div class="grid">
          ${field("hf_token", "HuggingFace token", `<input id="set-hf_token" data-token="hf_token" type="password" autocomplete="off">`, "For gated and private models. $HF_TOKEN overrides it; with neither, the token saved by hf auth login is used")}
          ${field("civitai_token", "Civitai API key", `<input id="set-civitai_token" data-token="civitai_token" type="password" autocomplete="off">`)}
        </div>
      </section>
      <section>
        <h3>This app's own files</h3>
        <p class="small muted">These settings, the download history and the cache of sample pictures are kept in this folder.</p>
        <div class="root-row quiet">${icon("drive")}<div class="root-main mono">${esc(state.settings.data_dir || "")}</div>
          <button class="icon-button" data-role="data-dir" title="Show in Explorer">${icon("external")}</button></div>
      </section>
    </div>`;
  fill(holder);
  wire(holder);
}

function fill(holder) {
  const s = state.settings;
  for (const key of FIELDS) {
    const input = holder.querySelector(`[data-field="${key}"]`);
    if (input) input.value = s[key] ?? "";
  }
  for (const key of CHECKS) {
    const input = holder.querySelector(`[data-check="${key}"]`);
    if (input) input.checked = !!s[key];
  }
  // In the order the server picks a token in, so the hint names the one actually in use.
  const hint = (key, env) => {
    if (s[`${key}_from_env`]) return `set from $${env}`;
    if (s[`${key}_set`]) return "saved — leave blank to keep";
    if (s[`${key}_from_login`]) return "not set — using the one from hf auth login";
    if (s[`${key}_login_expired`]) return "not set — the hf auth login one has expired, log in again";
    return "not set";
  };
  holder.querySelector('[data-token="hf_token"]').placeholder = hint("hf_token", "HF_TOKEN");
  holder.querySelector('[data-token="civitai_token"]').placeholder = hint("civitai_token", "CIVITAI_TOKEN");
}

function wire(holder) {
  const saved = holder.querySelector('[data-role="saved"]');
  // Everything is wired to the form, which is new on every visit to this page, and never to
  // the pane around it, which is not: a listener on the pane would be added again each time
  // and a click on "Take off the list" would take off one folder per visit.
  const form = holder.querySelector("#settings-form");
  const mark = () => { dirty = true; saved.textContent = "Unsaved changes"; saved.className = "small warn"; };
  form.addEventListener("input", (event) => {
    if (event.target.matches("[data-field], [data-check], [data-token]")) mark();
  });
  form.addEventListener("change", (event) => {
    if (event.target.matches("[data-field], [data-check]")) mark();
  });

  holder.querySelector('[data-role="save"]').onclick = async () => {
    const patch = {};
    for (const key of FIELDS) {
      const input = holder.querySelector(`[data-field="${key}"]`);
      if (!input) continue;
      patch[key] = input.type === "number" ? Number(input.value || 0) : input.value;
    }
    if (!patch.download_dir) patch.download_dir = "downloads";
    for (const key of CHECKS) patch[key] = holder.querySelector(`[data-check="${key}"]`).checked;
    for (const input of holder.querySelectorAll("[data-token]")) {
      if (input.value) patch[input.dataset.token] = input.value;
    }
    try {
      await act.saveSettings(patch);
      holder.querySelectorAll("[data-token]").forEach((i) => { i.value = ""; });
      dirty = false;
      saved.textContent = "Saved";
      saved.className = "small ok";
      fill(holder);
      await act.loadLibrary();
    } catch (error) { toastError(error); }
  };

  form.addEventListener("click", async (event) => {
    const root = event.target.closest("[data-root-do]");
    if (root) {
      const index = Number(root.dataset.index);
      const what = root.dataset.rootDo;
      if (what === "add") await act.addRoot();
      if (what === "remove") await act.removeRoot(index, { stay: true });
      if (what === "primary") await act.makePrimary(index, { stay: true });
      if (what === "reveal") await act.revealFolder(index, "");
      if (what !== "reveal") renderSettings();
      return;
    }
    const unhide = event.target.closest("[data-unhide]");
    if (unhide) { await act.unhideFolder(Number(unhide.dataset.unhide)); renderSettings(); return; }
    if (event.target.closest('[data-role="data-dir"]')) { post("/api/data-dir/reveal").catch(toastError); return; }
    const browseButton = event.target.closest("[data-browse]");
    if (browseButton) {
      const input = holder.querySelector(`[data-field="${browseButton.dataset.browse}"]`);
      const chosen = await act.pickSystemFolder(input.value.trim());
      if (chosen) { input.value = chosen; mark(); }
      return;
    }
    if (event.target.closest('[data-role="layout"]')) {
      const out = holder.querySelector('[data-role="layout-out"]');
      try {
        const layout = await get("/api/layout");
        out.hidden = false;
        if (!layout.root) { out.textContent = "No library folder is set — everything goes to one folder."; return; }
        const lines = Object.entries(layout.paths).map(([category, path]) =>
          `${layout.exists[category] ? "  " : "* "}${category.padEnd(17)} ${path}`);
        const ambiguous = Object.entries(layout.ambiguities || {}).map(([c, others]) => `  ${c}: also found ${others.join(", ")}`);
        out.textContent = lines.join("\n") + "\n\n* = would be created\n"
          + (ambiguous.length ? "\nseveral folders could have served:\n" + ambiguous.join("\n") : "");
      } catch (error) { toastError(error); }
    }
  });
}

// Leaving the page with changes nobody saved is asked about, once, rather than done.
state.leaving = () => {
  if (state.view.kind !== "settings" || !dirty) return true;
  if (!window.confirm("Leave Settings without saving your changes?")) return false;
  dirty = false;
  return true;
};
