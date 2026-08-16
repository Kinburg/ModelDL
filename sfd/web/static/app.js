// The page's behaviour, in one file: a queue rendered from the server's snapshots, patched
// in place by an event stream. No framework and no build step — the whole thing is served
// by the same process that does the downloading, and has to keep working from inside a
// PyInstaller bundle where there is nothing to build with.

const $ = (id) => document.getElementById(id);
const tasks = new Map();
let categories = [];

const fmtBytes = (n) => {
  if (n === null || n === undefined) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n, i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
};
// Task Manager shows bits per second; downloaders traditionally show bytes. Comparing the
// two without noticing costs an evening of thinking the download is eight times slower
// than it is, so both are one click apart.
let useBits = localStorage.getItem("units") === "bits";
const fmtSpeed = (bytesPerSecond) => {
  if (!bytesPerSecond) return "0";
  if (!useBits) return `${fmtBytes(bytesPerSecond)}/s`;
  let v = bytesPerSecond * 8, i = 0;
  const units = ["bit", "Kbit", "Mbit", "Gbit"];
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}/s`;
};

const fmtEta = (s) => {
  if (!s || !isFinite(s) || s > 86400) return "—";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(sec).padStart(2, "0")}`;
};
const fmtWhen = (epoch) => epoch
  ? new Date(epoch * 1000).toLocaleString(undefined,
      { dateStyle: "short", timeStyle: "medium" })
  : "—";
const fmtDuration = (s) => {
  if (!s || !isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${sec}s`;
};

function message(text, bad = false) {
  $("message").textContent = text;
  $("message").className = "small " + (bad ? "err" : "muted");
}

// Snapshots arrive from two places — the event stream and the reply to whatever request
// just made the change — and they can overtake each other. A task claimed by a worker while
// the POST reply is still in flight would otherwise be redrawn as pending, showing "PENDING"
// over a progress bar that is visibly moving.
function upsert(task) {
  const known = tasks.get(task.id);
  if (known && known.updated_at > task.updated_at) return false;
  tasks.set(task.id, { ...known, ...task });
  return true;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" }, ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(explain(body.detail) || response.statusText);
  return body;
}

// A rejected setting comes back as FastAPI's list of field errors. Rendered raw that reads
// as "[object Object]", which says nothing about the field that was actually wrong.
function explain(detail) {
  if (!Array.isArray(detail)) return detail;
  return detail
    .map((e) => `${(e.loc || []).slice(-1)[0] || "request"}: ${e.msg}`)
    .join("; ");
}

// --- what is on screen ------------------------------------------------------

// Finished downloads stay in the list until they are cleared, and after an evening of
// collecting they are most of it. They are shown as one line each until asked to open.
const expanded = new Set();

const STATES = {
  all: () => true,
  active: (t) => ["running", "pending", "paused"].includes(t.state),
  blocked: (t) => t.state === "blocked",
  failed: (t) => t.state === "failed",
  done: (t) => t.state === "done",
};

function filtering() {
  return $("filter-state").value !== "all" || $("filter-text").value.trim() !== "";
}

function matches(task) {
  const wanted = $("filter-text").value.trim().toLowerCase();
  if (!(STATES[$("filter-state").value] || STATES.all)(task)) return false;
  if (!wanted) return true;
  return [task.filename, task.label, task.source, task.dest]
    .some((field) => (field || "").toLowerCase().includes(wanted));
}

function render() {
  const order = (t) => (t.position ?? t.id);
  const all = [...tasks.values()].sort((a, b) => order(a) - order(b) || a.id - b.id);
  const list = all.filter(matches);

  const active = all.filter((t) => t.state === "running" || t.state === "pending").length;
  const blocked = all.filter((t) => t.state === "blocked").length;
  const waiting = all.filter((t) => t.state === "paused").length;
  // Counts describe the whole queue, never just the part being looked at: a filter that
  // also hid the totals would make it impossible to tell a filtered list from an empty one.
  $("summary").textContent = all.length
    ? `${all.length} in queue · ${active} active`
      + (waiting ? ` · ${waiting} waiting to start` : "")
      + (blocked ? ` · ${blocked} need a decision` : "")
      + (list.length !== all.length ? ` · showing ${list.length}` : "")
    : "";
  $("empty").hidden = list.length > 0;
  $("empty").textContent = all.length ? "Nothing matches that filter." : "Nothing queued yet.";
  // Only worth offering when several are held; one has its own Resume button.
  $("start-all").hidden = waiting < 2;
  $("start-all").textContent = `Start all (${waiting})`;

  const pinned = filtering();
  $("tasks").innerHTML = list.map((t) => taskHtml(t, pinned)).join("");
  for (const task of list) {
    const node = document.querySelector(`[data-id="${task.id}"]`);
    if (!node) continue;
    node.querySelectorAll("button").forEach((b) => {
      b.onclick = () => act(task.id, b.dataset.action, node);
    });
    if (!pinned) wireDrag(node);
  }
  updateTotals();
}

// --- reordering -------------------------------------------------------------

let dragging = null;

// Only wired when nothing is filtered out. The reorder call sends the rows on screen, and
// applying that order while some are hidden would shuffle them around the ones it cannot
// see — a queue quietly scrambled by a search box is not a trade worth making.
function wireDrag(node) {
  // The card is only draggable while the grip is held. A permanently draggable card lets
  // the browser claim every mouse-down for a drag, so a path or a trigger word cannot be
  // selected to copy — and reading those is most of what the card is for.
  const grip = node.querySelector(".grip");
  grip.addEventListener("mousedown", () => { node.draggable = true; });
  const release = () => { node.draggable = false; };
  grip.addEventListener("mouseup", release);
  node.addEventListener("dragend", release);

  node.addEventListener("dragstart", (e) => {
    dragging = node;
    node.classList.add("dragging");
    e.dataTransfer.effectAllowed = "move";
    // Firefox refuses to start a drag without payload.
    e.dataTransfer.setData("text/plain", node.dataset.id);
  });
  node.addEventListener("dragend", async () => {
    node.classList.remove("dragging");
    document.querySelectorAll(".drop-target").forEach((n) => n.classList.remove("drop-target"));
    dragging = null;
    const ids = [...document.querySelectorAll(".task")].map((n) => Number(n.dataset.id));
    try { await api("/api/tasks/reorder", { method: "POST", body: JSON.stringify({ ids }) }); }
    catch (e) { message(e.message, true); }
  });
  node.addEventListener("dragover", (e) => {
    e.preventDefault();
    if (!dragging || dragging === node) return;
    node.classList.add("drop-target");
    // Insert before or after depending on which half of the row the cursor is over, so a
    // row can be dropped at either end of the list.
    const box = node.getBoundingClientRect();
    const after = e.clientY > box.top + box.height / 2;
    node.parentNode.insertBefore(dragging, after ? node.nextSibling : node);
  });
  node.addEventListener("dragleave", () => node.classList.remove("drop-target"));
  node.addEventListener("drop", (e) => e.preventDefault());
}

function taskHtml(t, pinned = false) {
  const pct = t.fraction ? Math.round(t.fraction * 100) : 0;
  const grip = `<span class="grip" title="${pinned
    ? "Clear the filter to reorder" : "Drag to reorder"}">⣿</span>`;

  if (t.state === "done" && !expanded.has(t.id)) {
    return `
      <div class="task done compact ${pinned ? "pinned" : ""}" data-id="${t.id}">
        ${grip}
        <div class="task-head">
          <span class="name">${escapeHtml(t.filename || t.source)}</span>
          <span class="state done">done</span>
          <span class="small muted">${fmtBytes(t.size)}</span>
          <span class="actions">
            <button data-action="expand" title="Show the details">⌄</button>
            ${t.dest ? `<button data-action="open-folder" title="Show in File Explorer">Folder</button>` : ""}
            <button class="danger" data-action="cancel">Remove</button>
          </span>
        </div>
      </div>`;
  }

  const buttons = [];
  if (t.state === "blocked") buttons.push(`<button data-action="confirm">Accept</button>`);
  if (t.state === "running" || t.state === "pending") buttons.push(`<button data-action="pause">Pause</button>`);
  if (t.state === "paused") buttons.push(`<button data-action="resume">Resume</button>`);
  if (t.state === "failed") buttons.push(`<button data-action="retry">Retry</button>`);
  if (t.state === "done") {
    buttons.push(`<button data-action="expand" title="Collapse">⌃</button>`);
    if (t.dest) buttons.push(`<button data-action="open-folder" title="Show in File Explorer">Folder</button>`);
    buttons.push(`<button data-action="record">Info</button>`);
  }
  buttons.push(`<button class="danger" data-action="cancel">Remove</button>`);

  const picker = t.state === "blocked"
    ? `<select data-role="category">${categories.map((c) =>
        `<option value="${c}" ${c === t.category ? "selected" : ""}>${c}</option>`).join("")}</select>`
    : "";

  const why = t.reason
    ? `<div class="why small muted">${escapeHtml(t.category || "")}
         ${t.confidence ? `(${t.confidence})` : ""} — ${escapeHtml(t.reason)}</div>`
    : "";

  const triggers = (t.trigger_words || []).length
    ? `<div class="tags">${t.trigger_words.map((w) => `<span class="tag">${escapeHtml(w)}</span>`).join("")}</div>`
    : "";

  // What actually happened, once it has. Average speed is computed from bytes really
  // fetched, so a resumed or skipped file does not claim credit for the whole size.
  const timing = [];
  if (t.finished_at) {
    timing.push(`finished ${fmtWhen(t.finished_at)}`);
    if (t.duration) timing.push(`took ${fmtDuration(t.duration)}`);
    if (t.average_speed) timing.push(`avg ${fmtSpeed(t.average_speed)}`);
    if (t.transferred === 0) timing.push("already present");
    else if (t.size && t.transferred && t.transferred < t.size * 0.98) {
      timing.push(`${fmtBytes(t.transferred)} fetched — resumed`);
    }
  } else if (t.started_at && t.state === "running") {
    timing.push(`started ${fmtWhen(t.started_at)}`);
  } else {
    timing.push(`added ${fmtWhen(t.created_at)}`);
  }
  // A failure that is going to fix itself should say so, or the queue looks abandoned.
  if (t.state === "failed" && t.retry_at) timing.push(`retrying ${fmtWhen(t.retry_at)}`);
  else if (t.attempts) timing.push(`attempt ${t.attempts + (t.state === "done" ? 0 : 1)}`);

  return `
    <div class="task ${t.state} ${pinned ? "pinned" : ""}" data-id="${t.id}">
      ${grip}
      <div class="task-head">
        <span class="name">${escapeHtml(t.filename || t.source)}</span>
        <span class="state ${t.state}">${t.state}</span>
        <span class="actions">${picker}${buttons.join("")}</span>
      </div>
      <div class="track"><div class="fill ${t.state === "done" ? "done" : ""}" style="width:${pct}%"></div></div>
      <div class="row small muted">
        <span data-role="stats">${fmtBytes(t.downloaded)} / ${fmtBytes(t.size)}</span>
        <span style="flex:1"></span>
        <span>${escapeHtml(t.label || "")}</span>
      </div>
      ${t.dest ? `<div class="dest muted small">${escapeHtml(t.dest)}</div>` : ""}
      ${why}
      ${t.disagreement ? `<div class="small" style="color:var(--warn)">the service lists this as ${escapeHtml(t.disagreement)}</div>` : ""}
      ${t.error ? `<div class="small err">${escapeHtml(t.error)}</div>` : ""}
      <div class="timing small muted">${timing.map((x) => `<span>${escapeHtml(x)}</span>`).join("")}</div>
      ${triggers}
      <div data-role="record"></div>
    </div>`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// --- the stored record ------------------------------------------------------

function recordHtml(record, path) {
  const source = record.source || {};
  const usage = record.usage || {};
  const integrity = record.integrity || {};
  const classification = record.classification || {};

  const rows = [];
  const add = (label, value) => value && rows.push(
    `<dt>${escapeHtml(label)}</dt><dd>${value}</dd>`);

  if (source.page) {
    const page = escapeHtml(source.page);
    // rel=noreferrer: the model page has no business learning where the click came from.
    add("Page", `<a href="${page}" target="_blank" rel="noopener noreferrer">${page}</a>`);
  }
  add("Model", escapeHtml([source.model_name, source.version_name].filter(Boolean).join(" / ")));
  add("Repository", escapeHtml(source.repo_id || ""));
  add("Commit", source.commit ? `<span class="mono">${escapeHtml(source.commit)}</span>` : "");
  add("Downloaded", escapeHtml(record.downloaded_at || ""));
  add("SHA256", integrity.sha256
    ? `<span class="mono">${escapeHtml(integrity.sha256)}</span>` : "");
  add("Base model", escapeHtml(usage.base_model || ""));
  add("Filed as", escapeHtml(
    `${classification.category || "?"} (${classification.confidence || "?"})`
    + (classification.service_called_it
       ? ` — the service called it ${classification.service_called_it}` : "")));
  add("Why", escapeHtml(classification.reason || ""));

  const words = usage.trigger_words || [];
  if (words.length) {
    add("Triggers",
      `<span class="mono" data-role="triggers">${escapeHtml(words.join(", "))}</span>`
      + ` <button data-action="copy-triggers" style="padding:1px 8px;font-size:11.5px">Copy</button>`);
  }
  add("Record", `<span class="mono">${escapeHtml(path)}</span>`);

  return `<dl class="record">${rows.join("")}
    <pre hidden data-role="raw">${escapeHtml(JSON.stringify(record, null, 2))}</pre>
    <dt></dt><dd><button data-action="raw" style="padding:2px 9px;font-size:12px">Raw JSON</button></dd>
  </dl>`;
}

async function toggleRecord(id, node) {
  const holder = node.querySelector('[data-role="record"]');
  if (holder.innerHTML) { holder.innerHTML = ""; return; }
  const { record, path } = await api(`/api/tasks/${id}/record`);
  holder.innerHTML = recordHtml(record, path);

  holder.querySelector('[data-action="raw"]').onclick = (e) => {
    const raw = holder.querySelector('[data-role="raw"]');
    raw.hidden = !raw.hidden;
    e.target.textContent = raw.hidden ? "Raw JSON" : "Hide JSON";
  };
  const copy = holder.querySelector('[data-action="copy-triggers"]');
  if (copy) copy.onclick = async (e) => {
    await navigator.clipboard.writeText(
      holder.querySelector('[data-role="triggers"]').textContent);
    e.target.textContent = "Copied";
    setTimeout(() => (e.target.textContent = "Copy"), 1500);
  };
}

async function act(id, action, node) {
  try {
    if (action === "record") { await toggleRecord(id, node); return; }
    if (action === "expand") {
      if (!expanded.delete(id)) expanded.add(id);
      render();
      return;
    }
    // The server knows where the file went; it does not need to be told by the page.
    if (action === "open-folder") { await api(`/api/tasks/${id}/reveal`, { method: "POST" }); return; }
    if (action === "cancel") await api(`/api/tasks/${id}`, { method: "DELETE" });
    else if (action === "confirm") {
      const picked = node.querySelector('[data-role="category"]');
      await api(`/api/tasks/${id}/confirm`, {
        method: "POST", body: JSON.stringify({ category: picked ? picked.value : null }),
      });
    } else await api(`/api/tasks/${id}/${action}`, { method: "POST" });
    message("");
  } catch (e) { message(e.message, true); }
}

// --- live updates -----------------------------------------------------------

function connect() {
  const stream = new EventSource("/api/events");
  stream.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "task") { if (upsert(data.task)) render(); }
    else if (data.type === "removed") { tasks.delete(data.id); render(); }
    else if (data.type === "reload") load();
    else if (data.type === "progress") {
      const task = tasks.get(data.id);
      if (!task) return;
      task.downloaded = data.downloaded;
      task.size = data.total ?? task.size;
      task.fraction = task.size ? data.downloaded / task.size : 0;
      // Patch in place: re-rendering the whole list several times a second would fight
      // with the user's own clicks and scrolling.
      const node = document.querySelector(`[data-id="${data.id}"]`);
      if (!node) { render(); return; }
      task.speed = data.speed;
      node.querySelector(".fill").style.width = `${Math.round(task.fraction * 100)}%`;
      node.querySelector('[data-role="stats"]').textContent =
        `${fmtBytes(data.downloaded)} / ${fmtBytes(task.size)} · ${fmtSpeed(data.speed)}`
        + ` · ${data.connections} conn · ETA ${fmtEta(data.eta)}`;
      updateTotals();
    }
  };
  stream.onerror = () => { stream.close(); setTimeout(connect, 2000); };
}

async function load() {
  const { tasks: list } = await api("/api/tasks");
  tasks.clear();
  list.forEach((t) => tasks.set(t.id, t));
  render();
  refreshSpace();
}

// What the queue as a whole is doing. Per-file progress answers "is this one moving"; the
// question a 67 GB queue actually raises is whether it finishes before morning.
function updateTotals() {
  const node = $("totals");
  const unfinished = [...tasks.values()].filter((t) => t.state !== "done");
  const sized = unfinished.filter((t) => t.size);
  const total = sized.reduce((sum, t) => sum + t.size, 0);
  const remaining = sized.reduce((sum, t) => sum + Math.max(0, t.size - (t.downloaded || 0)), 0);
  const speed = unfinished.reduce(
    (sum, t) => sum + (t.state === "running" ? t.speed || 0 : 0), 0);

  if (!remaining) { node.hidden = true; return; }
  const parts = [`${fmtBytes(total - remaining)} of ${fmtBytes(total)}`];
  // Only worth an ETA while something is actually moving; a paused queue would otherwise
  // divide by zero and claim to be finishing forever.
  if (speed > 0) parts.push(fmtSpeed(speed), `ETA ${fmtEta(remaining / speed)}`);
  if (unfinished.length > sized.length) {
    parts.push(`${unfinished.length - sized.length} of unknown size`);
  }
  node.hidden = false;
  node.textContent = `· ${parts.join(" · ")}`;
}

// Said while the queue is being assembled, "this will not fit" is one line and a decision
// about what to drop. Said by the disk at four in the morning, it is a row of failures.
async function refreshSpace() {
  const node = $("space");
  try {
    const { needed, free, unknown, path } = await api("/api/space");
    const short = free !== null && needed > free;
    node.hidden = !short;
    if (short) {
      node.textContent = `· needs ${fmtBytes(needed)}, ${fmtBytes(free)} free on ${path}`
        + (unknown ? ` (${unknown} of unknown size)` : "");
    }
  } catch { node.hidden = true; }
}

// --- settings ---------------------------------------------------------------

const FIELDS = ["library_root", "profile", "connections", "concurrent_downloads",
                "disk_kind", "sidecar_dir", "hf_engine", "queue_position", "max_speed_kb"];
const CHECKS = ["group_by_base_model", "verify_hash", "write_sidecars",
                "write_compat_files", "write_trigger_txt", "hf_fallback", "auto_start",
                "auto_retry"];

const settingsOpen = () => $("settings-backdrop").classList.contains("open");

function openSettings() {
  $("settings-backdrop").classList.add("open");
  // The queue behind the dialog must not scroll under it, and typing should land in the
  // dialog rather than in the link box it is covering.
  document.body.style.overflow = "hidden";
  $("library_root").focus();
}

function closeSettings() {
  $("settings-backdrop").classList.remove("open");
  document.body.style.overflow = "";
  $("layout").textContent = "";
  $("source").focus();
}

async function loadSettings() {
  const { settings, categories: cats, error } = await api("/api/settings");
  categories = cats;
  if (error) { message(error, true); openSettings(); }
  FIELDS.forEach((k) => { if ($(k)) $(k).value = settings[k] ?? ""; });
  CHECKS.forEach((k) => { if ($(k)) $(k).checked = !!settings[k]; });
  $("hf_token").placeholder = settings.hf_token_from_env
    ? "set from $HF_TOKEN" : settings.hf_token_set ? "saved — leave blank to keep" : "not set";
  $("civitai_token").placeholder = settings.civitai_token_from_env
    ? "set from $CIVITAI_TOKEN" : settings.civitai_token_set ? "saved — leave blank to keep" : "not set";
}

$("save-settings").onclick = async () => {
  const patch = {};
  FIELDS.forEach((k) => { patch[k] = $(k).type === "number" ? Number($(k).value) : $(k).value; });
  CHECKS.forEach((k) => { patch[k] = $(k).checked; });
  patch.hf_token = $("hf_token").value;
  patch.civitai_token = $("civitai_token").value;
  try {
    await api("/api/settings", { method: "PUT", body: JSON.stringify(patch) });
    $("hf_token").value = ""; $("civitai_token").value = "";
    $("settings-saved").textContent = "saved";
    setTimeout(() => ($("settings-saved").textContent = ""), 2000);
    await loadSettings();
  } catch (e) { message(e.message, true); }
};

$("show-layout").onclick = async () => {
  const layout = await api("/api/layout");
  if (!layout.root) { $("layout").textContent = "No library root set — everything goes to one folder."; return; }
  const lines = Object.entries(layout.paths).map(([category, path]) =>
    `${layout.exists[category] ? "  " : "* "}${category.padEnd(17)} ${path}`);
  const ambiguous = Object.entries(layout.ambiguities || {}).map(([c, others]) =>
    `  ${c}: also found ${others.join(", ")}`);
  $("layout").textContent =
    lines.join("\n") + "\n\n* = would be created\n"
    + (ambiguous.length ? "\nseveral folders could have served:\n" + ambiguous.join("\n") : "");
};

$("toggle-settings").onclick = openSettings;
$("close-settings").onclick = closeSettings;
$("settings-backdrop").addEventListener("click", (e) => { if (e.target === $("settings-backdrop")) closeSettings(); });

$("add").onclick = async () => {
  const source = $("source").value.trim();
  if (!source) return;
  $("add").disabled = true;
  message("resolving…");
  try {
    const { tasks: created } = await api("/api/tasks", {
      method: "POST", body: JSON.stringify({ source }),
    });
    created.forEach(upsert);
    $("source").value = "";
    message(`added ${created.length} file(s)`);
    render();
    refreshSpace();
  } catch (e) { message(e.message, true); }
  finally { $("add").disabled = false; }
};

$("source").addEventListener("keydown", (e) => { if (e.key === "Enter") $("add").click(); });
$("clear-done").onclick = async () => { await api("/api/tasks/clear", { method: "POST" }); await load(); };

$("start-all").onclick = async () => {
  try {
    const { released } = await api("/api/tasks/start-all", { method: "POST" });
    await load();
    message(`started ${released} download(s)`);
  } catch (e) { message(e.message, true); }
};

// The state filter is remembered; the search box is not. Coming back to a queue that looks
// empty because of yesterday's search is worse than retyping four letters.
$("filter-state").value = localStorage.getItem("filter-state") || "all";
$("filter-state").onchange = () => {
  localStorage.setItem("filter-state", $("filter-state").value);
  render();
};
$("filter-text").oninput = render;

$("toggle-units").onclick = () => {
  useBits = !useBits;
  localStorage.setItem("units", useBits ? "bits" : "bytes");
  $("toggle-units").textContent = useBits ? "Mbit/s" : "MB/s";
};
$("toggle-units").textContent = useBits ? "Mbit/s" : "MB/s";

// --- desktop integration & shortcuts ----------------------------------------

async function pickFolder(initial = "") {
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.pick_folder) {
      return await window.pywebview.api.pick_folder(initial);
    }
    const res = await api("/api/utils/pick-folder", {
      method: "POST",
      body: JSON.stringify({ initial }),
    });
    return res.path || null;
  } catch (e) {
    message(e.message, true);
    return null;
  }
}

document.querySelectorAll(".btn-browse").forEach((btn) => {
  btn.onclick = async () => {
    const target = $(btn.dataset.target);
    if (!target) return;
    const selected = await pickFolder(target.value.trim());
    if (selected) {
      target.value = selected;
      target.dispatchEvent(new Event("input"));
    }
  };
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (settingsOpen()) closeSettings();
    else message("");
  }
  // Paste anywhere and the link lands in the box it was meant for — unless something is
  // already taking the keystroke, or the settings dialog is what is in front of you.
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "v" && !settingsOpen()) {
    const active = document.activeElement;
    if (!active || (active.tagName !== "INPUT" && active.tagName !== "TEXTAREA" && active.tagName !== "SELECT")) {
      $("source").focus();
    }
  }
});

loadSettings().then(load).then(connect).catch((e) => message(e.message, true));
