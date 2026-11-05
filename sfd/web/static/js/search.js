// Looking through Civitai for what is new: a search with filters the site does not have —
// tags to leave out, sold versions, authors hidden, versions already seen — kept as presets
// in the sidebar.
//
// The middle pane is built once for a search and then changed in parts: the box being typed
// into is never redrawn under the caret, and the list keeps where it was scrolled to while
// more of it is read in. A version counts as seen once it has been on screen for a moment,
// or scrolled past; it stays in the list until the next search, without its dot.

import { del, get, post, put } from "./api.js";
import { esc, fmtAgo, fmtBytes, fmtDate, plural, accessChip, paidSentence, paidNow, debounce, copyText } from "./util.js";
import { icon, kindIcon } from "./icons.js";
import { state, invalidate, remember, select } from "./store.js";
import { menuFrom } from "./menu.js";
import { confirmDialog, promptDialog } from "./dialogs.js";
import { covered, openLightbox, previewSrc, previewPair, previewsOf } from "./lightbox.js";
import { toast, toastError } from "./toasts.js";
import * as act from "./actions.js";

export const DEFAULTS = {
  query: "", base_models: [], types: [], period: "Week", sort: "Newest", site: "civitai.com",
  exclude_tags: [], paid: "all", seen: "show", hidden_authors: "hide",
};
const LORAS = ["LORA", "LoCon", "DoRA"];
const TYPE_LABELS = {
  LORA: "LoRA", TextualInversion: "Embedding", Controlnet: "ControlNet", MotionModule: "Motion module",
  TextEncoder: "Text encoder", CLIPVision: "CLIP vision", AestheticGradient: "Aesthetic gradient",
};
const typeLabel = (type) => TYPE_LABELS[type] || type || "Model";
const PERIODS = [["Day", "Day"], ["Week", "Week"], ["Month", "Month"], ["Year", "Year"], ["AllTime", "All"]];
const SORTS = [["Newest", "Newest"], ["Most Downloaded", "Downloads"], ["Highest Rated", "Rating"]];
const SITES = [["civitai.com", "civitai.com", "Safe for work only"], ["civitai.red", "civitai.red", "With NSFW — what civitai.red shows"]];
const PAID = [
  ["all", "Show all", "Sold and early-access versions are shown, marked"],
  ["no_sold", "Hide sold", "Versions sold for good are left out; early access stays, as it is free from a date. Bought ones stay too"],
  ["free", "Free only", "Only what can be downloaded for nothing now — and what your API key has bought"],
];
// Why a version was left out, as the list's foot says it.
const LEFT = {
  tags: "by tags", paid: "paid", "only on the site": "only usable on the site", seen: "seen",
  "hidden authors": "by hidden authors", "no version for the base models": "with no version for the base models",
};

// --- what the page keeps ---------------------------------------------------------------------

const S = {
  mounted: "",        // the view the pane was built for: "free", or "p<id>"
  saved: null,        // a preset's filters as saved, to tell a change by
  version: 0,
  rowsSig: "",
  panelSig: "",
  optionsAsked: false,
  // Marked not seen by hand: not marked seen again for being on screen.
  keepUnseen: new Set(),
};

// The list of each view — the free search, and every preset — as it was left: what it found,
// where it was scrolled to, what was selected, and the filters as they are in its panel. It
// is searched only when asked: opening a view shows what it had, or asks to be searched.
const lists = new Map();
const blank = (filters) => ({
  filters,            // as the panel has them, applied or not
  applied: null,      // as the list on screen was searched with; null before the first search
  ticket: 0,
  token: null,
  cards: [],
  byId: new Map(),
  done: true,
  scanned: 0,
  shown: 0,
  leftOut: {},
  loading: false,
  error: "",
  // Steps in a row that found nothing, while the list was too short to scroll: past a few,
  // reading on is left to a button rather than done on its own.
  empty: 0,
  scroll: 0,
  selection: [],
});
let L = blank({ ...DEFAULTS });

const copy = (filters) => JSON.parse(JSON.stringify({ ...DEFAULTS, ...(filters || {}) }));
// The same search whatever order its base models, types and tags were picked in: LoRA then
// Checkpoint asks Civitai exactly what Checkpoint then LoRA does.
const canon = (filters) => {
  const f = copy(filters);
  for (const key of Object.keys(f)) if (Array.isArray(f[key])) f[key] = [...f[key]].sort();
  return JSON.stringify(f);
};
const same = (a, b) => canon(a) === canon(b);
const viewKey = (view) => (view.preset ? `p${view.preset}` : "free");
export const presetOf = (id) => state.civitai.presets.find((p) => p.id === id) || null;
export const foundCard = (id) => L.byId.get(id) || null;
// The filters the list on screen was searched with — what its rows are, whatever the panel
// says now.
const shownWith = () => L.applied || L.filters;
const pending = () => !!L.applied && !same(L.filters, L.applied);

export async function loadCivitai() {
  try {
    const body = await get("/api/civitai/state");
    state.civitai.presets = body.presets || [];
    state.civitai.hidden = body.hidden_authors || [];
    state.civitai.loaded = true;
    invalidate("tree");
    if (state.view.kind === "civitai") invalidate("list");
  } catch (error) { toastError(error); }
}

async function loadOptions() {
  if (S.optionsAsked) return;
  S.optionsAsked = true;
  try {
    state.civitai.options = await get("/api/civitai/options");
  } catch {
    S.optionsAsked = false;
    return;
  }
  S.panelSig = "";
  invalidate("list");
}

function takePresets(presets) {
  if (!presets) return;
  state.civitai.presets = presets;
  invalidate("tree");
}

// --- searching -----------------------------------------------------------------------------

const changed = () => state.view.preset && S.saved && !same(L.filters, S.saved);

// A filter changed is only the panel changed: the list stays what it was until Apply.
function setFilters(patch) {
  L.filters = { ...L.filters, ...patch };
  if (viewKey(state.view) === "free") remember({ civitai_free: copy(L.filters) });
  paintHead();
  paintPanel();
}

// Search with the filters as the panel has them — the first search of a list, Apply, or
// Search again.
export function apply() {
  const input = document.querySelector('.search-pane [data-role="query"]');
  if (input && input.value.trim() !== L.filters.query) setFilters({ query: input.value.trim() });
  search();
}

async function search() {
  const list = L;
  const ticket = ++list.ticket;
  Object.assign(list, {
    token: null, cards: [], byId: new Map(), done: false, scanned: 0, shown: 0, leftOut: {},
    loading: true, error: "", empty: 0, scroll: 0, selection: [], applied: copy(list.filters),
  });
  S.version++;
  // Redrawn through the page's own redraw, which also tells the keyboard what the rows are.
  select([]);
  const preset = lists.get(viewKey(state.view)) === list ? state.view.preset || null : null;
  try {
    const body = await post("/api/civitai/search", { filters: list.applied, preset_id: preset });
    if (ticket === list.ticket) take(list, body);
  } catch (error) {
    if (ticket === list.ticket) { list.error = error.message; list.done = true; }
  } finally {
    if (ticket === list.ticket) { list.loading = false; S.version++; invalidate("list"); }
  }
}

async function more() {
  const list = L;
  if (list.loading || list.done || !list.token) return;
  const ticket = list.ticket;
  list.loading = true;
  paintFoot();
  try {
    const body = await post(`/api/civitai/search/${encodeURIComponent(list.token)}/more`);
    if (ticket === list.ticket) take(list, body);
  } catch (error) {
    if (ticket === list.ticket) { list.error = error.message; list.done = error.status === 404 || list.done; }
  } finally {
    if (ticket === list.ticket) { list.loading = false; S.version++; invalidate("list"); }
  }
}

function take(list, body) {
  list.token = body.token;
  let added = 0;
  for (const card of body.cards || []) {
    if (list.byId.has(card.version_id)) continue;
    list.byId.set(card.version_id, card);
    list.cards.push(card);
    added++;
  }
  list.empty = added ? 0 : list.empty + 1;
  list.done = !!body.done;
  list.scanned = body.scanned || 0;
  list.shown = body.shown || list.cards.length;
  list.leftOut = body.left_out || {};
  takePresets(body.presets);
}

// --- what the page does with what it found -------------------------------------------------

// Versions seen and not yet said, with the search each was seen in — the one whose preset
// counts it.
let pendingSeen = new Map();
const flushSeen = debounce(sendSeen, 1200);

function bySearch(batch) {
  const groups = new Map();
  for (const [id, token] of batch) {
    if (!groups.has(token)) groups.set(token, []);
    groups.get(token).push(id);
  }
  return groups;
}

async function sendSeen() {
  if (!pendingSeen.size) return;
  const batch = pendingSeen;
  pendingSeen = new Map();
  for (const [token, ids] of bySearch(batch)) {
    try {
      const body = await post("/api/civitai/seen", { ids, seen: true, token });
      takePresets(body.presets);
    } catch { /* seen again next time: nothing is lost but a dot */ }
  }
}

// On the way out, with `keepalive`, as the window's other unsaved bits are.
export function flushSeenOnExit() {
  if (!pendingSeen.size) return;
  for (const [token, ids] of bySearch(pendingSeen)) {
    try {
      fetch("/api/civitai/seen", {
        method: "POST", keepalive: true, headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ids, seen: true, token }),
      });
    } catch { /* closing either way */ }
  }
  pendingSeen = new Map();
}

// Seen, and everything above it with it: a list is read from the top, and a row scrolled
// past quickly was passed over all the same.
function seenUpTo(id) {
  const at = L.cards.findIndex((c) => c.version_id === id);
  if (at < 0) return;
  for (let i = 0; i <= at; i++) {
    const card = L.cards[i];
    if (card.seen || S.keepUnseen.has(card.version_id) || hiddenNow(card)) continue;
    card.seen = true;
    pendingSeen.set(card.version_id, L.token || "");
    const row = document.querySelector(`.row.found[data-key="s${card.version_id}"]`);
    if (row) { row.classList.add("seen"); row.querySelector(".new-dot")?.remove(); }
  }
  paintHead();
  flushSeen();
}

export async function markSeen(ids, seen = true) {
  const cards = ids.map((id) => L.byId.get(id)).filter(Boolean);
  for (const card of cards) {
    card.seen = seen;
    if (seen) S.keepUnseen.delete(card.version_id);
    else S.keepUnseen.add(card.version_id);
    pendingSeen.delete(card.version_id);
  }
  S.version++;
  invalidate("list", "inspector");
  try {
    const body = await post("/api/civitai/seen", { ids, seen, token: L.token || "" });
    takePresets(body.presets);
  } catch (error) { toastError(error); }
}

export async function hideAuthor(name, hidden = true) {
  if (!name) return;
  try {
    const body = await post("/api/civitai/authors", { username: name, hidden });
    state.civitai.hidden = body.hidden_authors || [];
    takePresets(body.presets);
  } catch (error) { toastError(error); return; }
  const lower = name.toLowerCase();
  // Every list kept, not only the one on screen: the author is hidden from all of them.
  for (const list of new Set([L, ...lists.values()])) {
    for (const card of list.cards) if (card.author.toLowerCase() === lower) card.author_hidden = hidden;
  }
  S.version++;
  // What left the list leaves the selection with it, rather than staying in the inspector.
  const kept = [...state.selection].filter((key) => key[0] !== "s" || !hiddenNow(L.byId.get(Number(key.slice(1))) || {}));
  if (kept.length !== state.selection.size) select(kept);
  invalidate("list", "inspector");
  toast(hidden ? `${name} is hidden from searches` : `${name} is shown in searches again`, {
    actions: [{ label: "Undo", run: () => hideAuthor(name, !hidden) }],
  });
}

// An author hidden while the list is on screen leaves it at once — unlike a version seen,
// it was asked to.
const hiddenNow = (card) => card.author_hidden && shownWith().hidden_authors === "hide";

export function downloadFound(id) {
  const card = L.byId.get(id);
  if (card) act.addSource(card.page);
}

export function revealFound(id) {
  state.revealed.add(`civitai${id}`);
  S.version++;
  invalidate("list", "inspector");
}

// --- presets ---------------------------------------------------------------------------------

export function openPreset(id) {
  act.go({ kind: "civitai", preset: id });
}

async function saveAsPreset() {
  const name = await promptDialog({
    title: "Save as a preset", label: "A name for this search, as the sidebar will list it", confirm: "Save",
    value: suggestName(L.filters),
  });
  if (!name || !name.trim()) return;
  const from = viewKey(state.view);
  const list = L;
  try {
    const body = await post("/api/civitai/presets", { name: name.trim(), filters: list.filters });
    takePresets(body.presets);
    const key = `p${body.preset.id}`;
    // A list searched with what was saved becomes the preset's, rather than being asked of
    // Civitai again: the new preset opens on it, and the view it was made in starts afresh.
    if (list.applied && same(list.filters, list.applied) && list.token && !list.loading) {
      lists.set(key, list);
      lists.delete(from);
      post(`/api/civitai/search/${encodeURIComponent(list.token)}/adopt`, { preset_id: body.preset.id })
        .then((answer) => takePresets(answer.presets)).catch(() => {});
    }
    act.go({ kind: "civitai", preset: body.preset.id });
    toast(`Saved as “${body.preset.name}”`, { level: "ok" });
  } catch (error) { toastError(error); }
}

function suggestName(filters) {
  const bases = filters.base_models.slice(0, 2).join(", ");
  const types = LORAS.every((t) => filters.types.includes(t)) ? "LoRA"
    : filters.types.length === 1 ? typeLabel(filters.types[0]) : filters.types.length ? "models" : "";
  return [bases, types].filter(Boolean).join(" · ") || filters.query || "New search";
}

async function saveChanges() {
  const id = state.view.preset;
  if (!id) return;
  try {
    const body = await put(`/api/civitai/presets/${id}`, { filters: L.filters });
    takePresets(body.presets);
    S.saved = copy(body.preset.filters);
    paintHead();
    toast("Preset saved", { level: "ok" });
  } catch (error) { toastError(error); }
}

// The panel back to what is saved. The list stays until Apply, like after any other change.
function undoChanges() {
  if (!S.saved) return;
  L.filters = copy(S.saved);
  const input = document.querySelector('[data-role="query"]');
  if (input) input.value = L.filters.query;
  S.panelSig = "";
  paintHead();
  paintPanel();
}

export async function renamePreset(id) {
  const preset = presetOf(id);
  if (!preset) return;
  const name = await promptDialog({ title: "Rename the preset", value: preset.name, confirm: "Rename" });
  if (!name || !name.trim() || name.trim() === preset.name) return;
  try {
    const body = await put(`/api/civitai/presets/${id}`, { name: name.trim() });
    takePresets(body.presets);
    if (state.view.kind === "civitai") paintHead();
  } catch (error) { toastError(error); }
}

export async function deletePreset(id) {
  const preset = presetOf(id);
  if (!preset) return;
  const yes = await confirmDialog({
    title: "Delete the preset", confirm: "Delete", danger: true,
    message: `Delete “${esc(preset.name)}”? The search it keeps goes with it; the versions it showed stay seen.`,
  });
  if (!yes) return;
  try {
    const body = await del(`/api/civitai/presets/${id}`);
    takePresets(body.presets);
    lists.delete(`p${id}`);
    if (state.view.kind === "civitai" && state.view.preset === id) act.go({ kind: "civitai" });
  } catch (error) { toastError(error); }
}

export async function movePreset(id, step) {
  const ids = state.civitai.presets.map((p) => p.id);
  const at = ids.indexOf(id);
  const to = at + step;
  if (at < 0 || to < 0 || to >= ids.length) return;
  [ids[at], ids[to]] = [ids[to], ids[at]];
  try {
    const body = await post("/api/civitai/presets/reorder", { ids });
    takePresets(body.presets);
  } catch (error) { toastError(error); }
}

function presetMenu(button) {
  const preset = state.view.preset ? presetOf(state.view.preset) : null;
  const dirty = changed();
  menuFrom(button, [
    preset && { label: "Save changes", icon: "save", disabled: !dirty, run: saveChanges },
    preset && { label: "Undo changes", icon: "back", disabled: !dirty, run: undoChanges },
    { label: preset ? "Save as a new preset…" : "Save as a preset…", icon: "star", run: saveAsPreset },
    preset && "-",
    preset && { label: "Rename…", icon: "edit", run: () => renamePreset(preset.id) },
    preset && { label: "Delete…", icon: "trash", danger: true, run: () => deletePreset(preset.id) },
  ]);
}

// --- drawing ---------------------------------------------------------------------------------

// The pane for a search: built when the view is opened, changed in parts after that. Returns
// the keys of the rows, for the keyboard to move through.
export function renderSearch() {
  const holder = document.getElementById("center");
  const view = state.view;
  const key = viewKey(view);
  if (view.preset && !state.civitai.loaded) {
    holder.innerHTML = `<div class="list-empty">Opening the preset…</div>`;
    S.mounted = "";
    return [];
  }
  if (view.preset && !presetOf(view.preset)) {
    // Deleted in another window, or gone with the database: the free search instead.
    queueMicrotask(() => act.go({ kind: "civitai" }));
    return [];
  }
  if (S.mounted !== key || !holder.querySelector(".search-pane")) {
    mount(holder, view, key);
    return keys();
  }
  paint();
  return keys();
}

const keys = () => L.cards.filter((c) => !hiddenNow(c)).map((c) => `s${c.version_id}`);

// Where the list on screen is scrolled to, kept with it before its pane is taken down — for
// another search, or another view. Read then rather than followed as it scrolls: a window in
// the background is sent no scroll events at all.
export function keepPlace() {
  const rows = document.querySelector(".search-pane #rows");
  if (rows && L.cards.length) L.scroll = rows.scrollTop;
}

// The view's list as it was left — or, the first time it is opened in a run of the app, an
// empty one that waits for Search: nothing is asked of Civitai for a view merely opened.
function mount(holder, view, key) {
  keepPlace();
  S.mounted = key;
  const preset = view.preset ? presetOf(view.preset) : null;
  S.saved = preset ? copy(preset.filters) : null;
  if (!lists.has(key)) lists.set(key, blank(copy(preset ? preset.filters : state.prefs.civitai_free)));
  L = lists.get(key);
  S.rowsSig = "";
  S.panelSig = "";
  S.version++;
  holder.innerHTML = `
    <div class="search-pane">
      <div class="list-head search-head">
        <div class="list-title"><span data-role="title"></span><span class="changed-dot" data-role="dot" title="Changed since it was saved" hidden></span></div>
        <button class="mini" data-sd="preset-menu" title="Save this search as a preset, or change the one it is">${icon("star")}Preset${icon("chevron-down")}</button>
        <label class="search-box grow">${icon("search")}
          <input data-role="query" type="search" spellcheck="false" autocomplete="off" placeholder="Search Civitai — or leave empty for everything the filters let through">
        </label>
        <button class="toggle" data-sd="filters" title="Show or hide the filters">${icon("layers")}Filters</button>
        <button data-sd="apply" data-role="apply"></button>
      </div>
      <div class="search-summary" data-role="summary"></div>
      <div class="search-panel" data-role="panel"></div>
      <div class="list" id="rows" tabindex="0" role="listbox" aria-multiselectable="true"></div>
    </div>`;
  const pane = holder.querySelector(".search-pane");
  const input = pane.querySelector('[data-role="query"]');
  input.value = L.filters.query;
  // Taken before the first drawing, which keeps them as they are on screen — the selection
  // the view was opened with is none, and a list just built is at its top.
  const scroll = L.scroll;
  const selected = L.selection.filter((k) => L.byId.has(Number(k.slice(1))));
  wirePane(pane, input);
  paint();
  const rows = pane.querySelector("#rows");
  rows.scrollTop = scroll;
  L.scroll = rows.scrollTop;
  // What was selected in it, selected again — the redraw this asks for draws it so.
  if (selected.length) {
    L.selection = selected;
    queueMicrotask(() => select(selected));
  }
  loadOptions();
}

function paint() {
  const pane = document.querySelector(".search-pane");
  if (!pane) return;
  paintHead();
  paintPanel();
  paintRows();
}

function paintHead() {
  const pane = document.querySelector(".search-pane");
  if (!pane) return;
  const preset = state.view.preset ? presetOf(state.view.preset) : null;
  pane.querySelector('[data-role="title"]').textContent = preset ? preset.name : "Search Civitai";
  pane.querySelector('[data-role="dot"]').hidden = !changed();
  const open = state.prefs.civitai_panel !== false;
  pane.querySelector('[data-sd="filters"]').classList.toggle("on", open);
  // One button for asking Civitai: Search before the list has been, Apply once the filters
  // are not the ones it was searched with, Search again otherwise.
  const button = pane.querySelector('[data-role="apply"]');
  const [label, look, title] = !L.applied ? ["Search", "search", "Ask Civitai with these filters"]
    : pending() ? ["Apply", "check", "Search again with the filters as they are now"]
      : ["Search again", "refresh", "Ask Civitai again with the same filters — for what is new since"];
  button.className = !L.applied || pending() ? "primary" : "";
  button.title = title;
  button.innerHTML = `${icon(look)}${label}`;
  const line = pane.querySelector('[data-role="summary"]');
  line.innerHTML = summary(open);
  line.hidden = !line.innerHTML.trim();
}

// The filters in one line, when the panel is folded away — and, either way, whether they are
// the ones the list shows, and the one thing done to the whole list.
function summary(open) {
  const f = L.filters;
  const types = LORAS.every((t) => f.types.includes(t))
    ? ["LoRA, LoCon, DoRA", ...f.types.filter((t) => !LORAS.includes(t)).map(typeLabel)]
    : f.types.map(typeLabel);
  const parts = [
    f.base_models.length ? f.base_models.join(", ") : "every base model",
    types.length ? types.join(", ") : "every type",
    `${(SORTS.find(([v]) => v === f.sort) || [, f.sort])[1].toLowerCase()}, ${f.period === "AllTime" ? "all time" : `past ${f.period.toLowerCase()}`}`,
    f.site,
    f.exclude_tags.length ? `without ${f.exclude_tags.join(", ")}` : "",
    f.paid === "no_sold" ? "no sold" : f.paid === "free" ? "free only" : "",
    f.seen === "hide" ? "seen hidden" : "",
    f.hidden_authors === "show" && state.civitai.hidden.length ? "hidden authors shown" : "",
  ].filter(Boolean);
  const unseen = L.cards.filter((c) => !c.seen && !hiddenNow(c)).length;
  const waiting = pending();
  if (open && !unseen && !waiting) return "";
  return `${open ? "" : `<span class="ellipsis" data-sd="filters" title="${esc(parts.join(" · "))} — click to change">${esc(parts.join(" · "))}</span>`}
    ${waiting ? `<span class="warn small nowrap" title="The list is still the one searched with the filters before">Not applied — the list is from before</span>` : ""}
    <span class="grow"></span>
    ${unseen ? `<button class="mini" data-sd="all-seen" title="Mark every version listed as seen">${icon("eye")}Mark ${unseen} as seen</button>` : ""}`;
}

function seg(name, choices, current) {
  return `<span class="seg">${choices.map(([value, label, title]) => `
    <button class="${value === current ? "on" : ""}" data-sf="${name}" data-value="${esc(value)}" ${title ? `title="${esc(title)}"` : ""}>${esc(label)}</button>`).join("")}</span>`;
}

const removable = (what, value, label = value, cls = "") =>
  `<span class="chip pick ${cls}">${esc(label)}<button data-sf="remove-${what}" data-value="${esc(value)}" aria-label="Remove">${icon("x")}</button></span>`;

function paintPanel() {
  const pane = document.querySelector(".search-pane");
  if (!pane) return;
  const panel = pane.querySelector('[data-role="panel"]');
  const open = state.prefs.civitai_panel !== false;
  const options = state.civitai.options;
  const sig = `${open}|${JSON.stringify(L.filters)}|${options ? options.base_models.length : 0}|${state.civitai.hidden.length}`;
  if (sig === S.panelSig) return;
  S.panelSig = sig;
  panel.hidden = !open;
  if (!open) { panel.innerHTML = ""; return; }
  const typing = panel.contains(document.activeElement) && document.activeElement.dataset.sf === "tag-input"
    ? document.activeElement.value : null;
  const f = L.filters;
  const bases = options ? options.base_models.filter((b) => !f.base_models.includes(b)) : [];
  const allLoras = LORAS.every((t) => f.types.includes(t));
  const others = f.types.filter((t) => !LORAS.includes(t) && t !== "Checkpoint");
  const offered = (options?.types || []).filter((t) => !LORAS.includes(t) && t !== "Checkpoint" && !f.types.includes(t));
  const hidden = state.civitai.hidden.length;
  panel.innerHTML = `
    <div class="filter-line"><span class="filter-label">Base model</span>
      ${f.base_models.map((b) => removable("base", b)).join("")}
      <select data-sf="add-base" aria-label="Add a base model">
        <option value="">${options ? (f.base_models.length ? "Add another…" : "Any — add one…") : "Asking Civitai for its list…"}</option>
        ${bases.map((b) => `<option value="${esc(b)}">${esc(b)}</option>`).join("")}
      </select></div>
    <div class="filter-line"><span class="filter-label">Type</span>
      <button class="toggle ${f.types.includes("Checkpoint") ? "on" : ""}" data-sf="type" data-value="Checkpoint">Checkpoint</button>
      <button class="toggle ${allLoras ? "on" : ""}" data-sf="loras" title="LoRA, LoCon and DoRA together — Civitai lists them as three types">LoRA, LoCon, DoRA</button>
      ${others.map((t) => removable("type", t, typeLabel(t))).join("")}
      ${offered.length ? `<select data-sf="add-type" aria-label="Add a type"><option value="">More types…</option>
        ${offered.map((t) => `<option value="${esc(t)}">${esc(typeLabel(t))}</option>`).join("")}</select>` : ""}
    </div>
    <div class="filter-line"><span class="filter-label">New in</span>${seg("period", PERIODS, f.period)}
      <span class="filter-gap"></span>${seg("sort", SORTS, f.sort)}</div>
    <div class="filter-line"><span class="filter-label">Site</span>${seg("site", SITES, f.site)}</div>
    <div class="filter-line"><span class="filter-label">Without tags</span>
      ${f.exclude_tags.map((t) => removable("tag", t, t, "no")).join("")}
      <input class="tag-input" data-sf="tag-input" spellcheck="false" autocomplete="off" placeholder="character* — Enter adds it" title="A tag the model must not carry; * stands for anything, so character* is every spelling of it">
    </div>
    <div class="filter-line"><span class="filter-label">Paid</span>${seg("paid", PAID, f.paid)}</div>
    <div class="filter-line"><span class="filter-label">Show</span>
      ${seg("seen", [["show", "Seen: show", "Versions already on screen in a search are listed too, without their dot"], ["hide", "hide", "Only versions not on screen in a search before"]], f.seen)}
      <span class="filter-gap"></span>
      ${seg("hidden_authors", [["show", `Hidden authors${hidden ? ` (${hidden})` : ""}: show`, "Their models are listed, dimmed, with Unhide"], ["hide", "hide", "Their models are left out"]], f.hidden_authors)}
    </div>`;
  if (typing !== null) {
    const input = panel.querySelector('[data-sf="tag-input"]');
    input.value = typing;
    input.focus();
  }
}

function paintRows() {
  const list = document.querySelector(".search-pane #rows");
  if (!list) return;
  const sig = [S.version, L.loading, L.error, [...state.selection].join(","), state.revealed.size,
    state.settings.blur_nsfw, state.settings.fetch_previews].join("|");
  if (sig === S.rowsSig) { paintFoot(); return; }
  S.rowsSig = sig;
  // Kept with the list, for coming back to it.
  L.selection = [...state.selection].filter((k) => k[0] === "s");
  const scroll = list.scrollTop;
  const cards = L.cards.filter((c) => !hiddenNow(c));
  list.innerHTML = cards.length
    ? `${cards.map(cardRow).join("")}<div class="search-foot" data-role="foot"></div>`
    : `<div class="list-empty">${emptyText()}</div><div class="search-foot" data-role="foot"></div>`;
  list.scrollTop = scroll;
  paintFoot();
  watch(list);
  fillUp(list);
}

function emptyText() {
  if (!L.applied && !L.loading) {
    const preset = state.view.preset ? presetOf(state.view.preset) : null;
    const last = preset?.unseen?.at
      ? `Last searched ${fmtAgo(preset.unseen.at)}${preset.unseen.count ? `, with ${preset.unseen.count}${preset.unseen.more ? "+" : ""} not seen then` : ""}.` : "";
    return `<div>Not searched yet. Nothing is asked of Civitai until you say so.</div>
      ${last ? `<div class="small muted">${esc(last)}</div>` : ""}
      <div class="empty-action"><button class="primary" data-sd="apply">${icon("search")}Search</button></div>`;
  }
  if (L.loading) return "Looking through Civitai…";
  if (L.error) return esc(L.error);
  if (!L.scanned) return "Civitai has nothing for this search.";
  return `Nothing in the ${plural(L.scanned, "model")} looked through is left after the filters.${L.done ? "" : " There is more to look through."}`;
}

function paintFoot() {
  const foot = document.querySelector('.search-pane [data-role="foot"]');
  if (!foot) return;
  const left = Object.entries(L.leftOut).filter(([, n]) => n)
    .map(([why, n]) => `${n} ${LEFT[why] || why}`);
  const line = L.scanned
    ? `Looked through ${L.scanned} · ${plural(L.cards.length, "version")} shown${left.length ? ` · left out: ${left.join(", ")}` : ""}`
    : "";
  foot.innerHTML = `
    <span class="muted small">${esc(line)}</span>
    ${L.error && L.cards.length ? `<span class="err small">${esc(L.error)}</span>` : ""}
    ${L.loading && L.cards.length ? `<span class="small accent">Looking through more…</span>`
      : !L.done && !L.loading ? `<button class="mini" data-sd="more">${icon("arrow-down")}Look through more</button>`
        : L.error ? `<button class="mini" data-sd="apply">${icon("refresh")}Search again</button>` : ""}`;
}

function thumbOf(card) {
  const source = { kind: "civitai", id: card.version_id };
  if (!card.previews || !state.settings.fetch_previews) {
    return `<div class="thumb empty">${icon(kindIcon(card.category))}</div>`;
  }
  const hide = covered(source, card.nsfw);
  return `<div class="thumb ${hide ? "covered" : ""}" data-thumb="s${card.version_id}">
    <img src="${previewSrc(source, 0, 160)}" loading="lazy" decoding="async" alt="" draggable="false"
         onerror="this.remove()" width="40" height="40"></div>`;
}

export function hereChip(card) {
  if (card.here === "this") return `<span class="chip link" title="This version is in your library">in library</span>`;
  if (card.here === "other") return `<span class="chip quiet" title="Another version of this model is in your library">other version here</span>`;
  return "";
}

function cardChips(card) {
  return [
    `<span class="chip kind">${esc(typeLabel(card.type))}</span>`,
    card.base_model ? `<span class="chip">${esc(card.base_model)}</span>` : "",
    accessChip(card.access),
    hereChip(card),
    card.author_hidden ? `<span class="chip quiet" title="You hid this author's models">hidden author</span>` : "",
  ].join("");
}

const published = (card) => Date.parse(card.published_at || "") / 1000 || 0;

function cardRow(card) {
  const key = `s${card.version_id}`;
  const when = published(card);
  return `
    <div class="row found ${card.seen ? "seen" : ""} ${card.author_hidden ? "dim" : ""} ${state.selection.has(key) ? "selected" : ""}" data-key="${key}">
      ${thumbOf(card)}
      <div class="row-main">
        <div class="row-title">${card.seen ? "" : `<span class="new-dot" title="Not on screen in a search before"></span>`}<span class="name">${esc(card.name)}</span>
          ${card.version_name ? `<span class="version">${esc(card.version_name)}</span>` : ""}</div>
        <div class="row-sub">${cardChips(card)}</div>
      </div>
      <div class="row-actions">
        ${card.author_hidden ? `<button class="mini" data-sd="unhide" title="Show ${esc(card.author)}'s models in searches again">Unhide</button>` : ""}
        <button class="icon-button" data-sd="download" title="Download… — asks where it goes first" aria-label="Download">${icon("download")}</button>
      </div>
      <div class="row-side">
        <span class="size ellipsis" title="By ${esc(card.author)}">${esc(card.author)}</span>
        <span class="date" title="${esc(when ? fmtDate(when) : "")}">${when ? fmtAgo(when) : ""}</span>
      </div>
    </div>`;
}

// --- seen by being on screen ---------------------------------------------------------------

let observer = null;
const timers = new Map();

function watch(list) {
  if (observer) observer.disconnect();
  for (const timer of timers.values()) clearTimeout(timer);
  timers.clear();
  observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const id = Number(entry.target.dataset.key.slice(1));
      if (entry.isIntersecting && entry.intersectionRatio >= 0.6) {
        if (!timers.has(id)) timers.set(id, setTimeout(() => { timers.delete(id); seenUpTo(id); }, 700));
      } else {
        clearTimeout(timers.get(id));
        timers.delete(id);
        // Gone off the top: scrolled past.
        if (entry.rootBounds && entry.boundingClientRect.bottom <= entry.rootBounds.top + 1) seenUpTo(id);
      }
    }
  }, { root: list, threshold: [0, 0.6] });
  list.querySelectorAll(".row.found:not(.seen)").forEach((row) => observer.observe(row));
}

// A list too short to scroll can never be scrolled to its end: it is read on by itself — but
// not for ever when the filters leave nothing of page after page.
function fillUp(list) {
  if (L.loading || L.done || L.error || L.empty >= 2) return;
  requestAnimationFrame(() => {
    if (list.isConnected && list.scrollHeight <= list.clientHeight + 40) more();
  });
}

// --- behaviour -----------------------------------------------------------------------------

function wirePane(pane, input) {
  // Typing changes the filters like any other control; Enter is Apply.
  const typed = debounce(() => setFilters({ query: input.value.trim() }), 250);
  input.addEventListener("input", typed);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") { event.preventDefault(); typed.flush(); apply(); }
    if (event.key === "Escape" && input.value) { event.stopPropagation(); input.value = ""; typed(); typed.flush(); }
  });

  pane.addEventListener("click", (event) => {
    const sd = event.target.closest("[data-sd]");
    if (sd) {
      event.stopPropagation();
      const what = sd.dataset.sd;
      const row = sd.closest(".row.found");
      const id = row ? Number(row.dataset.key.slice(1)) : null;
      if (what === "preset-menu") presetMenu(sd);
      else if (what === "filters") { remember({ civitai_panel: state.prefs.civitai_panel === false }); S.panelSig = ""; invalidate("list"); }
      else if (what === "apply") apply();
      else if (what === "more") { L.empty = 0; more(); }
      else if (what === "all-seen") markSeen(L.cards.filter((c) => !c.seen && !hiddenNow(c)).map((c) => c.version_id));
      else if (what === "download" && id) downloadFound(id);
      else if (what === "unhide" && id) hideAuthor(L.byId.get(id)?.author, false);
      return;
    }
    const sf = event.target.closest("button[data-sf]");
    if (sf) { event.stopPropagation(); filterClick(sf.dataset.sf, sf.dataset.value); return; }
    const picture = event.target.closest("[data-thumb]");
    if (picture && picture.classList.contains("covered")) {
      event.stopPropagation();
      revealFound(Number(picture.dataset.thumb.slice(1)));
    }
  });

  pane.addEventListener("change", (event) => {
    const sf = event.target.closest("select[data-sf]");
    if (!sf || !sf.value) return;
    const f = L.filters;
    if (sf.dataset.sf === "add-base") setFilters({ base_models: [...f.base_models, sf.value] });
    if (sf.dataset.sf === "add-type") setFilters({ types: [...f.types, sf.value] });
  });

  pane.addEventListener("keydown", (event) => {
    const tag = event.target.closest('[data-sf="tag-input"]');
    if (!tag || event.key !== "Enter") return;
    event.preventDefault();
    const words = tag.value.split(",").map((w) => w.trim().toLowerCase()).filter(Boolean);
    tag.value = "";
    const fresh = words.filter((w) => !L.filters.exclude_tags.includes(w));
    if (fresh.length) setFilters({ exclude_tags: [...L.filters.exclude_tags, ...fresh] });
  });

  const list = pane.querySelector("#rows");
  list.addEventListener("scroll", () => {
    // The pane taken off the page by another view still hears one last scroll, to the top of
    // nothing: that is not where the list was left.
    if (!list.isConnected) return;
    // Where the list is, for coming back to it at the same place.
    if (L.cards.length) L.scroll = list.scrollTop;
    if (list.scrollTop + list.clientHeight > list.scrollHeight - 500) more();
  }, { passive: true });
}

function filterClick(what, value) {
  const f = L.filters;
  const without = (list, item) => list.filter((x) => x !== item);
  if (["period", "sort", "site", "paid", "seen", "hidden_authors"].includes(what)) {
    if (f[what] !== value) setFilters({ [what]: value });
  } else if (what === "remove-base") setFilters({ base_models: without(f.base_models, value) });
  else if (what === "remove-type") setFilters({ types: without(f.types, value) });
  else if (what === "remove-tag") setFilters({ exclude_tags: without(f.exclude_tags, value) });
  else if (what === "type") setFilters({ types: f.types.includes(value) ? without(f.types, value) : [...f.types, value] });
  else if (what === "loras") {
    const all = LORAS.every((t) => f.types.includes(t));
    setFilters({ types: all ? f.types.filter((t) => !LORAS.includes(t)) : [...new Set([...f.types, ...LORAS])] });
  }
}

export function focusQuery() {
  const input = document.querySelector('.search-pane [data-role="query"]');
  if (input) { input.focus(); input.select(); }
}

// --- the right-click menu, and the inspector --------------------------------------------------

export function foundMenu(ids) {
  const cards = ids.map((id) => L.byId.get(id)).filter(Boolean);
  if (!cards.length) return [];
  const unseen = cards.filter((c) => !c.seen);
  if (cards.length > 1) {
    return [
      unseen.length && { label: `Mark ${unseen.length} as seen`, icon: "eye", run: () => markSeen(unseen.map((c) => c.version_id)) },
      cards.some((c) => c.seen) && { label: "Mark as not seen", icon: "eye-off", run: () => markSeen(cards.filter((c) => c.seen).map((c) => c.version_id), false) },
    ];
  }
  const [card] = cards;
  return [
    { label: "Download…", icon: "download", run: () => downloadFound(card.version_id) },
    card.previews && { label: "Samples", icon: "image", run: () => openLightbox({ kind: "civitai", id: card.version_id }, 0, card.name) },
    { label: "Open the page", icon: "external", run: () => window.open(card.page, "_blank", "noopener") },
    { label: "Copy the link", icon: "copy", run: () => copyText(card.page).then(() => toast("Link copied")) },
    "-",
    card.seen
      ? { label: "Mark as not seen", icon: "eye-off", run: () => markSeen([card.version_id], false) }
      : { label: "Mark as seen", icon: "eye", run: () => markSeen([card.version_id]) },
    card.author && (card.author_hidden
      ? { label: `Show ${card.author}'s models again`, icon: "eye", run: () => hideAuthor(card.author, false) }
      : { label: `Hide ${card.author}'s models`, icon: "eye-off", run: () => hideAuthor(card.author) }),
  ];
}

function facts(label, value) {
  return value ? `<dt>${esc(label)}</dt><dd>${value}</dd>` : "";
}

export function foundPanel(id) {
  const card = L.byId.get(id);
  if (!card) return `<div class="insp-empty">That version is not in this search any more.</div>`;
  const source = { kind: "civitai", id };
  const hide = covered(source, card.nsfw);
  const when = published(card);
  const files = card.files.map((f) => `<div class="small"><span class="mono">${esc(f.name)}</span>${f.size ? ` · ${esc(fmtBytes(f.size))}` : ""}${f.fp ? ` · ${esc(f.fp)}` : ""}${f.primary && card.files.length > 1 ? ` <span class="muted">· main</span>` : ""}</div>`).join("");
  const mine = card.here === "this" ? [...state.models.values()].find((m) => m.version_id === card.version_id && m.state === "present") : null;
  const banners = [];
  if (paidNow(card.access)) {
    banners.push(`<div class="banner banner-${card.access.owned === true ? "info" : "warn"}"><div class="banner-text">${accessChip(card.access)} ${esc(paidSentence(card.access))}</div></div>`);
  }
  if (card.here === "this") {
    banners.push(`<div class="banner banner-info"><div class="banner-text">This version is in your library.</div>
      ${mine ? `<div class="banner-actions"><button data-found="show">${icon("folder")}Show it</button></div>` : ""}</div>`);
  } else if (card.here === "other") {
    banners.push(`<div class="banner banner-info"><div class="banner-text">Another version of this model is in your library.</div></div>`);
  }
  if (card.author_hidden) {
    banners.push(`<div class="banner banner-info"><div class="banner-text">You hid ${esc(card.author)}'s models from searches.</div>
      <div class="banner-actions"><button data-found="unhide">${icon("eye")}Unhide</button></div></div>`);
  }
  const stats = card.stats || {};
  return `
    <div class="insp">
      ${card.previews && state.settings.fetch_previews ? `
        <button class="insp-preview ${hide ? "covered" : ""}" data-open="0" title="${hide ? "Marked adult — click to uncover" : card.previews > 1 ? `${card.previews} samples` : "Full size"}">
          ${previewPair(source, 0, 640)}
          ${card.previews > 1 ? `<span class="preview-count">${card.previews}</span>` : ""}
        </button>` : `<div class="insp-preview empty">${icon(kindIcon(card.category))}</div>`}
      <div class="insp-name"><span class="name-text">${esc(card.name)}</span></div>
      ${card.version_name ? `<div class="insp-title">${esc(card.version_name)}</div>` : ""}
      <div class="chips">${cardChips(card)}</div>
      <div class="insp-actions">
        <button class="primary" data-found="download" title="Asks where it goes first">${icon("download")}Download…</button>
        <a class="button" href="${esc(card.page)}" target="_blank" rel="noopener noreferrer">${icon("external")}Open the page</a>
        <button class="icon-button" data-found="more" aria-label="More">${icon("dots")}</button>
      </div>
      ${banners.join("")}
      ${card.trained_words.length ? `<section class="insp-section"><div class="insp-label">Trigger words</div>
        <div class="tags">${card.trained_words.map((w) => `<span class="tag">${esc(w)}</span>`).join("")}
        <button class="mini" data-copy-text="${esc(card.trained_words.join(", "))}">Copy</button></div></section>` : ""}
      ${card.previews > 1 && state.settings.fetch_previews ? `<div class="strip inline" data-role="strip"></div>` : ""}
      ${card.description ? `<section class="insp-section"><div class="insp-label">Description</div><div class="quote clamp">${esc(card.description)}</div></section>` : ""}
      <section class="insp-section"><dl class="facts">
        ${facts("Type", esc(typeLabel(card.type)))}
        ${facts("Base model", esc(card.base_model))}
        ${facts("Published", when ? esc(`${fmtDate(when)} · ${fmtAgo(when)}`) : "")}
        ${facts("Author", card.author ? `${esc(card.author)} <button class="mini" data-found="${card.author_hidden ? "unhide" : "hide"}">${card.author_hidden ? "Unhide" : "Hide their models"}</button>` : "")}
        ${facts("Downloads", stats.downloadCount ? esc(stats.downloadCount.toLocaleString("en-GB")) : "")}
        ${facts("Likes", stats.thumbsUpCount ? esc(stats.thumbsUpCount.toLocaleString("en-GB")) : "")}
        ${facts("Seen", card.seen ? "yes" : `no <button class="mini" data-found="seen">Mark as seen</button>`)}
        ${facts("Tags", card.tags.length ? esc(card.tags.join(", ")) : "")}
        ${facts("Site", esc(card.host))}
      </dl></section>
      ${files ? `<section class="insp-section"><div class="insp-label">${plural(card.files.length, "file")}</div>${files}</section>` : ""}
    </div>`;
}

export async function wireFound(holder, id) {
  const card = L.byId.get(id);
  if (!card) return;
  const source = { kind: "civitai", id };
  holder.onclick = (event) => {
    const opener = event.target.closest("[data-open]");
    if (opener) {
      if (opener.classList.contains("covered")) { revealFound(id); return; }
      openLightbox(source, Number(opener.dataset.open), card.name);
      return;
    }
    const target = event.target.closest("[data-found]");
    if (!target) return;
    const what = target.dataset.found;
    if (what === "download") downloadFound(id);
    else if (what === "hide") hideAuthor(card.author);
    else if (what === "unhide") hideAuthor(card.author, false);
    else if (what === "seen") markSeen([id]);
    else if (what === "more") menuFrom(target, foundMenu([id]));
    else if (what === "show") {
      const mine = [...state.models.values()].find((m) => m.version_id === card.version_id && m.state === "present");
      if (mine) act.showModel(mine.id);
    }
  };
  const strip = holder.querySelector('[data-role="strip"]');
  if (strip) {
    try {
      const { previews } = await previewsOf(source);
      strip.innerHTML = previews.map((p, i) => `
        <img class="shot ${covered(source, p.nsfw) ? "covered" : ""}" src="${previewSrc(source, i, 240)}"
             data-open="${i}" loading="lazy" alt="" draggable="false">`).join("");
    } catch { strip.remove(); }
  }
}

export function foundMulti(ids) {
  const cards = ids.map((id) => L.byId.get(id)).filter(Boolean);
  const unseen = cards.filter((c) => !c.seen).length;
  return `
    <div class="insp">
      <div class="insp-name"><span class="name-text">${plural(cards.length, "version")} selected</span></div>
      <section class="insp-section">${cards.slice(0, 8).map((c) => `<div class="small ellipsis">${esc(c.name)} <span class="muted">${esc(c.version_name)}</span></div>`).join("")}
        ${cards.length > 8 ? `<div class="small muted">and ${cards.length - 8} more</div>` : ""}</section>
      <div class="insp-actions column">
        ${unseen ? `<button data-found-multi="seen">${icon("eye")}Mark ${unseen} as seen</button>` : ""}
        ${cards.length - unseen ? `<button data-found-multi="unseen">${icon("eye-off")}Mark ${cards.length - unseen} as not seen</button>` : ""}
      </div>
    </div>`;
}

export function wireFoundMulti(holder, ids) {
  holder.onclick = (event) => {
    const target = event.target.closest("[data-found-multi]");
    if (!target) return;
    const cards = ids.map((id) => L.byId.get(id)).filter(Boolean);
    if (target.dataset.foundMulti === "seen") markSeen(cards.filter((c) => !c.seen).map((c) => c.version_id));
    else markSeen(cards.filter((c) => c.seen).map((c) => c.version_id), false);
  };
}

export const SEARCH_HELP = "Civitai, searched with the filters its site does not have: tags to leave out, sold versions, versions that can only be used on the site, authors you hid. Nothing is asked of Civitai until you press Search, or Apply after changing the filters; each preset keeps its list, and where it was scrolled to, until then. A version counts as seen once it has been on screen; Seen: hide leaves those out of the next search. Save a search as a preset to keep it in the sidebar, with a count of what it found new.";
