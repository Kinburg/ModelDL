// The middle pane: whatever is being looked at, as a list — a folder's models, the downloads,
// the history, the missing, the duplicates, the leftovers — with selection, sorting, a drag
// onto the tree, and the right-click menu.

import { post } from "./api.js";
import {
  esc, fmtBytes, fmtSpeed, fmtEta, fmtAgo, fmtDate, fmtTime, fmtDuration, dayLabel, kindLabel,
  plural, baseName, copyText,
} from "./util.js";
import { icon, kindIcon } from "./icons.js";
import {
  state, invalidate, onRender, modelsIn, downloadsIn, activeTasks, missingModels,
  unidentifiedModels, select, folderLabel, remember, placementOf, selectedModels,
} from "./store.js";
import { showMenu, menuFrom } from "./menu.js";
import { covered, openLightbox, previewSrc } from "./lightbox.js";
import { toast } from "./toasts.js";
import { renderSettings } from "./settings.js";
import * as act from "./actions.js";

let visible = [];

// --- what each view lists ---------------------------------------------------------------

const SORTS = {
  name: (a, b) => nameOf(a).localeCompare(nameOf(b), undefined, { numeric: true, sensitivity: "base" }),
  date: (a, b) => dateOf(b) - dateOf(a),
  size: (a, b) => (b.size || 0) - (a.size || 0),
  kind: (a, b) => kindLabel(a.category).localeCompare(kindLabel(b.category)) || SORTS.name(a, b),
};
const nameOf = (item) => item.filename || item.source || "";
const dateOf = (item) => item.finished_at || item.first_seen || item.created_at || 0;

function sorted(items) {
  const compare = SORTS[state.sort.key] || SORTS.name;
  const list = [...items].sort(compare);
  return state.sort.dir < 0 ? list.reverse() : list;
}

// Identifying is for a model that came from somewhere else and that nothing has described.
// One downloaded here already says where it came from, and that is not Civitai's to rewrite.
export const canIdentify = (m) =>
  m.state === "present" && m.origin === "found" && !m.identified && !m.parts;

export function matches(model, query) {
  if (!query) return true;
  const haystack = [
    model.filename, model.title, model.version_name, model.note, model.base_model,
    kindLabel(model.category), model.host, model.relative, model.path,
    ...(model.trigger_words || []),
  ].join("\n").toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every((word) => haystack.includes(word));
}

function taskMatches(task, query) {
  if (!query) return true;
  const model = task.model_id ? state.models.get(task.model_id) : null;
  const haystack = [task.filename, task.original_filename, task.label, task.source, task.dest,
    task.origin, task.note, task.model_name, task.version_name, model?.title].join("\n").toLowerCase();
  return query.toLowerCase().split(/\s+/).filter(Boolean).every((word) => haystack.includes(word));
}

function buildView() {
  const view = state.view;
  const query = state.search.trim();

  if (view.kind === "folder") {
    if (query) {
      const found = sorted([...state.models.values()].filter((m) => matches(m, query)));
      return {
        title: `Search the library for “${esc(query)}”`,
        meta: `${plural(found.length, "model")}`,
        rows: found.map((m) => ({ key: `m${m.id}`, html: modelRow(m, { where: true }) })),
        empty: "Nothing in the library matches. The search looks at names, notes, trigger words, base models and folders.",
      };
    }
    const models = modelsIn(view.root, view.relative);
    const landing = downloadsIn(view.root, view.relative);
    const present = models.filter((m) => m.state !== "missing");
    const size = present.reduce((sum, m) => sum + (m.size || 0), 0);
    const rows = [
      ...landing.map((t) => ({ key: `t${t.id}`, html: ghostRow(t) })),
      ...sorted(models).map((m) => ({
        key: `m${m.id}`,
        html: modelRow(m, {
          where: view.root !== "outside" && m.relative !== view.relative ? "inner" : false,
          base: view.relative,
        }),
      })),
    ];
    return {
      title: crumbs(view),
      meta: `${plural(present.length, "model")} · ${fmtBytes(size)}`
        + (models.length > present.length ? ` · <span class="warn">${models.length - present.length} missing</span>` : "")
        + (landing.length ? ` · <span class="accent">${landing.length} downloading</span>` : ""),
      tools: folderTools(view),
      rows,
      empty: view.root === "outside"
        ? "Nothing is outside the library."
        : state.roots.length ? "No models in this folder." : "Add a folder to the library to see what is in it.",
    };
  }

  if (view.kind === "downloads") {
    const tasks = activeTasks().filter((t) => taskMatches(t, query))
      .sort((a, b) => (a.position ?? a.id) - (b.position ?? b.id) || a.id - b.id);
    const running = tasks.filter((t) => t.state === "running" || t.state === "pending").length;
    const waiting = tasks.filter((t) => t.state === "paused").length;
    const blocked = tasks.filter((t) => t.state === "blocked").length;
    const done = tasks.filter((t) => t.state === "done").length;
    return {
      title: "Downloads",
      meta: [plural(tasks.length, "file"), running && `${running} active`, waiting && `${waiting} waiting`,
        blocked && `<span class="warn">${blocked} need a decision</span>`].filter(Boolean).join(" · "),
      tools: [
        waiting ? `<button data-tool="start-all" class="primary">${icon("play")}Start all</button>` : "",
        running ? `<button data-tool="pause-all">${icon("pause")}Pause all</button>` : "",
        done ? `<button data-tool="clear-done" title="Move finished downloads to the history">${icon("history")}Clear finished</button>` : "",
      ].join(""),
      rows: tasks.map((t) => ({ key: `t${t.id}`, html: taskRow(t, !query) })),
      empty: query ? "No download matches." : "Nothing is downloading. Paste a link above — or anywhere, with Ctrl+V.",
      sortable: false,
    };
  }

  if (view.kind === "history") {
    const tasks = [...state.tasks.values()].filter((t) => t.state === "done" && taskMatches(t, query))
      .sort((a, b) => (b.finished_at || b.created_at) - (a.finished_at || a.created_at));
    const rows = [];
    let day = null;
    for (const task of tasks) {
      const label = dayLabel(task.finished_at || task.created_at);
      if (label !== day) { rows.push({ key: null, html: `<div class="group-head">${esc(label)}</div>` }); day = label; }
      rows.push({ key: `t${task.id}`, html: historyRow(task) });
    }
    const size = tasks.reduce((sum, t) => sum + (t.size || 0), 0);
    return {
      title: "History",
      meta: `${plural(tasks.length, "download")} · ${fmtBytes(size)}`,
      rows,
      empty: query ? "Nothing in the history matches." : "Nothing has been downloaded yet.",
      sortable: false,
    };
  }

  if (view.kind === "missing") {
    const models = missingModels().filter((m) => matches(m, query))
      .sort((a, b) => (b.missing_since || 0) - (a.missing_since || 0));
    const again = models.filter((m) => m.source);
    return {
      title: "Missing from disk",
      meta: `${plural(models.length, "model")} the library remembers but cannot find`
        + (again.length ? ` · ${again.length} can be downloaded again` : ""),
      help: "Moved or deleted outside ModelDL. A model that turns up elsewhere under the same name and size is linked again on its own; for the rest, download it again, point to the file, or forget it.",
      tools: `${again.length ? `<button data-tool="again-all" title="Each back into the folder it was in">${icon("download")}Download all again…</button>` : ""}
              ${models.length ? `<button data-tool="forget-all" class="danger">${icon("trash")}Forget all…</button>` : ""}`,
      rows: models.map((m) => ({ key: `m${m.id}`, html: modelRow(m, { where: true }) })),
      empty: "Every model the library knows about is where it should be.",
      sortable: false,
    };
  }

  if (view.kind === "unidentified") {
    const models = sorted(unidentifiedModels().filter((m) => matches(m, query)));
    const unhashed = models.filter((m) => !m.sha256).reduce((s, m) => s + (m.size || 0), 0);
    return {
      title: "Unidentified",
      meta: `${plural(models.length, "model")} not downloaded by ModelDL, and not described by anything beside them`,
      help: "What they are is read from the files themselves — their headers, and the folders they are in. Identify looks each one up on Civitai by its hash and on HuggingFace by its name and size, and fills in the rest; a big file is read in full only to prove a match.",
      tools: models.length ? `<button data-tool="identify-all" title="At most ${fmtBytes(unhashed)} to read">${icon("hash")}Identify all</button>` : "",
      rows: models.map((m) => ({ key: `m${m.id}`, html: modelRow(m, { where: true }) })),
      empty: "Every model in the library came with a description.",
    };
  }

  if (view.kind === "duplicates") {
    const data = state.duplicates;
    if (!data || data.loading) {
      return { title: "Duplicates", meta: "Looking…", rows: [], empty: "Comparing the files that share a size…" };
    }
    const all = data.groups || [];
    const wanted = (g) => !query || g.models.some((m) => matches(state.models.get(m.id) || m, query));
    const groups = all.filter(wanted);
    const linked = (data.linked || []).filter(wanted);
    const rows = [];
    for (const group of groups) {
      rows.push({ key: null, html: duplicateHead(group) });
      const keep = act.keeperOf(group);
      for (const m of group.models) {
        rows.push({
          key: `m${m.id}`,
          html: modelRow(state.models.get(m.id) || m, { where: true, keep: { group, chosen: m.id === keep } }),
        });
      }
    }
    if (linked.length) {
      rows.push({ key: null, html: `<div class="group-head">Linked — one file under several names</div>` });
      for (const group of linked) {
        rows.push({ key: null, html: linkedHead(group) });
        for (const m of group.models) {
          rows.push({ key: `m${m.id}`, html: modelRow(state.models.get(m.id) || m, { where: true }) });
        }
      }
    }
    const same = all.filter((g) => g.status === "same");
    const likely = all.filter((g) => g.status === "likely");
    const toHash = likely.flatMap((g) => g.to_hash);
    const toRead = likely.reduce((sum, g) => sum + g.to_read, 0);
    return {
      title: "Duplicates",
      meta: [
        same.length ? `${plural(same.length, "file")} kept more than once · ${fmtBytes(data.wasted)} to gain` : "",
        likely.length ? `<span class="warn">${likely.length} very likely · ${fmtBytes(data.likely)}</span>` : "",
        (data.linked || []).length ? `${fmtBytes(data.saved)} saved by links` : "",
      ].filter(Boolean).join(" · ") || "No copies",
      help: "Files that share a size are first compared by a few small pieces of each"
        + (data.told_apart ? ` — ${plural(data.told_apart, "set")} of them turned out to be different models` : "")
        + ". Certain once the hashes of the whole files match. A workflow names the file it loads, folder"
        + " included: after deleting a copy, point the workflows that used it at the one you kept.",
      tools: `${toHash.length ? `<button data-tool="hash-likely" data-ids="${toHash.join(",")}" title="Reads ${fmtBytes(toRead)}">${icon("hash")}Confirm all by hash</button>` : ""}
              <button data-tool="reload-dups">${icon("refresh")}Look again</button>`,
      rows,
      empty: data.error ? esc(data.error)
        : query ? "No copy matches."
        : (data.linked || []).length ? "No file is kept twice."
        : `No file is kept twice.${data.shared_sizes ? ` ${plural(data.shared_sizes, "set")} of files share a size, and all of them are different models.` : ""}`,
      sortable: false,
    };
  }

  if (view.kind === "cleanup") {
    const data = state.cleanup;
    if (!data || data.loading) return { title: "Cleanup", meta: "Looking…", rows: [], empty: "Looking through the folders…" };
    const items = data.items.filter((i) => !query || `${i.name} ${i.folder} ${i.why}`.toLowerCase().includes(query.toLowerCase()));
    return {
      title: "Cleanup",
      meta: `${plural(items.length, "item")} · ${fmtBytes(items.reduce((s, i) => s + i.size, 0))}`,
      help: "Unfinished downloads nobody is coming back for, and files named after models that are not there any more.",
      tools: `${items.length ? `<button data-tool="cleanup-all" class="danger">${icon("trash")}Delete all…</button>` : ""}
              <button data-tool="reload-cleanup">${icon("refresh")}Look again</button>`,
      rows: items.map((i) => ({ key: `c${i.id}`, html: cleanupRow(i) })),
      empty: data.error ? esc(data.error) : "Nothing to clean up.",
      sortable: false,
    };
  }

  return { title: "", rows: [], empty: "" };
}

function crumbs(view) {
  if (view.root === "outside") return "Outside the library";
  const root = state.roots[view.root];
  if (!root) return "Library";
  const parts = view.relative ? view.relative.split("/") : [];
  const links = [`<button class="crumb" data-crumb="">${esc(root.name)}</button>`];
  parts.forEach((part, index) => {
    links.push(`<span class="crumb-sep">›</span><button class="crumb" data-crumb="${esc(parts.slice(0, index + 1).join("/"))}">${esc(part)}</button>`);
  });
  return links.join("");
}

function folderTools(view) {
  const deep = `<button class="toggle ${state.deep ? "on" : ""}" data-tool="deep" title="Show what is in the folders inside this one too">${icon("layers")}Subfolders</button>`;
  const more = view.root === "outside" ? "" : `<button class="icon-button" data-tool="folder-menu" aria-label="More">${icon("dots")}</button>`;
  return deep + more;
}

// --- rows ---------------------------------------------------------------------------------

function thumb(model, size = 40) {
  const source = { kind: "model", id: model.id };
  if (model.previews && model.state !== "missing") {
    const hide = covered(source, model.nsfw);
    // Not draggable: a picture dragged by a shaky click carries its own URL, which the
    // window would otherwise take for a link dropped in to download.
    return `<div class="thumb ${hide ? "covered" : ""}" data-thumb="m${model.id}">
      <img src="${previewSrc(source, 0, 160)}" loading="lazy" decoding="async" alt="" draggable="false"
           onerror="this.remove()" width="${size}" height="${size}"></div>`;
  }
  return `<div class="thumb empty">${icon(model.state === "missing" ? "file-off" : kindIcon(model.category))}</div>`;
}

export function chips(model) {
  const list = [];
  if (model.category) list.push(`<span class="chip kind">${esc(kindLabel(model.category))}</span>`);
  if (model.base_model) list.push(`<span class="chip">${esc(model.base_model)}</span>`);
  if (model.precision) list.push(`<span class="chip quiet">${esc(model.precision)}</span>`);
  if (model.links > 1) {
    list.push(`<span class="chip link" title="One file under ${model.links} names — deleting this one frees nothing while another is left">linked · ${model.links}</span>`);
  }
  if (model.origin === "found") {
    list.push(model.identified
      ? `<span class="chip quiet" title="Not downloaded by ModelDL, identified afterwards">found</span>`
      : `<span class="chip quiet" title="Not downloaded by ModelDL">not from ModelDL</span>`);
  } else if (model.host) {
    list.push(`<span class="chip quiet">${esc(model.host)}</span>`);
  }
  return list.join("");
}

// The head of one set of copies in the Duplicates view: how sure, how much it would free, and
// the one thing to do next — prove it with the hashes, or delete the copies not kept.
function duplicateHead(group) {
  const first = state.models.get(group.models[0].id) || group.models[0];
  const name = first.title && first.title !== first.filename ? first.title : first.filename;
  const sure = group.status === "same";
  const others = group.copies - 1;
  const linkable = sure ? act.linkable(group) : [];
  const action = sure
    ? `${linkable.length ? `<button class="mini" data-dup="link" data-group="${esc(group.key)}" title="The other copies become names of the one kept: every path keeps working, and their room is freed">${icon("link")}Link the copies…</button>` : ""}
       <button class="mini danger" data-dup="dedupe" data-group="${esc(group.key)}">${icon("trash")}Delete ${others === 1 ? "the other copy" : `${others} other copies`}…</button>`
    : `<button class="mini" data-dup="confirm" data-group="${esc(group.key)}" title="Reads ${fmtBytes(group.to_read)}">${icon("hash")}Confirm by hash</button>`;
  return `
    <div class="dup-head">
      <div class="dup-line">
        <span class="dup-title" title="${esc(name)}">${esc(name)}</span>
        <span class="state ${sure ? "ok" : "warn"}" title="${sure ? "The hashes of the whole files match" : "The pieces compared agree — only the hashes of the whole files are proof"}">${sure ? "identical" : "very likely identical"}</span>
      </div>
      <div class="dup-line">
        <span class="dup-meta">${plural(group.copies, "copy", "copies")} · ${fmtBytes(group.wasted)} to gain</span>
        <span class="grow"></span>${action}
      </div>
      ${group.caution ? `<div class="dup-caution">${icon("alert")}<span>${esc(group.caution)}</span></div>` : ""}
    </div>`;
}

// A file that is already under several names: nothing to gain, and the way back to copies.
function linkedHead(group) {
  const first = state.models.get(group.models[0].id) || group.models[0];
  const name = first.title && first.title !== first.filename ? first.title : first.filename;
  return `
    <div class="dup-head">
      <div class="dup-line">
        <span class="dup-title" title="${esc(name)}">${esc(name)}</span>
        <span class="state ok" title="Hard links: one file on the disk, under several names">linked</span>
      </div>
      <div class="dup-line">
        <span class="dup-meta">one file, ${group.names} names · ${fmtBytes(group.saved)} saved${group.outside ? ` · ${plural(group.outside, "name")} outside the library` : ""}</span>
        <span class="grow"></span>
        <button class="mini" data-dup="separate" data-group="${esc(group.key)}" title="Give every name a copy of its own again">${icon("copy")}Make separate copies…</button>
      </div>
    </div>`;
}

function keepToggle({ group, chosen }, id) {
  if (chosen) {
    const why = group.keep === id && group.keep_why ? ` — ${group.keep_why}` : "";
    return `<span class="keep on" title="${esc(`Kept when the other copies are deleted${why}`)}">${icon("star")}Keep</span>`;
  }
  return `<button class="keep" data-keep="${esc(group.key)}" data-id="${id}" title="Keep this copy instead">${icon("copy")}Copy</button>`;
}

function modelRow(model, { where = false, base = "", keep = null } = {}) {
  const selected = state.selection.has(`m${model.id}`);
  const missing = model.state === "missing";
  const place = model.root === null || model.root === undefined
    ? baseName(model.folder)
    : folderLabel(model.root, model.relative);
  const inner = base && model.relative.startsWith(`${base}/`) ? model.relative.slice(base.length + 1) : model.relative;
  const marks = [
    model.update ? `<span class="mark accent" title="A newer version: ${esc(model.update.version_name || "")}">${icon("update")}</span>` : "",
    model.left_behind ? `<span class="mark warn" title="Files named after it were left in the folder it came from">${icon("back")}</span>` : "",
    model.note ? `<span class="mark note" title="${esc(model.note)}">${icon("note")}</span>` : "",
    model.trigger_words?.length ? `<span class="mark" title="${esc(model.trigger_words.join(", "))}">${icon("tag")}</span>` : "",
    model.parts ? `<span class="mark" title="${model.parts} parts">${icon("layers")}</span>` : "",
  ].join("");
  const shown = where === "inner" ? inner : place;
  const sub = missing
    ? `<span class="warn">Missing since ${fmtDate(model.missing_since)}</span> · last seen in ${esc(place)}`
    : `${chips(model)}${where && shown ? `<span class="where" title="${esc(model.path)}">${icon("folder")}${esc(shown)}</span>` : ""}`;
  return `
    <div class="row model ${missing ? "missing" : ""} ${selected ? "selected" : ""}" data-key="m${model.id}"
         draggable="${missing ? "false" : "true"}">
      ${thumb(model)}
      <div class="row-main">
        <div class="row-title"><span class="name">${esc(model.filename)}</span>${marks}</div>
        <div class="row-sub">${sub}</div>
      </div>
      ${keep ? keepToggle(keep, model.id) : ""}
      <div class="row-side">
        <span class="size">${fmtBytes(model.size)}</span>
        <span class="date" title="${esc(fmtDate(model.first_seen))}">${missing ? "" : fmtAgo(model.first_seen)}</span>
      </div>
    </div>`;
}

function stats(task) {
  if (task.state === "running") {
    return `${fmtBytes(task.downloaded)} of ${fmtBytes(task.size)}`
      + (task.speed ? ` · ${fmtSpeed(task.speed)}` : "")
      + (task.eta ? ` · ETA ${fmtEta(task.eta)}` : "");
  }
  if (task.state === "failed") return esc(task.error || "failed");
  if (task.state === "blocked") return `<span class="warn">Needs a decision on where it goes</span>`;
  if (task.state === "paused") return `${fmtBytes(task.downloaded)} of ${fmtBytes(task.size)} · paused`;
  if (task.state === "pending") return `waiting · ${fmtBytes(task.size)}`;
  return fmtBytes(task.size);
}

function taskThumb(task, size = 40) {
  if (task.previews) {
    const source = { kind: "task", id: task.id };
    const hide = covered(source, task.nsfw);
    return `<div class="thumb ${hide ? "covered" : ""}" data-thumb="t${task.id}">
      <img src="${previewSrc(source, 0, 160)}" loading="lazy" decoding="async" alt="" draggable="false"
           onerror="this.remove()" width="${size}" height="${size}"></div>`;
  }
  return `<div class="thumb empty">${icon("download")}</div>`;
}

function ghostRow(task) {
  const pct = task.fraction ? Math.round(task.fraction * 100) : 0;
  return `
    <div class="row task ghost ${state.selection.has(`t${task.id}`) ? "selected" : ""}" data-key="t${task.id}">
      ${taskThumb(task)}
      <div class="row-main">
        <div class="row-title"><span class="name">${esc(task.filename || task.source)}</span>
          <span class="state ${task.state}">${task.state}</span></div>
        <div class="track"><div class="fill" style="width:${pct}%"></div></div>
        <div class="row-sub" data-role="stats">${stats(task)}</div>
      </div>
      <div class="row-side"><span class="size">${fmtBytes(task.size)}</span></div>
    </div>`;
}

function taskRow(task, draggable) {
  const pct = task.fraction ? Math.round(task.fraction * 100) : 0;
  const done = task.state === "done";
  const buttons = [];
  if (task.state === "blocked") {
    buttons.push(`<button class="mini primary" data-cmd="confirm" title="File it where the card says">Accept</button>`);
    buttons.push(`<button class="mini" data-cmd="where" title="Choose a folder in the library">Elsewhere…</button>`);
  }
  if (task.state === "running" || task.state === "pending") buttons.push(`<button class="icon-button" data-cmd="pause" aria-label="Pause" title="Pause">${icon("pause")}</button>`);
  if (task.state === "paused") buttons.push(`<button class="icon-button" data-cmd="resume" aria-label="Resume" title="Resume">${icon("play")}</button>`);
  if (task.state === "failed") buttons.push(`<button class="icon-button" data-cmd="retry" aria-label="Retry" title="Retry">${icon("retry")}</button>`);
  const where = task.dest ? placementOf(task.dest) : { root: null };
  const folder = where.root !== null ? folderLabel(where.root, where.relative) : "";
  // A finished download says what became of its file: done is only true while it is there.
  const fate = done ? historyStatus(task) : null;
  const label = task.state === "blocked" ? "needs a decision"
    : fate && fate.label && fate.label !== "in library" ? fate.label : task.state;
  const cls = fate && fate.label && fate.label !== "in library" ? fate.cls : task.state;
  // Once it has landed, the picture is the model's: the one beside the file on disk, which
  // is there whether or not the service published any.
  const picture = fate && fate.model ? thumb(fate.model) : taskThumb(task);
  return `
    <div class="row task ${task.state} ${state.selection.has(`t${task.id}`) ? "selected" : ""}" data-key="t${task.id}">
      ${draggable ? `<span class="grip" title="Drag to change what downloads next">${icon("grip")}</span>` : ""}
      ${picture}
      <div class="row-main">
        <div class="row-title"><span class="name">${esc(task.filename || task.source)}</span>
          <span class="state ${cls}">${label}</span>
          ${task.note ? `<span class="mark note" title="${esc(task.note)}">${icon("note")}</span>` : ""}</div>
        ${done ? "" : `<div class="track"><div class="fill ${task.state}" style="width:${pct}%"></div></div>`}
        <div class="row-sub"><span data-role="stats">${stats(task)}</span>${folder ? `<span class="where">${icon("folder")}${esc(folder)}</span>` : ""}</div>
      </div>
      <div class="row-actions">${buttons.join("")}</div>
      <div class="row-side"><span class="size">${esc(task.origin || "")}</span>
        <span class="date">${done ? fmtAgo(task.finished_at) : ""}</span></div>
    </div>`;
}

export function historyStatus(task) {
  const model = task.model_id ? state.models.get(task.model_id) : null;
  if (model && model.state === "present") return { label: "in library", cls: "ok", model };
  if (model && model.state === "missing") return { label: "missing", cls: "warn", model };
  if (task.fate === "forgotten") return { label: "forgotten", cls: "muted", model: null };
  if (task.fate === "deleted" || !model) return { label: "deleted", cls: "muted", model: null };
  return { label: "", cls: "", model };
}

function historyRow(task) {
  const status = historyStatus(task);
  const model = status.model;
  const name = model ? model.filename : task.filename;
  const renamed = task.original_filename && task.original_filename !== name;
  return `
    <div class="row history ${status.cls === "muted" ? "gone" : ""} ${state.selection.has(`t${task.id}`) ? "selected" : ""}" data-key="t${task.id}">
      <span class="time">${fmtTime(task.finished_at || task.created_at)}</span>
      ${model ? thumb(model, 32) : taskThumb(task, 32)}
      <div class="row-main">
        <div class="row-title"><span class="name">${esc(name)}</span>
          <span class="state ${status.cls}">${status.label}</span></div>
        <div class="row-sub">${[
          task.model_name ? esc([task.model_name, task.version_name].filter(Boolean).join(" / ")) : "",
          esc(task.origin || ""),
          renamed ? `downloaded as <span class="mono">${esc(task.original_filename)}</span>` : "",
        ].filter(Boolean).join(" · ")}</div>
      </div>
      <div class="row-side"><span class="size">${fmtBytes(task.size)}</span>
        <span class="date">${task.duration ? `took ${fmtDuration(task.duration)}` : ""}</span></div>
    </div>`;
}

function cleanupRow(item) {
  const icons = { fragment: "download", orphan: "file-off", record: "note" };
  const where = placementOf(`${item.folder}${state.sep}x`);
  const folder = where.root !== null ? folderLabel(where.root, where.relative) : item.folder;
  return `
    <div class="row cleanup ${state.selection.has(`c${item.id}`) ? "selected" : ""}" data-key="c${item.id}">
      <div class="thumb empty">${icon(icons[item.kind] || "file")}</div>
      <div class="row-main">
        <div class="row-title"><span class="name">${esc(item.name)}</span></div>
        <div class="row-sub">${esc(item.why)} · ${plural(item.files.length, "file")}
          <span class="where" title="${esc(item.folder)}">${icon("folder")}${esc(folder)}</span></div>
      </div>
      <div class="row-side"><span class="size">${fmtBytes(item.size)}</span></div>
    </div>`;
}

// --- drawing ---------------------------------------------------------------------------------

let shownView = "";

function renderList() {
  const holder = document.getElementById("center");
  if (state.view.kind === "settings") {
    // Nothing on this page is a row: what the keyboard can select must not be the rows of
    // whatever list was on screen before it.
    visible = [];
    renderSettings();
    return;
  }
  const built = buildView();
  visible = built.rows.map((r) => r.key).filter(Boolean);
  const list = holder.querySelector("#rows");
  // A redraw of the same list keeps where it was scrolled to; another folder or another
  // view starts at its top.
  const view = `${state.view.kind}|${state.view.root}|${state.view.relative}|${state.search.trim()}`;
  const scroll = list && view === shownView ? list.scrollTop : 0;
  shownView = view;
  const sortable = built.sortable !== false;
  holder.innerHTML = `
    <div class="list-head">
      <div class="list-title">${built.title}</div>
      <span class="list-meta">${built.meta || ""}</span>
      <span class="grow"></span>
      <div class="list-tools">${built.tools || ""}
        ${sortable ? `<select data-tool="sort" aria-label="Sort by" title="Sort by">
          ${[["name", "Name"], ["date", "Newest"], ["size", "Size"], ["kind", "Kind"]].map(([k, l]) =>
            `<option value="${k}" ${state.sort.key === k ? "selected" : ""}>${l}</option>`).join("")}
        </select>` : ""}
      </div>
    </div>
    ${built.help ? `<div class="list-help">${esc(built.help)}</div>` : ""}
    <div class="list" id="rows" tabindex="0" role="listbox" aria-multiselectable="true">
      ${built.rows.length ? built.rows.map((r) => r.html).join("") : `<div class="list-empty">${built.empty || ""}</div>`}
    </div>`;
  holder.querySelector("#rows").scrollTop = scroll;
  if (state.view.kind === "downloads" && !state.search.trim()) wireReorder(holder.querySelector("#rows"));
}

// Progress arrives several times a second; redrawing the list for each would fight the
// person scrolling it. The row's own bar and numbers are changed in place instead.
export function patchProgress(task) {
  for (const row of document.querySelectorAll(`[data-key="t${task.id}"]`)) {
    const fill = row.querySelector(".fill");
    if (fill) fill.style.width = `${Math.round((task.fraction || 0) * 100)}%`;
    const numbers = row.querySelector('[data-role="stats"]');
    if (numbers) numbers.innerHTML = stats(task);
  }
}

// --- behaviour -----------------------------------------------------------------------------

export const visibleKeys = () => visible;

function clickRow(event, key) {
  if (event.shiftKey && state.anchor && visible.includes(state.anchor)) {
    const [a, b] = [visible.indexOf(state.anchor), visible.indexOf(key)].sort((x, y) => x - y);
    const range = visible.slice(a, b + 1);
    const keys = event.ctrlKey || event.metaKey ? [...new Set([...state.selection, ...range])] : range;
    select(keys, { anchor: state.anchor, focus: key });
    return;
  }
  if (event.ctrlKey || event.metaKey) {
    const next = new Set(state.selection);
    if (next.has(key)) next.delete(key); else next.add(key);
    select([...next], { anchor: key, focus: key });
    return;
  }
  select([key], { anchor: key, focus: key });
}

export function moveSelection(step, extend = false) {
  if (!visible.length) return;
  const from = state.focus && visible.includes(state.focus) ? visible.indexOf(state.focus) : -1;
  const to = Math.max(0, Math.min(visible.length - 1, from + step));
  const key = visible[to];
  if (extend && state.anchor && visible.includes(state.anchor)) {
    const [a, b] = [visible.indexOf(state.anchor), to].sort((x, y) => x - y);
    select(visible.slice(a, b + 1), { anchor: state.anchor, focus: key });
  } else {
    select([key], { anchor: key, focus: key });
  }
  requestAnimationFrame(() => document.querySelector(`[data-key="${key}"]`)?.scrollIntoView({ block: "nearest" }));
}

export function selectAll() {
  select(visible, { anchor: visible[0], focus: visible[visible.length - 1] });
}

function ids(prefix) {
  return [...state.selection].filter((k) => k[0] === prefix).map((k) => Number(k.slice(1)));
}

export function rowMenu(event, key) {
  if (!state.selection.has(key)) select([key], { anchor: key, focus: key });
  const kind = key[0];
  let items = [];
  if (kind === "m") items = modelMenu(selectedModels());
  else if (kind === "t") items = taskMenu(ids("t").map((id) => state.tasks.get(id)).filter(Boolean));
  else if (kind === "c") items = [
    { label: "Delete…", icon: "trash", danger: true, run: () => act.deleteCleanup(ids("c")) },
  ];
  showMenu(event.clientX, event.clientY, items);
}

export function modelMenu(models) {
  if (!models.length) return [];
  const idsOf = models.map((m) => m.id);
  const present = models.filter((m) => m.state === "present");
  const missing = models.filter((m) => m.state === "missing");
  if (models.length > 1) {
    const again = missing.filter((m) => m.source);
    return [
      present.length && { label: `Move ${present.length} to…`, icon: "move", run: () => act.moveModelsDialog(present.map((m) => m.id)) },
      again.length && { label: `Download ${again.length} again…`, icon: "download", run: () => act.downloadAllAgain(again.map((m) => m.id)) },
      models.some(canIdentify) && { label: "Identify", icon: "hash", run: () => act.identify(models.filter(canIdentify).map((m) => m.id)) },
      { label: "Check for newer versions", icon: "update", run: () => act.checkUpdates(idsOf) },
      "-",
      present.length && { label: `Delete ${present.length} from disk…`, icon: "trash", danger: true, hint: "Del", run: () => act.deleteModels(present.map((m) => m.id)) },
      missing.length && { label: `Forget ${missing.length} missing…`, icon: "x", danger: true, run: () => act.forgetModels(missing.map((m) => m.id)) },
    ];
  }
  const [model] = models;
  if (model.state === "missing") {
    return [
      model.source && { label: "Download again…", icon: "download", run: () => act.downloadAgain(model.id) },
      { label: "Find online…", icon: "globe", run: () => act.findOnline(model.id) },
      { label: "Find the file…", icon: "search", run: () => act.locateModel(model.id) },
      { label: "Open the folder it was in", icon: "external", run: () => act.revealModel(model.id) },
      "-",
      { label: "Forget…", icon: "x", danger: true, hint: "Del", run: () => act.forgetModels([model.id]) },
    ];
  }
  return [
    { label: "Show in Explorer", icon: "external", run: () => act.revealModel(model.id) },
    model.previews && { label: "Samples", icon: "image", run: () => openLightbox({ kind: "model", id: model.id }, 0, model.filename) },
    "-",
    { label: "Rename", icon: "edit", hint: "F2", run: () => document.dispatchEvent(new CustomEvent("rename-model", { detail: model.id })) },
    { label: "Move to…", icon: "move", run: () => act.moveModelsDialog([model.id]) },
    model.links > 1 && { label: "Make a separate copy…", icon: "copy", run: () => act.separateModels([model.id]) },
    "-",
    model.trigger_words?.length && { label: "Copy trigger words", icon: "copy", run: () => copyText(model.trigger_words.join(", ")).then(() => toast("Trigger words copied")) },
    { label: "Copy path", icon: "copy", run: () => copyText(model.path).then(() => toast("Path copied")) },
    "-",
    canIdentify(model) && { label: "Identify", icon: "hash", run: () => act.identify([model.id]) },
    model.hash_source === "download" && { label: "Verify the file", icon: "check", run: () => act.verify([model.id]) },
    (model.provider === "civitai" || model.provider === "huggingface") && { label: "Check for a newer version", icon: "update", run: () => act.checkUpdates([model.id]) },
    "-",
    { label: "Delete from disk…", icon: "trash", danger: true, hint: "Del", run: () => act.deleteModels([model.id]) },
  ];
}

function taskMenu(tasks) {
  if (!tasks.length) return [];
  if (state.view.kind === "history") {
    const [task] = tasks;
    const status = historyStatus(task);
    return [
      status.model && { label: "Show in the library", icon: "folder", run: () => act.showModel(status.model.id) },
      !status.model || status.model.state === "missing"
        ? { label: "Download it again", icon: "download", run: () => act.redownload(task.id) } : null,
      "-",
      { label: tasks.length > 1 ? `Remove ${tasks.length} from the history` : "Remove from the history", icon: "x", danger: true, run: () => act.removeFromHistory(tasks.map((t) => t.id)) },
    ];
  }
  const [task] = tasks;
  const items = [];
  if (tasks.length === 1) {
    if (task.state === "blocked") {
      items.push({ label: "Accept the placement", icon: "check", run: () => act.taskCommand(task.id, "confirm") });
      items.push({ label: "Elsewhere…", icon: "folder", run: () => act.placeTask(task.id) });
    }
    if (task.state === "running" || task.state === "pending") items.push({ label: "Pause", icon: "pause", run: () => act.taskCommand(task.id, "pause") });
    if (task.state === "paused") items.push({ label: "Resume", icon: "play", run: () => act.taskCommand(task.id, "resume") });
    if (task.state === "failed") items.push({ label: "Retry", icon: "retry", run: () => act.taskCommand(task.id, "retry") });
    if (task.state === "done" && task.model_id) items.push({ label: "Show in the library", icon: "folder", run: () => act.showModel(task.model_id) });
    if (task.dest) items.push({ label: "Show in Explorer", icon: "external", run: () => post(`/api/tasks/${task.id}/reveal`).catch(() => {}) });
    items.push("-");
    if (task.dest && task.state !== "done") items.push({ label: "Delete its files…", icon: "trash", danger: true, run: () => act.deleteTaskFiles(task.id) });
  }
  items.push({
    label: tasks.length > 1 ? `Remove ${tasks.length} from the list` : "Remove from the list",
    icon: "x", danger: true,
    run: () => tasks.forEach((t) => act.taskCommand(t.id, "remove")),
  });
  return items;
}

export function wireList() {
  const holder = document.getElementById("center");

  holder.addEventListener("click", (event) => {
    const tool = event.target.closest("[data-tool]");
    if (tool && tool.tagName !== "SELECT") { runTool(tool); return; }
    const crumb = event.target.closest("[data-crumb]");
    if (crumb) { act.go({ kind: "folder", root: state.view.root, relative: crumb.dataset.crumb }); return; }
    const dup = event.target.closest("[data-dup]");
    if (dup) { event.stopPropagation(); act.duplicateCommand(dup.dataset.dup, dup.dataset.group); return; }
    const keep = event.target.closest("[data-keep]");
    if (keep) { event.stopPropagation(); act.chooseKeeper(keep.dataset.keep, Number(keep.dataset.id)); return; }
    const command = event.target.closest("[data-cmd]");
    const row = event.target.closest(".row[data-key]");
    if (command && row) {
      event.stopPropagation();
      const id = Number(row.dataset.key.slice(1));
      if (command.dataset.cmd === "where") act.placeTask(id);
      else act.taskCommand(id, command.dataset.cmd);
      return;
    }
    const picture = event.target.closest("[data-thumb]");
    if (picture && picture.classList.contains("covered")) {
      event.stopPropagation();
      const key = picture.dataset.thumb;
      state.revealed.add(key[0] === "m" ? `model${key.slice(1)}` : `task${key.slice(1)}`);
      invalidate("list", "inspector");
      return;
    }
    if (row) clickRow(event, row.dataset.key);
    else if (event.target.closest("#rows")) select([]);
  });

  holder.addEventListener("dblclick", (event) => {
    const row = event.target.closest(".row[data-key]");
    if (!row) return;
    const key = row.dataset.key;
    const id = Number(key.slice(1));
    if (key[0] === "m") {
      const model = state.models.get(id);
      if (!model) return;
      if (model.previews && model.state === "present") openLightbox({ kind: "model", id }, 0, model.filename);
      else act.revealModel(id);
    } else if (key[0] === "t") {
      const task = state.tasks.get(id);
      if (task?.model_id && state.view.kind !== "downloads") act.showModel(task.model_id);
    }
  });

  holder.addEventListener("change", (event) => {
    const tool = event.target.closest('[data-tool="sort"]');
    if (!tool) return;
    state.sort = { key: tool.value, dir: 1 };
    remember({ sort: state.sort });
    invalidate("list");
  });

  holder.addEventListener("contextmenu", (event) => {
    const row = event.target.closest(".row[data-key]");
    if (!row) return;
    event.preventDefault();
    rowMenu(event, row.dataset.key);
  });

  holder.addEventListener("dragstart", (event) => {
    const row = event.target.closest(".row.model[data-key]");
    if (!row) return;
    const key = row.dataset.key;
    const chosen = state.selection.has(key) ? selectedModels().filter((m) => m.state === "present").map((m) => m.id)
      : [Number(key.slice(1))];
    if (!chosen.length) { event.preventDefault(); return; }
    event.dataTransfer.effectAllowed = "move";
    event.dataTransfer.setData("application/x-modeldl-models", JSON.stringify(chosen));
    event.dataTransfer.setData("text/plain", chosen.map((id) => state.models.get(id)?.path).join("\n"));
    const label = document.createElement("div");
    label.className = "drag-label";
    label.textContent = chosen.length > 1 ? `${chosen.length} models` : state.models.get(chosen[0])?.filename || "";
    document.body.appendChild(label);
    event.dataTransfer.setDragImage(label, 12, 12);
    setTimeout(() => label.remove(), 0);
    document.body.classList.add("dragging-models");
  });
  holder.addEventListener("dragend", () => document.body.classList.remove("dragging-models"));
}

async function runTool(tool) {
  const name = tool.dataset.tool;
  const view = state.view;
  if (name === "deep") { state.deep = !state.deep; remember({ deep: state.deep }); invalidate("list"); return; }
  if (name === "start-all") return act.startAll();
  if (name === "pause-all") return act.pauseAll();
  if (name === "clear-done") return act.clearFinished();
  if (name === "reload-cleanup") return act.loadCleanup();
  if (name === "reload-dups") return act.loadDuplicates();
  if (name === "hash-likely") return act.hashModels(tool.dataset.ids.split(",").map(Number));
  if (name === "cleanup-all") return act.deleteCleanup((state.cleanup?.items || []).map((i) => i.id));
  if (name === "forget-all") return act.forgetModels(missingModels().map((m) => m.id));
  if (name === "again-all") return act.downloadAllAgain(missingModels().filter((m) => m.source).map((m) => m.id));
  if (name === "identify-all") return act.identify(unidentifiedModels().map((m) => m.id));
  if (name === "folder-menu") {
    const models = modelsIn(view.root, view.relative);
    const present = models.filter((m) => m.state === "present");
    const unknown = present.filter((m) => m.origin === "found" && !m.identified);
    const known = present.filter((m) => m.provider === "civitai" || m.provider === "huggingface");
    menuFrom(tool, [
      { label: "Show in Explorer", icon: "external", run: () => act.revealFolder(view.root, view.relative) },
      { label: "New folder…", icon: "plus", run: () => act.newFolder(view.root, view.relative) },
      "-",
      { label: `Identify ${unknown.length} unidentified here`, icon: "hash", disabled: !unknown.length, run: () => act.identify(unknown.map((m) => m.id)) },
      { label: `Check ${known.length} here for newer versions`, icon: "update", disabled: !known.length, run: () => act.checkUpdates(known.map((m) => m.id)) },
      "-",
      view.relative && { label: "Hide this folder from the library", icon: "eye-off", run: () => act.hideFolder(view.root, view.relative) },
    ]);
  }
}

// Dragging a download by its grip changes what runs next. Only with nothing filtered out:
// the reorder sends the rows on screen, and applying it while some are hidden would shuffle
// them around the ones it cannot see.
function wireReorder(list) {
  let dragging = null;
  list.querySelectorAll(".row.task").forEach((row) => {
    const grip = row.querySelector(".grip");
    if (!grip) return;
    grip.addEventListener("mousedown", () => { row.draggable = true; });
    const release = () => { row.draggable = false; };
    grip.addEventListener("mouseup", release);
    row.addEventListener("dragstart", (event) => {
      dragging = row;
      row.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
      // A type of its own: plain text is what a link dragged in from a browser carries, and
      // the window would dim itself for a drop that is only a row changing places.
      event.dataTransfer.setData("application/x-modeldl-reorder", row.dataset.key);
    });
    row.addEventListener("dragend", async () => {
      release();
      row.classList.remove("dragging");
      if (!dragging) return;
      dragging = null;
      const order = [...list.querySelectorAll(".row.task")].map((n) => Number(n.dataset.key.slice(1)));
      try { await post("/api/tasks/reorder", { ids: order }); } catch (error) { toast(error.message, { level: "error" }); }
    });
    row.addEventListener("dragover", (event) => {
      if (!dragging || dragging === row) return;
      event.preventDefault();
      const box = row.getBoundingClientRect();
      const after = event.clientY > box.top + box.height / 2;
      list.insertBefore(dragging, after ? row.nextSibling : row);
    });
  });
}

onRender("list", renderList);
