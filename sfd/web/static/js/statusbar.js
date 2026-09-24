// The strip along the bottom: what the queue as a whole is doing, what is being copied or
// hashed right now (with the button that stops it), and how much room is left.

import { get, post } from "./api.js";
import { esc, fmtAgo, fmtBytes, fmtEta, fmtSpeed, setBits, usingBits, debounce, plural } from "./util.js";
import { icon } from "./icons.js";
import { state, onRender, invalidate, remember } from "./store.js";
import { toastError } from "./toasts.js";
import * as act from "./actions.js";

let space = null;

export const refreshSpace = debounce(async () => {
  try { space = await get("/api/space"); } catch { space = null; }
  invalidate("status");
}, 1500);

function queueSummary() {
  const unfinished = [...state.tasks.values()].filter((t) => t.state !== "done" && !t.archived);
  if (!unfinished.length) return "";
  const running = unfinished.filter((t) => t.state === "running");
  const blocked = unfinished.filter((t) => t.state === "blocked").length;
  const sized = unfinished.filter((t) => t.size);
  const total = sized.reduce((sum, t) => sum + t.size, 0);
  const left = sized.reduce((sum, t) => sum + Math.max(0, t.size - (t.downloaded || 0)), 0);
  const speed = running.reduce((sum, t) => sum + (t.speed || 0), 0);
  const parts = [running.length ? `${running.length} downloading` : `${plural(unfinished.length, "download")} waiting`];
  if (blocked) parts.push(`<span class="warn">${blocked} need a decision</span>`);
  if (total) parts.push(`${fmtBytes(total - left)} of ${fmtBytes(total)}`);
  if (speed > 0) parts.push(fmtSpeed(speed), `ETA ${fmtEta(left / speed)}`);
  return `<button class="status-item link" data-status="downloads">${icon("download")}${parts.join(" · ")}</button>`;
}

function workSummary() {
  const bits = [];
  if (state.moving) {
    const share = state.moving.total ? Math.round((state.moving.copied / state.moving.total) * 100) : 0;
    bits.push(`<span class="status-item busy">${icon(state.moving.verb ? "copy" : "move")}${esc(state.moving.verb || "Moving")}… ${fmtBytes(state.moving.copied)} of ${fmtBytes(state.moving.total)} (${share}%)
      <button class="mini danger" data-status="stop-move">${state.moving.stopping ? "stopping…" : "Stop"}</button></span>`);
  }
  const job = state.jobs.current;
  if (job) {
    const verb = { identify: "Identifying", verify: "Verifying", hash: "Hashing" }[job.kind] || "Hashing";
    const share = job.total ? ` ${Math.round((job.done / job.total) * 100)}%` : "";
    const queued = state.jobs.queued?.length ? ` · ${state.jobs.queued.length} more` : "";
    bits.push(`<span class="status-item busy" title="${esc(job.name)}">${icon("hash")}${verb} ${esc(job.name)}${share}${queued}
      <button class="mini" data-status="stop-jobs">Stop</button></span>`);
  }
  return bits.join("");
}

function renderStatus() {
  const holder = document.getElementById("status");
  const primary = state.roots[0];
  const short = space && space.free !== null && space.needed > space.free;
  holder.innerHTML = `
    ${queueSummary()}
    ${workSummary()}
    <span class="grow"></span>
    ${short ? `<span class="status-item warn" title="What the queue still has to fetch, against what the disk has left">${icon("alert")}needs ${fmtBytes(space.needed)}, ${fmtBytes(space.free)} free</span>` : ""}
    ${primary && primary.free != null && !short ? `<span class="status-item muted" title="${esc(primary.path)}">${icon("drive")}${fmtBytes(primary.free)} free</span>` : ""}
    <button class="status-item link" data-status="units" title="Switch speed between bytes and bits">${usingBits() ? "Mbit/s" : "MB/s"}</button>
    <button class="status-item link" data-status="rescan" title="Read the library's folders again">${icon("refresh")}${state.syncedAt ? `read ${esc(fmtAgo(state.syncedAt))}` : "reading…"}</button>`;
}

export function wireStatus() {
  const holder = document.getElementById("status");
  holder.addEventListener("click", async (event) => {
    const target = event.target.closest("[data-status]");
    if (!target) return;
    const what = target.dataset.status;
    if (what === "downloads") act.go({ kind: "downloads" });
    if (what === "units") {
      setBits(!usingBits());
      remember({ units: usingBits() ? "bits" : "bytes" });
      invalidate("status", "list", "inspector");
    }
    if (what === "rescan") act.rescan({ quiet: false });
    if (what === "stop-jobs") act.stopJobs();
    if (what === "stop-move" && state.moving && !state.moving.stopping) {
      state.moving.stopping = true;
      renderStatus();
      const url = state.moving.model_id !== undefined
        ? `/api/models/${state.moving.model_id}/move/stop`
        : `/api/tasks/${state.moving.id}/move/stop`;
      try { await post(url); } catch (error) { toastError(error); state.moving.stopping = false; }
    }
  });
  // "read 3 min ago" goes stale on its own.
  setInterval(() => invalidate("status"), 30000);
}

onRender("status", renderStatus);
