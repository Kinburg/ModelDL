// Small things every part of the page needs: formatting, escaping, the clipboard.

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

export function fmtBytes(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n, i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v < 10 && i > 0 ? 1 : 0)} ${units[i]}`;
}

// Task Manager shows bits per second; downloaders traditionally show bytes. Comparing the
// two without noticing costs an evening of thinking the download is eight times slower
// than it is, so both are one click apart.
let bits = false;
export const setBits = (value) => { bits = !!value; };
export const usingBits = () => bits;

export function fmtSpeed(bytesPerSecond) {
  if (!bytesPerSecond) return "0";
  if (!bits) return `${fmtBytes(bytesPerSecond)}/s`;
  let v = bytesPerSecond * 8, i = 0;
  const units = ["bit", "Kbit", "Mbit", "Gbit"];
  while (v >= 1000 && i < units.length - 1) { v /= 1000; i++; }
  return `${v.toFixed(v < 10 ? 1 : 0)} ${units[i]}/s`;
}

export function fmtEta(s) {
  if (!s || !isFinite(s) || s > 86400 * 3) return "—";
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}:${String(sec).padStart(2, "0")}`;
}

export function fmtDuration(s) {
  if (!s || !isFinite(s)) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60), sec = Math.round(s % 60);
  return m >= 60 ? `${Math.floor(m / 60)}h ${m % 60}m` : `${m}m ${sec}s`;
}

// Dates in the language of the rest of the page. "Yesterday" beside a weekday in whatever
// language the machine is set to reads as two different apps.
const LOCALE = "en-GB";

export const fmtWhen = (epoch) => epoch
  ? new Date(epoch * 1000).toLocaleString(LOCALE, { dateStyle: "medium", timeStyle: "short" })
  : "—";

export const fmtTime = (epoch) => epoch
  ? new Date(epoch * 1000).toLocaleTimeString(LOCALE, { hour: "2-digit", minute: "2-digit" })
  : "";

export const fmtDate = (epoch) => epoch
  ? new Date(epoch * 1000).toLocaleDateString(LOCALE, { day: "numeric", month: "short", year: "numeric" })
  : "—";

// "3 days ago" reads faster than a date for anything recent, and worse than one for
// anything that is not.
export function fmtAgo(epoch) {
  if (!epoch) return "—";
  const seconds = Date.now() / 1000 - epoch;
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 86400 * 14) return `${Math.floor(seconds / 86400)} days ago`;
  return fmtDate(epoch);
}

// The history groups by day, and a day is best named the way a person would: today,
// yesterday, the weekday for this week, the date after that.
export function dayLabel(epoch) {
  const date = new Date(epoch * 1000);
  const today = new Date();
  const start = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const days = Math.round((start(today) - start(date)) / 86400000);
  if (days <= 0) return "Today";
  if (days === 1) return "Yesterday";
  if (days < 7) return date.toLocaleDateString(LOCALE, { weekday: "long" });
  return date.toLocaleDateString(LOCALE, {
    day: "numeric", month: "long", year: date.getFullYear() === today.getFullYear() ? undefined : "numeric",
  });
}

export const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;

// --- what a Civitai version costs -----------------------------------------------------

// Whether it has to be bought to be downloaded, now: sold for good, or in early access that
// has not ended. The day early access ends it is free, whenever it was last asked about.
export function paidNow(access) {
  if (!access) return false;
  if (access.permanent) return true;
  const until = Date.parse(access.until || "");
  return Number.isFinite(until) ? until > Date.now() : true;
}

// Paid · bought, Early access · free from 1 Oct 2026 · not bought — or "" for free.
export function accessLabel(access) {
  if (!paidNow(access)) return "";
  const until = Date.parse(access.until || "");
  const what = access.permanent ? "Paid"
    : Number.isFinite(until) ? `Early access · free from ${fmtDate(until / 1000)}` : "Early access";
  return what + (access.owned === true ? " · bought" : access.owned === false ? " · not bought" : "");
}

// Whether downloading it would be refused: to be bought, and not known to be.
export const refused = (access) => paidNow(access) && access.owned !== true;

// The same, as a sentence, for wherever there is room to say it before Download is pressed.
export function paidSentence(access) {
  if (!paidNow(access)) return "";
  const until = Date.parse(access.until || "");
  const what = access.permanent ? "Sold on Civitai"
    : `In early access on Civitai${Number.isFinite(until) ? ` — free from ${fmtDate(until / 1000)}` : ""}`;
  const owned = access.owned === true ? "bought with your API key, so it downloads like any other."
    : access.owned === false ? "not bought with your API key: Civitai refuses the download until it is bought on its page."
      : "and whether it is bought can be told only with a Civitai API key, in Settings under Accounts.";
  return `${what}, ${owned}`;
}

// The chip that says so, coloured by whether it is bought.
export function accessChip(access) {
  const label = accessLabel(access);
  if (!label) return "";
  const why = access.owned === true ? "Bought by the account of your Civitai API key"
    : access.owned === false ? "Not bought by the account of your Civitai API key: Civitai refuses the download until it is"
      : "Whether it is bought could not be told: that takes a Civitai API key, in Settings under Accounts";
  return `<span class="chip paid ${access.owned === true ? "owned" : ""}" title="${esc(why)}">${esc(label)}</span>`;
}

export function debounce(fn, wait) {
  let timer = null;
  const wrapped = (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => { timer = null; fn(...args); }, wait);
  };
  wrapped.flush = (...args) => { if (timer) { clearTimeout(timer); timer = null; fn(...args); } };
  return wrapped;
}

// What each kind is called on screen. The internal names are stable identifiers, not
// labels anybody should have to read.
const KINDS = {
  checkpoint: "Checkpoint", diffusion_model: "Diffusion model", lora: "LoRA", vae: "VAE",
  text_encoder: "Text encoder", clip_vision: "CLIP vision", controlnet: "ControlNet",
  embedding: "Embedding", upscaler: "Upscaler", ipadapter: "IP-Adapter",
  style_model: "Style model", hypernetwork: "Hypernetwork", motion_module: "Motion module",
  llm: "LLM", detection: "Detector", other: "Other",
};
export const kindLabel = (kind) => KINDS[kind] || (kind ? kind : "Unknown");

export const stemOf = (name) => {
  const dot = String(name || "").lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
};
export const suffixOf = (name) => {
  const dot = String(name || "").lastIndexOf(".");
  return dot > 0 ? name.slice(dot) : "";
};

export const folderOf = (path) => String(path || "").replace(/[\\/][^\\/]*$/, "");
export const baseName = (path) => String(path || "").split(/[\\/]/).pop();

// Putting something on the clipboard is allowed straight off a click and asks nobody, but a
// webview told otherwise makes it throw — and a copy button that silently does nothing is
// worse than one that says so. The textarea is how this was done before there was an API
// for it, and it still works where the API is refused.
export async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch { /* refused or absent — the old way below */ }
  const holder = document.createElement("textarea");
  holder.value = text;
  holder.setAttribute("readonly", "");
  holder.style.cssText = "position:fixed;top:0;left:0;opacity:0";
  document.body.appendChild(holder);
  holder.select();
  let copied = false;
  try { copied = document.execCommand("copy"); } catch { copied = false; }
  holder.remove();
  return copied;
}

export const $ = (id) => document.getElementById(id);

export function el(html) {
  const holder = document.createElement("template");
  holder.innerHTML = html.trim();
  return holder.content.firstElementChild;
}
