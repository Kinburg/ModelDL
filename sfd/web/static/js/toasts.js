// Messages that come and go in the corner. An error stays until it is dismissed: a failure
// that vanished on its own while nobody was looking is a failure nobody saw.

import { esc } from "./util.js";
import { icon } from "./icons.js";

const LIMIT = 5;
let holder = null;

function container() {
  if (!holder) {
    holder = document.createElement("div");
    holder.id = "toasts";
    holder.setAttribute("role", "status");
    holder.setAttribute("aria-live", "polite");
    document.body.appendChild(holder);
  }
  return holder;
}

// `actions` are `{label, run}`; running one closes the message.
export function toast(message, { level = "info", actions = [], timeout } = {}) {
  const box = document.createElement("div");
  // Prefixed: a bare `ok` or `warn` is also the page's colour for text, and the whole
  // message would take it.
  box.className = `toast toast-${level}`;
  box.innerHTML = `
    <span class="toast-icon">${icon(level === "error" ? "alert" : level === "ok" ? "check" : "help")}</span>
    <span class="toast-text">${esc(message)}</span>
    <span class="toast-actions"></span>
    <button class="icon-button toast-close" aria-label="Dismiss">${icon("x")}</button>`;
  const row = box.querySelector(".toast-actions");
  for (const action of actions) {
    const button = document.createElement("button");
    button.className = "link-button";
    button.textContent = action.label;
    button.onclick = async () => { close(); await action.run(); };
    row.appendChild(button);
  }
  const close = () => {
    box.classList.add("leaving");
    setTimeout(() => box.remove(), 180);
  };
  box.querySelector(".toast-close").onclick = close;

  const list = container();
  list.appendChild(box);
  while (list.children.length > LIMIT) list.firstElementChild.remove();

  const wait = timeout ?? (level === "error" ? 0 : actions.length ? 9000 : 4500);
  if (wait) {
    let timer = setTimeout(close, wait);
    box.addEventListener("mouseenter", () => clearTimeout(timer));
    box.addEventListener("mouseleave", () => { timer = setTimeout(close, 2500); });
  }
  return close;
}

export const toastError = (error) => toast(error?.message || String(error), { level: "error" });
