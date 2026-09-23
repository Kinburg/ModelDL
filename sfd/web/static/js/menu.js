// The right-click menu. The page draws its own: the window's built-in one belongs to a
// browser — reload, save as, inspect — and this window is not a browser. Over a text field
// the built-in editing menu is still what a right-click brings up; see `wireContextMenus`.

import { esc } from "./util.js";
import { icon } from "./icons.js";

let open = null;

// `items` are `{label, icon, run, danger, disabled, hint}` or `"-"` for a separator.
export function showMenu(x, y, items) {
  closeMenu();
  const visible = items.filter(Boolean);
  if (!visible.length) return;
  const menu = document.createElement("div");
  menu.className = "menu";
  menu.setAttribute("role", "menu");
  let separatorPending = false;
  for (const item of visible) {
    if (item === "-") { separatorPending = menu.children.length > 0; continue; }
    if (separatorPending) {
      const line = document.createElement("div");
      line.className = "menu-separator";
      menu.appendChild(line);
      separatorPending = false;
    }
    const button = document.createElement("button");
    button.className = `menu-item${item.danger ? " danger" : ""}`;
    button.setAttribute("role", "menuitem");
    button.disabled = !!item.disabled;
    button.innerHTML = `${icon(item.icon || "dots", "menu-icon")}
      <span class="menu-label">${esc(item.label)}</span>
      ${item.hint ? `<span class="menu-hint">${esc(item.hint)}</span>` : ""}`;
    button.onclick = (event) => {
      event.stopPropagation();
      closeMenu();
      item.run && item.run();
    };
    menu.appendChild(button);
  }
  document.body.appendChild(menu);
  const box = menu.getBoundingClientRect();
  const left = Math.min(x, window.innerWidth - box.width - 6);
  const top = y + box.height > window.innerHeight - 6 ? Math.max(6, y - box.height) : y;
  menu.style.left = `${Math.max(6, left)}px`;
  menu.style.top = `${top}px`;
  open = menu;
  const first = menu.querySelector(".menu-item:not(:disabled)");
  if (first) first.focus({ preventScroll: true });
}

export function closeMenu() {
  if (open) { open.remove(); open = null; }
}

export const menuOpen = () => !!open;

// A menu opened by a button rather than a right-click: placed under the button.
export function menuFrom(button, items) {
  const box = button.getBoundingClientRect();
  showMenu(box.left, box.bottom + 4, items);
}

document.addEventListener("mousedown", (event) => {
  if (open && !open.contains(event.target)) closeMenu();
}, true);
window.addEventListener("blur", closeMenu);
window.addEventListener("resize", closeMenu);
// Closed by the person scrolling, not by a scroll event: a list redrawn under an open menu —
// a download moving on a percent — puts its scroll position back, which is a scroll event
// too, and a menu that vanished on every one of those could hardly be used.
document.addEventListener("wheel", (event) => {
  if (open && !open.contains(event.target)) closeMenu();
}, { capture: true, passive: true });
document.addEventListener("keydown", (event) => {
  if (!open) return;
  const items = [...open.querySelectorAll(".menu-item:not(:disabled)")];
  const index = items.indexOf(document.activeElement);
  // The menu has the keyboard while it is open: the same keys also move the list's
  // selection, and a press meant for one must not do both.
  if (event.key === "ArrowDown") { event.preventDefault(); event.stopPropagation(); (items[index + 1] || items[0])?.focus(); }
  else if (event.key === "ArrowUp") { event.preventDefault(); event.stopPropagation(); (items[index - 1] || items[items.length - 1])?.focus(); }
  else if (event.key === "Escape") { event.preventDefault(); event.stopPropagation(); closeMenu(); }
  else if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    event.stopPropagation();
    if (items.includes(document.activeElement)) document.activeElement.click();
  }
}, true);
