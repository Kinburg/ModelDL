// Line icons, drawn here rather than fetched: the app runs offline and from inside a bundle,
// and a font or a sprite sheet is one more file that can go missing. Each is a 24x24 path
// stroked in the current text colour, so it takes the colour of whatever it sits in.

const PATHS = {
  folder: "M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z",
  "folder-open": "M3 17.5V7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h6.5A2.5 2.5 0 0 1 20 9.5V10 M3 17.5 5.6 11.6A2 2 0 0 1 7.4 10.5H21l-2.8 7.6a2.5 2.5 0 0 1-2.3 1.9H5.5A2.5 2.5 0 0 1 3 17.5z",
  "chevron-right": "M9.5 6l6 6-6 6",
  "chevron-down": "M6 9.5l6 6 6-6",
  download: "M12 4v11 M7 10.5l5 5 5-5 M5 20h14",
  history: "M12 8v4.5l3 1.8 M3.6 13A8.5 8.5 0 1 0 5.2 7 M3.5 3.5V8h4.5",
  alert: "M12 9.5v4 M12 17h.01 M10.3 4.2 2.7 17.5A2 2 0 0 0 4.4 20.5h15.2a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z",
  help: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M9.6 9.2a2.5 2.5 0 1 1 3.4 2.3c-.7.3-1 .9-1 1.6v.4 M12 17h.01",
  copy: "M9 9h10v11H9z M15 9V5.5A1.5 1.5 0 0 0 13.5 4h-8A1.5 1.5 0 0 0 4 5.5v8A1.5 1.5 0 0 0 5.5 15H9",
  trash: "M4 7h16 M10 11v6 M14 11v6 M6 7l1 12.5A1.5 1.5 0 0 0 8.5 21h7a1.5 1.5 0 0 0 1.5-1.5L18 7 M9 7V4.5A.5.5 0 0 1 9.5 4h5a.5.5 0 0 1 .5.5V7",
  edit: "M4 20h4L19.5 8.5a2.1 2.1 0 0 0-3-3L5 17v3z M14.5 7.5l2 2",
  move: "M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v8a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 17.5z M9 13.5h6 M12.5 11l2.5 2.5-2.5 2.5",
  external: "M14 4h6v6 M20 4l-9 9 M18 14v4.5a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 4 18.5v-11A1.5 1.5 0 0 1 5.5 6H10",
  search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14z M20 20l-4-4",
  settings: "M4 6h9 M17 6h3 M4 12h3 M11 12h9 M4 18h11 M19 18h1 M15 8a2 2 0 1 0 0-4 2 2 0 0 0 0 4z M9 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4z M17 20a2 2 0 1 0 0-4 2 2 0 0 0 0 4z",
  x: "M6 6l12 12 M18 6L6 18",
  dots: "M5 12h.01 M12 12h.01 M19 12h.01",
  refresh: "M19.5 12a7.5 7.5 0 1 1-2.2-5.3 M19.5 4.5V9H15",
  plus: "M12 5v14 M5 12h14",
  hash: "M5 9h14 M5 15h14 M10.5 4 8.5 20 M15.5 4l-2 16",
  check: "M5 12.5l4.5 4.5L19 7.5",
  pause: "M8.5 5v14 M15.5 5v14",
  play: "M7 4.5v15l12.5-7.5z",
  retry: "M4.5 12a7.5 7.5 0 1 0 2.2-5.3 M4.5 4.5V9H9",
  file: "M6.5 3h7l4 4v12.5a1.5 1.5 0 0 1-1.5 1.5H6.5A1.5 1.5 0 0 1 5 19.5v-15A1.5 1.5 0 0 1 6.5 3z M13.5 3v4h4",
  "file-off": "M6.5 3h7l4 4v8 M17.5 19.5a1.5 1.5 0 0 1-1.5 1.5H6.5A1.5 1.5 0 0 1 5 19.5V5 M3 3l18 18",
  image: "M5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5v-13A1.5 1.5 0 0 1 5.5 4z M9 11a2 2 0 1 0 0-4 2 2 0 0 0 0 4z M4 17l5-5 4 4 2.5-2.5L20 18",
  link: "M10 14a4.2 4.2 0 0 0 6 0l3-3a4.2 4.2 0 0 0-6-6l-1 1 M14 10a4.2 4.2 0 0 0-6 0l-3 3a4.2 4.2 0 0 0 6 6l1-1",
  drive: "M4 13.5h16v5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5z M4 13.5 6.6 6A1.5 1.5 0 0 1 8 5h8a1.5 1.5 0 0 1 1.4 1l2.6 7.5 M8 17h.01 M11.5 17h.01",
  layers: "M12 4l8.5 4.3L12 12.6 3.5 8.3z M3.5 12.2l8.5 4.3 8.5-4.3 M3.5 16l8.5 4.3 8.5-4.3",
  broom: "M19.5 3.5 12 11 M9.5 10.5l4 4 M9.5 10.5c-3.2.4-5.5 3.4-6 8.5 0 0 4.7.9 8.6-2.4a6.5 6.5 0 0 0 1.4-2.1",
  note: "M5.5 4h13A1.5 1.5 0 0 1 20 5.5V15l-5 5H5.5A1.5 1.5 0 0 1 4 18.5v-13A1.5 1.5 0 0 1 5.5 4z M14.5 20v-4.5a.5.5 0 0 1 .5-.5h5 M8 9h8 M8 12.5h5",
  tag: "M3.5 12V5a1.5 1.5 0 0 1 1.5-1.5h7L21 12.5 12.5 21z M8 8.5h.01",
  update: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 16V8.5 M8.5 11.5 12 8l3.5 3.5",
  stop: "M7.5 6h9A1.5 1.5 0 0 1 18 7.5v9a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 6 16.5v-9A1.5 1.5 0 0 1 7.5 6z",
  back: "M9.5 14 4.5 9l5-5 M4.5 9h11a4.5 4.5 0 0 1 0 9H12",
  globe: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M3.5 12h17 M12 3c2.6 2.6 2.6 15.4 0 18 M12 3c-2.6 2.6-2.6 15.4 0 18",
  eye: "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
  "eye-off": "M3 3l18 18 M10.6 5.6c.5-.1.9-.1 1.4-.1 6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-2.4 3.2 M6.6 6.6A16 16 0 0 0 2.5 12S6 18.5 12 18.5c1.6 0 3-.4 4.2-1 M10 10a3 3 0 0 0 4.2 4.2",
  cube: "M12 3.5 20 8v8l-8 4.5L4 16V8z M4 8l8 4.5L20 8 M12 12.5v8",
  grip: "M9 6h.01 M15 6h.01 M9 12h.01 M15 12h.01 M9 18h.01 M15 18h.01",
  "arrow-up": "M12 19V5 M6 11l6-6 6 6",
  "arrow-down": "M12 5v14 M6 13l6 6 6-6",
  star: "M12 3.8l2.5 5.2 5.7.8-4.1 4 1 5.6-5.1-2.7-5.1 2.7 1-5.6-4.1-4 5.7-.8z",
  question: "M9 9.2a3 3 0 1 1 4.2 2.8c-.8.4-1.2 1-1.2 1.9v.6 M12 18h.01",
  // Three nodes and their links: a ComfyUI graph.
  workflow: "M3.5 4.5h5v4h-5z M15.5 4.5h5v4h-5z M9.5 15.5h5v4h-5z M8.5 6.5h7 M6 8.5v6.5a2.5 2.5 0 0 0 2.5 2.5h1 M18 8.5v6.5a2.5 2.5 0 0 1-2.5 2.5h-1",
  save: "M5.5 4h10.5l4 4v10.5a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5v-13A1.5 1.5 0 0 1 5.5 4z M8 4v4.5h7V4 M8 20v-6h8v6",
};

export function icon(name, cls = "") {
  const path = PATHS[name] || PATHS.file;
  return `<svg class="i ${cls}" viewBox="0 0 24 24" aria-hidden="true" focusable="false">`
    + `<path d="${path}"/></svg>`;
}

// A thing's kind as a small picture, for rows with no sample image of their own.
const KIND_ICONS = {
  lora: "tag", checkpoint: "cube", diffusion_model: "cube", vae: "layers",
  text_encoder: "note", clip_vision: "eye", controlnet: "link", embedding: "tag",
  upscaler: "image", llm: "note", detection: "search", other: "file",
};
export const kindIcon = (kind) => KIND_ICONS[kind] || "file";
