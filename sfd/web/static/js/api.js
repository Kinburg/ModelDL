// Talking to the server. Every request is JSON both ways; a refusal comes back as an Error
// whose message is the server's own explanation, because that is what a person can act on.

export async function api(path, options = {}) {
  const init = { headers: { "Content-Type": "application/json" }, ...options };
  if (init.body !== undefined && typeof init.body !== "string") init.body = JSON.stringify(init.body);
  const response = await fetch(path, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(explain(body.detail) || response.statusText || `error ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

export const get = (path) => api(path);
export const post = (path, body = {}) => api(path, { method: "POST", body });
export const put = (path, body = {}) => api(path, { method: "PUT", body });
export const del = (path) => api(path, { method: "DELETE" });

// A rejected setting comes back as FastAPI's list of field errors. Rendered raw that reads
// as "[object Object]", which says nothing about the field that was actually wrong.
function explain(detail) {
  if (!Array.isArray(detail)) return detail;
  return detail.map((e) => `${(e.loc || []).slice(-1)[0] || "request"}: ${e.msg}`).join("; ");
}
