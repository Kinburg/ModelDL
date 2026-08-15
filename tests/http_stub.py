"""A local HTTP server that misbehaves on demand.

Models the shape of HuggingFace and Civitai: a canonical `/resolve` URL that 302s to a
`/cdn/...?sig=<signature>` target, where the signature can be rotated out from under an
in-flight download. On top of that it can be told to drop connections mid-body, ignore
Range headers, or serve a login page — the three ways real services corrupt downloads.
"""

from __future__ import annotations

import hashlib
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlsplit

_RANGE = re.compile(r"bytes=(\d+)-(\d*)")


class StubState:
    def __init__(self, data: bytes) -> None:
        self.data = data
        # The origin publishes the real content hash on its redirect...
        self.linked_etag = hashlib.sha256(data).hexdigest()
        # ...while the CDN serves an ETag that looks identical in shape but means something
        # else entirely. On Xet-backed HuggingFace repos this is the Xet content id. Any
        # code that mistakes it for a checksum will reject good downloads.
        self.cdn_etag = hashlib.sha256(b"xet-content-id-decoy").hexdigest()
        self.signature = "sig-0"
        self._sig_counter = 0

        # Knobs the tests turn.
        self.ignore_range = False
        # Honour small ranges but ignore anything bigger. Lets the resolve probe see a
        # well-behaved 206 while real chunk requests come back as a full-body 200 — the
        # shape of a misconfigured CDN edge.
        self.ignore_range_over: int | None = None
        self.head_supported = True
        # Reproduces HuggingFace's Xet bridge: the signed URL is minted for the byte range
        # of the request that triggered the redirect, and answers anything else with
        # "403 Auth failed: invalid range". A probe that sends a Range therefore poisons
        # the very URL it returns.
        self.bind_range = False
        self.serve_html = False
        self.fail_after: int | None = None   # bytes to send before dropping the connection
        self.fail_times = 0                  # how many responses get truncated that way
        self.rotate_every: int | None = None  # invalidate the signature every N data responses

        self.data_requests = 0
        self.resolve_requests = 0
        self.bytes_served = 0
        self.lock = threading.Lock()

    def rotate(self) -> None:
        self._sig_counter += 1
        self.signature = f"sig-{self._sig_counter}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: StubState  # injected on the server instance

    def log_message(self, *args) -> None:  # noqa: D102 - silence the test run
        pass

    @property
    def _state(self) -> StubState:
        return self.server.state  # type: ignore[attr-defined]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/resolve":
            self._do_resolve()
        elif parsed.path.startswith("/cdn/"):
            self._do_cdn(parsed.query)
        else:
            self._send_simple(404, b"not found")

    def do_HEAD(self) -> None:  # noqa: N802
        st = self._state
        if not st.head_supported:
            # Some origins genuinely refuse HEAD; the provider must fall back to a ranged
            # GET rather than giving up.
            self.send_response(405)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        parsed = urlsplit(self.path)
        if parsed.path == "/resolve":
            self._do_resolve()
            return
        if not parsed.path.startswith("/cdn/"):
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        with st.lock:
            if parse_qs(parsed.query).get("sig", [""])[0] != st.signature:
                self.send_response(403)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if st.serve_html:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            total, etag, ranges = len(st.data), st.cdn_etag, not st.ignore_range

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(total))
        self.send_header("ETag", f'"{etag}"')
        self.send_header("Content-Disposition", 'attachment; filename="model.safetensors"')
        if ranges:
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    # --- endpoints --------------------------------------------------------

    def _do_resolve(self) -> None:
        st = self._state
        with st.lock:
            st.resolve_requests += 1
            signature = st.signature
            bind_range = st.bind_range

        location = f"/cdn/model.safetensors?sig={signature}"
        if bind_range:
            location += f"&rng={quote(self.headers.get('Range') or 'any')}"
        self.send_response(302)
        self.send_header("Location", location)
        # Metadata that only exists on this hop, exactly as huggingface.co does it.
        self.send_header("X-Linked-Etag", f'"{st.linked_etag}"')
        self.send_header("X-Linked-Size", str(len(st.data)))
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _do_cdn(self, query: str) -> None:
        st = self._state
        params = parse_qs(query)
        supplied = params.get("sig", [""])[0]

        bound = params.get("rng", [None])[0]
        if bound is not None and unquote(bound) != "any":
            if (self.headers.get("Range") or "") != unquote(bound):
                self._send_simple(403, b"Auth failed: invalid range")
                return

        with st.lock:
            if supplied != st.signature:
                # Exactly what a CDN does with an expired presigned URL.
                self._send_simple(403, b"<Error><Code>AccessDenied</Code></Error>")
                return
            if st.serve_html:
                self._send_simple(200, b"<html><body>Please log in</body></html>", "text/html")
                return

            st.data_requests += 1
            if st.rotate_every and st.data_requests % st.rotate_every == 0:
                st.rotate()
            data, etag, ignore_range = st.data, st.cdn_etag, st.ignore_range
            ignore_over = st.ignore_range_over

        total = len(data)
        header = self.headers.get("Range")
        start, end, status = 0, total - 1, 200
        if header and not ignore_range:
            match = _RANGE.match(header)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else total - 1
                end = min(end, total - 1)
                if start > end:
                    self._send_simple(416, b"range not satisfiable")
                    return
                if ignore_over is not None and (end - start + 1) > ignore_over:
                    start, end, status = 0, total - 1, 200
                else:
                    status = 206

        body = data[start : end + 1]

        # Decide truncation only once the body is known, and only when it would actually
        # cut something short. Otherwise the one-byte probe that `resolve()` issues would
        # silently eat the budget without ever exercising a dropped connection.
        truncate = None
        with st.lock:
            if st.fail_after is not None and st.fail_times > 0 and len(body) > st.fail_after:
                truncate = st.fail_after
                st.fail_times -= 1

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", f'"{etag}"')
        self.send_header(
            "Content-Disposition", 'attachment; filename="model.safetensors"'
        )
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
        self.end_headers()

        if truncate is not None:
            self.wfile.write(body[:truncate])
            self.wfile.flush()
            self.close_connection = True
            with st.lock:
                st.bytes_served += min(truncate, len(body))
            return
        self.wfile.write(body)
        with st.lock:
            st.bytes_served += len(body)

    # --- helpers ----------------------------------------------------------

    def _send_simple(self, status: int, body: bytes, content_type: str = "text/plain") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class StubServer:
    def __init__(self, data: bytes) -> None:
        self.state = StubState(data)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.state = self.state  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> StubServer:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/resolve"
