"""Error taxonomy.

The split that matters is retryable vs terminal. A retryable error means "drop this
connection, get a fresh signed URL, continue from where we stopped" — the bytes already
on disk stay valid. A terminal error means the download cannot proceed without the user
doing something (supplying a token, accepting a licence).
"""

from __future__ import annotations


class SfdError(Exception):
    """Base for everything this package raises."""


class Retryable(SfdError):
    """Transient. Re-resolve the URL and resume from the current offset."""


class Terminal(SfdError):
    """Not fixable by retrying. Surface to the user."""


# --- Terminal ---------------------------------------------------------------


class AuthRequired(Terminal):
    """The service wants a token we do not have (401, or an anonymous redirect to a login page)."""


class AccessDenied(Terminal):
    """Token is present but insufficient — gated repo, unaccepted licence, early access."""


class NotBinaryContent(Terminal):
    """Got HTML where a model was expected.

    Civitai serves its login page with 200 text/html when the request is unauthenticated,
    so without this check you end up with a 12 KB web page named model.safetensors.
    """


class RangeNotHonored(Terminal):
    """Asked to resume from a non-zero offset and the server sent the whole file instead.

    Writing this response would corrupt the file — either by appending a full copy onto a
    partial one, or by silently overwriting good bytes. We refuse to write anything.
    """


class ChecksumMismatch(Terminal):
    """The completed file does not match the hash the service advertised."""


class TransferFailed(Terminal):
    """A chunk exhausted its retry budget. Carries the last underlying cause."""


class RemoteChanged(Terminal):
    """ETag or size changed since the partial download started; resuming is unsafe."""


# --- Retryable --------------------------------------------------------------


class SignatureExpired(Retryable):
    """The CDN rejected the signed URL (403/410). Re-resolve from the canonical URL."""


class Stalled(Retryable):
    """Throughput stayed below the floor for longer than the watchdog window.

    Data was still trickling in, so no socket timeout would ever have fired. This is the
    failure mode where a download appears alive for hours while moving nothing.
    """


class TransportError(Retryable):
    """Connection reset, timeout, DNS blip, 5xx."""


class ResolveError(Retryable):
    """Could not turn a canonical reference into a fresh download URL."""
