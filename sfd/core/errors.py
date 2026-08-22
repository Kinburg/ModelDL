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
    """The byte range we asked for can never be served, and no retry changes that.

    A 416 means our own arithmetic is out of step with the file; a redirect where a
    resolved URL was promised means the provider handed us something we must not follow.
    Both need a person, not another attempt. A server that merely *ignored* the range on
    one response is a different matter — see RangeIgnored.
    """


class ChecksumMismatch(Terminal):
    """The completed file does not match the hash the service advertised."""


class NotEnoughSpace(Terminal):
    """The volume cannot hold what is left of this file.

    Worth its own check because preallocation is sparse: nothing is reserved up front, so a
    full disk would otherwise surface as an OSError from a chunk writer hours into a
    download, with the bytes already fetched thrown away on the retry.
    """


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


class RangeIgnored(Retryable):
    """The response came back without the byte range we asked for.

    Writing it would corrupt the file — appending a full copy onto a partial one, or
    overwriting verified bytes — so the response is dropped unread. But dropping it is the
    whole of the danger: the check runs before the first byte is written, so the partial on
    disk is exactly as it was and another attempt costs nothing but a round trip.

    Worth retrying because this is nearly always one CDN edge misbehaving rather than the
    file losing range support. The connection that comes back lands on a different node,
    and the fresh signed URL may point somewhere else entirely. A server that has genuinely
    stopped serving ranges simply runs the attempt budget out and fails with the reason
    intact.
    """


class ResolveError(Retryable):
    """Could not turn a canonical reference into a fresh download URL."""
