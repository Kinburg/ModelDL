"""Looking through Civitai for what is new, the way a person keeps coming back to look.

Civitai's own search filters by everything that narrows what it returns — base models, types,
how recent, a text query — and by nothing that leaves something out. Those are done here,
on each page it sends: tags a model is marked with, versions that are sold, versions that can
only be used on the site, authors hidden, versions already on screen before. Each of them
thins a page, so a step of the search reads pages until it has enough to show or has read as
many as it is allowed to, and says how many it looked through to show what it shows.

Two of them cost a request of their own. Whether a version can be downloaded at all is not in
the search's answer — such a version lists its files and a download link like any other — but
it is in the answer to a lookup by hash, a hundred versions to a request. And whether a sold
version has been bought is known only to the download link, asked with the user's key, one
version at a time; it is asked only of the sold versions a filter would otherwise hide.

civitai.com and civitai.red answer the API identically; what differs is `nsfw=true`, which
the red one is asked with. Links and downloads keep to the domain the search was made on.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import re
import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import httpx

from ..core import polite
from ..library import versions
from ..library.classify import from_provider
from ..providers.civitai import CivitaiProvider, describe_previews
from ..settings import Settings
from .db import Database

SITES = ("civitai.com", "civitai.red")
SORTS = ("Newest", "Most Downloaded", "Highest Rated")
PERIODS = ("Day", "Week", "Month", "Year", "AllTime")
# Civitai's model types, as its API spells them.
TYPES = (
    "Checkpoint", "LORA", "LoCon", "DoRA", "TextualInversion", "Hypernetwork", "Controlnet",
    "Upscaler", "VAE", "MotionModule", "TextEncoder", "UNet", "CLIPVision", "Detection",
    "Poses", "Wildcards", "Workflows", "Other",
)
# Show every version whatever it costs; leave out the ones sold for good but keep early
# access, which is free from a date; or keep only what can be downloaded for nothing now.
PAID = ("all", "no_sold", "free")
SHOW = ("show", "hide")

# The base models a search can name, for when Civitai cannot be asked for its own list.
FALLBACK_BASE_MODELS = (
    "Krea 2", "Krea 2 Turbo", "MOSS Image", "Flux.1 D", "Flux.1 S", "Flux.1 Krea",
    "Flux.1 Kontext", "Flux.2 D", "Qwen", "Chroma", "Illustrious", "Pony", "NoobAI", "SDXL 1.0",
    "SD 1.5", "Wan Video 2.2 T2V-A14B", "Wan Video 2.2 I2V-A14B", "ZImageTurbo",
)

PAGE_SIZE = 100
# One step of a search — the first screenful, or the next one when the list is scrolled to its
# end — stops once it has this many to show, or once it has read this many pages for them.
WANTED = 40
PAGES_PER_STEP = 4
# The page keeps the list of each preset, and of the free search, until it is searched again:
# the search behind it is kept as long, one for each of them, so that coming back to a list
# left an hour ago still reads on and still shows its pictures. Past this many lists, the one
# looked at longest ago goes.
SESSIONS_KEPT = 64
# Cards no list on screen shows any more are let go past this many.
CARDS_KEPT = 5000
# How long a bought-or-not answer is believed. Buying happens on the site, and a card asked
# about before is asked again after this, not on every scroll.
OWNED_FOR = 600.0
ENUMS_FOR = 6 * 3600.0
# Civitai's batch lookup takes at most this many hashes.
HASH_BATCH = 100
DESCRIPTION_LIMIT = 6000

_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"<\s*(br|/p|/div|/li|/h\d)\s*/?\s*>", re.IGNORECASE)


# --- what a search asks -----------------------------------------------------------------


def _choice(value: Any, allowed: tuple[str, ...], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default


def _words(value: Any, limit: int, length: int = 80) -> tuple[str, ...]:
    """A list of short strings, trimmed, without repeats or blanks — whatever was sent."""
    if not isinstance(value, (list, tuple)):
        return ()
    kept: dict[str, None] = {}
    for item in value:
        if isinstance(item, str) and item.strip():
            kept.setdefault(item.strip()[:length], None)
        if len(kept) >= limit:
            break
    return tuple(kept)


@dataclass(frozen=True, slots=True)
class Filters:
    query: str = ""
    base_models: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    period: str = "Week"
    sort: str = "Newest"
    # SFW unless asked: the app has other users, and civitai.red is chosen, not fallen into.
    site: str = "civitai.com"
    # Tags a model must not carry; `*` stands for anything, so `character*` is every spelling.
    exclude_tags: tuple[str, ...] = ()
    paid: str = "all"
    seen: str = "show"
    hidden_authors: str = "hide"

    @classmethod
    def read(cls, data: Mapping[str, Any] | None) -> Filters:
        """Filters from what the page sent, or a preset kept: anything not understood is the
        default rather than a refusal — a preset saved by an older build still opens."""
        data = data if isinstance(data, Mapping) else {}
        query = data.get("query")
        return cls(
            query=query.strip()[:200] if isinstance(query, str) else "",
            base_models=_words(data.get("base_models"), 40),
            types=tuple(t for t in _words(data.get("types"), len(TYPES)) if t in TYPES),
            period=_choice(data.get("period"), PERIODS, "Week"),
            sort=_choice(data.get("sort"), SORTS, "Newest"),
            site=_choice(data.get("site"), SITES, "civitai.com"),
            exclude_tags=tuple(t.lower() for t in _words(data.get("exclude_tags"), 100)),
            paid=_choice(data.get("paid"), PAID, "all"),
            seen=_choice(data.get("seen"), SHOW, "show"),
            hidden_authors=_choice(data.get("hidden_authors"), SHOW, "hide"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "query": self.query, "base_models": list(self.base_models), "types": list(self.types),
            "period": self.period, "sort": self.sort, "site": self.site,
            "exclude_tags": list(self.exclude_tags), "paid": self.paid, "seen": self.seen,
            "hidden_authors": self.hidden_authors,
        }

    @property
    def nsfw(self) -> bool:
        return self.site == "civitai.red"

    def params(self) -> list[tuple[str, str]]:
        """What Civitai is asked: everything it can filter by itself."""
        params: list[tuple[str, str]] = [
            ("limit", str(PAGE_SIZE)), ("sort", self.sort), ("period", self.period),
        ]
        if self.query:
            params.append(("query", self.query))
        params += [("types", t) for t in self.types]
        params += [("baseModels", b) for b in self.base_models]
        if self.nsfw:
            params.append(("nsfw", "true"))
        return params


def tag_matcher(patterns: Iterable[str]):
    """Whether any of a model's tags is one of these — `*` for any run of characters, the
    rest as written, case not counting."""
    compiled = [
        re.compile(".*".join(re.escape(part) for part in pattern.split("*")), re.IGNORECASE)
        for pattern in patterns if pattern.strip("* ")
    ]

    def excluded(tags: Iterable[str]) -> str | None:
        for tag in tags:
            for pattern in compiled:
                if pattern.fullmatch(tag.strip()):
                    return tag
        return None

    return excluded


# --- what a search finds -----------------------------------------------------------------


@dataclass(slots=True)
class Card:
    """One version found, as the list shows it: a model, at the version the search is about —
    its newest for the base models asked for."""

    version_id: int
    model_id: int
    host: str
    name: str
    version_name: str
    type: str
    base_model: str
    published_at: str
    author: str
    tags: list[str]
    nsfw: bool
    poi: bool
    access: dict[str, Any] | None
    stats: dict[str, Any]
    trained_words: list[str]
    description: str
    files: list[dict[str, Any]]
    previews: list[dict[str, Any]]
    sha256: str | None
    usage: str | None = None
    owned: bool | None = None

    @property
    def page(self) -> str:
        return f"https://{self.host}/models/{self.model_id}?modelVersionId={self.version_id}"

    def to_json(self) -> dict[str, Any]:
        verdict = from_provider({"model_type": self.type})
        return {
            "version_id": self.version_id,
            "model_id": self.model_id,
            "host": self.host,
            "name": self.name,
            "version_name": self.version_name,
            "type": self.type,
            "category": verdict.category.value if verdict else "other",
            "base_model": self.base_model,
            "published_at": self.published_at,
            "author": self.author,
            "tags": self.tags,
            "nsfw": self.nsfw,
            "poi": self.poi,
            "access": {**self.access, "owned": self.owned} if self.access else None,
            "stats": self.stats,
            "trained_words": self.trained_words,
            "description": self.description,
            "files": self.files,
            "previews": len(self.previews),
            "page": self.page,
        }


def pick_version(item: Mapping[str, Any], base_models: Iterable[str]) -> dict[str, Any] | None:
    """The version a search shows of a model: its newest, of the base models asked for. The
    list it comes in is newest first."""
    wanted = set(base_models)
    for version in item.get("modelVersions") or []:
        if not isinstance(version, dict) or not isinstance(version.get("id"), int):
            continue
        if not wanted or version.get("baseModel") in wanted:
            return version
    return None


def describe(item: Mapping[str, Any], filters: Filters) -> Card | None:
    version = pick_version(item, filters.base_models)
    model_id = item.get("id")
    if version is None or not isinstance(model_id, int):
        return None
    files = [f for f in version.get("files") or [] if isinstance(f, dict)]
    primary = next((f for f in files if f.get("primary")), files[0] if files else None)
    digest = ((primary or {}).get("hashes") or {}).get("SHA256")
    previews = describe_previews(version)
    creator = item.get("creator") if isinstance(item.get("creator"), dict) else {}
    stats = item.get("stats") if isinstance(item.get("stats"), dict) else {}
    return Card(
        version_id=version["id"],
        model_id=model_id,
        host=filters.site,
        name=str(item.get("name") or ""),
        version_name=str(version.get("name") or ""),
        type=str(item.get("type") or ""),
        base_model=str(version.get("baseModel") or ""),
        published_at=str(version.get("publishedAt") or ""),
        author=str(creator.get("username") or ""),
        tags=[t for t in item.get("tags") or [] if isinstance(t, str)],
        # Covered until clicked when the model is marked adult or its first picture is.
        nsfw=bool(item.get("nsfw")) or bool(previews and previews[0].get("nsfw")),
        poi=bool(item.get("poi")),
        access=versions.access(version),
        stats={k: stats.get(k) for k in ("downloadCount", "thumbsUpCount", "commentCount")
               if isinstance(stats.get(k), (int, float))},
        trained_words=[w for w in version.get("trainedWords") or [] if isinstance(w, str)][:40],
        description=plain_text(item.get("description"))[:DESCRIPTION_LIMIT],
        files=[_file(f) for f in files],
        previews=previews,
        sha256=digest.upper() if isinstance(digest, str) else None,
    )


def _file(entry: Mapping[str, Any]) -> dict[str, Any]:
    metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
    size_kb = entry.get("sizeKB")
    return {
        "id": entry.get("id") if isinstance(entry.get("id"), int) else None,
        "name": str(entry.get("name") or ""),
        "size": int(size_kb * 1024) if isinstance(size_kb, (int, float)) else None,
        "type": entry.get("type"),
        "fp": metadata.get("fp"),
        "format": metadata.get("format"),
        "primary": bool(entry.get("primary")),
    }


def plain_text(markup: Any) -> str:
    """A description as text. Civitai's are HTML, and the page shows text: the paragraphs are
    kept, the markup is not."""
    if not isinstance(markup, str) or not markup:
        return ""
    text = _BREAK.sub("\n", markup)
    text = html.unescape(_TAG.sub("", text))
    lines = [line.strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def downloadable(usage: str | None) -> bool:
    """Whether a version can be downloaded at all, by what Civitai says of how it may be used.
    Not knowing is not a reason to hide one."""
    return usage in (None, "", "Download")


def paid_out(card: Card, mode: str) -> bool:
    """Whether the paid filter leaves a card out. A version bought is kept whatever the mode:
    it downloads like any other."""
    access = card.access
    if mode == "all" or access is None or card.owned is True:
        return False
    if mode == "no_sold":
        return bool(access.get("permanent"))
    return True


def needs_ownership(card: Card, mode: str) -> bool:
    """Whether the filter's answer turns on the card having been bought."""
    if mode == "all" or card.access is None:
        return False
    return mode == "free" or bool(card.access.get("permanent"))


@dataclass(slots=True)
class Session:
    """A search under way: what it asks, where Civitai's list is up to, what it has shown."""

    token: str
    filters: Filters
    preset_id: int | None
    cursor: str | None = None
    done: bool = False
    scanned: int = 0
    # Left out, by why — what the list's head says it looked through to show what it shows.
    left_out: dict[str, int] = field(default_factory=dict)
    shown: list[int] = field(default_factory=list)
    at: float = field(default_factory=time.time)

    def left(self, why: str, count: int = 1) -> None:
        self.left_out[why] = self.left_out.get(why, 0) + count

    @property
    def view(self) -> str:
        """The list on the page it is the search of: a preset's, or the free search's."""
        return f"p{self.preset_id}" if self.preset_id is not None else "free"


class Search:
    def __init__(self, settings: Settings, database: Database, versions_here) -> None:
        self.settings = settings
        self.db = database
        # Which Civitai versions the library has, by model — see `Library._versions_here`.
        self._versions_here = versions_here
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._cards: OrderedDict[int, Card] = OrderedDict()
        self._owned: dict[int, tuple[bool | None, float]] = {}
        self._usage: dict[int, str | None] = {}
        self._enums: tuple[float, dict[str, Any]] | None = None
        self._client: httpx.AsyncClient | None = None
        # One step at a time for each search: a second scroll to the bottom while the first
        # is still reading pages would read the same pages twice.
        self._locks: dict[str, asyncio.Lock] = {}

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(connect=15.0, read=45.0, write=15.0, pool=15.0),
                follow_redirects=True,
                headers={"User-Agent": "model-dl"},
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _auth(self) -> dict[str, str]:
        token = self.settings.effective_civitai_token
        return {"Authorization": f"Bearer {token}"} if token else {}

    # --- what the page needs to build a search -------------------------------------------

    async def options(self) -> dict[str, Any]:
        """The base models Civitai has now — its own list, asked once in a while — and the rest
        of what the filters offer."""
        now = time.time()
        if self._enums is None or now - self._enums[0] > ENUMS_FOR:
            try:
                response = await polite.get(self._http(), "https://civitai.com/api/v1/enums")
                response.raise_for_status()
                body = response.json()
                if not isinstance(body, dict):
                    raise ValueError("not an object")
                self._enums = (now, body)
            except (httpx.HTTPError, ValueError):
                if self._enums is None:
                    return self._options({}, fallback=True)
        return self._options(self._enums[1] if self._enums else {})

    @staticmethod
    def _options(enums: Mapping[str, Any], fallback: bool = False) -> dict[str, Any]:
        listed = enums.get("ActiveBaseModel") or enums.get("BaseModel")
        bases = [b for b in listed if isinstance(b, str)] if isinstance(listed, list) else []
        return {
            "base_models": bases or list(FALLBACK_BASE_MODELS),
            "base_models_fallback": fallback or not bases,
            "types": list(TYPES),
            "sorts": list(SORTS),
            "periods": list(PERIODS),
            "sites": list(SITES),
        }

    def state(self) -> dict[str, Any]:
        return {"presets": self.db.list_presets(), "hidden_authors": self.db.hidden_authors()}

    # --- searching -------------------------------------------------------------------------

    async def start(self, filters: Filters, preset_id: int | None = None) -> dict[str, Any]:
        """A new search, and its first screenful. It takes the place of the one its list was
        showing before."""
        token = secrets.token_urlsafe(12)
        session = Session(token, filters, preset_id)
        self._settle(session)
        return await self.more(token)

    def adopt(self, token: str, preset_id: int) -> dict[str, Any]:
        """Make a search the one of a preset — the search a new preset was saved from, which
        its list goes on showing rather than being searched for again."""
        session = self._sessions.get(token)
        if session is None:
            raise KeyError(token)
        del self._sessions[token]
        session.preset_id = preset_id
        self._settle(session)
        self._count_unseen(session)
        return {"presets": self.db.list_presets()}

    def forget(self, preset_id: int) -> None:
        """A preset deleted: its list is gone from the page, and its search goes with it."""
        for token, session in list(self._sessions.items()):
            if session.preset_id == preset_id:
                del self._sessions[token]
                self._locks.pop(token, None)
        self._prune_cards()

    def _settle(self, session: Session) -> None:
        for token, other in list(self._sessions.items()):
            if other.view == session.view:
                del self._sessions[token]
                self._locks.pop(token, None)
        self._sessions[session.token] = session
        while len(self._sessions) > SESSIONS_KEPT:
            old, _ = self._sessions.popitem(last=False)
            self._locks.pop(old, None)
        self._prune_cards()

    def _prune_cards(self) -> None:
        """Let go of the cards no kept search shows, the oldest first, once there are too many.
        A card a list still shows is kept however many there are: its pictures are asked for
        whenever that list is scrolled."""
        if len(self._cards) <= CARDS_KEPT:
            return
        shown = {v for session in self._sessions.values() for v in session.shown}
        for version_id in list(self._cards):
            if len(self._cards) <= CARDS_KEPT:
                break
            if version_id not in shown:
                del self._cards[version_id]

    async def more(self, token: str) -> dict[str, Any]:
        """The next screenful of a search, read from as many pages as it takes — within the
        limit of a step."""
        session = self._sessions.get(token)
        if session is None:
            raise KeyError(token)
        self._sessions.move_to_end(token)
        lock = self._locks.setdefault(token, asyncio.Lock())
        async with lock:
            found: list[Card] = []
            pages = 0
            while not session.done and pages < PAGES_PER_STEP and len(found) < WANTED:
                items = await self._page(session)
                pages += 1
                found += await self._keep(session, items)
            session.at = time.time()
            here = self._versions_here()
            seen = self.db.seen_versions(c.version_id for c in found)
            hidden = {a.lower() for a in self.db.hidden_authors()}
            cards = [self._card_json(c, here, seen, hidden) for c in found]
            self._count_unseen(session)
            return {
                "token": token,
                "cards": cards,
                "done": session.done,
                "scanned": session.scanned,
                "shown": len(session.shown),
                "left_out": session.left_out,
                "filters": session.filters.to_json(),
                "preset_id": session.preset_id,
                # With the count of the preset searched brought up to date.
                "presets": self.db.list_presets(),
            }

    async def _page(self, session: Session) -> list[dict[str, Any]]:
        params = session.filters.params()
        if session.cursor:
            params.append(("cursor", session.cursor))
        url = f"https://{session.filters.site}/api/v1/models"
        try:
            response = await polite.get(self._http(), url, params=params, headers=self._auth())
        except httpx.HTTPError as exc:
            raise SearchError(f"Civitai could not be reached: {exc}") from exc
        if response.status_code == 429:
            raise SearchError("Civitai asked for fewer requests, and kept asking — try again in a minute")
        if response.status_code >= 400:
            raise SearchError(f"Civitai refused the search ({response.status_code}): {_reason(response)}")
        try:
            body = response.json()
        except ValueError as exc:
            raise SearchError("Civitai answered with something that is not a list of models") from exc
        items = body.get("items") if isinstance(body, dict) else None
        metadata = body.get("metadata") if isinstance(body, dict) else None
        cursor = metadata.get("nextCursor") if isinstance(metadata, dict) else None
        session.cursor = str(cursor) if cursor not in (None, "") else None
        items = [i for i in items or [] if isinstance(i, dict)]
        session.scanned += len(items)
        # A page with nothing on it and a cursor anyway would otherwise be read for ever.
        if session.cursor is None or not items:
            session.done = True
        return items

    async def _keep(self, session: Session, items: list[dict[str, Any]]) -> list[Card]:
        """The cards of a page the filters leave in, cheapest filter first: what is in the
        answer itself, then what the database knows, then what takes a request."""
        filters = session.filters
        excluded = tag_matcher(filters.exclude_tags)
        hidden = {a.lower() for a in self.db.hidden_authors()} if filters.hidden_authors == "hide" else set()
        shown = set(session.shown)
        cards: list[Card] = []
        for item in items:
            card = describe(item, filters)
            if card is None:
                session.left("no version for the base models")
                continue
            if card.version_id in shown:
                continue
            if excluded(card.tags):
                session.left("tags")
                continue
            if card.author.lower() in hidden:
                session.left("hidden authors")
                continue
            cards.append(card)
            shown.add(card.version_id)

        if filters.seen == "hide" and cards:
            seen = self.db.seen_versions(c.version_id for c in cards)
            if seen:
                session.left("seen", sum(1 for c in cards if c.version_id in seen))
                cards = [c for c in cards if c.version_id not in seen]

        await self._learn_usage(cards, filters.site)
        usable = [c for c in cards if downloadable(c.usage)]
        if len(usable) < len(cards):
            session.left("only on the site", len(cards) - len(usable))
        cards = usable

        await self._learn_owned([c for c in cards if needs_ownership(c, filters.paid)], filters.site)
        kept = [c for c in cards if not paid_out(c, filters.paid)]
        if len(kept) < len(cards):
            session.left("paid", len(cards) - len(kept))

        for card in kept:
            session.shown.append(card.version_id)
            self._cards[card.version_id] = card
            self._cards.move_to_end(card.version_id)
        self._prune_cards()
        return kept

    async def _learn_usage(self, cards: list[Card], host: str) -> None:
        """How each version may be used, for the ones not asked about before: by hash, a
        hundred to a request; one at a time for a version whose file has none."""
        for card in cards:
            if card.version_id in self._usage:
                card.usage = self._usage[card.version_id]
        unknown = [c for c in cards if c.version_id not in self._usage]
        hashed = [c for c in unknown if c.sha256]
        for start in range(0, len(hashed), HASH_BATCH):
            chunk = hashed[start:start + HASH_BATCH]
            by_version = await self._usage_by_hash(host, [c.sha256 for c in chunk])   # type: ignore[misc]
            for card in chunk:
                card.usage = by_version.get(card.version_id)
                # An answer that left a version out is not an answer about it.
                if card.version_id in by_version:
                    self._usage[card.version_id] = card.usage
        for card in unknown:
            if card.sha256:
                continue
            with contextlib.suppress(httpx.HTTPError, ValueError):
                response = await polite.get(
                    self._http(), f"https://{host}/api/v1/model-versions/{card.version_id}",
                    headers=self._auth(),
                )
                if response.status_code == 200:
                    body = response.json()
                    card.usage = body.get("usageControl") if isinstance(body, dict) else None
                    self._usage[card.version_id] = card.usage

    async def _usage_by_hash(self, host: str, digests: list[str]) -> dict[int, str | None]:
        client = self._http()
        request = client.build_request(
            "POST", f"https://{host}/api/v1/model-versions/by-hash", json=digests, headers=self._auth()
        )
        try:
            response = await polite.send(client, request)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return {}
        found: dict[int, str | None] = {}
        for entry in body if isinstance(body, list) else []:
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                usage = entry.get("usageControl")
                found[entry["id"]] = usage if isinstance(usage, str) else None
        return found

    async def _learn_owned(self, cards: list[Card], host: str) -> None:
        """Whether the account of the key has bought each of these: one request for each, to
        its download link, and none at all without a key."""
        token = self.settings.effective_civitai_token
        if not cards or not token:
            return
        provider = CivitaiProvider(token, host)
        now = time.time()
        for card in cards:
            known = self._owned.get(card.version_id)
            if known is not None and now - known[1] < OWNED_FOR:
                card.owned = known[0]
                continue
            primary = next((f for f in card.files if f.get("primary")), None)
            card.owned = await provider.owned(card.version_id, (primary or {}).get("id"), self._http())
            self._owned[card.version_id] = (card.owned, now)

    def _card_json(self, card: Card, here: dict[int, set[int]], seen: set[int], hidden: set[str]) -> dict[str, Any]:
        mine = here.get(card.model_id) or set()
        return {
            **card.to_json(),
            "here": "this" if card.version_id in mine else "other" if mine else None,
            "seen": card.version_id in seen,
            "author_hidden": card.author.lower() in hidden,
        }

    def _count_unseen(self, session: Session) -> None:
        """Keep the count the sidebar shows beside a preset: what its last look found that has
        not been on screen, of what it shows."""
        if session.preset_id is None:
            return
        seen = self.db.seen_versions(session.shown)
        hidden = {a.lower() for a in self.db.hidden_authors()}
        unseen = sum(
            1 for version_id in session.shown
            if version_id not in seen
            and not (session.filters.hidden_authors == "hide"
                     and (card := self._cards.get(version_id)) is not None
                     and card.author.lower() in hidden)
        )
        self.db.update_preset(session.preset_id, unseen={
            "count": unseen, "more": not session.done, "at": time.time(),
        })

    # --- after a search: what the page does with what it found ----------------------------

    def card(self, version_id: int) -> Card:
        card = self._cards.get(version_id)
        if card is None:
            raise KeyError(version_id)
        return card

    def mark_seen(self, version_ids: list[int], seen: bool = True, token: str | None = None) -> dict[str, Any]:
        pairs = [(v, self._cards[v].model_id if v in self._cards else None) for v in version_ids]
        changed = self.db.mark_seen(pairs, seen)
        session = self._sessions.get(token or "")
        if session is not None:
            self._count_unseen(session)
        return {"changed": changed, "presets": self.db.list_presets()}

    def hide_author(self, username: str, hidden: bool = True) -> dict[str, Any]:
        changed = self.db.set_author_hidden(username, hidden)
        for session in self._sessions.values():
            if session.preset_id is not None:
                self._count_unseen(session)
        return {"changed": changed, "hidden_authors": self.db.hidden_authors(),
                "presets": self.db.list_presets()}


class SearchError(RuntimeError):
    """Civitai could not answer a search — said as it would be to the person who asked."""


def _reason(response: httpx.Response) -> str:
    with contextlib.suppress(ValueError):
        body = response.json()
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                error = error.get("message") or error.get("name")
            if error:
                return str(error)[:300]
    return response.reason_phrase or "no reason given"
