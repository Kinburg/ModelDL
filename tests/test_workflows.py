"""The ComfyUI workflow inside a sample picture: finding it, reading little, keeping it.

What is pinned down here is what a person would notice going wrong: a workflow that is in
the picture and not found, a found one that ComfyUI's paste handler then quietly ignores, a
full-size picture downloaded to answer a question its first few kilobytes answer, and a save
that overwrites something.
"""

from __future__ import annotations

import asyncio
import json
import struct
import zlib
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.library import previews, workflows
from sfd.library.workflows import API, GRAPH
from sfd.settings import Settings
from sfd.web.app import create_app

GRAPH_JSON = {
    "last_node_id": 2,
    "nodes": [{"id": 1, "type": "KSampler"}, {"id": 2, "type": "SaveImage"}],
    "links": [],
    "extra": {"ds": {"scale": 1}},
    "version": 0.4,
}
PROMPT_JSON = {
    "1": {"class_type": "KSampler", "inputs": {"seed": 1}},
    "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
}
GRAPH_TEXT = json.dumps(GRAPH_JSON)
PROMPT_TEXT = json.dumps(PROMPT_JSON)
IMAGE_URL = "https://image.civitai.com/bucket/one/original=true/143524783.jpeg"


# --- building pictures ----------------------------------------------------------------


def chunk(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))


IHDR = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
IDAT = chunk(b"IDAT", zlib.compress(b"\0\0\0\0"))
IEND = chunk(b"IEND", b"")


def text(key: str, value: str) -> bytes:
    return chunk(b"tEXt", key.encode("latin-1") + b"\0" + value.encode("latin-1"))


def itext(key: str, value: str, compress: bool = False) -> bytes:
    raw = value.encode("utf-8")
    body = key.encode() + b"\0" + bytes([int(compress), 0]) + b"en\0" + b"\0"
    return chunk(b"iTXt", body + (zlib.compress(raw) if compress else raw))


def ztext(key: str, value: str) -> bytes:
    return chunk(b"zTXt", key.encode() + b"\0\0" + zlib.compress(value.encode("latin-1")))


def png(*chunks: bytes) -> bytes:
    return workflows.PNG_SIGNATURE + IHDR + b"".join(chunks)


def tiff(values: dict[int, str], order: str = "<") -> bytes:
    """A TIFF block with ASCII values in its first directory, as ComfyUI writes EXIF."""
    count = len(values)
    data_at = 8 + 2 + 12 * count + 4
    ifd, blobs = struct.pack(order + "H", count), b""
    for tag, value in sorted(values.items()):
        raw = value.encode("utf-8") + b"\0"
        if len(raw) <= 4:
            field = raw.ljust(4, b"\0")
        else:
            field = struct.pack(order + "I", data_at + len(blobs))
            blobs += raw
        ifd += struct.pack(order + "HHI", tag, 2, len(raw)) + field
    head = (b"II" if order == "<" else b"MM") + struct.pack(order + "HI", 42, 8)
    return head + ifd + struct.pack(order + "I", 0) + blobs


def jpeg(exif: bytes | None) -> bytes:
    out = b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    if exif is not None:
        payload = b"Exif\0\0" + exif
        out += b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    return out + b"\xff\xda" + struct.pack(">H", 8) + b"\x01\x01\x00\x00\x3f\x00" + b"\x12\x34" * 64 + b"\xff\xd9"


def webp(exif: bytes | None, *, header: bool = False) -> bytes:
    flags = 0x08 if exif is not None else 0
    body = b"VP8X" + struct.pack("<I", 10) + bytes([flags]) + b"\0" * 9
    image = b"\x2f" + b"\0" * 32
    body += b"VP8L" + struct.pack("<I", len(image)) + image + b"\0"
    if exif is not None:
        payload = (b"Exif\0\0" if header else b"") + exif
        body += b"EXIF" + struct.pack("<I", len(payload)) + payload + (b"\0" if len(payload) % 2 else b"")
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WEBP" + body


COMFY_PNG = png(text("prompt", PROMPT_TEXT), text("workflow", GRAPH_TEXT), IDAT, IEND)


# --- finding it -----------------------------------------------------------------------


def test_a_comfyui_png_gives_its_graph_as_written():
    found = workflows.read_bytes(COMFY_PNG)
    assert found.kind == GRAPH and found.nodes == 2
    # Byte for byte: nothing needed changing, so nothing was.
    assert found.text == GRAPH_TEXT


def test_the_api_format_alone_is_still_found():
    found = workflows.read_bytes(png(text("prompt", PROMPT_TEXT), IDAT, IEND))
    assert found.kind == API and found.nodes == 2 and json.loads(found.text) == PROMPT_JSON


@pytest.mark.parametrize("carrier", [
    itext("workflow", json.dumps(dict(GRAPH_JSON, extra={"title": "Портрет · 肖像"}), ensure_ascii=False)),
    itext("workflow", GRAPH_TEXT, compress=True),
    ztext("workflow", GRAPH_TEXT),
])
def test_text_in_utf8_or_compressed_is_read_too(carrier: bytes):
    found = workflows.read_bytes(png(carrier, IDAT, IEND))
    assert found.kind == GRAPH and found.nodes == 2


def test_somebody_else_s_parameters_are_not_a_workflow():
    assert workflows.read_bytes(png(text("parameters", "a cat, Steps: 20"), IDAT, IEND)) is None


def test_reading_stops_where_the_pixels_begin():
    """Text after the pixels is legal PNG, but nothing ComfyUI saves puts it there; a remote
    read does not download a whole picture to look for it. A file on disk is read whole."""
    late = png(IDAT, text("workflow", GRAPH_TEXT), IEND)
    assert workflows.scan(late, complete=False) == (True, None)
    assert workflows.read_bytes(late).kind == GRAPH


def test_half_a_chunk_is_not_an_answer():
    cut = COMFY_PNG.index(b"tEXtworkflow") + 40
    assert workflows.scan(COMFY_PNG[:cut], complete=False) == (False, None)
    assert workflows.scan(COMFY_PNG[:5], complete=False) == (False, None)
    settled, found = workflows.scan(COMFY_PNG, complete=False)
    assert settled and found.kind == GRAPH


def test_the_api_format_alone_waits_for_the_graph_that_may_follow():
    head = png(text("prompt", PROMPT_TEXT))
    assert workflows.scan(head, complete=False) == (False, None)


@pytest.mark.parametrize("order", ["<", ">"])
def test_a_jpeg_keeps_it_in_exif(order: str):
    exif = tiff({0x010F: "workflow:" + GRAPH_TEXT, 0x0110: "prompt:" + PROMPT_TEXT}, order)
    found = workflows.read_bytes(jpeg(exif))
    assert found.kind == GRAPH and found.text == GRAPH_TEXT


def test_a_jpeg_with_a_camera_s_exif_has_none():
    assert workflows.read_bytes(jpeg(tiff({0x010F: "Canon", 0x0110: "EOS"}))) is None
    assert workflows.read_bytes(jpeg(None)) is None


@pytest.mark.parametrize("header", [False, True])
def test_a_webp_keeps_it_after_the_image(header: bool):
    data = webp(tiff({0x010F: "workflow:" + GRAPH_TEXT}), header=header)
    assert workflows.read_bytes(data).kind == GRAPH
    # Cut before the EXIF: the answer is still to come.
    assert workflows.scan(data[: data.index(b"EXIF")], complete=False) == (False, None)


def test_a_webp_says_up_front_when_it_has_none():
    """Otherwise a WebP without metadata would be downloaded whole to learn nothing."""
    data = webp(None)
    assert workflows.scan(data[:40], complete=False) == (True, None)


def test_a_graph_without_extra_is_made_pasteable():
    """ComfyUI's paste handler wants `version`, `nodes` and `extra`, and ignores anything
    short of that without a word."""
    bare = {"nodes": [{"id": 1, "type": "KSampler"}], "links": []}
    found = workflows.read_bytes(png(text("workflow", json.dumps(bare)), IDAT, IEND))
    pasted = json.loads(found.text)
    assert pasted["extra"] == {} and pasted["version"] == 0.4 and pasted["nodes"] == bare["nodes"]


def test_nan_becomes_null_so_a_paste_can_parse_it():
    raw = GRAPH_TEXT.replace('"scale": 1', '"scale": NaN')
    found = workflows.read_bytes(png(text("workflow", raw), IDAT, IEND))

    def strict(name: str):
        raise ValueError(name)

    assert json.loads(found.text, parse_constant=strict)["extra"]["ds"]["scale"] is None


@pytest.mark.parametrize("workflow", ["{}", "[1, 2]", '{"nodes": []}', "{broken", ""])
def test_json_that_is_not_a_graph_is_nothing(workflow: str):
    assert workflows.read_bytes(png(text("workflow", workflow), IDAT, IEND)) is None


def test_an_api_prompt_must_look_like_one():
    odd = json.dumps({"1": {"inputs": {}}, "2": "SaveImage"})
    assert workflows.read_bytes(png(text("prompt", odd), IDAT, IEND)) is None


def test_what_is_not_a_picture_is_nothing():
    assert workflows.scan(b"GIF89a" + b"\0" * 64) == (True, None)
    assert workflows.scan(b"") == (True, None)


# --- reading as little as possible ------------------------------------------------------


class Counted(httpx.AsyncByteStream):
    """A response body handed over in small pieces, counting how much was taken."""

    def __init__(self, data: bytes, piece: int = 16 * 1024) -> None:
        self.data, self.piece, self.taken = data, piece, 0

    async def __aiter__(self):
        for start in range(0, len(self.data), self.piece):
            part = self.data[start : start + self.piece]
            self.taken += len(part)
            yield part


def serve(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run(coroutine):
    return asyncio.run(coroutine)


def test_only_the_start_of_a_picture_is_downloaded():
    body = Counted(COMFY_PNG[:-len(IDAT + IEND)] + chunk(b"IDAT", b"\0" * (4 * 1024 * 1024)) + IEND)

    async def go():
        async with serve(lambda request: httpx.Response(
                200, headers={"Content-Type": "image/png"}, stream=body)) as client:
            return await workflows.read(IMAGE_URL, client)

    assert run(go()).kind == GRAPH
    assert body.taken < 64 * 1024


def test_a_host_we_do_not_download_from_is_never_asked():
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"{request.url} should never have been requested")

    async def go():
        async with serve(explode) as client:
            return await workflows.read("https://evil.example/x.png", client)

    assert run(go()) is None


def test_a_picture_that_is_gone_is_an_answer():
    async def go():
        async with serve(lambda request: httpx.Response(404)) as client:
            return await workflows.read(IMAGE_URL, client)

    assert run(go()) is None


@pytest.mark.parametrize("response", [
    httpx.Response(500),
    httpx.Response(403),
    httpx.Response(200, headers={"Content-Type": "text/html"}, content=b"<html>checking your browser</html>"),
])
def test_a_refusal_is_not_an_answer(response: httpx.Response):
    async def go():
        async with serve(lambda request: response) as client:
            return await workflows.read(IMAGE_URL, client)

    with pytest.raises(workflows.Unreadable):
        run(go())


def test_a_connection_that_fails_is_not_an_answer():
    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    async def go():
        async with serve(fail) as client:
            return await workflows.read(IMAGE_URL, client)

    with pytest.raises(workflows.Unreadable):
        run(go())


# --- remembering ----------------------------------------------------------------------


def test_the_answer_is_remembered_either_way(tmp_path: Path):
    found = workflows.read_bytes(COMFY_PNG)
    workflows.remember(tmp_path, IMAGE_URL, found)
    assert workflows.cached(tmp_path, IMAGE_URL) == (True, found)

    other = IMAGE_URL.replace("one", "two")
    workflows.remember(tmp_path, other, None)
    assert workflows.cached(tmp_path, other) == (True, None)
    assert workflows.cached(tmp_path, IMAGE_URL.replace("one", "three")) == (False, None)


def test_a_cache_file_nobody_can_read_is_asked_again(tmp_path: Path):
    workflows.cache_path(tmp_path, IMAGE_URL).write_text("{not json", "utf-8")
    assert workflows.cached(tmp_path, IMAGE_URL) == (False, None)


def test_two_questions_about_one_picture_make_one_read(tmp_path: Path):
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=COMFY_PNG)

    async def go():
        async with serve(respond) as client:
            finder = workflows.Finder(client)
            first, second = await asyncio.gather(
                finder.find(IMAGE_URL, tmp_path), finder.find(IMAGE_URL, tmp_path))
            third = await finder.find(IMAGE_URL, tmp_path)
            return first, second, third

    first, second, third = run(go())
    assert first == second == third and first.kind == GRAPH
    assert len(calls) == 1


def test_a_resized_copy_is_asked_for_as_the_original():
    """A copy the CDN resized was re-encoded and lost its metadata. Older API answers, and
    the `.civitai.info` files other tools wrote from them, name such a copy."""
    resized = "https://image.civitai.com/bucket/one/width=450/143524783.jpeg"
    assert previews.original_url(resized) == IMAGE_URL
    assert previews.original_url(IMAGE_URL) == IMAGE_URL
    assert previews.original_url(
        "https://image.civitai.com/b/one/original=true,quality=90/a.jpeg").endswith("/one/original=true/a.jpeg")
    hub = "https://huggingface.co/org/repo/resolve/main/sample.png"
    assert previews.original_url(hub) == hub


def test_the_finder_reads_the_original(tmp_path: Path):
    asked: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "image/png"}, content=COMFY_PNG)

    async def go():
        async with serve(respond) as client:
            return await workflows.Finder(client).find(
                "https://image.civitai.com/bucket/one/width=450/143524783.jpeg", tmp_path)

    assert run(go()).kind == GRAPH
    assert asked == [IMAGE_URL]


def test_a_failure_is_not_remembered(tmp_path: Path):
    calls: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(503)

    async def go():
        async with serve(respond) as client:
            finder = workflows.Finder(client)
            for _ in range(2):
                with pytest.raises(workflows.Unreadable):
                    await finder.find(IMAGE_URL, tmp_path)

    run(go())
    assert len(calls) == 2
    assert not workflows.cache_path(tmp_path, IMAGE_URL).exists()


# --- saving ---------------------------------------------------------------------------


def test_the_name_says_which_model_and_which_sample():
    assert workflows.file_name("lenovo_qwen21.safetensors", 1, GRAPH) == "lenovo_qwen21 - sample 2.json"
    assert workflows.file_name("style.bf16.safetensors", 0, API) == "style.bf16 - sample 1 (API).json"
    assert workflows.file_name('odd:name?.safetensors', 0, GRAPH) == "odd_name_ - sample 1.json"
    assert workflows.file_name("", 0, GRAPH) == "workflow - sample 1.json"


def test_saving_the_same_one_twice_makes_one_file(tmp_path: Path):
    first = workflows.save(tmp_path / "workflows", "style - sample 1.json", GRAPH_TEXT)
    again = workflows.save(tmp_path / "workflows", "style - sample 1.json", GRAPH_TEXT)
    assert first == (tmp_path / "workflows" / "style - sample 1.json", False)
    assert again == (first[0], True)
    assert first[0].read_text("utf-8") == GRAPH_TEXT


def test_a_different_workflow_never_replaces_one_saved_before(tmp_path: Path):
    """Edited in ComfyUI since, or another download of the same model: either way not ours
    to overwrite."""
    mine = tmp_path / "style - sample 1.json"
    mine.write_text('{"edited": true}', "utf-8")
    path, existed = workflows.save(tmp_path, "style - sample 1.json", GRAPH_TEXT)
    assert path.name == "style - sample 1 (2).json" and not existed
    assert mine.read_text("utf-8") == '{"edited": true}'


def test_comfyui_s_own_folder_is_found_beside_its_models(tmp_path: Path):
    install = tmp_path / "ComfyUI"
    (install / "models").mkdir(parents=True)
    shared = tmp_path / "Shared"
    shared.mkdir()
    assert workflows.comfy_folder([shared, install / "models"]) is None

    (install / "user" / "default").mkdir(parents=True)
    assert workflows.comfy_folder([shared, install / "models"]) == install / "user" / "default" / "workflows"


# --- over HTTP ------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    install = tmp_path / "ComfyUI"
    (install / "models").mkdir(parents=True)
    (install / "user" / "default").mkdir(parents=True)
    settings = Settings(
        library_root=str(install / "models"),
        download_dir=str(tmp_path / "downloads"),
        preview_dir=str(tmp_path / "previews"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database                                 # type: ignore[attr-defined]
        test.settings = settings                                 # type: ignore[attr-defined]
        test.root = install / "models"                           # type: ignore[attr-defined]
        test.saved = install / "user" / "default" / "workflows"  # type: ignore[attr-defined]
        test.previews = tmp_path / "previews"                    # type: ignore[attr-defined]
        yield test


def queue(client, previews: list[dict]):
    """A download whose first sample was already looked inside, so nothing goes online."""
    workflows.remember(client.previews, IMAGE_URL, workflows.read_bytes(COMFY_PNG))
    return client.database.add(
        source="https://civitai.com/models/7",
        provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 42, "file_id": 1}},
        filename="ollie.safetensors",
        size=1024,
        dest="/tmp/ollie.safetensors",
        meta={"previews": previews},
    )


SAMPLES = [
    {"url": IMAGE_URL, "type": "image"},
    {"url": "https://image.civitai.com/bucket/two/original=true/b.mp4", "type": "video"},
    # Not a host previews come from: refused before any request, and so a quiet "none".
    {"url": "https://evil.example/c.png", "type": "image"},
]


def test_the_viewer_learns_what_every_sample_carries(client):
    task = queue(client, SAMPLES)
    body = client.get(f"/api/tasks/{task.id}/workflows").json()
    assert body["workflows"] == [
        {"index": 0, "kind": "graph", "nodes": 2},
        {"index": 1, "kind": "skipped"},
        {"index": 2, "kind": "none"},
    ]


def test_copy_is_given_the_workflow_itself(client):
    task = queue(client, SAMPLES)
    body = client.get(f"/api/tasks/{task.id}/workflows/0").json()
    assert body == {"kind": "graph", "nodes": 2, "text": GRAPH_TEXT}
    for index in (1, 2, 9, -1):
        assert client.get(f"/api/tasks/{task.id}/workflows/{index}").status_code == 404


def test_save_goes_into_comfyui_s_own_folder_named_after_the_model(client):
    task = queue(client, SAMPLES)
    first = client.post(f"/api/tasks/{task.id}/workflows/0/save").json()
    assert first["ok"] and not first["existed"] and first["name"] == "ollie - sample 1.json"
    assert (client.saved / "ollie - sample 1.json").read_text("utf-8") == GRAPH_TEXT

    again = client.post(f"/api/tasks/{task.id}/workflows/0/save").json()
    assert again["existed"] and again["name"] == "ollie - sample 1.json"
    assert len(list(client.saved.iterdir())) == 1


def test_the_folder_in_the_settings_wins(client, tmp_path: Path):
    task = queue(client, SAMPLES)
    client.put("/api/settings", json={"workflow_dir": str(tmp_path / "mine")})
    body = client.post(f"/api/tasks/{task.id}/workflows/0/save").json()
    assert Path(body["path"]) == tmp_path / "mine" / "ollie - sample 1.json"
    assert not client.saved.exists()


def test_with_nowhere_to_save_the_page_is_asked_to_choose(client, tmp_path: Path):
    task = queue(client, SAMPLES)
    elsewhere = tmp_path / "library"
    elsewhere.mkdir()
    client.put("/api/settings", json={"library_root": str(elsewhere)})
    assert client.post(f"/api/tasks/{task.id}/workflows/0/save").json() == {
        "ok": False, "needs_folder": True}


def test_the_settings_say_what_an_empty_folder_means(client):
    body = client.get("/api/settings").json()["settings"]
    assert body["workflow_dir"] == "" and Path(body["workflow_dir_found"]) == client.saved


def test_with_previews_turned_off_no_picture_is_looked_at(client):
    task = queue(client, SAMPLES)
    client.put("/api/settings", json={"fetch_previews": False})
    kinds = [w["kind"] for w in client.get(f"/api/tasks/{task.id}/workflows").json()["workflows"]]
    assert kinds == ["skipped", "skipped", "skipped"]


def test_the_picture_beside_a_model_is_read_from_disk(client):
    model_file = client.root / "loras" / "style.safetensors"
    model_file.parent.mkdir(parents=True)
    model_file.write_bytes(b"weights")
    model_file.with_name("style.preview.png").write_bytes(COMFY_PNG)
    client.post("/api/library/rescan")
    (model,) = client.get("/api/library").json()["models"]

    body = client.get(f"/api/models/{model['id']}/workflows").json()
    assert body["workflows"] == [{"index": 0, "kind": "graph", "nodes": 2}]
    saved = client.post(f"/api/models/{model['id']}/workflows/0/save").json()
    assert saved["name"] == "style - sample 1.json"


def test_reveal_names_a_saved_workflow_and_nothing_else(client, monkeypatch):
    shown: list[str] = []
    monkeypatch.setattr("sfd.desktop.open_system_path", lambda path: shown.append(path) or True)
    task = queue(client, SAMPLES)
    name = client.post(f"/api/tasks/{task.id}/workflows/0/save").json()["name"]

    for wrong in ("..\\settings.json", "../settings.json", "missing.json", "ollie.safetensors"):
        assert client.post("/api/workflows/reveal", json={"name": wrong}).status_code == 404
    assert client.post("/api/workflows/reveal", json={"name": name}).json() == {"ok": True}
    assert shown == [str(client.saved / name)]
