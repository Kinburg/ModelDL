"""Sample images: what is kept from a service's payload, and how it reaches the page.

The two things worth pinning down are the URL rewriting — a thumbnail is a full-size image
with one path segment changed, and getting that wrong means either a broken picture or a
three-megabyte download per row — and the host allowlist, which is the only thing between a
remote API response and an HTTP client running on someone's machine.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from sfd.jobs.db import Database
from sfd.library import previews
from sfd.providers.civitai import PREVIEW_LIMIT, _describe_files
from sfd.settings import Settings
from sfd.web.app import create_app

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
IMAGE = "https://image.civitai.com/bucket/de305d54/width=450/sample.jpeg"

VERSION = {
    "id": 42,
    "name": "v1",
    "modelId": 7,
    "baseModel": "Illustrious",
    "trainedWords": ["sick ollie"],
    "model": {"name": "Sick Ollie", "type": "LORA", "nsfw": False},
    "images": [
        {
            "url": "https://image.civitai.com/bucket/one/width=450/a.jpeg",
            "type": "image", "width": 832, "height": 1216, "nsfwLevel": 1,
            "meta": {
                "prompt": "sick ollie, a cat on a skateboard",
                "negativePrompt": "blurry",
                "sampler": "Euler a", "steps": 28, "cfgScale": 4.5, "seed": 1234567,
                "Model": "illustriousXL", "clipSkip": 2,
                # Not ours to keep: belongs to the picture, not to the model.
                "resources": [{"name": "something"}],
            },
        },
        {
            "url": "https://image.civitai.com/bucket/two/width=450/b.mp4",
            "type": "video", "nsfwLevel": 4, "meta": None,
        },
    ],
    "files": [
        {
            "id": 1, "primary": True, "type": "Model", "name": "ollie.safetensors",
            "sizeKB": 1024, "metadata": {"format": "SafeTensor", "fp": "fp16"},
            "hashes": {"SHA256": "AB" * 32},
        }
    ],
}


def meta_of(version: dict = VERSION) -> dict:
    return _describe_files(version)[0].meta


# --- what is kept from the payload ------------------------------------------


def test_the_samples_come_through_with_their_prompts():
    first = meta_of()["previews"][0]
    assert first["url"].endswith("a.jpeg")
    assert first["meta"]["prompt"] == "sick ollie, a cat on a skateboard"
    assert first["meta"]["negative_prompt"] == "blurry"
    assert first["meta"]["cfg_scale"] == 4.5
    assert first["meta"]["seed"] == 1234567
    # The picture's own resource list is not the model's business.
    assert "resources" not in first["meta"]


def test_a_video_sample_is_marked_as_one():
    kept = meta_of()["previews"]
    assert kept[1]["type"] == "video"
    assert kept[1]["meta"] == {}


def test_the_compatibility_preview_skips_a_video():
    """`<model>.preview.png` has to be a still: the model managers put it in an <img>."""
    version = dict(VERSION, images=list(reversed(VERSION["images"])))
    assert meta_of(version)["preview_url"].endswith("a.jpeg")


def test_adult_samples_are_flagged():
    kept = meta_of()["previews"]
    # nsfwLevel 1 is the safe level, not a missing one.
    assert kept[0]["nsfw"] is False
    assert kept[1]["nsfw"] is True


def test_the_older_word_for_it_is_understood_too():
    version = dict(VERSION, images=[
        {"url": "https://image.civitai.com/b/one/width=450/a.jpeg", "nsfw": "None"},
        {"url": "https://image.civitai.com/b/two/width=450/b.jpeg", "nsfw": "Mature"},
    ])
    assert [p["nsfw"] for p in meta_of(version)["previews"]] == [False, True]


def test_a_wall_of_samples_is_cut_short():
    version = dict(VERSION, images=[
        {"url": f"https://image.civitai.com/b/{n}/width=450/{n}.jpeg"} for n in range(40)
    ])
    assert len(meta_of(version)["previews"]) == PREVIEW_LIMIT


def test_a_version_with_no_images_says_so():
    kept = meta_of(dict(VERSION, images=[]))
    assert kept["previews"] == [] and kept["preview_url"] is None


# --- reading them back ------------------------------------------------------


def test_a_task_from_an_older_build_still_has_its_one_picture():
    """Queues already on disk predate the list, and hold a single `preview_url`."""
    assert previews.entries({"preview_url": IMAGE}) == [{"url": IMAGE, "type": "image"}]


def test_entries_prefers_the_list_when_there_is_one():
    meta = {"preview_url": IMAGE, "previews": [{"url": "https://x/y.png", "type": "image"}]}
    assert previews.entries(meta)[0]["url"] == "https://x/y.png"


def test_entries_of_nothing_is_nothing():
    assert previews.entries({}) == []
    assert previews.entries({"previews": [{"nope": 1}]}) == []


# --- asking the CDN for a smaller copy --------------------------------------


def test_a_width_replaces_the_one_in_the_path():
    assert previews.variant_url(IMAGE, 320) == (
        "https://image.civitai.com/bucket/de305d54/width=320/sample.jpeg")


def test_any_transformation_is_replaced_not_appended():
    url = "https://image.civitai.com/bucket/de305d54/original=true,quality=90/sample.jpeg"
    assert previews.variant_url(url, 320).endswith("/de305d54/width=320/sample.jpeg")


def test_a_url_without_a_transformation_gets_one():
    url = "https://image.civitai.com/bucket/de305d54/sample.jpeg"
    assert previews.variant_url(url, 320) == (
        "https://image.civitai.com/bucket/de305d54/width=320/sample.jpeg")


def test_a_video_thumbnail_is_asked_for_as_a_still():
    assert "width=320,anim=false" in previews.variant_url(IMAGE, 320, still=True)


def test_someone_elses_host_is_left_alone():
    """Only Civitai's CDN reads path segments as instructions; rewriting anybody else's URL
    turns a working picture into a 404."""
    url = "https://huggingface.co/org/repo/resolve/main/sample.png"
    assert previews.variant_url(url, 320) == url


def test_asking_for_nothing_changes_nothing():
    assert previews.variant_url(IMAGE) == IMAGE


# --- the allowlist ----------------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://image.civitai.com/x/y/width=450/z.jpeg",
    "https://civitai.com/images/1",
    "https://cdn.huggingface.co/x.png",
    "https://hf.co/x.png",
])
def test_the_services_we_download_from_are_allowed(url: str):
    assert previews.allowed(url)


@pytest.mark.parametrize("url", [
    "http://image.civitai.com/x.png",              # plaintext, so tamperable in flight
    "https://civitai.com.attacker.example/x.png",  # the name only starts the same way
    "https://169.254.169.254/latest/meta-data",    # the cloud metadata service
    "https://127.0.0.1:9200/_search",              # something else on this machine
    "file:///C:/Windows/win.ini",
    "https://evil.example/x.png",
])
def test_anything_else_is_refused(url: str):
    """These URLs arrive inside an API response, which is remote data. A local server that
    fetched whatever they named would be a proxy into the machine it runs on."""
    assert not previews.allowed(url)


def test_a_refused_url_is_never_requested(tmp_path: Path):
    def explode(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"{request.url} should never have been requested")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(explode)) as client:
            return await previews.fetch("https://evil.example/x.png", tmp_path, client)

    assert asyncio.run(run()) is None


# --- the cache --------------------------------------------------------------


def test_one_copy_per_url(tmp_path: Path):
    """Five quantisations of one version share their pictures; the cache should too."""
    assert previews.cache_path(tmp_path, IMAGE) == previews.cache_path(tmp_path, IMAGE)
    assert previews.cache_path(tmp_path, IMAGE) != previews.cache_path(
        tmp_path, previews.variant_url(IMAGE, 320))


def test_fetching_stores_it_once(tmp_path: Path):
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=PNG, headers={"Content-Type": "image/png"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            first = await previews.fetch(IMAGE, tmp_path, client)
            second = await previews.fetch(IMAGE, tmp_path, client)
            return first, second

    first, second = asyncio.run(run())
    assert first == second == previews.cache_path(tmp_path, IMAGE)
    assert first.read_bytes() == PNG
    assert len(calls) == 1


def test_a_login_page_is_not_an_image(tmp_path: Path):
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>sign in</html>",
                              headers={"Content-Type": "text/html"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await previews.fetch(IMAGE, tmp_path, client)

    assert asyncio.run(run()) is None
    assert not previews.cache_path(tmp_path, IMAGE).exists()


def test_something_enormous_is_dropped_rather_than_stored(tmp_path: Path):
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\x00" * (previews.MAX_BYTES + 1),
                              headers={"Content-Type": "image/png"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            return await previews.fetch(IMAGE, tmp_path, client)

    assert asyncio.run(run()) is None
    assert not previews.cache_path(tmp_path, IMAGE).exists()


def test_what_a_file_is_comes_from_its_bytes(tmp_path: Path):
    """Civitai serves JPEG from URLs ending in .png often enough that the name is no
    evidence, and the cache names everything `.img` anyway."""
    path = tmp_path / "x.img"
    path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
    assert previews.media_type(path) == "image/jpeg"
    assert previews.sniff(b"\x89PNG\r\n\x1a\n") == "image/png"
    assert previews.sniff(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert previews.sniff(b"\x00\x00\x00 ftypisom") == "video/mp4"
    assert previews.sniff(b"not a picture") == "application/octet-stream"


# --- over HTTP --------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        download_dir=str(tmp_path / "downloads"),
        preview_dir=str(tmp_path / "previews"),
        concurrent_downloads=1,
        _path=str(tmp_path / "settings.json"),
    )
    database = Database(tmp_path / "queue.db")
    with TestClient(create_app(settings, database), base_url="http://127.0.0.1:7788") as test:
        test.database = database          # type: ignore[attr-defined]
        yield test


def queue(database: Database, **meta):
    return database.add(
        source="https://civitai.com/models/7",
        provider="civitai",
        identity={"provider": "civitai", "ref": {"version_id": 42, "file_id": 1}},
        filename="ollie.safetensors",
        size=1024,
        dest="/tmp/ollie.safetensors",
        meta=meta,
    )


def test_the_queue_says_how_many_pictures_there_are(client):
    queue(client.database, **meta_of())
    assert client.get("/api/tasks").json()["tasks"][0]["previews"] == 2


def test_a_download_with_no_pictures_says_zero(client):
    queue(client.database)
    assert client.get("/api/tasks").json()["tasks"][0]["previews"] == 0


def test_the_prompts_are_served_on_request(client):
    task = queue(client.database, **meta_of())
    body = client.get(f"/api/tasks/{task.id}/previews").json()
    assert [p["type"] for p in body["previews"]] == ["image", "video"]
    assert body["previews"][0]["meta"]["sampler"] == "Euler a"
    assert body["previews"][1]["nsfw"] is True


def test_a_cached_picture_is_served_with_the_type_its_bytes_say(client, tmp_path: Path):
    task = queue(client.database, **meta_of())
    wanted = previews.variant_url(
        "https://image.civitai.com/bucket/one/width=450/a.jpeg", 320)
    cached = previews.cache_path(tmp_path / "previews", wanted)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(PNG)

    response = client.get(f"/api/tasks/{task.id}/preview?w=320")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == PNG
    # The list is redrawn on every state change of every task; without this the queue would
    # re-request every thumbnail each time.
    assert "max-age" in response.headers["cache-control"]


def test_a_picture_that_cannot_be_had_is_not_a_broken_page(client):
    queued = queue(client.database, previews=[{"url": "https://evil.example/x.png"}])
    assert client.get(f"/api/tasks/{queued.id}/preview").status_code == 502


def test_there_is_no_third_picture(client):
    task = queue(client.database, **meta_of())
    assert client.get(f"/api/tasks/{task.id}/preview/9").status_code == 404
    assert client.get(f"/api/tasks/{task.id}/preview/-1").status_code == 404


def test_a_task_that_does_not_exist_has_no_pictures(client):
    assert client.get("/api/tasks/999/preview").status_code == 404
    assert client.get("/api/tasks/999/previews").status_code == 404


def test_turning_them_off_stops_them_being_served(client):
    task = queue(client.database, **meta_of())
    client.put("/api/settings", json={"fetch_previews": False})
    assert client.get(f"/api/tasks/{task.id}/preview").status_code == 404


def test_a_silly_width_is_refused(client):
    """The width goes into a URL this server then fetches, so it is a number in a range,
    not whatever the query string says."""
    task = queue(client.database, **meta_of())
    assert client.get(f"/api/tasks/{task.id}/preview?w=99999").status_code == 422
    assert client.get(f"/api/tasks/{task.id}/preview?w=nonsense").status_code == 422
