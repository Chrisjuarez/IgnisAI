"""Guards on the prediction queue.

Three concurrent forecasts took the 512 MB instance past its memory limit and
the restart dropped all three. The queue is what keeps that from recurring, so
these check that it actually bounds concurrency and that it does not hold up
the cheap routes - a health check stuck behind a forecast would get the
instance restarted for a different reason.
"""
import asyncio

import pytest

from services.tilesvc.concurrency_limit import ConcurrencyLimit


class _RecordingApp:
    """Inner ASGI app that records how many requests it is handling at once."""

    def __init__(self, release: asyncio.Event):
        self.release = release
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed = []

    async def __call__(self, scope, receive, send):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await self.release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"{}"})
            self.completed.append(scope["path"])
        finally:
            self.in_flight -= 1


async def _request(app, path):
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(_message):
        pass

    await app({"type": "http", "path": path}, receive, send)


async def _run_concurrently(limit, paths, *, release_after=0.05):
    release = asyncio.Event()
    inner = _RecordingApp(release)
    gated = ConcurrencyLimit(inner, path_prefixes=("/predict", "/spread_bands"), limit=limit)
    requests = asyncio.gather(*(_request(gated, path) for path in paths))
    await asyncio.sleep(release_after)
    observed_while_held = inner.max_in_flight
    release.set()
    await requests
    return inner, observed_while_held


def test_matching_requests_run_one_at_a_time():
    paths = ["/predict_multistep", "/spread_bands", "/predict_raster_json", "/predict"]
    inner, _ = asyncio.run(_run_concurrently(1, paths))

    assert inner.max_in_flight == 1
    assert sorted(inner.completed) == sorted(paths)


def test_limit_above_one_allows_that_many_in_flight():
    inner, observed = asyncio.run(_run_concurrently(2, ["/predict_multistep"] * 5))

    assert observed == 2
    assert inner.max_in_flight == 2
    assert len(inner.completed) == 5


def test_other_routes_are_not_held_behind_a_prediction():
    async def scenario():
        held = asyncio.Event()
        inner = _RecordingApp(held)
        gated = ConcurrencyLimit(inner, path_prefixes=("/predict",), limit=1)

        prediction = asyncio.create_task(_request(gated, "/predict_multistep"))
        queued = asyncio.create_task(_request(gated, "/predict_multistep"))
        await asyncio.sleep(0.05)

        # Only one prediction is in flight and the second is queued, yet a
        # health check passes straight through to the app.
        assert inner.max_in_flight == 1
        health = asyncio.create_task(_request(gated, "/healthz"))
        await asyncio.sleep(0.05)
        assert inner.in_flight == 2

        held.set()
        await asyncio.gather(prediction, queued, health)
        return inner

    inner = asyncio.run(scenario())
    assert sorted(inner.completed) == ["/healthz", "/predict_multistep", "/predict_multistep"]


def test_non_http_scopes_pass_through():
    calls = []

    async def inner(scope, receive, send):
        calls.append(scope["type"])

    gated = ConcurrencyLimit(inner, path_prefixes=("/predict",), limit=1)
    asyncio.run(gated({"type": "lifespan"}, None, None))

    assert calls == ["lifespan"]


@pytest.mark.parametrize("limit", [0, -1])
def test_rejects_a_limit_that_would_block_every_prediction(limit):
    with pytest.raises(ValueError):
        ConcurrencyLimit(lambda *_: None, path_prefixes=("/predict",), limit=limit)
