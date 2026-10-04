"""Bounds how many requests under given path prefixes run at once.

A prediction holds about 150 MB above idle while it builds inputs, runs the
rollout and draws the bands. On a 512 MB instance three at once exceed the
limit, and the restart drops every request in flight with it. Queueing the
excess instead keeps the peak at what one prediction needs; a queued request
holds nothing but its connection. It is also faster: the work is CPU-bound, so
running predictions side by side finishes them all at about the same late time,
while a queue hands the first one back as soon as it is done.

Plain ASGI rather than an HTTP middleware so the slot is held until the
response has been sent, not just until the handler returns.
"""
import asyncio
from typing import Iterable


class ConcurrencyLimit:
    """ASGI middleware: at most `limit` matching requests in flight, the rest wait."""

    def __init__(self, app, *, path_prefixes: Iterable[str], limit: int):
        if limit < 1:
            raise ValueError(f"limit must be at least 1, got {limit}")
        self.app = app
        self.path_prefixes = tuple(path_prefixes)
        self.slots = asyncio.Semaphore(limit)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith(self.path_prefixes):
            await self.app(scope, receive, send)
            return
        async with self.slots:
            await self.app(scope, receive, send)
