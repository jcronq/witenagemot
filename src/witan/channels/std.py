"""stdin/stdout channel for local development and CLI use.

:class:`StdChannel` reads inbound turns from stdin and writes
outbound messages to stdout. Two receive modes: ``"line"`` (one
turn per newline-terminated line) and ``"block"`` (the whole stdin
buffer up to EOF, delivered as a single turn). The block mode is
what the ``witan run`` CLI uses so multi-line prompts pipe cleanly.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TextIO, runtime_checkable

__all__ = ["Channel", "StdChannel", "Turn"]


@dataclass(frozen=True)
class Turn:
    """An inbound message plus any transport metadata.

    ``text`` is the plain user content. ``metadata`` is a free-form
    dict — transports use it for envelope info (Signal thread id,
    HTTP headers, etc.) without polluting the primary API.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Channel(Protocol):
    """The transport interface every :class:`Channel` impl satisfies.

    :meth:`receive` yields :class:`Turn` values as they arrive;
    concrete impls decide whether the iterator ever terminates
    (stdin/EOF-terminated vs. long-running Signal loop). :meth:`send`
    delivers an outbound message; ``reply_to`` lets transports thread
    the response into the same conversation when the underlying
    protocol supports it.
    """

    def receive(self) -> AsyncIterator[Turn]:
        """Return an async iterator of inbound turns."""
        ...

    async def send(self, text: str, *, reply_to: Turn | None = None) -> None:
        """Send ``text`` as an outbound message."""
        ...


class StdChannel:
    """stdin/stdout implementation of :class:`Channel`.

    ``mode="line"`` yields one :class:`Turn` per stripped, newline-
    terminated stdin line. ``mode="block"`` reads until EOF and
    yields a single :class:`Turn` with the whole buffer. Test suites
    inject a custom ``stdin``/``stdout`` (any file-like object with
    ``readline``/``read``/``write``); production code omits them and
    the channel binds to :data:`sys.stdin` and :data:`sys.stdout`.
    """

    def __init__(
        self,
        *,
        mode: Literal["line", "block"] = "line",
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        self._mode = mode
        self._stdin_override = stdin
        self._stdout_override = stdout

    @property
    def _stdin(self) -> TextIO:
        return self._stdin_override if self._stdin_override is not None else sys.stdin

    @property
    def _stdout(self) -> TextIO:
        return self._stdout_override if self._stdout_override is not None else sys.stdout

    async def receive(self) -> AsyncIterator[Turn]:
        """Yield inbound :class:`Turn` values from stdin.

        Runs blocking reads in a thread via :func:`asyncio.to_thread`
        so the caller's event loop stays responsive.
        """
        if self._mode == "block":
            text = await asyncio.to_thread(self._stdin.read)
            if text:
                yield Turn(text=text)
            return
        while True:
            line = await asyncio.to_thread(self._stdin.readline)
            if not line:
                return
            stripped = line.rstrip("\n").rstrip("\r")
            if not stripped:
                continue
            yield Turn(text=stripped)

    async def send(self, text: str, *, reply_to: Turn | None = None) -> None:
        """Write ``text`` to stdout followed by a newline and flush.

        ``reply_to`` is accepted for interface parity with richer
        transports; stdout has no notion of threaded replies so the
        argument is intentionally ignored here.
        """
        del reply_to  # unused; kept for Channel-protocol parity
        await asyncio.to_thread(self._write_and_flush, text)

    def _write_and_flush(self, text: str) -> None:
        self._stdout.write(text)
        if not text.endswith("\n"):
            self._stdout.write("\n")
        self._stdout.flush()
