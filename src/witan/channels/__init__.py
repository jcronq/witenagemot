"""Transport abstractions for witan agents.

A :class:`Channel` is anything an agent can receive turns from and
send messages to — a human via CLI or Signal, another agent via a
structured inbox, an HTTP endpoint. v0.1 ships :class:`StdChannel`
only; :class:`SignalChannel` and :class:`SurfaceChannel` land in
v0.2.

:class:`Turn` is the immutable envelope for one inbound message.
"""

from __future__ import annotations

from .std import Channel, StdChannel, Turn

__all__ = ["Channel", "StdChannel", "Turn"]
