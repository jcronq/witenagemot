"""Anthropic backend for the witan Kernel protocol.

Uses the ``anthropic`` SDK's ``messages.create`` endpoint — no
streaming, no MCP, no tool execution loop. The kernel translates
:class:`witan.KernelSpec` into API arguments, awaits the response,
and returns a normalized :class:`witan.KernelResult`. Tool
execution and multi-turn agent loops are the caller's responsibility
in v0.1; witan owns the invocation, not the orchestration.
"""

from __future__ import annotations

import os
from typing import Any

from ..types import KernelResult, KernelSpec, UsageInfo

__all__ = ["AnthropicKernel"]


# Default token budget when the caller doesn't set one via env. The
# Anthropic SDK requires ``max_tokens`` on every call; witan doesn't
# take a stance on the right ceiling, so we pick a generous default
# and let operators override it via ``WITAN_ANTHROPIC_MAX_TOKENS``.
_DEFAULT_MAX_TOKENS = 4096


def _thinking_to_native(level: str | None) -> dict[str, Any] | None:
    """Map a witan thinking level to Anthropic's ``ThinkingConfigParam``.

    ``"off"`` disables extended thinking; the other levels enable it
    with a coarse token budget. Anthropic's own API takes an explicit
    ``budget_tokens``; the mapping here follows the convention used
    by other productized kernels (small / medium / large budgets).
    """
    if level is None:
        return None
    if level == "off":
        return {"type": "disabled"}
    budgets = {
        "minimal": 1024,
        "low": 2048,
        "medium": 4096,
        "high": 8192,
    }
    budget = budgets.get(level)
    if budget is None:
        return None
    return {"type": "enabled", "budget_tokens": budget}


def _extract_text(message: Any) -> str:
    """Concatenate every ``text`` block on an Anthropic ``Message``.

    Robust to the SDK returning either dataclass-like objects (with
    ``.type`` / ``.text`` attributes) or plain dicts — both shapes
    appear in tests and real responses.
    """
    parts: list[str] = []
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    for block in content or ():
        block_type = getattr(block, "type", None)
        if block_type is None and isinstance(block, dict):
            block_type = block.get("type")
        if block_type != "text":
            continue
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(text)
    return "".join(parts)


def _extract_usage(message: Any) -> UsageInfo | None:
    """Translate an Anthropic ``Usage`` block into :class:`UsageInfo`."""
    usage = getattr(message, "usage", None)
    if usage is None and isinstance(message, dict):
        usage = message.get("usage")
    if usage is None:
        return None

    def _get(key: str) -> Any:
        if hasattr(usage, key):
            return getattr(usage, key)
        if isinstance(usage, dict):
            return usage.get(key)
        return None

    input_tokens = _get("input_tokens") or 0
    output_tokens = _get("output_tokens") or 0
    return UsageInfo(
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        cache_read_input_tokens=_get("cache_read_input_tokens"),
        cache_creation_input_tokens=_get("cache_creation_input_tokens"),
        total_tokens=int(input_tokens) + int(output_tokens),
    )


class AnthropicKernel:
    """Single-turn Anthropic Messages API backend.

    Optional constructor args let tests substitute a fake client and
    pin the token budget. In normal use the kernel constructs an
    ``AsyncAnthropic`` client at first call — API key resolution is
    delegated to the SDK's default behavior (``ANTHROPIC_API_KEY``
    env var or the OAuth flow the SDK ships with).
    """

    def __init__(
        self,
        client: Any | None = None,
        *,
        max_tokens: int | None = None,
    ) -> None:
        self._client = client
        if max_tokens is not None:
            self._max_tokens = max_tokens
        else:
            env_val = os.environ.get("WITAN_ANTHROPIC_MAX_TOKENS")
            self._max_tokens = int(env_val) if env_val else _DEFAULT_MAX_TOKENS

    def _get_client(self) -> Any:
        if self._client is None:
            # Local import so importing this module is cheap in tests
            # that patch the client and never touch real network.
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic()
        return self._client

    async def run(self, spec: KernelSpec) -> KernelResult:
        """Dispatch one turn and return the normalized result.

        Reads the user message off :attr:`KernelSpec.user_prompt`; the
        runner sets that field before dispatch. Direct callers set it
        via :func:`dataclasses.replace`. Kept minimal on purpose:
        this is a translator, not an agent loop — multi-step tool use
        is v0.2 territory.
        """
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": spec.model,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "user", "content": spec.user_prompt}
            ],
        }
        if spec.append_system_prompt:
            kwargs["system"] = spec.append_system_prompt
        if spec.allowed_tools:
            # v0.1: witan doesn't execute tools. The list is forwarded
            # so downstream orchestrators that DO execute tools can
            # inspect what the model was told about — no schemas yet.
            kwargs["tools"] = [
                {"name": name, "description": "", "input_schema": {"type": "object"}}
                for name in spec.allowed_tools
            ]
        thinking = _thinking_to_native(spec.thinking)
        if thinking is not None:
            kwargs["thinking"] = thinking
        if spec.max_seconds > 0:
            kwargs["timeout"] = float(spec.max_seconds)

        message = await client.messages.create(**kwargs)
        return KernelResult(
            text=_extract_text(message),
            usage=_extract_usage(message),
            raw=message,
        )


