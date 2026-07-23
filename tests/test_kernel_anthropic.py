"""Tests for :mod:`witan.kernels.anthropic`.

Uses a stub client — no real network. Verifies the KernelSpec →
API args translation and the KernelResult construction path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from witan import KernelSpec, make_kernel
from witan.kernel import Kernel
from witan.kernels.anthropic import AnthropicKernel


@dataclass
class _StubUsage:
    input_tokens: int = 7
    output_tokens: int = 11
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None


@dataclass
class _StubTextBlock:
    text: str
    type: str = "text"


@dataclass
class _StubMessage:
    content: list[Any]
    usage: _StubUsage


class _StubMessages:
    def __init__(self, response: _StubMessage) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _StubMessage:
        self.calls.append(kwargs)
        return self.response


class _StubClient:
    def __init__(self, response: _StubMessage) -> None:
        self.messages = _StubMessages(response)


@pytest.fixture
def stub_client() -> _StubClient:
    return _StubClient(
        _StubMessage(
            content=[_StubTextBlock(text="hello back")],
            usage=_StubUsage(),
        )
    )


async def test_run_translates_spec_to_api_args(stub_client: _StubClient) -> None:
    kernel = AnthropicKernel(client=stub_client, max_tokens=512)
    spec = KernelSpec(
        model="claude-haiku-4-5",
        allowed_tools=("read_file",),
        append_system_prompt="Be terse.",
        thinking="low",
        user_prompt="hello",
    )
    result = await kernel.run(spec)
    call = stub_client.messages.calls[0]
    assert call["model"] == "claude-haiku-4-5"
    assert call["max_tokens"] == 512
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["system"] == "Be terse."
    assert call["tools"][0]["name"] == "read_file"
    assert call["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    assert result.text == "hello back"
    assert result.usage is not None
    assert result.usage.input_tokens == 7
    assert result.usage.output_tokens == 11
    assert result.usage.total_tokens == 18
    assert result.raw is stub_client.messages.response


async def test_run_thinking_off_maps_to_disabled(stub_client: _StubClient) -> None:
    kernel = AnthropicKernel(client=stub_client)
    spec = KernelSpec(model="claude-haiku-4-5", thinking="off", user_prompt="hi")
    await kernel.run(spec)
    call = stub_client.messages.calls[0]
    assert call["thinking"] == {"type": "disabled"}


async def test_run_omits_optional_fields_when_unset(stub_client: _StubClient) -> None:
    kernel = AnthropicKernel(client=stub_client)
    spec = KernelSpec(model="claude-haiku-4-5", user_prompt="hi")
    await kernel.run(spec)
    call = stub_client.messages.calls[0]
    assert "system" not in call
    assert "tools" not in call
    assert "thinking" not in call
    assert "timeout" not in call


async def test_run_max_seconds_sets_timeout(stub_client: _StubClient) -> None:
    kernel = AnthropicKernel(client=stub_client)
    spec = KernelSpec(model="claude-haiku-4-5", max_seconds=30, user_prompt="hi")
    await kernel.run(spec)
    assert stub_client.messages.calls[0]["timeout"] == 30.0


async def test_run_handles_dict_shaped_content(stub_client: _StubClient) -> None:
    stub_client.messages.response = _StubMessage(
        content=[{"type": "text", "text": "dict form"}],
        usage=_StubUsage(input_tokens=1, output_tokens=2),
    )
    kernel = AnthropicKernel(client=stub_client)
    spec = KernelSpec(model="claude-haiku-4-5", user_prompt="hi")
    result = await kernel.run(spec)
    assert result.text == "dict form"


def test_make_kernel_returns_kernel_protocol_instance() -> None:
    kernel = make_kernel("anthropic")
    assert isinstance(kernel, Kernel)


def test_make_kernel_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown kernel backend"):
        make_kernel("nope")
