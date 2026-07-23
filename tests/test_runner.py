"""Tests for :mod:`witan.runner`."""

from __future__ import annotations

import pytest

from witan import (
    AgentSpec,
    BehavioralRule,
    KernelResult,
    KernelSpec,
    Registry,
    Turn,
    run_agent,
)


class FakeKernel:
    """Records the ``(spec, prompt)`` pairs it was handed."""

    def __init__(self) -> None:
        self.calls: list[tuple[KernelSpec, str]] = []

    async def run(self, spec: KernelSpec, prompt: str) -> KernelResult:
        self.calls.append((spec, prompt))
        return KernelResult(text=f"fake:{prompt}", usage=None, raw=None)


@pytest.fixture
def fake_kernel_patch(monkeypatch: pytest.MonkeyPatch) -> FakeKernel:
    """Patch :func:`witan.kernel.make_kernel` to yield a shared fake."""
    fake = FakeKernel()

    def _factory(_backend: str) -> FakeKernel:
        return fake

    monkeypatch.setattr("witan.runner.make_kernel", _factory)
    return fake


async def test_run_agent_dispatches_turn_text(fake_kernel_patch: FakeKernel) -> None:
    reg = Registry()
    reg.register(
        AgentSpec(
            name="a",
            persona="unit",
            runtime="short-lived",
            kernel=KernelSpec(model="claude-haiku-4-5"),
        )
    )
    result = await run_agent("a", turn=Turn(text="hello"), registry=reg)
    assert result.text == "fake:hello"
    assert len(fake_kernel_patch.calls) == 1
    spec, prompt = fake_kernel_patch.calls[0]
    assert prompt == "hello"
    assert spec.model == "claude-haiku-4-5"


async def test_run_agent_applies_behavioral_rules(fake_kernel_patch: FakeKernel) -> None:
    reg = Registry()
    reg.register(
        AgentSpec(
            name="terse",
            persona="unit",
            runtime="short-lived",
            kernel=KernelSpec(model="claude-haiku-4-5"),
            behavioral_rules=(BehavioralRule(id="terse", injection="Be terse."),),
        )
    )
    await run_agent("terse", turn=Turn(text="hi"), registry=reg)
    spec, _prompt = fake_kernel_patch.calls[0]
    assert spec.append_system_prompt is not None
    assert "terse" in spec.append_system_prompt


async def test_run_agent_uses_default_registry(
    fake_kernel_patch: FakeKernel,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from witan import registry as registry_module

    private = Registry()
    private.register(
        AgentSpec(
            name="only-in-default",
            persona="unit",
            runtime="short-lived",
            kernel=KernelSpec(model="claude-haiku-4-5"),
        )
    )
    monkeypatch.setattr(registry_module, "default_registry", private)
    monkeypatch.setattr("witan.runner.default_registry", private)

    result = await run_agent("only-in-default", turn=Turn(text="ping"))
    assert result.text == "fake:ping"


async def test_run_agent_raises_for_unknown_agent(
    fake_kernel_patch: FakeKernel,
) -> None:
    reg = Registry()
    with pytest.raises(KeyError):
        await run_agent("missing", turn=Turn(text="x"), registry=reg)
