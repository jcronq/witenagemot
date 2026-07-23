"""Tests for :mod:`witan.registry`."""

from __future__ import annotations

import pytest

from witan import AgentSpec, KernelSpec, Registry


def _spec(name: str) -> AgentSpec:
    return AgentSpec(
        name=name,
        persona="unit",
        runtime="short-lived",
        kernel=KernelSpec(model="claude-haiku-4-5"),
    )


def test_register_and_get() -> None:
    reg = Registry()
    spec = _spec("alpha")
    reg.register(spec)
    assert reg.get("alpha") is spec
    assert "alpha" in reg
    assert len(reg) == 1


def test_duplicate_registration_raises_value_error() -> None:
    reg = Registry()
    reg.register(_spec("dup"))
    with pytest.raises(ValueError, match="already registered"):
        reg.register(_spec("dup"))


def test_get_unknown_raises_key_error_with_known_list() -> None:
    reg = Registry()
    reg.register(_spec("known-1"))
    reg.register(_spec("known-2"))
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    msg = str(excinfo.value)
    assert "missing" in msg
    assert "known-1" in msg


def test_get_unknown_on_empty_registry() -> None:
    reg = Registry()
    with pytest.raises(KeyError, match="none"):
        reg.get("anything")


def test_replace_overwrites() -> None:
    reg = Registry()
    reg.register(_spec("a"))
    new = _spec("a")
    reg.replace(new)
    assert reg.get("a") is new


def test_names_sorted() -> None:
    reg = Registry()
    reg.register(_spec("zeta"))
    reg.register(_spec("alpha"))
    reg.register(_spec("mu"))
    assert reg.names() == ["alpha", "mu", "zeta"]


def test_default_registry_is_singleton() -> None:
    from witan import default_registry
    from witan.registry import default_registry as second_ref

    assert default_registry is second_ref
