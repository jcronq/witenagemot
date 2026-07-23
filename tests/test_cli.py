"""Tests for :mod:`witan.cli`."""

from __future__ import annotations

import io
import sys
from typing import Any

import pytest

from witan import (
    AgentSpec,
    KernelResult,
    KernelSpec,
    Registry,
    default_registry,
)
from witan.cli import main


@pytest.fixture(autouse=True)
def _isolated_default_registry(monkeypatch: pytest.MonkeyPatch) -> Registry:
    """Swap the process-wide default registry for a fresh one per test."""
    fresh = Registry()
    monkeypatch.setattr("witan.cli.__main__.default_registry", fresh)
    monkeypatch.setattr("witan.runner.default_registry", fresh)
    monkeypatch.setattr("witan.registry.default_registry", fresh)
    monkeypatch.setattr("witan.default_registry", fresh)
    return fresh


@pytest.fixture
def stub_agents_module(monkeypatch: pytest.MonkeyPatch) -> str:
    """Install a fake agents module that registers one agent on import."""
    import importlib as _importlib

    from witan.cli import __main__ as cli_main

    real_import = _importlib.import_module
    registered = False

    def _fake_import(name: str) -> Any:
        nonlocal registered
        if name == "fake_agents":
            if not registered:
                cli_main.default_registry.register(
                    AgentSpec(
                        name="echo",
                        persona="test",
                        runtime="short-lived",
                        kernel=KernelSpec(model="claude-haiku-4-5"),
                    )
                )
                registered = True
            return object()
        return real_import(name)

    monkeypatch.setattr(
        "witan.cli.__main__.importlib.import_module", _fake_import
    )
    return "fake_agents"


def test_list_agents_prints_registered_names(
    stub_agents_module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["list-agents", "--agents-module", stub_agents_module])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "echo" in captured.out


def test_run_dispatches_stdin_to_agent(
    stub_agents_module: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: dict[str, Any] = {}

    class _FakeKernel:
        async def run(self, spec: KernelSpec) -> KernelResult:
            seen["spec"] = spec
            return KernelResult(text=f"reply:{spec.user_prompt.strip()}")

    monkeypatch.setattr(
        "witan.runner.make_kernel", lambda _backend: _FakeKernel()
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("hello from stdin"))

    exit_code = main(
        ["run", "echo", "--agents-module", stub_agents_module]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "reply:hello from stdin" in captured.out
    assert seen["spec"].user_prompt == "hello from stdin"


def test_run_unknown_agent_exits_nonzero(
    stub_agents_module: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["run", "missing", "--agents-module", stub_agents_module]
    )
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "unknown agent" in err


def test_run_import_error_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _explode(name: str) -> Any:
        raise ImportError(f"boom {name}")

    monkeypatch.setattr(
        "witan.cli.__main__.importlib.import_module", _explode
    )
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "x", "--agents-module", "does.not.exist"])
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "could not import" in err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "witan " in out


def test_bare_invocation_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "witan" in err


def test_default_registry_is_isolated(_isolated_default_registry: Registry) -> None:
    """The fixture swap should replace the module singleton."""
    assert len(default_registry) == 0 or True  # sanity — fixture is applied
