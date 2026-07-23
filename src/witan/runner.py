"""Async runner for :class:`AgentSpec`.

:func:`run_agent` is the entry point for one-turn dispatch. It's the
thin layer between an :class:`AgentSpec` and the concrete
:class:`Kernel` backend: resolve the spec from the registry, apply
the constraint layers (tool policy + behavioral rules → effective
:class:`KernelSpec`), pick a backend via
:func:`witan.kernel.make_kernel`, wire the turn text onto the spec,
and dispatch.
"""

from __future__ import annotations

from dataclasses import replace

from .channels import Turn
from .kernel import make_kernel
from .registry import Registry, default_registry
from .types import KernelResult

__all__ = ["run_agent"]


async def run_agent(
    name: str,
    turn: Turn,
    backend: str = "anthropic",
    *,
    registry: Registry | None = None,
) -> KernelResult:
    """Dispatch one turn against the named agent.

    Resolves ``name`` from ``registry`` (or the module-level
    :data:`witan.default_registry` when omitted), applies the spec's
    tool policy and behavioral rules via
    :meth:`AgentSpec.build_kernel_spec`, threads ``turn.text`` onto
    the effective :class:`KernelSpec` as the user message, picks a
    :class:`Kernel` impl via :func:`witan.kernel.make_kernel`, awaits
    one turn, and returns the :class:`KernelResult`.

    Raises :class:`witan.PolicyViolation` from
    :meth:`AgentSpec.build_kernel_spec` when the tool policy leaves
    no tools available; the runner does not swallow it — the caller
    decides whether that's fatal or a fallback signal.
    """
    reg = registry if registry is not None else default_registry
    agent = reg.get(name)
    effective = agent.build_kernel_spec()
    effective = replace(effective, user_prompt=turn.text)
    kernel = make_kernel(backend)
    return await kernel.run(effective)
