"""Kernel Protocol and backend factory.

The :class:`Kernel` Protocol is the contract every backend must
satisfy. Agent code (:mod:`witan.runner`, user code) depends only on
this interface — never on a concrete impl. :func:`make_kernel`
resolves a backend name to an instance.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import KernelResult, KernelSpec

__all__ = ["Kernel", "make_kernel"]


@runtime_checkable
class Kernel(Protocol):
    """Backend-agnostic single-turn LLM invocation contract.

    Impls translate the :class:`KernelSpec` fields into their native
    request shape, run one turn, and return a normalized
    :class:`KernelResult`. Streaming is not part of this contract in
    v0.1 — implementations block until the turn completes.
    """

    async def run(self, spec: KernelSpec) -> KernelResult:
        """Run one turn against this backend and return the result."""
        ...


def make_kernel(backend: str) -> Kernel:
    """Resolve ``backend`` (e.g. ``"anthropic"``) to a :class:`Kernel`.

    v0.1 ships only the anthropic backend. Unknown backend names
    raise :class:`ValueError` with the list of supported names so the
    error surfaces during config loading rather than at dispatch
    time.
    """
    if backend == "anthropic":
        # Local import so the factory module stays importable without
        # the backend's transitive deps loaded — useful for tests and
        # for tools that only need the type surface.
        from .kernels.anthropic import AnthropicKernel

        return AnthropicKernel()
    raise ValueError(
        f"unknown kernel backend {backend!r}; supported: 'anthropic'"
    )
