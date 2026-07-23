"""Agent registry — name → :class:`AgentSpec` lookup.

v0.1 keeps the registry as an in-memory dict. Callers use the
module-level :data:`default_registry` singleton; tests can build a
private :class:`Registry` for isolation. A file-based loader that
hydrates the same interface from TOML lands in v0.2.
"""

from __future__ import annotations

from .types import AgentSpec

__all__ = ["Registry", "default_registry"]


class Registry:
    """A small in-process name → :class:`AgentSpec` store.

    Not thread-safe by design — registration happens at import time
    on a single thread. Concurrent registration would just wrap the
    dict in a :class:`threading.Lock`; nothing in v0.1 exercises
    that path.
    """

    def __init__(self) -> None:
        self._specs: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        """Register ``spec`` under :attr:`AgentSpec.name`.

        Raises :class:`ValueError` on duplicate names so a typo in
        the agents module surfaces at import time rather than as a
        silent overwrite.
        """
        if spec.name in self._specs:
            raise ValueError(
                f"agent {spec.name!r} is already registered; "
                f"call Registry.replace if overwrite is intentional"
            )
        self._specs[spec.name] = spec

    def replace(self, spec: AgentSpec) -> None:
        """Overwrite an existing registration.

        Tests use this to substitute a spec for a single test case
        without rebuilding the whole registry; production code
        should treat this as a knowingly-destructive operation.
        """
        self._specs[spec.name] = spec

    def get(self, name: str) -> AgentSpec:
        """Look up an :class:`AgentSpec` by name.

        Raises :class:`KeyError` for unknown names — the registry is
        an explicit allowlist, not an open namespace.
        """
        if name not in self._specs:
            known = ", ".join(sorted(self._specs)) or "(none)"
            raise KeyError(
                f"no agent registered as {name!r} (known: {known})"
            )
        return self._specs[name]

    def names(self) -> list[str]:
        """Return registered agent names in sorted order."""
        return sorted(self._specs)

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)


default_registry = Registry()
"""Process-wide default registry.

Agents modules populate this at import time; :func:`witan.run_agent`
falls back to it when no explicit registry is passed.
"""
