"""Core dataclasses for the witan public surface.

The four primitives (:class:`AgentSpec`, :class:`ToolPolicy`,
:class:`BehavioralRule`, :class:`OutputSchema`) and the kernel-side
value objects (:class:`KernelSpec`, :class:`KernelResult`,
:class:`UsageInfo`) all live here so the top-level ``witan`` package
can re-export them from a single module. Everything is a frozen
dataclass so registered specs are safe to share across the process
without accidental mutation. Behavior lives in
:mod:`witan.runner` and the concrete kernels.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

__all__ = [
    "AgentSpec",
    "BehavioralRule",
    "KernelResult",
    "KernelSpec",
    "OutputSchema",
    "PolicyViolation",
    "ThinkingLevel",
    "ToolPolicy",
    "UsageInfo",
]


ThinkingLevel = Literal["off", "minimal", "low", "medium", "high"]
"""Normalized reasoning-effort levels; backends translate to native shape."""

RuntimeKind = Literal["long-running", "short-lived", "batch"]
"""Coarse hint about how an agent is expected to be invoked."""

ToolPolicyType = Literal["allow", "deny"]
"""``allow`` treats :attr:`ToolPolicy.allowlist` as the permitted set;
``deny`` treats :attr:`ToolPolicy.denylist` as the blocked set."""


class PolicyViolation(RuntimeError):  # noqa: N818 — public API name is set by the design doc
    """Raised when :class:`AgentSpec` constraints leave no tools available."""


@dataclass(frozen=True)
class ToolPolicy:
    """Pre-execution tool filter — either an allowlist or a denylist.

    :meth:`evaluate` is pure: it takes the set of tools the kernel spec
    requested and returns the subset permitted under this policy.
    The runner surfaces a :class:`PolicyViolation` if the resulting
    set is empty so callers can decide between falling back and
    failing loud.
    """

    type: ToolPolicyType
    allowlist: frozenset[str] = field(default_factory=frozenset)
    denylist: frozenset[str] = field(default_factory=frozenset)

    def evaluate(self, requested: set[str]) -> set[str]:
        """Return the subset of ``requested`` allowed by this policy."""
        if self.type == "allow":
            return set(requested) & set(self.allowlist)
        return set(requested) - set(self.denylist)


@dataclass(frozen=True)
class BehavioralRule:
    """Prompt-level constraint injected into the system prompt.

    ``id`` surfaces as a heading in the rendered prompt so an operator
    reading the assembled text can see which rules are active.
    ``injection`` is raw text — no templating. In v0.1 only
    ``condition="always"`` is supported; conditional injection lands
    later.
    """

    id: str
    injection: str
    condition: Literal["always"] = "always"

    def render(self) -> str:
        """Return the text block this rule contributes to the prompt."""
        return f"## Constraint: {self.id}\n{self.injection}".rstrip()


@dataclass(frozen=True)
class OutputSchema:
    """Post-execution output validation config.

    In v0.1 this is a placeholder — the JSON schema is stored on the
    spec so the registry records the expected output shape, but no
    validation runs. Actual validation wires in during v0.3.
    """

    schema_json: dict


@dataclass(frozen=True)
class UsageInfo:
    """Normalized token-usage summary.

    Field names mirror Anthropic's wire format so downstream event
    consumers can parse a single shape regardless of backend. Cache
    fields are optional because not every backend reports them.
    """

    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class KernelSpec:
    """Inputs to one :meth:`Kernel.run` invocation, backend-agnostic.

    ``model`` is required — construction fails loud if it's omitted
    or empty. Each backend translates the fields (``allowed_tools``,
    ``append_system_prompt``, ``thinking``) to its native option
    shape. ``max_seconds`` of 0 means unbounded.
    """

    model: str
    allowed_tools: tuple[str, ...] = ()
    max_seconds: int = 0
    append_system_prompt: str | None = None
    thinking: ThinkingLevel | None = None
    # The user-turn text for this dispatch. Populated by the runner
    # after :meth:`AgentSpec.build_kernel_spec` — kept on the spec so
    # the :class:`Kernel` Protocol's ``run(spec)`` signature is a
    # single-argument contract. Callers using the kernel directly
    # (tests, embedded use) can set this via :func:`dataclasses.replace`.
    user_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.model or not isinstance(self.model, str):
            raise ValueError(
                "KernelSpec.model is required and must be a non-empty string; "
                "witan has no default model by design."
            )


@dataclass(frozen=True)
class KernelResult:
    """What :meth:`Kernel.run` returns, backend-agnostic.

    ``text`` is the concatenated assistant text content. ``usage``
    may be ``None`` if the backend didn't surface usage for this
    call. ``raw`` is the native backend response, kept as an escape
    hatch — code above the kernel abstraction should not depend on
    its shape.
    """

    text: str
    usage: UsageInfo | None = None
    raw: Any = None


@dataclass(frozen=True)
class AgentSpec:
    """A named, constraint-wrapped configuration for one agent flavor.

    Four descriptive axes — ``persona``, ``runtime``, ``scope``,
    ``lifecycle`` — sit alongside the wrapped :class:`KernelSpec`.
    Three optional constraint layers (:class:`ToolPolicy`,
    :class:`BehavioralRule` tuple, :class:`OutputSchema`) are applied
    via :meth:`build_kernel_spec` to yield the effective spec that
    gets dispatched. Frozen; use :func:`dataclasses.replace` for
    derived specs.
    """

    name: str
    persona: str
    runtime: RuntimeKind
    kernel: KernelSpec
    scope: str = ""
    lifecycle: str = ""
    tool_policy: ToolPolicy | None = None
    behavioral_rules: tuple[BehavioralRule, ...] = ()
    output_schema: OutputSchema | None = None

    def effective_tools(self) -> tuple[str, ...]:
        """Return the tool tuple after :attr:`tool_policy` filtering.

        Sorted for determinism. Raises :class:`PolicyViolation` if
        the policy drains a non-empty requested set — dispatching a
        kernel with no tools when the caller asked for some is almost
        certainly a bug and should surface before the LLM call fires.
        """
        requested = set(self.kernel.allowed_tools)
        if self.tool_policy is None:
            return tuple(sorted(requested))
        effective = self.tool_policy.evaluate(requested)
        if requested and not effective:
            raise PolicyViolation(
                f"agent {self.name!r}: no tools available under "
                f"policy {self.tool_policy.type!r}"
            )
        return tuple(sorted(effective))

    def assembled_system_prompt(self) -> str | None:
        """Merge the kernel's base ``append_system_prompt`` with every
        active behavioral rule's rendered block. Returns ``None`` if
        the result is empty — passing ``""`` would still hit the
        backend as a real system-prompt append.
        """
        parts: list[str] = []
        base = self.kernel.append_system_prompt
        if base:
            parts.append(base.rstrip())
        for rule in self.behavioral_rules:
            if rule.condition == "always":
                parts.append(rule.render())
        if not parts:
            return None
        return "\n\n".join(parts)

    def build_kernel_spec(self) -> KernelSpec:
        """Return the effective :class:`KernelSpec` for dispatch.

        Applies the tool policy to :attr:`KernelSpec.allowed_tools`
        and merges behavioral rules into ``append_system_prompt``.
        Pure — does not mutate :attr:`kernel`.
        """
        return replace(
            self.kernel,
            allowed_tools=self.effective_tools(),
            append_system_prompt=self.assembled_system_prompt(),
        )


# Alias — some callers prefer the shorter name; the design doc uses both.
Agent = AgentSpec
