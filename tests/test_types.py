"""Tests for :mod:`witan.types`."""

from __future__ import annotations

import pytest

from witan import (
    AgentSpec,
    BehavioralRule,
    KernelSpec,
    OutputSchema,
    PolicyViolation,
    ToolPolicy,
)


def test_kernel_spec_requires_non_empty_model() -> None:
    with pytest.raises(ValueError, match="required"):
        KernelSpec(model="")  # type: ignore[arg-type]


def test_kernel_spec_missing_model_raises() -> None:
    # Positional / keyword omission is a TypeError from the dataclass
    # signature — but empty string is what most callers actually
    # supply (env var not set), so we surface it as ValueError above.
    with pytest.raises(TypeError):
        KernelSpec()  # type: ignore[call-arg]


def test_kernel_spec_defaults() -> None:
    spec = KernelSpec(model="claude-haiku-4-5")
    assert spec.model == "claude-haiku-4-5"
    assert spec.allowed_tools == ()
    assert spec.max_seconds == 0
    assert spec.append_system_prompt is None
    assert spec.thinking is None


def test_kernel_spec_has_no_user_prompt_field() -> None:
    """Prompt is per-turn — it belongs on Kernel.run(spec, prompt), not on the spec."""
    from dataclasses import fields

    spec = KernelSpec(model="claude-haiku-4-5")
    assert not hasattr(spec, "user_prompt")
    field_names = {f.name for f in fields(KernelSpec)}
    assert "user_prompt" not in field_names


def test_tool_policy_allow() -> None:
    policy = ToolPolicy(type="allow", allowlist=frozenset({"read", "write"}))
    assert policy.evaluate({"read", "delete"}) == {"read"}


def test_tool_policy_deny() -> None:
    policy = ToolPolicy(type="deny", denylist=frozenset({"delete"}))
    assert policy.evaluate({"read", "delete"}) == {"read"}


def test_behavioral_rule_render_includes_id() -> None:
    rule = BehavioralRule(id="terse", injection="Reply <= 3 sentences.")
    rendered = rule.render()
    assert "terse" in rendered
    assert "Reply <= 3 sentences." in rendered


def test_output_schema_stores_json() -> None:
    schema = OutputSchema(schema_json={"type": "object"})
    assert schema.schema_json == {"type": "object"}


def _make_spec(**overrides: object) -> AgentSpec:
    base: dict[str, object] = {
        "name": "test",
        "persona": "unit",
        "runtime": "short-lived",
        "kernel": KernelSpec(model="claude-haiku-4-5"),
    }
    base.update(overrides)
    return AgentSpec(**base)  # type: ignore[arg-type]


def test_agent_spec_build_kernel_spec_no_constraints() -> None:
    agent = _make_spec()
    effective = agent.build_kernel_spec()
    assert effective.model == "claude-haiku-4-5"
    assert effective.allowed_tools == ()
    assert effective.append_system_prompt is None


def test_agent_spec_applies_tool_policy() -> None:
    agent = _make_spec(
        kernel=KernelSpec(
            model="claude-haiku-4-5",
            allowed_tools=("read", "write", "delete"),
        ),
        tool_policy=ToolPolicy(
            type="allow", allowlist=frozenset({"read", "write"})
        ),
    )
    effective = agent.build_kernel_spec()
    assert effective.allowed_tools == ("read", "write")


def test_agent_spec_policy_violation_when_policy_drains_tools() -> None:
    agent = _make_spec(
        kernel=KernelSpec(
            model="claude-haiku-4-5",
            allowed_tools=("read",),
        ),
        tool_policy=ToolPolicy(
            type="allow", allowlist=frozenset({"nothing"})
        ),
    )
    with pytest.raises(PolicyViolation):
        agent.build_kernel_spec()


def test_agent_spec_merges_behavioral_rules_into_system_prompt() -> None:
    agent = _make_spec(
        kernel=KernelSpec(
            model="claude-haiku-4-5",
            append_system_prompt="Base system.",
        ),
        behavioral_rules=(
            BehavioralRule(id="terse", injection="Be terse."),
            BehavioralRule(id="polite", injection="Be polite."),
        ),
    )
    effective = agent.build_kernel_spec()
    assert effective.append_system_prompt is not None
    assert "Base system." in effective.append_system_prompt
    assert "terse" in effective.append_system_prompt
    assert "polite" in effective.append_system_prompt


def test_agent_spec_empty_prompt_returns_none() -> None:
    agent = _make_spec()
    assert agent.assembled_system_prompt() is None


def test_agent_spec_is_frozen() -> None:
    agent = _make_spec()
    # frozen dataclass raises FrozenInstanceError, a subclass of AttributeError
    with pytest.raises(AttributeError):
        agent.name = "renamed"  # type: ignore[misc]


def test_policy_violation_is_runtime_error() -> None:
    assert issubclass(PolicyViolation, RuntimeError)
