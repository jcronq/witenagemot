"""Minimal registration example — a single-turn ``echo`` agent.

Run:

.. code-block:: bash

    echo "hello world" | python -m witan.cli run echo --agents-module examples.echo

Import-time side-effect: the module registers the ``echo`` agent on
:data:`witan.default_registry`. No orchestration happens on import —
the CLI drives dispatch.
"""

from __future__ import annotations

from witan import AgentSpec, BehavioralRule, KernelSpec, default_registry

default_registry.register(
    AgentSpec(
        name="echo",
        persona="test-echo",
        runtime="short-lived",
        kernel=KernelSpec(model="claude-haiku-4-5"),
        behavioral_rules=(
            BehavioralRule(
                id="echo-only",
                injection=(
                    "You are an echo test agent. Reply with exactly the user's "
                    "message and nothing else."
                ),
            ),
        ),
    )
)
