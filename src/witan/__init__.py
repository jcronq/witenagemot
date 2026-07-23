"""Witenagemot — a multi-agent framework.

Public API surface. Import everything you need from the top level:

.. code-block:: python

    from witan import (
        Agent, AgentSpec, KernelSpec, ToolPolicy, BehavioralRule,
        OutputSchema, PolicyViolation, KernelResult, Registry,
        default_registry, run_agent, Channel, Turn, StdChannel,
    )

Submodules (:mod:`witan.kernels`, :mod:`witan.channels`,
:mod:`witan.cli`) exist so implementations can grow without cluttering
this namespace, but the top-level names above are the stable v0.1
surface.
"""

from __future__ import annotations

from .channels import Channel, StdChannel, Turn
from .kernel import Kernel, make_kernel
from .registry import Registry, default_registry
from .runner import run_agent
from .types import (
    Agent,
    AgentSpec,
    BehavioralRule,
    KernelResult,
    KernelSpec,
    OutputSchema,
    PolicyViolation,
    ThinkingLevel,
    ToolPolicy,
    UsageInfo,
)

__version__ = "0.1.0"


__all__ = [
    "Agent",
    "AgentSpec",
    "BehavioralRule",
    "Channel",
    "Kernel",
    "KernelResult",
    "KernelSpec",
    "OutputSchema",
    "PolicyViolation",
    "Registry",
    "StdChannel",
    "ThinkingLevel",
    "ToolPolicy",
    "Turn",
    "UsageInfo",
    "__version__",
    "default_registry",
    "make_kernel",
    "run_agent",
]
