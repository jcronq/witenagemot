# Witenagemot — Design

> Meeting of the wise. A council of counselors, gathered to decide.

## Vision

Build a Python framework that makes multi-agent systems tractable. The concrete north star: someone should be able to compose a small collection of `witan` agents into a working automated organization — customer support, code review, ops rotation, research pipelines — without rebuilding the transport, the provider abstraction, or the agent lifecycle each time.

Constraints:

- **Mix and match agents and models.** An agent isn't tied to a model; a model isn't tied to an agent. Adding a new model provider is a translator, not a rewrite. Adding a new agent is a config.
- **Transport is first-class.** Most frameworks stop at "call an LLM with tools." Witan includes a Channel abstraction that lets agents talk to humans (Signal, HTTP, stdin) and to each other (surface passing, structured inbox) with the same interface.
- **Composable but not magical.** The primitives are dataclasses and protocols. If you can read the type signatures, you can wire the system.

## Core primitives

There are exactly four concepts. The whole framework is these four and how they compose.

### 1. `AgentSpec`

A declarative description of an agent. Four axes:

- **persona** — the role (e.g., "customer-support-tier-1", "code-reviewer")
- **runtime** — long-running / short-lived / batch
- **scope** — what tools it can call, what surfaces it can read, what channels it can send on
- **lifecycle** — how a turn starts and ends (idle-flush semantics, session boundaries)

Plus three constraint layers:

- **`ToolPolicy`** — allowlist or denylist over tool names. Pre-execution filter; a violation surfaces before the LLM call fires.
- **`BehavioralRule`** — prompt-level constraints injected into the system prompt. Rules carry an `id` so an operator reading the rendered prompt can see which rules are active.
- **`OutputSchema`** — post-execution validation of the agent's output shape. Optional; enables downstream code to trust structured returns.

The AgentSpec wraps a `KernelSpec` — the runtime doesn't replace it, it decorates it.

### 2. `Kernel`

The backend-agnostic LLM invocation layer. Every backend takes the same `(KernelSpec, prompt)` pair — spec is the *durable* configuration (model + allowed tools + system-prompt appends + thinking level), reusable across many turns; `prompt` is the *per-turn* user message text. The split keeps a single `KernelSpec` shareable across a session without reconstructing it every dispatch. Every backend returns the same `KernelResult` and emits the same `TurnSummary`/`SystemEvent` stream. Provider-specific types (Anthropic's `ResultMessage`, OpenAI's response objects, local model responses) are translated by each backend before crossing the abstraction boundary.

The protocol signature is `async def run(self, spec: KernelSpec, prompt: str) -> KernelResult`.

**Backends shipped v0.1:**
- `anthropic` — via the Claude Agent SDK
- `openai` — via the openai SDK
- `local` — via any OpenAI-compatible endpoint (llama.cpp server, ollama, vllm)

Adding a new backend = writing a translator. Agent code, runner code, and Channel code never see backend-native types.

### 3. `Channel`

The transport abstraction. A Channel is anything an agent can send messages to or receive messages from — a human, another agent, a pub/sub topic. All Channels expose the same interface:

- `receive() -> Turn` — get the next inbound turn (blocking or async)
- `send(message, ...)` — send an outbound message
- `close_flush(reason)` — an idle/close signal for cleanup

Two Channel classes shipped v0.1:

- **`SignalChannel`** — human interface via signal-cli (the abstraction Alice already uses in production)
- **`SurfaceChannel`** — agent-to-agent structured file passing (inspired by Alice's `inner/surface/` + `inner/notes/` inbox pattern)

And a stub:

- **`StdChannel`** — stdin/stdout, for local dev

The unified interface is the point. A team of agents wired via `SurfaceChannel` behaves like a team wired via SignalChannel wired via HTTP — same primitives, different transport.

### 4. `Registry`

A name → `AgentSpec` lookup. Ships as an in-memory dict with `register()` / `get()` / `replace()` / `names()`. Explicit allowlist, no open namespace.

Later phase: a file-based loader that hydrates the registry from `agents/*.yaml` files without changing the interface.

## How they compose

```python
from witan import AgentSpec, Registry, run_agent
from witan.channels import SignalChannel
from witan.kernels import anthropic

# Register an agent
default_registry.register(AgentSpec(
    name="triage",
    persona="customer-support-tier-1",
    tool_policy=ToolPolicy.allow({"read_file", "search_kb", "send_message"}),
    behavioral_rules=[
        BehavioralRule(id="terse", injection="Reply in <= 3 sentences."),
    ],
    kernel=KernelSpec(model="claude-sonnet-4-6", ...),
))

# Wire a transport (v0.1 ships StdChannel; SignalChannel lands v0.2)
channel = StdChannel()

# User composes the loop — no framework magic
async for turn in channel.receive():
    result = await run_agent("triage", turn=turn)
    await channel.send(result.text, reply_to=turn)
```

Same shape works for any `Channel`. To fan in from multiple channels, users write their own merge loop with `asyncio.wait` or a helper library — Witan doesn't hide it (see the Decisions section).

## The runner

`run_agent(name, turn, backend=None)` is the entry point. It:

1. Resolves the agent from the registry
2. Applies `ToolPolicy` to trim the requested tool set (raises `PolicyViolation` if it empties)
3. Injects active `BehavioralRule`s into the system prompt
4. Picks a `Kernel` impl from `backend` (or `_default_backend_for_agent`)
5. Dispatches one turn via `kernel.run(effective_spec, turn.text)` — the durable spec plus the per-turn prompt as separate arguments
6. Applies `OutputSchema` validation if set
7. Returns `KernelResult`

The runner is thin. The whole point is that dispatch mechanics don't leak into agent authoring.

## Ergonomic surface

### Python API

Direct import as shown above. First-class citizen.

### CLI

```
witan run <agent>         # v0.1: run a single turn from stdin
witan serve <agent>       # v0.2: long-running listener on a chosen channel
witan list-agents         # what's in the registry
witan validate            # lint agents/*.toml
```

### Config files

Agents defined in TOML under `agents/*.toml`:

```toml
name = "triage"
persona = "customer-support-tier-1"

[kernel]
model = "claude-sonnet-4-6"   # required — no default; must be explicit
allowed_tools = ["read_file", "search_kb", "send_message"]

[tool_policy]
type = "allow"
allowlist = ["read_file", "search_kb", "send_message"]

[[behavioral_rules]]
id = "terse"
injection = "Reply in <= 3 sentences."
```

Same shape as the dataclass. TOML is the canonical persistence format; Python is the runtime API.

## What NOT to build (yet)

Explicit non-goals for v0.1 so scope doesn't creep:

- No orchestrator / graph engine. Compose agents by calling `run_agent` from other agents' tools. If a graph engine turns out to be needed, it goes in v1.x on top of the primitives, not into them.
- No RAG / vector store abstractions. That's an infra choice for the user; witan agents can call retrieval as a tool, but witan doesn't ship a vector store.
- No LangChain-style prompt template DSL. `BehavioralRule.injection` is raw text.
- No streaming interfaces in v0.1 — `run_agent` returns after the turn completes. Streaming lands when the backends' streaming primitives stabilize.

## Roadmap

**v0.0 (scaffold — this branch)** — repo layout, pyproject, README, DESIGN, no code.

**v0.1 (MVP)** — core primitives extracted fresh from Alice's design:
- `AgentSpec`, `ToolPolicy`, `BehavioralRule`, `OutputSchema`
- `Kernel` protocol + anthropic backend
- `Registry` + `run_agent`
- `StdChannel` for local dev
- CLI: `witan run` (one-shot from stdin)
- One example: a stdin echo agent that demonstrates the primitives

**v0.2** — OpenAI + local backends. `SignalChannel` + `SurfaceChannel`. TOML config loading. Two more examples (dispatcher pattern, agent-to-agent handoff). `witan serve`.

**v0.3** — `OutputSchema` validation. HTTP channel. Convenience helper `MultiChannelRunner` if user demand shows the merge pattern is common. Docs site.

**v1.0** — stable API. Full test coverage. Production-ready for the automated-company use case.

## Decisions (2026-07-23)

Answered by Jason:

1. **License:** MIT.
2. **Visibility:** Public. Live at https://github.com/jcronq/witenagemot.
3. **Class naming:** Plain — `Agent`, `Registry`, `Kernel`, `Channel`. Theme lives in the project name and docs, not the code surface.
4. **Config format:** TOML. Same file family as `pyproject.toml`; no extra parser dep.
5. **Default model:** None. Every `AgentSpec` must declare its model explicitly. Fails loud if omitted.
6. **Multi-channel dispatch:** User composes. Channels expose async iterators; users write their own merge loop. A `MultiChannelRunner` convenience helper is v0.3 territory, added only if demand emerges.

## Open questions

None currently blocking v0.1. Adjustments will land as commits to this doc alongside code that motivates them.
