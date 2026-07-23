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

The backend-agnostic LLM invocation layer. Every backend takes the same `KernelSpec`, returns the same `KernelResult`, and emits the same `TurnSummary`/`SystemEvent` stream. Provider-specific types (Anthropic's `ResultMessage`, OpenAI's response objects, local model responses) are translated by each backend before crossing the abstraction boundary.

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

# Wire a transport
channel = SignalChannel(account="+15555550000")

# Run
async for turn in channel.receive_loop():
    result = await run_agent("triage", turn=turn)
    await channel.send(result.text, reply_to=turn)
```

Same code, `Channel = SurfaceChannel(inbox_dir=...)` — agent now runs on inter-agent surface protocol instead of Signal. No agent code changes.

## The runner

`run_agent(name, turn, backend=None)` is the entry point. It:

1. Resolves the agent from the registry
2. Applies `ToolPolicy` to trim the requested tool set (raises `PolicyViolation` if it empties)
3. Injects active `BehavioralRule`s into the system prompt
4. Picks a `Kernel` impl from `backend` (or `_default_backend_for_agent`)
5. Dispatches one turn
6. Applies `OutputSchema` validation if set
7. Returns `KernelResult`

The runner is thin. The whole point is that dispatch mechanics don't leak into agent authoring.

## Ergonomic surface

### Python API

Direct import as shown above. First-class citizen.

### CLI

```
witan serve <agent>       # start the agent as a long-running Signal listener
witan run <agent> <input> # run a single turn from stdin
witan list-agents         # what's in the registry
witan validate            # lint agents/*.yaml
```

### Config files

Agents defined in YAML under `agents/*.yaml`:

```yaml
name: triage
persona: customer-support-tier-1
kernel:
  model: claude-sonnet-4-6
  allowed_tools: [read_file, search_kb, send_message]
tool_policy:
  type: allow
  allowlist: [read_file, search_kb, send_message]
behavioral_rules:
  - id: terse
    injection: "Reply in <= 3 sentences."
```

Same shape as the dataclass. YAML is the canonical persistence format; Python is the runtime API.

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
- `SignalChannel` + `StdChannel`
- CLI: `witan serve`, `witan run`
- One example: a Signal echo agent

**v0.2** — OpenAI + local backends. `SurfaceChannel`. YAML config loading. Two more examples (dispatcher pattern, agent-to-agent handoff).

**v0.3** — `OutputSchema` validation. HTTP channel. Docs.

**v1.0** — stable API. Full test coverage. Production-ready for the automated-company use case.

## Open design questions for Jason

1. **License** — MIT is proposed; is that right, or private-then-open, or Apache-2 for patent clauses?
2. **Repo visibility** — public jcronq/witenagemot or private for now?
3. **Naming inside the lib** — `witan` as the package name; but do we want the "witan" concept surfaced in class names too (`Councilor` instead of `Agent`? `Witenagemot` instead of `Registry`?) or is that too cute?
4. **Config format** — YAML above. Fine, or prefer TOML / Python-only?
5. **Base model default** — what should agents get if they don't specify a model? `claude-sonnet-4-6` is a reasonable current default; anything else?
6. **The channel-plurality question** — a single agent might read from multiple channels (Signal + Surface). Does the runner handle multi-channel dispatch, or is it the user's job to compose channels?

Once these land, v0.1 implementation starts.
