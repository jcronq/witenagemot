# Witenagemot

*Old English: "meeting of the wise." A council of counselors, gathered to decide.*

A Python framework for building multi-agent systems: mix and match agents and models, with transport built in. Designed to scale from a single scripted agent to an entire automated organization.

## Status

Pre-alpha. Design conversation in progress; see `docs/DESIGN.md`.

## The pitch

Most agent frameworks focus on one thing: a single LLM call with tools. Witenagemot is designed around three primitives that make multi-agent systems tractable:

- **`AgentSpec`** — a declarative description of an agent: which model, which tools, what behavioral constraints, what output shape. Composable, registerable, portable across backends.
- **`Kernel`** — a backend-agnostic LLM invocation layer. One `KernelSpec` in, one `KernelResult` out. Add a new provider by writing a translator; agent code never changes.
- **`Channel`** — a transport abstraction that lets agents talk to humans (Signal, HTTP, stdin) and to each other (surface passing, structured inbox). Same interface either way.

## Quickstart

Coming soon.

## Design origin

The `AgentSpec` + `Kernel` + `Registry` + `runner` pattern is extracted from and inspired by Alice, a personal AI assistant built by Jason Cronquist. Alice's `core/agent_library/` is where these abstractions were originally proven; Witenagemot productizes them as a standalone library with a first-class transport layer added on top.

## License

MIT.
