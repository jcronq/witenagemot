# CLAUDE.md — Witenagemot

## What this project is

A Python framework for multi-agent systems. Pre-alpha. See `docs/DESIGN.md` for the architectural blueprint.

## Design origin

The core abstractions (`AgentSpec`, `Kernel`, `Registry`, `runner`) are extracted from and inspired by Alice at `~/alice/src/core/`. Read that first if you're going to touch the agent library — it's the reference implementation this project is productizing.

## Development notes

- Python 3.11+
- `pyproject.toml` uses uv-compatible layout; `uv sync` should Just Work once uv is installed
- Package name is `witan`; project name is `witenagemot`. All CLI, imports, and module paths use the short name
- Layout is src-based (`src/witan/`) — don't hoist packages to the root

## Design conversation

Design decisions land in `docs/DESIGN.md` before code. If you're proposing an abstraction change, update the design doc in the same PR.

## No source code yet

As of first commit, this is scaffold only. Implementation is gated on the design doc's review with Jason.
