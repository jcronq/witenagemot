"""``witan`` CLI.

Three subcommands ship in v0.1:

- ``witan run <agent>`` — dispatch one turn read from stdin and
  print the result text to stdout.
- ``witan list-agents`` — list registered agents from an imported
  agents module.
- ``witan --version`` — print the installed package version.

Agents must be registered by importing a Python module named via
``--agents-module``. The module is expected to populate
:data:`witan.default_registry` at import time.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Sequence

from .. import __version__
from ..channels import StdChannel
from ..registry import default_registry
from ..runner import run_agent

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Kept factored out so tests can construct the parser and assert on
    its behavior without shelling into ``main``.
    """
    parser = argparse.ArgumentParser(
        prog="witan",
        description="Witenagemot — mix-and-match agent dispatch.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"witan {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Dispatch a single turn from stdin against the named agent.",
    )
    run_parser.add_argument("agent", help="Registered agent name.")
    run_parser.add_argument(
        "--agents-module",
        required=True,
        help="Python module to import that registers agents.",
    )
    run_parser.add_argument(
        "--backend",
        default="anthropic",
        help="Kernel backend to dispatch through (default: anthropic).",
    )

    list_parser = subparsers.add_parser(
        "list-agents",
        help="Print registered agent names.",
    )
    list_parser.add_argument(
        "--agents-module",
        required=True,
        help="Python module to import that registers agents.",
    )

    return parser


def _load_agents_module(module_name: str) -> None:
    """Import ``module_name`` for its registration side-effects.

    Raises :class:`SystemExit` (exit code 2) with a friendly message
    if the module can't be imported so the operator sees a real error
    instead of an ImportError traceback in the middle of a pipe.
    """
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        print(
            f"witan: could not import agents module {module_name!r}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def _cmd_run(args: argparse.Namespace) -> int:
    _load_agents_module(args.agents_module)
    if args.agent not in default_registry:
        known = ", ".join(default_registry.names()) or "(none)"
        print(
            f"witan: unknown agent {args.agent!r} (known: {known})",
            file=sys.stderr,
        )
        return 2
    channel = StdChannel(mode="block")

    async def _run() -> int:
        async for turn in channel.receive():
            result = await run_agent(
                args.agent, turn=turn, backend=args.backend
            )
            await channel.send(result.text, reply_to=turn)
            return 0
        # Empty stdin — nothing to do.
        return 0

    return asyncio.run(_run())


def _cmd_list_agents(args: argparse.Namespace) -> int:
    _load_agents_module(args.agents_module)
    for name in default_registry.names():
        print(name)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse ``argv`` and dispatch to a subcommand handler.

    Returns the process exit code; ``__main__`` shim below wraps this
    in :func:`sys.exit`.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "list-agents":
        return _cmd_list_agents(args)
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover - trivial shim
    raise SystemExit(main())
