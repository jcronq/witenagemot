"""CLI entry point for witan.

Invoke via ``python -m witan.cli`` or the ``witan`` script installed
by the package. See :mod:`witan.cli.__main__` for command handlers.
"""

from __future__ import annotations


def main(*args: object, **kwargs: object) -> int:
    """Thin re-export of :func:`witan.cli.__main__.main`.

    Imported lazily so ``python -m witan.cli`` does not warn about
    ``witan.cli.__main__`` being in :data:`sys.modules` twice.
    """
    from .__main__ import main as _main

    return _main(*args, **kwargs)  # type: ignore[arg-type]


__all__ = ["main"]
