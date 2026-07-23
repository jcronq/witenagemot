"""Tests for :mod:`witan.channels.std`."""

from __future__ import annotations

import io

from witan import StdChannel, Turn


async def test_line_mode_yields_one_turn_per_line() -> None:
    stdin = io.StringIO("hello\nworld\n")
    stdout = io.StringIO()
    channel = StdChannel(mode="line", stdin=stdin, stdout=stdout)
    turns = [t async for t in channel.receive()]
    assert [t.text for t in turns] == ["hello", "world"]


async def test_line_mode_skips_blank_lines() -> None:
    stdin = io.StringIO("hello\n\nworld\n")
    channel = StdChannel(mode="line", stdin=stdin, stdout=io.StringIO())
    turns = [t async for t in channel.receive()]
    assert [t.text for t in turns] == ["hello", "world"]


async def test_line_mode_terminates_on_eof() -> None:
    stdin = io.StringIO("")
    channel = StdChannel(mode="line", stdin=stdin, stdout=io.StringIO())
    turns = [t async for t in channel.receive()]
    assert turns == []


async def test_block_mode_yields_whole_buffer_as_one_turn() -> None:
    stdin = io.StringIO("line 1\nline 2\nline 3\n")
    channel = StdChannel(mode="block", stdin=stdin, stdout=io.StringIO())
    turns = [t async for t in channel.receive()]
    assert len(turns) == 1
    assert turns[0].text == "line 1\nline 2\nline 3\n"


async def test_block_mode_empty_stdin_yields_no_turns() -> None:
    channel = StdChannel(mode="block", stdin=io.StringIO(""), stdout=io.StringIO())
    turns = [t async for t in channel.receive()]
    assert turns == []


async def test_send_writes_text_with_newline() -> None:
    stdout = io.StringIO()
    channel = StdChannel(stdin=io.StringIO(""), stdout=stdout)
    await channel.send("hello world")
    assert stdout.getvalue() == "hello world\n"


async def test_send_does_not_double_newline() -> None:
    stdout = io.StringIO()
    channel = StdChannel(stdin=io.StringIO(""), stdout=stdout)
    await channel.send("has-newline\n")
    assert stdout.getvalue() == "has-newline\n"


async def test_send_reply_to_argument_is_accepted() -> None:
    stdout = io.StringIO()
    channel = StdChannel(stdin=io.StringIO(""), stdout=stdout)
    await channel.send("resp", reply_to=Turn(text="incoming"))
    assert stdout.getvalue() == "resp\n"
