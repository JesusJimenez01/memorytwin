"""
Tests for the `mt` command-line interface
=========================================

Covers argument parsing and the non-destructive `mt setup` command.
"""

import json
from argparse import Namespace

import pytest

from memorytwin.escriba import cli


class TestParser:
    """Tests for the argument parser."""

    def test_capture_accepts_positional_text(self):
        args = cli.build_parser().parse_args(["capture", "Chose FastAPI for async", "-p", "api"])

        assert args.command == "capture"
        assert args.text == "Chose FastAPI for async"
        assert args.project == "api"

    def test_consolidate_requires_project(self):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["consolidate"])

    def test_every_command_has_a_handler(self):
        parser = cli.build_parser()
        subparsers = next(a for a in parser._actions if a.dest == "command")

        assert set(subparsers.choices) == set(cli.HANDLERS)

    def test_no_command_prints_help(self, capsys):
        cli.main([])

        assert "usage: mt" in capsys.readouterr().out


class TestSetup:
    """Tests for `mt setup`."""

    def _run_setup(self, path, force=False):
        cli.handle_setup(Namespace(path=str(path), force=force))

    def test_creates_project_files(self, tmp_path):
        self._run_setup(tmp_path)

        assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8").startswith("# Memory Twin")
        assert "OPENROUTER_API_KEY" in (tmp_path / ".env").read_text(encoding="utf-8")
        config = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))
        assert config["servers"]["memorytwin"]["type"] == "stdio"
        gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
        assert ".env" in gitignore and "data/" in gitignore

    def test_preserves_existing_files(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("# My own agent rules\n", encoding="utf-8")
        (tmp_path / ".env").write_text("SECRET=1\n", encoding="utf-8")
        (tmp_path / ".vscode").mkdir()
        (tmp_path / ".vscode" / "mcp.json").write_text(
            json.dumps({"servers": {"other": {"command": "other-server", "type": "stdio"}}}),
            encoding="utf-8",
        )

        self._run_setup(tmp_path)

        assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8") == "# My own agent rules\n"
        assert (tmp_path / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
        servers = json.loads((tmp_path / ".vscode" / "mcp.json").read_text(encoding="utf-8"))["servers"]
        assert set(servers) == {"other", "memorytwin"}

    def test_force_overwrites_agents_file(self, tmp_path):
        (tmp_path / "AGENTS.md").write_text("old", encoding="utf-8")

        self._run_setup(tmp_path, force=True)

        assert (tmp_path / "AGENTS.md").read_text(encoding="utf-8").startswith("# Memory Twin")

    def test_gitignore_is_only_extended_once(self, tmp_path):
        (tmp_path / ".gitignore").write_text("node_modules\n", encoding="utf-8")

        assert cli._ensure_gitignore(tmp_path) == "updated"
        assert cli._ensure_gitignore(tmp_path) is None

        content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
        assert content.count("data/") == 1
        assert content.startswith("node_modules\n")

    def test_invalid_mcp_json_is_not_overwritten(self, tmp_path):
        mcp_path = tmp_path / ".vscode" / "mcp.json"
        mcp_path.parent.mkdir()
        mcp_path.write_text("{ not json", encoding="utf-8")

        with pytest.raises(ValueError):
            cli._merge_mcp_config(mcp_path, "mt", ["mcp"])

        assert mcp_path.read_text(encoding="utf-8") == "{ not json"
