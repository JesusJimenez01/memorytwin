"""
Tests for the project analyzer (onboarding)
===========================================
"""

import json

import pytest

from memorytwin.escriba.project_analyzer import ProjectAnalyzer, _requirement_name


@pytest.mark.parametrize("spec, expected", [
    ("requests", "requests"),
    ("requests>=2.31", "requests"),
    ("pydantic[email]~=2.0", "pydantic"),
    ("tomli; python_version < '3.11'", "tomli"),
    ("mypkg @ git+https://github.com/org/mypkg", "mypkg"),
    ("  numpy!=1.0  ", "numpy"),
])
def test_requirement_name(spec, expected):
    assert _requirement_name(spec) == expected


@pytest.fixture
def sample_project(tmp_path):
    """A small Python + Docker project layout."""
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "node_modules").mkdir()  # must be ignored
    (tmp_path / "node_modules" / "junk.js").write_text("x")
    (tmp_path / "src" / "app.py").write_text("print('hi')")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12")
    (tmp_path / "README.md").write_text("# Sample\n\nA sample service.\n\n## Usage\nRun it.")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "sample"\ndescription = "Demo"\nrequires-python = ">=3.11"\n'
        'dependencies = ["fastapi>=0.110", "sqlalchemy[asyncio]>=2"]\n'
    )
    (tmp_path / "package.json").write_text(json.dumps({"name": "web", "dependencies": {"react": "^18"}}))
    return tmp_path


class TestProjectAnalyzer:
    """Tests for ProjectAnalyzer."""

    def test_analyze_detects_stack_and_dependencies(self, sample_project):
        analysis = ProjectAnalyzer(str(sample_project)).analyze()

        technologies = {item["technology"] for item in analysis["stack"]}
        assert {"Python", "Docker", "Node.js/JavaScript"} <= technologies
        assert {"fastapi", "sqlalchemy", "react"} <= set(analysis["dependencies"]["main"])
        assert analysis["config"]["project_info"]["name"] == "sample"
        assert "A sample service." in analysis["config"]["readme_summary"]
        assert "Usage" not in analysis["config"]["readme_summary"]

    def test_ignored_directories_are_skipped(self, sample_project):
        structure = ProjectAnalyzer(str(sample_project)).analyze()["structure"]

        assert "node_modules" not in structure["main_directories"]
        assert ".js" not in structure["file_types"]

    def test_onboarding_text_mentions_key_facts(self, sample_project):
        analyzer = ProjectAnalyzer(str(sample_project))
        text = analyzer.generate_onboarding_text(analyzer.analyze())

        assert text.startswith(f"# Onboarding Analysis: {sample_project.name}")
        assert "## Technology Stack" in text
        assert "fastapi" in text
