"""
Project Analyzer - Initial analysis of existing projects
========================================================

Generates an "onboarding" episode by analyzing the structure,
technology stack, and conventions of an existing project.
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Fix encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

console = Console()


# File and directory patterns to ignore
IGNORE_PATTERNS = {
    'directories': {
        '.git', '.venv', 'venv', 'env', '__pycache__', 'node_modules',
        '.next', '.nuxt', 'dist', 'build', '.idea', '.vscode', '.github',
        'coverage', '.pytest_cache', '.mypy_cache', '.tox', 'egg-info',
        '.eggs', 'site-packages', 'target', 'bin', 'obj', '.gradle'
    },
    'files': {
        '.DS_Store', 'Thumbs.db', '.env', '.env.local', '*.pyc', '*.pyo',
        '*.lock', 'package-lock.json', 'yarn.lock', 'poetry.lock'
    }
}

# Technology stack detection by characteristic files
STACK_INDICATORS = {
    # Python
    'pyproject.toml': ('Python', 'Modern project configuration'),
    'setup.py': ('Python', 'Traditional setup'),
    'requirements.txt': ('Python', 'pip dependencies'),
    'Pipfile': ('Python', 'Pipenv'),
    'poetry.lock': ('Python', 'Poetry'),
    'manage.py': ('Django', 'Python web framework'),

    # JavaScript/TypeScript
    'package.json': ('Node.js/JavaScript', 'NPM project'),
    'tsconfig.json': ('TypeScript', 'TypeScript configuration'),
    'next.config.js': ('Next.js', 'React SSR framework'),
    'next.config.mjs': ('Next.js', 'React SSR framework'),
    'nuxt.config.js': ('Nuxt.js', 'Vue SSR framework'),
    'vite.config.js': ('Vite', 'Modern build tool'),
    'vite.config.ts': ('Vite', 'Modern build tool'),
    'angular.json': ('Angular', 'SPA framework'),
    'vue.config.js': ('Vue.js', 'Progressive framework'),

    # Java/Kotlin
    'pom.xml': ('Java/Maven', 'Maven build'),
    'build.gradle': ('Java/Kotlin/Gradle', 'Gradle build'),

    # .NET
    '*.csproj': ('C#/.NET', '.NET project'),
    '*.sln': ('C#/.NET', 'Visual Studio solution'),

    # Go
    'go.mod': ('Go', 'Go module'),

    # Rust
    'Cargo.toml': ('Rust', 'Cargo project'),

    # Docker/Infra
    'Dockerfile': ('Docker', 'Containerization'),
    'docker-compose.yml': ('Docker Compose', 'Container orchestration'),
    'docker-compose.yaml': ('Docker Compose', 'Container orchestration'),
    'kubernetes': ('Kubernetes', 'K8s orchestration'),

    # CI/CD
    '.github/workflows': ('GitHub Actions', 'CI/CD'),
    '.gitlab-ci.yml': ('GitLab CI', 'CI/CD'),
    'Jenkinsfile': ('Jenkins', 'CI/CD'),
}

# Common architecture patterns
ARCHITECTURE_PATTERNS = {
    'src/': 'Separated source code',
    'lib/': 'Internal libraries',
    'tests/': 'Unit tests',
    'test/': 'Unit tests',
    'spec/': 'Tests (BDD style)',
    'docs/': 'Documentation',
    'scripts/': 'Utility scripts',
    'config/': 'Separated configuration',
    'migrations/': 'Database migrations',
    'api/': 'API layer',
    'models/': 'Data models',
    'controllers/': 'Controllers (MVC)',
    'views/': 'Views (MVC)',
    'services/': 'Business services',
    'repositories/': 'Data access layer',
    'domain/': 'Domain logic (DDD)',
    'infrastructure/': 'Infrastructure (DDD)',
    'application/': 'Application layer (DDD)',
    'components/': 'Components (Frontend)',
    'pages/': 'Pages (Next.js/Nuxt)',
    'hooks/': 'Custom hooks (React)',
    'utils/': 'Utilities',
    'helpers/': 'Helpers',
}


class ProjectAnalyzer:
    """Analyzes the structure and characteristics of an existing project."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.project_name = self.project_path.name

    def analyze(self) -> dict:
        """
        Run full project analysis.

        Returns:
            Dictionary with the structured analysis
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing project...", total=None)

            # Gather information
            progress.update(task, description="Scanning structure...")
            structure = self._analyze_structure()

            progress.update(task, description="Detecting technology stack...")
            stack = self._detect_stack()

            progress.update(task, description="Identifying patterns...")
            patterns = self._identify_patterns()

            progress.update(task, description="Reading configuration...")
            config_info = self._read_config_files()

            progress.update(task, description="Analyzing dependencies...")
            dependencies = self._analyze_dependencies()

            progress.update(task, description="Detecting conventions...")
            conventions = self._detect_conventions()

        return {
            'project_name': self.project_name,
            'project_path': str(self.project_path),
            'structure': structure,
            'stack': stack,
            'patterns': patterns,
            'config': config_info,
            'dependencies': dependencies,
            'conventions': conventions
        }

    def _analyze_structure(self) -> dict:
        """Analyze directory and file structure."""
        dirs = []
        file_extensions = Counter()

        for root, directories, filenames in os.walk(self.project_path):
            # Filter ignored directories
            directories[:] = [d for d in directories if d not in IGNORE_PATTERNS['directories']
                             and not d.endswith('.egg-info')]

            rel_root = Path(root).relative_to(self.project_path)

            # Only first depth level for main directories
            if len(rel_root.parts) <= 1:
                for d in directories:
                    dirs.append(str(rel_root / d) if str(rel_root) != '.' else d)

            for f in filenames:
                if not any(f.endswith(ext) for ext in ['.pyc', '.pyo']):
                    ext = Path(f).suffix.lower()
                    if ext:
                        file_extensions[ext] += 1

        return {
            'main_directories': dirs[:20],  # Limit
            'file_types': dict(file_extensions.most_common(10)),
            'total_dirs': len(dirs),
        }

    def _detect_stack(self) -> list:
        """Detect the project's technology stack."""
        detected = []

        for indicator, (tech, description) in STACK_INDICATORS.items():
            if indicator.startswith('*'):
                # Glob pattern
                pattern = indicator
                if list(self.project_path.glob(pattern)):
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})
            elif '/' in indicator:
                # It's a path
                if (self.project_path / indicator).exists():
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})
            else:
                # It's a file
                if (self.project_path / indicator).exists():
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})

        return detected

    def _identify_patterns(self) -> list:
        """Identify architectural patterns."""
        patterns = []

        for pattern_dir, description in ARCHITECTURE_PATTERNS.items():
            if (self.project_path / pattern_dir.rstrip('/')).exists():
                patterns.append({'directory': pattern_dir, 'pattern': description})

        # Detect specific patterns
        if self._has_pattern(['domain/', 'infrastructure/', 'application/']):
            patterns.append({'pattern': 'Domain-Driven Design (DDD)', 'confidence': 'high'})
        elif self._has_pattern(['models/', 'views/', 'controllers/']):
            patterns.append({'pattern': 'MVC (Model-View-Controller)', 'confidence': 'high'})
        elif self._has_pattern(['components/', 'hooks/', 'pages/']):
            patterns.append({'pattern': 'React/Next.js Structure', 'confidence': 'high'})
        elif self._has_pattern(['services/', 'repositories/']):
            patterns.append({'pattern': 'Layered Architecture', 'confidence': 'medium'})

        return patterns

    def _has_pattern(self, dirs: list) -> bool:
        """Check if multiple directories from the pattern exist."""
        found = sum(1 for d in dirs if (self.project_path / d.rstrip('/')).exists())
        return found >= 2

    def _read_config_files(self) -> dict:
        """Read information from configuration files."""
        config = {}

        # README
        for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
            readme_path = self.project_path / readme_name
            if readme_path.exists():
                content = readme_path.read_text(encoding='utf-8', errors='ignore')
                # Extract first section (summary)
                lines = content.split('\n')
                summary_lines = []
                for line in lines[:30]:  # First 30 lines
                    if line.startswith('## ') and summary_lines:
                        break
                    summary_lines.append(line)
                config['readme_summary'] = '\n'.join(summary_lines)[:1000]
                break

        # pyproject.toml
        pyproject_path = self.project_path / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                import tomllib
                content = pyproject_path.read_text(encoding='utf-8')
                data = tomllib.loads(content)
                project_info = data.get('project', data.get('tool', {}).get('poetry', {}))
                config['project_info'] = {
                    'name': project_info.get('name'),
                    'description': project_info.get('description'),
                    'version': project_info.get('version'),
                    'python_requires': project_info.get('requires-python'),
                }
            except Exception:
                pass

        # package.json
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                data = json.loads(package_json_path.read_text(encoding='utf-8'))
                config['package_info'] = {
                    'name': data.get('name'),
                    'description': data.get('description'),
                    'version': data.get('version'),
                    'main': data.get('main'),
                    'scripts': list(data.get('scripts', {}).keys())[:10],
                }
            except Exception:
                pass

        return config

    def _analyze_dependencies(self) -> dict:
        """Analyze main dependencies."""
        deps = {'main': [], 'dev': []}

        # Python - requirements.txt
        req_path = self.project_path / 'requirements.txt'
        if req_path.exists():
            try:
                content = req_path.read_text(encoding='utf-8')
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#') and not line.startswith('-'):
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                        if pkg:
                            deps['main'].append(pkg)
            except Exception:
                pass

        # Python - pyproject.toml
        pyproject_path = self.project_path / 'pyproject.toml'
        if pyproject_path.exists():
            try:
                import tomllib
                content = pyproject_path.read_text(encoding='utf-8')
                data = tomllib.loads(content)
                project_deps = data.get('project', {}).get('dependencies', [])
                for dep in project_deps:
                    pkg = dep.split('==')[0].split('>=')[0].split('<=')[0].split('[')[0]
                    if pkg and pkg not in deps['main']:
                        deps['main'].append(pkg)
            except Exception:
                pass

        # Node.js - package.json
        package_json_path = self.project_path / 'package.json'
        if package_json_path.exists():
            try:
                data = json.loads(package_json_path.read_text(encoding='utf-8'))
                deps['main'].extend(list(data.get('dependencies', {}).keys())[:15])
                deps['dev'].extend(list(data.get('devDependencies', {}).keys())[:10])
            except Exception:
                pass

        return {
            'main': deps['main'][:20],  # Limitar
            'dev': deps['dev'][:10]
        }

    def _detect_conventions(self) -> dict:
        """Detect code conventions in use."""
        conventions = {}

        # Detect linting/formatting tools
        linting_tools = {
            '.eslintrc': 'ESLint',
            '.eslintrc.js': 'ESLint',
            '.eslintrc.json': 'ESLint',
            'eslint.config.js': 'ESLint (flat config)',
            '.prettierrc': 'Prettier',
            'prettier.config.js': 'Prettier',
            '.flake8': 'Flake8',
            'setup.cfg': 'Flake8/other (setup.cfg)',
            'pyproject.toml': 'Ruff/Black/isort (check pyproject)',
            '.editorconfig': 'EditorConfig',
            'tslint.json': 'TSLint (deprecated)',
            '.stylelintrc': 'Stylelint',
        }

        detected_tools = []
        for file, tool in linting_tools.items():
            if (self.project_path / file).exists():
                detected_tools.append(tool)

        conventions['linting_formatting'] = list(set(detected_tools))

        # Detect testing
        testing_tools = {
            'pytest.ini': 'pytest',
            'conftest.py': 'pytest',
            'tests/': 'Structured tests',
            'jest.config.js': 'Jest',
            'jest.config.ts': 'Jest',
            'vitest.config.js': 'Vitest',
            'vitest.config.ts': 'Vitest',
            'cypress.config.js': 'Cypress (E2E)',
            'playwright.config.ts': 'Playwright (E2E)',
        }

        detected_testing = []
        for file, tool in testing_tools.items():
            path = self.project_path / file.rstrip('/')
            if path.exists():
                detected_testing.append(tool)

        conventions['testing'] = list(set(detected_testing))

        return conventions

    def generate_onboarding_text(self, analysis: dict) -> str:
        """
        Generate structured text for the onboarding episode.

        Args:
            analysis: Analysis result

        Returns:
            Formatted text for capture
        """
        sections = []

        # Header
        sections.append(f"# Onboarding Analysis: {analysis['project_name']}")
        sections.append(f"\nPath: {analysis['project_path']}")

        # Technology stack
        if analysis['stack']:
            sections.append("\n## Technology Stack")
            for item in analysis['stack']:
                sections.append(
                    f"- **{item['technology']}**: {item['description']} "
                    f"(detected by `{item['indicator']}`)"
                )

        # Structure
        sections.append("\n## Project Structure")
        if analysis['structure']['main_directories']:
            sections.append("Main directories:")
            for d in analysis['structure']['main_directories']:
                sections.append(f"- `{d}/`")

        if analysis['structure']['file_types']:
            sections.append("\nMost common file types:")
            for ext, count in analysis['structure']['file_types'].items():
                sections.append(f"- `{ext}`: {count} files")

        # Architectural patterns
        if analysis['patterns']:
            sections.append("\n## Architectural Patterns")
            for p in analysis['patterns']:
                if 'confidence' in p:
                    sections.append(f"- **{p['pattern']}** (confidence: {p['confidence']})")
                else:
                    sections.append(f"- `{p['directory']}`: {p['pattern']}")

        # Project configuration
        if analysis['config']:
            sections.append("\n## Project Configuration")
            if 'project_info' in analysis['config']:
                info = analysis['config']['project_info']
                if info.get('name'):
                    sections.append(f"- Name: {info['name']}")
                if info.get('description'):
                    sections.append(f"- Description: {info['description']}")
                if info.get('python_requires'):
                    sections.append(f"- Python required: {info['python_requires']}")

            if 'readme_summary' in analysis['config']:
                sections.append("\n### README Summary:")
                sections.append(f"```\n{analysis['config']['readme_summary'][:500]}...\n```")

        # Dependencies
        if analysis['dependencies']['main']:
            sections.append("\n## Main Dependencies")
            sections.append(", ".join(analysis['dependencies']['main']))

        # Conventions
        if analysis['conventions']:
            sections.append("\n## Detected Conventions")
            if analysis['conventions'].get('linting_formatting'):
                sections.append(
                    f"- **Linting/Formatting**: "
                    f"{', '.join(analysis['conventions']['linting_formatting'])}"
                )
            if analysis['conventions'].get('testing'):
                sections.append(
                    f"- **Testing**: {', '.join(analysis['conventions']['testing'])}"
                )

        # Recommendations
        sections.append("\n## Recommendations for the Agent")
        sections.append("- Follow the detected code conventions")
        sections.append("- Respect the existing directory structure")
        if analysis['conventions'].get('testing'):
            sections.append("- Add tests for new code")
        sections.append("- Consult this memory before making architectural decisions")

        return "\n".join(sections)


async def onboard_project(
    project_path: str,
    project_name: Optional[str] = None,
    source_assistant: str = "analyzer"
) -> dict:
    """
    Run full project onboarding.

    Args:
        project_path: Path to the project
        project_name: Project name (auto-detected if not provided)
        source_assistant: Assistant identifier

    Returns:
        Created episode with the analysis
    """
    from memorytwin.escriba import Escriba

    analyzer = ProjectAnalyzer(project_path)
    analysis = analyzer.analyze()

    # Use detected name if not provided
    if not project_name:
        project_name = analysis['project_name']

    # Generate onboarding text
    onboarding_text = analyzer.generate_onboarding_text(analysis)

    # Capture as episode
    escriba = Escriba(project_name=project_name)
    episode = await escriba.capture_thinking(
        thinking_text=onboarding_text,
        user_prompt=f"Automatic onboarding of project {project_name}",
        source_assistant=source_assistant,
        project_name=project_name
    )

    return {
        'episode_id': str(episode.id),
        'project_name': project_name,
        'analysis': analysis,
        'onboarding_text': onboarding_text
    }
