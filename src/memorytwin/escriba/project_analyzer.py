"""
Project Analyzer - Análisis inicial de proyectos existentes
===========================================================

Genera un episodio de "onboarding" analizando la estructura,
stack tecnológico y convenciones de un proyecto existente.
"""

import os
import json
from pathlib import Path
from typing import Optional
from collections import Counter

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


# Patrones de archivos y carpetas a ignorar
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

# Detección de stack tecnológico por archivos característicos
STACK_INDICATORS = {
    # Python
    'pyproject.toml': ('Python', 'Configuración moderna de proyecto'),
    'setup.py': ('Python', 'Setup tradicional'),
    'requirements.txt': ('Python', 'Dependencias pip'),
    'Pipfile': ('Python', 'Pipenv'),
    'poetry.lock': ('Python', 'Poetry'),
    'manage.py': ('Django', 'Framework web Python'),
    
    # JavaScript/TypeScript
    'package.json': ('Node.js/JavaScript', 'Proyecto NPM'),
    'tsconfig.json': ('TypeScript', 'Configuración TypeScript'),
    'next.config.js': ('Next.js', 'Framework React SSR'),
    'next.config.mjs': ('Next.js', 'Framework React SSR'),
    'nuxt.config.js': ('Nuxt.js', 'Framework Vue SSR'),
    'vite.config.js': ('Vite', 'Build tool moderno'),
    'vite.config.ts': ('Vite', 'Build tool moderno'),
    'angular.json': ('Angular', 'Framework SPA'),
    'vue.config.js': ('Vue.js', 'Framework progresivo'),
    
    # Java/Kotlin
    'pom.xml': ('Java/Maven', 'Build con Maven'),
    'build.gradle': ('Java/Kotlin/Gradle', 'Build con Gradle'),
    
    # .NET
    '*.csproj': ('C#/.NET', 'Proyecto .NET'),
    '*.sln': ('C#/.NET', 'Solución Visual Studio'),
    
    # Go
    'go.mod': ('Go', 'Módulo Go'),
    
    # Rust
    'Cargo.toml': ('Rust', 'Proyecto Cargo'),
    
    # Docker/Infra
    'Dockerfile': ('Docker', 'Contenedorización'),
    'docker-compose.yml': ('Docker Compose', 'Orquestación de contenedores'),
    'docker-compose.yaml': ('Docker Compose', 'Orquestación de contenedores'),
    'kubernetes': ('Kubernetes', 'Orquestación K8s'),
    
    # CI/CD
    '.github/workflows': ('GitHub Actions', 'CI/CD'),
    '.gitlab-ci.yml': ('GitLab CI', 'CI/CD'),
    'Jenkinsfile': ('Jenkins', 'CI/CD'),
}

# Patrones de arquitectura comunes
ARCHITECTURE_PATTERNS = {
    'src/': 'Código fuente separado',
    'lib/': 'Librerías internas',
    'tests/': 'Tests unitarios',
    'test/': 'Tests unitarios',
    'spec/': 'Tests (estilo BDD)',
    'docs/': 'Documentación',
    'scripts/': 'Scripts de utilidad',
    'config/': 'Configuración separada',
    'migrations/': 'Migraciones de BD',
    'api/': 'Capa API',
    'models/': 'Modelos de datos',
    'controllers/': 'Controladores (MVC)',
    'views/': 'Vistas (MVC)',
    'services/': 'Servicios de negocio',
    'repositories/': 'Capa de acceso a datos',
    'domain/': 'Lógica de dominio (DDD)',
    'infrastructure/': 'Infraestructura (DDD)',
    'application/': 'Capa de aplicación (DDD)',
    'components/': 'Componentes (Frontend)',
    'pages/': 'Páginas (Next.js/Nuxt)',
    'hooks/': 'Custom hooks (React)',
    'utils/': 'Utilidades',
    'helpers/': 'Helpers',
}


class ProjectAnalyzer:
    """Analiza la estructura y características de un proyecto existente."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.project_name = self.project_path.name
        
    def analyze(self) -> dict:
        """
        Ejecutar análisis completo del proyecto.
        
        Returns:
            Diccionario con el análisis estructurado
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Analizando proyecto...", total=None)
            
            # Recopilar información
            progress.update(task, description="Escaneando estructura...")
            structure = self._analyze_structure()
            
            progress.update(task, description="Detectando stack tecnológico...")
            stack = self._detect_stack()
            
            progress.update(task, description="Identificando patrones...")
            patterns = self._identify_patterns()
            
            progress.update(task, description="Leyendo configuración...")
            config_info = self._read_config_files()
            
            progress.update(task, description="Analizando dependencias...")
            dependencies = self._analyze_dependencies()
            
            progress.update(task, description="Detectando convenciones...")
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
        """Analizar estructura de directorios y archivos."""
        dirs = []
        files = []
        file_extensions = Counter()
        
        for root, directories, filenames in os.walk(self.project_path):
            # Filtrar directorios ignorados
            directories[:] = [d for d in directories if d not in IGNORE_PATTERNS['directories']
                             and not d.endswith('.egg-info')]
            
            rel_root = Path(root).relative_to(self.project_path)
            
            # Solo primer nivel de profundidad para directorios principales
            if len(rel_root.parts) <= 1:
                for d in directories:
                    dirs.append(str(rel_root / d) if str(rel_root) != '.' else d)
            
            for f in filenames:
                if not any(f.endswith(ext) for ext in ['.pyc', '.pyo']):
                    ext = Path(f).suffix.lower()
                    if ext:
                        file_extensions[ext] += 1
        
        return {
            'main_directories': dirs[:20],  # Limitar
            'file_types': dict(file_extensions.most_common(10)),
            'total_dirs': len(dirs),
        }
    
    def _detect_stack(self) -> list:
        """Detectar stack tecnológico del proyecto."""
        detected = []
        
        for indicator, (tech, description) in STACK_INDICATORS.items():
            if indicator.startswith('*'):
                # Patrón glob
                pattern = indicator
                if list(self.project_path.glob(pattern)):
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})
            elif '/' in indicator:
                # Es una ruta
                if (self.project_path / indicator).exists():
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})
            else:
                # Es un archivo
                if (self.project_path / indicator).exists():
                    detected.append({'technology': tech, 'indicator': indicator, 'description': description})
        
        return detected
    
    def _identify_patterns(self) -> list:
        """Identificar patrones arquitectónicos."""
        patterns = []
        
        for pattern_dir, description in ARCHITECTURE_PATTERNS.items():
            if (self.project_path / pattern_dir.rstrip('/')).exists():
                patterns.append({'directory': pattern_dir, 'pattern': description})
        
        # Detectar patrones específicos
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
        """Verificar si existen múltiples directorios del patrón."""
        found = sum(1 for d in dirs if (self.project_path / d.rstrip('/')).exists())
        return found >= 2
    
    def _read_config_files(self) -> dict:
        """Leer información de archivos de configuración."""
        config = {}
        
        # README
        for readme_name in ['README.md', 'README.rst', 'README.txt', 'README']:
            readme_path = self.project_path / readme_name
            if readme_path.exists():
                content = readme_path.read_text(encoding='utf-8', errors='ignore')
                # Extraer primera sección (resumen)
                lines = content.split('\n')
                summary_lines = []
                for line in lines[:30]:  # Primeras 30 líneas
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
        """Analizar dependencias principales."""
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
        """Detectar convenciones de código usadas."""
        conventions = {}
        
        # Detectar herramientas de linting/formateo
        linting_tools = {
            '.eslintrc': 'ESLint',
            '.eslintrc.js': 'ESLint',
            '.eslintrc.json': 'ESLint',
            'eslint.config.js': 'ESLint (flat config)',
            '.prettierrc': 'Prettier',
            'prettier.config.js': 'Prettier',
            '.flake8': 'Flake8',
            'setup.cfg': 'Flake8/otros (setup.cfg)',
            'pyproject.toml': 'Ruff/Black/isort (revisar pyproject)',
            '.editorconfig': 'EditorConfig',
            'tslint.json': 'TSLint (deprecated)',
            '.stylelintrc': 'Stylelint',
        }
        
        detected_tools = []
        for file, tool in linting_tools.items():
            if (self.project_path / file).exists():
                detected_tools.append(tool)
        
        conventions['linting_formatting'] = list(set(detected_tools))
        
        # Detectar testing
        testing_tools = {
            'pytest.ini': 'pytest',
            'conftest.py': 'pytest',
            'tests/': 'Tests estructurados',
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
        Generar texto estructurado para el episodio de onboarding.
        
        Args:
            analysis: Resultado del análisis
            
        Returns:
            Texto formateado para captura
        """
        sections = []
        
        # Header
        sections.append(f"# Análisis de Onboarding: {analysis['project_name']}")
        sections.append(f"\nRuta: {analysis['project_path']}")
        
        # Stack tecnológico
        if analysis['stack']:
            sections.append("\n## Stack Tecnológico")
            for item in analysis['stack']:
                sections.append(f"- **{item['technology']}**: {item['description']} (detectado por `{item['indicator']}`)")
        
        # Estructura
        sections.append("\n## Estructura del Proyecto")
        if analysis['structure']['main_directories']:
            sections.append("Directorios principales:")
            for d in analysis['structure']['main_directories']:
                sections.append(f"- `{d}/`")
        
        if analysis['structure']['file_types']:
            sections.append("\nTipos de archivo más comunes:")
            for ext, count in analysis['structure']['file_types'].items():
                sections.append(f"- `{ext}`: {count} archivos")
        
        # Patrones arquitectónicos
        if analysis['patterns']:
            sections.append("\n## Patrones Arquitectónicos")
            for p in analysis['patterns']:
                if 'confidence' in p:
                    sections.append(f"- **{p['pattern']}** (confianza: {p['confidence']})")
                else:
                    sections.append(f"- `{p['directory']}`: {p['pattern']}")
        
        # Configuración del proyecto
        if analysis['config']:
            sections.append("\n## Configuración del Proyecto")
            if 'project_info' in analysis['config']:
                info = analysis['config']['project_info']
                if info.get('name'):
                    sections.append(f"- Nombre: {info['name']}")
                if info.get('description'):
                    sections.append(f"- Descripción: {info['description']}")
                if info.get('python_requires'):
                    sections.append(f"- Python requerido: {info['python_requires']}")
            
            if 'readme_summary' in analysis['config']:
                sections.append("\n### Resumen del README:")
                sections.append(f"```\n{analysis['config']['readme_summary'][:500]}...\n```")
        
        # Dependencias
        if analysis['dependencies']['main']:
            sections.append("\n## Dependencias Principales")
            sections.append(", ".join(analysis['dependencies']['main']))
        
        # Convenciones
        if analysis['conventions']:
            sections.append("\n## Convenciones Detectadas")
            if analysis['conventions'].get('linting_formatting'):
                sections.append(f"- **Linting/Formateo**: {', '.join(analysis['conventions']['linting_formatting'])}")
            if analysis['conventions'].get('testing'):
                sections.append(f"- **Testing**: {', '.join(analysis['conventions']['testing'])}")
        
        # Recomendaciones
        sections.append("\n## Recomendaciones para el Agente")
        sections.append("- Seguir las convenciones de código detectadas")
        sections.append("- Respetar la estructura de directorios existente")
        if analysis['conventions'].get('testing'):
            sections.append("- Añadir tests para nuevo código")
        sections.append("- Consultar esta memoria antes de tomar decisiones arquitectónicas")
        
        return "\n".join(sections)


async def onboard_project(
    project_path: str,
    project_name: Optional[str] = None,
    source_assistant: str = "analyzer"
) -> dict:
    """
    Ejecutar onboarding completo de un proyecto.
    
    Args:
        project_path: Ruta al proyecto
        project_name: Nombre del proyecto (se detecta si no se proporciona)
        source_assistant: Identificador del asistente
        
    Returns:
        Episodio creado con el análisis
    """
    from memorytwin.escriba import Escriba
    
    analyzer = ProjectAnalyzer(project_path)
    analysis = analyzer.analyze()
    
    # Usar nombre detectado si no se proporciona
    if not project_name:
        project_name = analysis['project_name']
    
    # Generar texto de onboarding
    onboarding_text = analyzer.generate_onboarding_text(analysis)
    
    # Capturar como episodio
    escriba = Escriba(project_name=project_name)
    episode = await escriba.capture_thinking(
        thinking_text=onboarding_text,
        user_prompt=f"Onboarding automático del proyecto {project_name}",
        source_assistant=source_assistant,
        project_name=project_name
    )
    
    return {
        'episode_id': str(episode.id),
        'project_name': project_name,
        'analysis': analysis,
        'onboarding_text': onboarding_text
    }
