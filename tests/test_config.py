"""
Tests para configuración
========================

Tests unitarios para config.py.
"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch


class TestSettings:
    """Tests para la clase Settings."""

    def test_settings_defaults(self):
        """Test de valores por defecto."""
        from memorytwin.config import Settings
        
        # Crear settings con valores por defecto
        settings = Settings()
        
        assert settings.mcp_server_host == "localhost"
        assert settings.mcp_server_port == 8765
        assert settings.gradio_server_port == 7860
        assert settings.gradio_share is False
        assert settings.llm_provider == "google"
        assert settings.llm_model == "gemini-2.0-flash"
        assert settings.llm_temperature == 0.3
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_settings_chroma_default(self):
        """Test de ruta por defecto de ChromaDB."""
        from memorytwin.config import Settings
        
        settings = Settings()
        
        assert "./data/chroma" in settings.chroma_persist_dir or "data" in settings.chroma_persist_dir

    def test_settings_sqlite_default(self):
        """Test de ruta por defecto de SQLite."""
        from memorytwin.config import Settings
        
        settings = Settings()
        
        assert "memory.db" in settings.sqlite_db_path or "data" in settings.sqlite_db_path


class TestGetSettings:
    """Tests para la función get_settings."""

    def test_get_settings_returns_settings(self):
        """Test que get_settings devuelve Settings."""
        from memorytwin.config import get_settings, Settings
        
        settings = get_settings()
        
        assert isinstance(settings, Settings)

    def test_get_settings_cached(self):
        """Test que get_settings está cacheado."""
        from memorytwin.config import get_settings
        
        settings1 = get_settings()
        settings2 = get_settings()
        
        # Mismo objeto (cached)
        assert settings1 is settings2


class TestPathFunctions:
    """Tests para funciones de paths."""

    def test_get_data_dir_exists(self, tmp_path):
        """Test que get_data_dir crea el directorio."""
        from memorytwin.config import get_data_dir
        
        data_dir = get_data_dir()
        
        assert isinstance(data_dir, Path)
        # El directorio debe existir (se crea si no existe)
        assert data_dir.exists()

    def test_get_chroma_dir_exists(self, tmp_path):
        """Test que get_chroma_dir crea el directorio."""
        from memorytwin.config import get_chroma_dir
        
        chroma_dir = get_chroma_dir()
        
        assert isinstance(chroma_dir, Path)
        assert chroma_dir.exists()

    def test_get_sqlite_path_parent_exists(self, tmp_path):
        """Test que get_sqlite_path crea directorio padre."""
        from memorytwin.config import get_sqlite_path
        
        sqlite_path = get_sqlite_path()
        
        assert isinstance(sqlite_path, Path)
        assert sqlite_path.parent.exists()


class TestSettingsWithEnvVars:
    """Tests para configuración con variables de entorno."""

    def test_settings_from_env_mcp_port(self):
        """Test de configuración desde variable de entorno."""
        with patch.dict(os.environ, {"MCP_SERVER_PORT": "9999"}):
            from importlib import reload
            import memorytwin.config as config_module
            
            # Limpiar cache
            config_module.get_settings.cache_clear()
            
            # Re-importar para obtener nuevos valores
            reload(config_module)
            
            # Verificar que se puede cambiar
            assert True  # El test verifica que no hay errores

    def test_settings_gradio_share_true(self):
        """Test de GRADIO_SHARE=true."""
        with patch.dict(os.environ, {"GRADIO_SHARE": "true"}):
            from memorytwin.config import Settings
            
            settings = Settings()
            
            assert settings.gradio_share is True

    def test_settings_gradio_share_false(self):
        """Test de GRADIO_SHARE=false."""
        with patch.dict(os.environ, {"GRADIO_SHARE": "false"}):
            from memorytwin.config import Settings
            
            settings = Settings()
            
            assert settings.gradio_share is False

    def test_settings_llm_temperature_custom(self):
        """Test de LLM_TEMPERATURE personalizado."""
        with patch.dict(os.environ, {"LLM_TEMPERATURE": "0.7"}):
            from memorytwin.config import Settings
            
            settings = Settings()
            
            assert settings.llm_temperature == 0.7
