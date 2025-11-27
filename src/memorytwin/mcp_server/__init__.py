"""
MCP Server - Model Context Protocol para Memory Twin
=====================================================

Servidor MCP que expone herramientas para capturar y
consultar memorias técnicas desde cualquier cliente
compatible (VS Code, Claude, etc.).
"""

# Import lazy para evitar RuntimeWarning al ejecutar como módulo
# El warning ocurre cuando se hace `python -m memorytwin.mcp_server.server`
# porque el __init__.py se ejecuta antes que server.py

def get_server():
    """Obtener instancia del servidor MCP (import lazy)."""
    from memorytwin.mcp_server.server import MemoryTwinMCPServer
    return MemoryTwinMCPServer

__all__ = ["get_server"]
