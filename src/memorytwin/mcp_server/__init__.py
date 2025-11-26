"""
MCP Server - Model Context Protocol para Memory Twin
=====================================================

Servidor MCP que expone herramientas para capturar y
consultar memorias técnicas desde cualquier cliente
compatible (VS Code, Claude, etc.).
"""

from memorytwin.mcp_server.server import MemoryTwinMCPServer

__all__ = ["MemoryTwinMCPServer"]
