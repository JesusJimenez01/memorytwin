"""
MCP Server - Model Context Protocol for Memory Twin
====================================================

MCP server that exposes tools for capturing and
querying technical memories from any compatible client
(VS Code, Claude, etc.).
"""

# Lazy import to avoid RuntimeWarning when running as module
# The warning occurs when running `python -m memorytwin.mcp_server.server`
# because __init__.py executes before server.py

def get_server():
    """Get MCP server instance (lazy import)."""
    from memorytwin.mcp_server.server import MemoryTwinMCPServer
    return MemoryTwinMCPServer

__all__ = ["get_server"]
