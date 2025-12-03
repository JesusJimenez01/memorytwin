"""
Configuración global de pytest para Memory Twin.
"""
import os

# Desactivar Langfuse durante tests para evitar ruido en trazas de producción
# Usamos LANGFUSE_HOST vacío para que falle silenciosamente al intentar conectar
os.environ["LANGFUSE_HOST"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
