"""
Global pytest configuration for Memory Twin.
"""
import os

# Disable Langfuse during tests to avoid noise in production traces
# We use an empty LANGFUSE_HOST so it fails silently when trying to connect
os.environ["LANGFUSE_HOST"] = ""
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
