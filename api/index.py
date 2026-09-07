"""
Vercel Serverless Function Entry Point for FastAPI.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import the FastAPI application instance
from main import app

# Vercel looks for the ASGI 'app' object
__all__ = ["app"]
