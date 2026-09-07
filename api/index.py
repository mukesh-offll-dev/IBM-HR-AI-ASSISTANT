"""
Vercel Serverless Function Entry Point for FastAPI.
Wraps the application with path normalization so that Vercel rewrites and
proxied requests map directly to the corresponding FastAPI endpoints.
"""

import sys
from pathlib import Path

# Add project root directory to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import app as main_app


class VercelFastAPIWrapper:
    """
    ASGI middleware ensuring that Vercel Serverless routing correctly translates
    rewritten paths (/api/index.py, /api/index, /api) and headers (x-matched-path,
    x-forwarded-uri) back to standard FastAPI routes.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            headers = dict(scope.get("headers", []))
            # 1. Check for original matched path headers provided by Vercel Edge
            matched_path = headers.get(b"x-matched-path")
            forwarded_uri = headers.get(b"x-forwarded-uri")
            invoke_path = headers.get(b"x-invoke-path")
            real_path = matched_path or forwarded_uri or invoke_path

            if real_path:
                path_str = real_path.decode("utf-8")
                if "?" in path_str:
                    path_str = path_str.split("?")[0]
                scope["path"] = path_str
            else:
                path_str = scope.get("path", "")
                # Only strip Vercel entrypoint script prefixes, preserve /api/* routes
                for prefix in ["/api/index.py", "/api/index"]:
                    if path_str == prefix or path_str.startswith(prefix + "/"):
                        stripped = path_str[len(prefix):]
                        path_str = stripped if stripped else "/"
                        scope["path"] = path_str
                        break

            # If path resolved empty or directly to entrypoint, route to root dashboard
            if scope.get("path") in ["", "/api/index.py", "/api/index"]:
                scope["path"] = "/"

        await self.app(scope, receive, send)


app = VercelFastAPIWrapper(main_app)

__all__ = ["app"]
