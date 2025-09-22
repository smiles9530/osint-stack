#!/usr/bin/env python3
"""
Simple startup script for testing without model downloads
"""

import os
import sys
from pathlib import Path

# Ensure the app directory is in Python path
app_dir = Path("/app")
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Set PYTHONPATH environment variable
os.environ["PYTHONPATH"] = "/app"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
