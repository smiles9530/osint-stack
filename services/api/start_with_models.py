#!/usr/bin/env python3
"""
Startup script that downloads models and starts the API server
"""

import asyncio
import subprocess
import sys
import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def download_models():
    """Download transformer models if not already present"""
    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)
    
    # Check if key models already exist
    required_models = [
        "models--cardiffnlp--twitter-roberta-base-sentiment-latest",
        "models--ynie--roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "models--unitary--unbiased-toxic-roberta",
        "models--BAAI--bge-m3"
    ]
    
    existing_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
    missing_models = [model for model in required_models if model not in existing_models]
    
    if not missing_models:
        logger.info("✅ All required models already downloaded, skipping download")
        return True
    
    logger.info(f"🚀 Downloading missing models: {missing_models}")
    try:
        result = subprocess.run([sys.executable, "download_models.py"], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info("✅ Models downloaded successfully")
            return True
        else:
            logger.warning(f"⚠️  Model download had issues: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("⚠️  Model download timed out, continuing without models")
        return False
    except Exception as e:
        logger.warning(f"⚠️  Model download failed: {e}, continuing without models")
        return False

def start_api():
    """Start the API server"""
    logger.info("🚀 Starting API server...")
    
    # Ensure the app directory is in Python path
    import sys
    from pathlib import Path
    app_dir = Path("/app")
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    
    try:
        # Use exec to replace the current process
        os.execv(sys.executable, [sys.executable, "-m", "uvicorn", "app.main:app", 
                                 "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"])
    except Exception as e:
        logger.error(f"❌ Failed to start API server: {e}")
        sys.exit(1)

async def main():
    """Main startup sequence"""
    logger.info("🚀 Starting OSINT Stack API with Advanced Analysis...")
    
    # Download models first
    await download_models()
    
    # Start API server
    start_api()

if __name__ == "__main__":
    asyncio.run(main())
