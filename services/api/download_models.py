#!/usr/bin/env python3
"""
Model download script for Docker build
Downloads the required transformer models for stance, sentiment, and bias analysis
"""

import os
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_model(model_name, model_type="transformers"):
    """Download a specific model"""
    try:
        logger.info(f"Downloading {model_name}...")
        
        if model_type == "transformers":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"✅ Downloaded tokenizer for {model_name}")
            
            # Download model
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"✅ Downloaded model for {model_name}")
            
        elif model_type == "sentence_transformers":
            try:
                from sentence_transformers import SentenceTransformer
                
                # Download sentence transformer model
                model = SentenceTransformer(model_name)
                logger.info(f"✅ Downloaded sentence transformer model {model_name}")
            except Exception as e:
                logger.warning(f"⚠️  Sentence transformer download failed: {e}")
                # Fallback: just download the model files without loading
                from transformers import AutoTokenizer, AutoModel
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name)
                    logger.info(f"✅ Downloaded model files for {model_name}")
                except Exception as e2:
                    logger.error(f"❌ Fallback download also failed: {e2}")
                    return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {model_name}: {e}")
        return False

def download_spacy_model():
    """Download spaCy model"""
    try:
        logger.info("📦 Downloading spaCy model: en_core_web_sm")
        import subprocess
        result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            logger.info("✅ spaCy model downloaded successfully")
            return True
        else:
            logger.warning(f"⚠️  spaCy model download failed: {result.stderr}")
            return False
    except Exception as e:
        logger.warning(f"⚠️  spaCy model download failed: {e}")
        return False

def main():
    """Download all required models"""
    logger.info("🚀 Starting model download process...")
    
    # Create models directory
    models_dir = Path("/app/models")
    models_dir.mkdir(exist_ok=True)
    
    # Set environment variables for model caching
    os.environ["TRANSFORMERS_CACHE"] = str(models_dir)
    os.environ["HF_HOME"] = str(models_dir)
    os.environ["TORCH_HOME"] = str(models_dir)
    
    # Download spaCy model first
    spacy_success = download_spacy_model()
    
    # Define models to download
    models = [
        {
            "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "type": "transformers",
            "description": "Sentiment analysis model"
        },
        {
            "name": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
            "type": "transformers", 
            "description": "NLI model for stance detection"
        },
        {
            "name": "unitary/unbiased-toxic-roberta",
            "type": "transformers",
            "description": "Toxicity detection model"
        },
        {
            "name": "BAAI/bge-m3",
            "type": "sentence_transformers",
            "description": "Multilingual embedding model"
        },
        {
            "name": "BAAI/bge-reranker-base",
            "type": "sentence_transformers",
            "description": "Reranking model"
        },
        {
            "name": "sentence-transformers/all-MiniLM-L6-v2",
            "type": "transformers",
            "description": "Fallback embedding model"
        }
    ]
    
    # Download each model
    success_count = 0
    total_models = len(models) + 1  # +1 for spaCy model
    
    # Count spaCy success
    if spacy_success:
        success_count += 1
    
    for model_info in models:
        logger.info(f"📦 {model_info['description']}: {model_info['name']}")
        
        if download_model(model_info["name"], model_info["type"]):
            success_count += 1
        else:
            logger.warning(f"⚠️  Skipping {model_info['name']} due to download failure")
    
    # Summary
    logger.info(f"📊 Model download complete: {success_count}/{total_models} models downloaded successfully")
    
    if success_count == total_models:
        logger.info("🎉 All models downloaded successfully!")
        return True
    else:
        logger.warning(f"⚠️  {total_models - success_count} models failed to download")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
